"""
02_train_mae.py
─────────────────────────────────────────────────────────────────────────────
BV-Vec Step 2: ViT-Small + Masked Autoencoder (MAE) 학습

[모델 구조]
 입력: [N, 64, 46]  (64 = 8×8 패치, 46 = 채널 수)
   |
   | patch_embed (Linear)
   v
 [N, 64, 256]  — 패치 임베딩 + 2D sincos 위치 임베딩
   |
   | 마스킹 (유효 패치의 75% → MASK 토큰으로 교체)
   v
 Encoder: TransformerEncoder (depth=6, heads=8, dim=256)
   |
 Decoder: TransformerEncoder (depth=2, heads=4, dim=128)
   |
 예측 헤드: Linear → [N, 64, 46]
   |
 손실: MSE(예측, 원본) — MASK된 유효 패치에만 적용

[S2Vec 대비 주요 차이점]
 - PAD 토큰 도입: 수도권 경계 불규칙으로 인한 빈 패치 처리
 - 입력 피처: 위성/OSM 대신 행정 통계 46채널
 - 임베딩 차원: 512 → 256 (데이터 규모 적합)
 - 학습률: 1e-4 고정 (S2Vec의 base_lr=1.5e-4 linear scaling과 유사)

실행:
    python 02_train_mae.py
    python 02_train_mae.py --epochs 50 --batch-size 32

출력:
    checkpoints/bv_vec_best.pt   # 최적 모델 체크포인트
"""

import math
import pickle
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ─── 경로 ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

NPZ_PATH  = DATA_DIR / "sample_grids.npz"
META_PATH = DATA_DIR / "sample_grids_meta.pkl"
CKPT_PATH = CKPT_DIR / "bv_vec_best.pt"

# ─── 하이퍼파라미터 ───────────────────────────────────────────────────────────
# S2Vec 논문 설정을 참고하되, 소규모 데이터에 맞게 조정
CFG = {
    "N_FEATURES":    46,    # 입력 채널 수
    "GRID_SIZE":     8,     # 8×8 = 64 패치 (800m 부모셀 / 100m 패치)
    "EMBED_DIM":     256,   # 임베딩 차원 (S2Vec: 512)
    "ENCODER_DEPTH": 6,     # 인코더 Transformer 레이어 수 (ViT-Small)
    "N_HEADS":       8,     # Multi-head Attention 헤드 수
    "DECODER_DIM":   128,   # 디코더 내부 차원
    "DECODER_DEPTH": 2,     # 디코더 Transformer 레이어 수
    "MASK_RATIO":    0.75,  # 마스킹 비율 (S2Vec 동일)
    "BATCH_SIZE":    32,    # 샘플 데이터용 (실제: 128)
    "EPOCHS":        100,   # 샘플 데이터용 (실제: 200)
    "LR":            1e-4,  # 학습률 (S2Vec base_lr=1.5e-4, batch 256 기준)
    "WEIGHT_DECAY":  0.05,  # AdamW 가중치 감쇠
    "BETAS":         (0.9, 0.95),  # AdamW 모멘텀 계수
    "WARMUP_EPOCHS": 10,    # 학습률 워밍업 에포크 (실제: 20)
    "VAL_RATIO":     0.1,   # 검증 데이터 비율
    "SEED":          42,
    "MIN_LR":        1e-6,  # 코사인 감쇠 최솟값
}


# ─── 2D Sinusoidal 위치 임베딩 ────────────────────────────────────────────────
def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int = 8) -> torch.Tensor:
    """
    2D 공간 구조를 반영한 고정 위치 임베딩 생성.
    Returns [grid_size*grid_size, embed_dim] — requires_grad=False

    일반 1D 위치 임베딩과 달리, 격자의 행(y)과 열(x)을 각각 인코딩해
    공간적 인접성을 보존합니다. (S2Vec에서도 동일 방식 사용)
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    half = embed_dim // 4
    # 주파수: 10000^(-k/half) for k=0..half-1
    omega = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))

    # 격자 좌표 (row=y, col=x)
    gy, gx = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    pos_x = torch.outer(gx.reshape(-1), omega)  # [64, half]
    pos_y = torch.outer(gy.reshape(-1), omega)

    # [sin(x), cos(x), sin(y), cos(y)] 연결
    return torch.cat([
        torch.sin(pos_x), torch.cos(pos_x),
        torch.sin(pos_y), torch.cos(pos_y),
    ], dim=-1)  # [64, embed_dim]


# ─── BV-Vec MAE 모델 ──────────────────────────────────────────────────────────
class BVVecMAE(nn.Module):
    """
    도시 격자 데이터용 Masked Autoencoder.

    S2Vec 구조를 참고해 설계됨:
     - 인코더: ViT-Small (Pre-LN Transformer)
     - 디코더: 경량 Transformer (복원 전용, 추론 시 불필요)
     - 마스킹: PAD 제외 유효 패치의 75% 무작위 제거
     - 임베딩: PAD-제외 평균 풀링 → 256차원 지역 벡터
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D   = cfg["EMBED_DIM"]      # 인코더 차원
        H   = cfg["N_HEADS"]        # Attention 헤드 수
        L   = cfg["ENCODER_DEPTH"]  # 인코더 깊이
        Dd  = cfg["DECODER_DIM"]    # 디코더 차원
        Ld  = cfg["DECODER_DEPTH"]  # 디코더 깊이
        F_  = cfg["N_FEATURES"]     # 입력 채널 수 (46)
        G   = cfg["GRID_SIZE"]      # 격자 크기 (8)

        # ── 1. 패치 임베딩: 46차원 피처 → D차원
        self.patch_embed = nn.Linear(F_, D)

        # ── 2. 고정 2D sincos 위치 임베딩 [1, 64, D]
        pos = build_2d_sincos_pos_embed(D, G)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        # ── 3. 특수 토큰
        # mask_token: MAE에서 가려진 패치를 나타내는 학습 가능 벡터
        # pad_token:  데이터 없는 위치(수도권 외곽 등)를 나타내는 벡터
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pad_token  = nn.Parameter(torch.zeros(1, 1, D))

        # ── 4. 인코더: Pre-LN Transformer (안정적 학습)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=H, dim_feedforward=D * 4,
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,  # Pre-LN
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=L)
        self.enc_norm = nn.LayerNorm(D)

        # ── 5. 디코더: 경량 Transformer (복원 전용)
        self.dec_proj = nn.Linear(D, Dd)  # 인코더 차원 → 디코더 차원 축소
        dec_layer = nn.TransformerEncoderLayer(
            d_model=Dd, nhead=max(1, H // 2), dim_feedforward=Dd * 4,
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder  = nn.TransformerEncoder(dec_layer, num_layers=Ld)
        self.dec_norm = nn.LayerNorm(Dd)
        self.dec_pred = nn.Linear(Dd, F_)  # 원본 피처 차원으로 복원

        self._init_weights()

    def _init_weights(self):
        """Truncated normal 초기화 (MAE 논문 권장)."""
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pad_token,  std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _apply_masking(
        self,
        x: torch.Tensor,        # [N, 64, D] 임베딩 후 입력
        pad_mask: torch.Tensor, # [N, 64] bool — True=PAD
        mask_ratio: float,
    ):
        """
        벡터화된 랜덤 마스킹.

        핵심 아이디어:
         - PAD 패치는 마스킹 대상에서 제외 (score=inf)
         - 유효 패치의 mask_ratio 만큼 MASK 토큰으로 교체
         - 배치 내 샘플마다 유효 패치 수가 다를 수 있으므로 per-sample 임계값 계산

        Args:
            x         : 패치 임베딩 텐서
            pad_mask  : PAD 위치 마스크
            mask_ratio: 유효 패치 중 가릴 비율 (0.75 = 75%)

        Returns:
            x         : MASK 토큰이 삽입된 텐서
            is_masked : [N, 64] bool — 실제로 마스킹된 위치
        """
        N, SEQ, D = x.shape

        # 유효 패치에만 랜덤 점수 부여 (PAD는 inf → 절대 선택 안됨)
        rand = torch.rand(N, SEQ, device=x.device)
        rand = rand.masked_fill(pad_mask, float("inf"))

        # 샘플별 마스킹할 패치 수 계산
        valid_counts = (~pad_mask).sum(dim=1).float()         # [N]
        n_mask = (valid_counts * mask_ratio).long().clamp(min=1)  # [N]

        # n_mask번째 점수 = 임계값 (이하인 패치를 마스킹)
        sorted_rand, _ = rand.sort(dim=1)
        thresholds = sorted_rand[torch.arange(N, device=x.device), n_mask - 1]  # [N]
        is_masked = rand <= thresholds.unsqueeze(1)   # [N, 64] bool

        # MASK 토큰 삽입: mask_token + 위치 임베딩 (위치 정보 유지)
        mask_emb = (self.mask_token + self.pos_embed).expand(N, -1, -1)  # [N, 64, D]
        x = torch.where(is_masked.unsqueeze(-1), mask_emb, x)

        return x, is_masked

    def forward(
        self,
        feats: torch.Tensor,     # [N, 64, F] — 정규화된 입력 피처
        pad_mask: torch.Tensor,  # [N, 64] bool
        mask_ratio: float = 0.75,
    ):
        """
        MAE forward pass (학습 시 사용).

        처리 흐름:
         입력 → 패치 임베딩 → PAD 치환 → 마스킹 → 인코더 → 디코더 → MSE 손실
        """
        N, SEQ, _ = feats.shape

        # 1. 패치 임베딩: [N, 64, F] → [N, 64, D]
        #    + 2D 위치 임베딩 추가 (공간 구조 인식)
        x = self.patch_embed(feats) + self.pos_embed

        # 2. PAD 위치를 pad_token으로 교체
        #    (PAD 패치도 위치 정보는 유지)
        pad_emb = (self.pad_token + self.pos_embed).expand(N, -1, -1)
        x = torch.where(pad_mask.unsqueeze(-1), pad_emb, x)

        # 3. 유효 패치의 75% 마스킹 (무작위)
        x, is_masked = self._apply_masking(x, pad_mask, mask_ratio)

        # 4. 인코더: PAD 위치는 key_padding_mask로 Attention에서 제외
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.enc_norm(x)

        # 5. 디코더: 인코더 출력 → 원본 피처 복원
        x = self.dec_proj(x)   # 차원 축소 (D → Dd)
        x = self.decoder(x, src_key_padding_mask=pad_mask)
        x = self.dec_norm(x)
        pred = self.dec_pred(x)  # [N, 64, F]

        # 6. 손실: MASK된 유효 패치에 대해서만 MSE 계산
        #    (PAD 위치 제외 — is_masked는 이미 PAD를 포함하지 않음)
        loss = F.mse_loss(pred[is_masked], feats[is_masked])

        return loss, pred

    @torch.no_grad()
    def encode(
        self,
        feats: torch.Tensor,    # [N, 64, F]
        pad_mask: torch.Tensor, # [N, 64] bool
    ) -> torch.Tensor:
        """
        임베딩 추출 (추론 시 사용, 마스킹 없음).

        PAD를 제외한 유효 패치들의 인코더 출력을 평균 풀링해
        256차원 지역 임베딩 벡터를 반환합니다.

        Returns: [N, EMBED_DIM] float32
        """
        N = feats.shape[0]

        x = self.patch_embed(feats) + self.pos_embed
        pad_emb = (self.pad_token + self.pos_embed).expand(N, -1, -1)
        x = torch.where(pad_mask.unsqueeze(-1), pad_emb, x)

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.enc_norm(x)

        # PAD 제외 평균 풀링
        valid = (~pad_mask).float().unsqueeze(-1)       # [N, 64, 1]
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return pooled  # [N, EMBED_DIM]


# ─── 데이터셋 ─────────────────────────────────────────────────────────────────
class GridDataset(Dataset):
    """
    부모셀 격자 데이터셋.

    입력 NPZ에서 [N, 8, 8, F] → [N, 64, F]로 reshape해 반환.
    """
    def __init__(self, images: np.ndarray, pad_masks: np.ndarray):
        N, H, W, C = images.shape
        # 공간 차원(8,8) → 시퀀스(64)로 flatten
        self.x        = torch.from_numpy(images.reshape(N, H * W, C))
        self.pad_mask = torch.from_numpy(pad_masks.reshape(N, H * W))

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.pad_mask[idx]


# ─── 코사인 워밍업 LR 스케줄러 ───────────────────────────────────────────────
class CosineWarmupScheduler:
    """
    선형 워밍업 + 코사인 감쇠 스케줄러.

    S2Vec 논문과 동일한 스케줄:
     - 처음 warmup 에포크: 0 → base_lr 선형 증가
     - 이후: base_lr → min_lr 코사인 감쇠
    """
    def __init__(self, optimizer, warmup, total, base_lr, min_lr):
        self.opt     = optimizer
        self.warmup  = warmup
        self.total   = total
        self.base    = base_lr
        self.min     = min_lr

    def step(self, epoch: int) -> float:
        if epoch < self.warmup:
            lr = self.base * (epoch + 1) / self.warmup
        else:
            p  = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * p))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ─── 학습 루프 ────────────────────────────────────────────────────────────────
def train(cfg: dict, npz_path: Path, meta_path: Path, ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[02] BV-Vec MAE 학습")
    print(f"     디바이스: {device}")
    if device.type == "cuda":
        print(f"     GPU: {torch.cuda.get_device_name(0)}")

    # ── 데이터 로드
    print(f"\n데이터 로드: {npz_path}")
    data      = np.load(npz_path)
    images    = data["images"]    # [N, 8, 8, 46] 정규화됨
    pad_masks = data["pad_masks"] # [N, 8, 8] bool

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    N = len(images)
    print(f"  셀 수: {N:,}  /  유효 패치 비율: "
          f"{(~pad_masks).sum()/(N*64)*100:.1f}%")

    # ── 학습/검증 분할
    torch.manual_seed(cfg["SEED"])
    ds     = GridDataset(images, pad_masks)
    n_val  = max(1, int(N * cfg["VAL_RATIO"]))
    n_tr   = N - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val])

    tr_dl  = DataLoader(tr_ds, batch_size=cfg["BATCH_SIZE"],
                        shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    val_dl = DataLoader(val_ds, batch_size=cfg["BATCH_SIZE"],
                        shuffle=False, num_workers=0)

    print(f"  학습: {n_tr}개 / 검증: {n_val}개")

    # ── 모델
    model = BVVecMAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n모델 파라미터: {n_params:,}")
    print(f"  EMBED_DIM={cfg['EMBED_DIM']}, ENCODER_DEPTH={cfg['ENCODER_DEPTH']}, "
          f"N_HEADS={cfg['N_HEADS']}")

    # ── 옵티마이저: AdamW (S2Vec 동일 설정)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
        betas=cfg["BETAS"],
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup=cfg["WARMUP_EPOCHS"],
        total=cfg["EPOCHS"],
        base_lr=cfg["LR"],
        min_lr=cfg["MIN_LR"],
    )

    # ── 학습
    best_val_loss = float("inf")
    print(f"\n학습 시작: {cfg['EPOCHS']} 에포크, LR={cfg['LR']}, "
          f"MASK_RATIO={cfg['MASK_RATIO']}")
    print("-" * 60)

    for epoch in range(cfg["EPOCHS"]):
        t0 = time.time()
        lr = scheduler.step(epoch)

        # -- train
        model.train()
        tr_loss = 0.0
        for x, pm in tr_dl:
            x, pm = x.to(device), pm.to(device)
            loss, _ = model(x, pm, cfg["MASK_RATIO"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)

        # -- validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, pm in val_dl:
                x, pm = x.to(device), pm.to(device)
                loss, _ = model(x, pm, cfg["MASK_RATIO"])
                val_loss += loss.item()
        val_loss /= max(len(val_dl), 1)

        elapsed = time.time() - t0

        # 체크포인트 저장 (val_loss 개선 시)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "cfg":         cfg,
            }, ckpt_path)
            star = " *"  # 최적 갱신 표시
        else:
            star = ""

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{cfg['EPOCHS']}  "
                  f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={lr:.2e}  {elapsed:.1f}s{star}")

    print("-" * 60)
    print(f"[완료] 최적 val_loss={best_val_loss:.4f}")
    print(f"  체크포인트: {ckpt_path}")
    print(f"\n다음 단계: python 03_extract_eval.py")


def main():
    parser = argparse.ArgumentParser(description="BV-Vec MAE 학습")
    parser.add_argument("--epochs",     type=int, default=CFG["EPOCHS"])
    parser.add_argument("--batch-size", type=int, default=CFG["BATCH_SIZE"])
    parser.add_argument("--lr",         type=float, default=CFG["LR"])
    args = parser.parse_args()

    cfg = dict(CFG)
    cfg["EPOCHS"]     = args.epochs
    cfg["BATCH_SIZE"] = args.batch_size
    cfg["LR"]         = args.lr

    if not NPZ_PATH.exists():
        print(f"데이터 없음: {NPZ_PATH}")
        print("먼저 01_generate_sample.py를 실행하세요.")
        return

    train(cfg, NPZ_PATH, META_PATH, CKPT_PATH)


if __name__ == "__main__":
    main()
