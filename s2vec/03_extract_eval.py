"""
03_extract_eval.py
─────────────────────────────────────────────────────────────────────────────
BV-Vec Step 3: 임베딩 추출 + K-Means 클러스터링 + 유사지역 검색 평가

[처리 흐름]
 학습된 MAE 모델
   |
   | encode() — 마스킹 없이 full forward → PAD 제외 평균 풀링
   v
 [N, 256] 임베딩 → L2 정규화 → 코사인 유사도 검색 가능
   |
   | K-Means 클러스터링 (k=10)
   v
 clustered_k10.parquet — 각 셀의 클러스터 ID + 좌표

[왜 L2 정규화?]
 L2 정규화 후 내적 = 코사인 유사도
 → 단순 행렬 곱(emb @ query)으로 전체 셀 유사도 일괄 계산 가능 (O(N))

실행:
    python 03_extract_eval.py
    python 03_extract_eval.py --n-clusters 15

출력:
    embeddings/embeddings_raw.npy       # [N, 256] 원본 임베딩
    embeddings/embeddings_norm.npy      # [N, 256] L2 정규화 임베딩
    embeddings/embedding_map.parquet    # 인덱스 ↔ UTM 좌표 매핑
    embeddings/clustered_k10.parquet    # K-Means 클러스터 결과
"""

import argparse
import pickle
import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize

# ─── 경로 ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
EMB_DIR  = ROOT / "embeddings"
EMB_DIR.mkdir(exist_ok=True)

NPZ_PATH  = DATA_DIR / "sample_grids.npz"
META_PATH = DATA_DIR / "sample_grids_meta.pkl"
CKPT_PATH = CKPT_DIR / "bv_vec_best.pt"


# ─── 임베딩 추출 ──────────────────────────────────────────────────────────────
def extract_embeddings(
    model,
    images: np.ndarray,    # [N, 8, 8, F]
    pad_masks: np.ndarray, # [N, 8, 8]
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    학습된 MAE 인코더로 임베딩 추출.

    마스킹 없이 전체 패치를 인코더에 통과시킨 후,
    PAD 제외 평균 풀링으로 셀당 1개 벡터 생성.

    Returns: [N, EMBED_DIM] float32
    """
    N, H, W, C = images.shape

    # [N, 8, 8, 46] → [N, 64, 46]
    x_t  = torch.from_numpy(images.reshape(N, H * W, C))
    pm_t = torch.from_numpy(pad_masks.reshape(N, H * W))

    ds = TensorDataset(x_t, pm_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_emb = []
    with torch.no_grad():
        for x_batch, pm_batch in dl:
            x_batch  = x_batch.to(device)
            pm_batch = pm_batch.to(device)
            emb = model.encode(x_batch, pm_batch)  # [B, EMBED_DIM]
            all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb, axis=0).astype(np.float32)


# ─── 좌표 매핑 테이블 ─────────────────────────────────────────────────────────
def build_embedding_map(parent_ids: list) -> pd.DataFrame:
    """
    parent_ids [(px, py), ...] → UTM 좌표 + 임베딩 인덱스 매핑 DataFrame

    좌표 변환:
     center_x = px * 800 + 400  (부모셀 중심 UTM-X)
     center_y = py * 800 + 400  (부모셀 중심 UTM-Y)

    이후 pyproj로 EPSG:5179 → EPSG:4326(WGS84) 변환해 지도에 표시.
    """
    rows = []
    for idx, (px, py) in enumerate(parent_ids):
        rows.append({
            "emb_idx":      idx,
            "parent_x":     px,
            "parent_y":     py,
            "center_x_utm": px * 800 + 400,  # UTM-K 미터 단위 중심 좌표
            "center_y_utm": py * 800 + 400,
        })
    return pd.DataFrame(rows)


# ─── K-Means 클러스터링 ───────────────────────────────────────────────────────
def cluster_embeddings(
    embeddings_norm: np.ndarray,  # L2 정규화된 임베딩 [N, 256]
    n_clusters: int,
    emb_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    코사인 공간에서 K-Means 클러스터링.

    L2 정규화 후 유클리드 거리 = 코사인 거리이므로
    표준 K-Means로 코사인 클러스터링 효과를 냄.

    N > 50,000이면 MiniBatchKMeans 자동 전환.
    """
    print(f"\nK-Means 클러스터링: k={n_clusters}")

    if len(embeddings_norm) > 50_000:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                             batch_size=4096, n_init=10)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    labels = km.fit_predict(embeddings_norm)
    result = emb_map.copy()
    result["cluster"] = labels
    return result


# ─── 코사인 유사도 검색 ───────────────────────────────────────────────────────
def find_similar(
    query_emb: np.ndarray,        # [EMBED_DIM] 쿼리 임베딩
    embeddings_norm: np.ndarray,  # [N, EMBED_DIM] L2 정규화됨
    emb_map: pd.DataFrame,
    top_k: int = 5,
    exclude_idx: int = -1,
) -> pd.DataFrame:
    """
    행렬 곱 기반 코사인 유사도 검색 (O(N·D)).

    L2 정규화된 임베딩 간 내적 = 코사인 유사도이므로
    단순 행렬 곱으로 전체 셀과의 유사도를 한 번에 계산.
    """
    # 쿼리도 L2 정규화
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

    # 전체 셀과 코사인 유사도 계산
    sims = embeddings_norm @ q_norm  # [N] — O(N·D) 행렬 곱

    # 쿼리 셀 자신 제외
    if exclude_idx >= 0:
        sims[exclude_idx] = -999.0

    top_indices = np.argsort(sims)[::-1][:top_k]
    result = emb_map.iloc[top_indices].copy()
    result["cosine_sim"] = sims[top_indices]
    return result.reset_index(drop=True)


# ─── 정성 평가 ────────────────────────────────────────────────────────────────
def qualitative_eval(embeddings_norm: np.ndarray, emb_map: pd.DataFrame):
    """
    랜덤 셀 선택 → top-5 유사 셀 출력.

    실제 환경에서는 강남역, 여의도 등 주요 랜드마크 좌표를 지정하지만,
    샘플 데이터에서는 랜덤 셀로 대체합니다.
    """
    print("\n" + "=" * 60)
    print("정성 평가: 유사 셀 검색")
    print("=" * 60)

    rng = np.random.default_rng(0)
    sample_idxs = rng.choice(len(emb_map), size=min(3, len(emb_map)), replace=False)

    for idx in sample_idxs:
        row = emb_map.iloc[idx]
        similar = find_similar(
            embeddings_norm[idx], embeddings_norm, emb_map,
            top_k=6, exclude_idx=idx
        )
        print(f"\n[쿼리 셀 #{idx}]  UTM=({int(row['center_x_utm'])}, {int(row['center_y_utm'])})")
        print("  Top-5 유사 셀:")
        for _, r in similar.head(5).iterrows():
            print(f"    idx={int(r['emb_idx']):4d} | "
                  f"UTM=({int(r['center_x_utm'])},{int(r['center_y_utm'])}) | "
                  f"cos={r['cosine_sim']:.4f}")


# ─── 진입점 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BV-Vec 임베딩 추출 및 클러스터링")
    parser.add_argument("--n-clusters", type=int, default=10,
                        help="K-Means 클러스터 수 (기본: 10)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[03] 임베딩 추출 + 클러스터링")
    print(f"     디바이스: {device}")

    # ── 데이터 로드
    print(f"\n데이터 로드: {NPZ_PATH}")
    data      = np.load(NPZ_PATH)
    images    = data["images"]
    pad_masks = data["pad_masks"]
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    parent_ids = meta["parent_ids"]
    print(f"  셀 수: {len(images):,}")

    # ── 모델 로드 (02_train_mae.py의 BVVecMAE 동적 임포트)
    print(f"\n모델 로드: {CKPT_PATH}")
    if not CKPT_PATH.exists():
        print(f"체크포인트 없음: {CKPT_PATH}")
        print("먼저 02_train_mae.py를 실행하세요.")
        return

    spec   = importlib.util.spec_from_file_location("train_mae", ROOT / "02_train_mae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_mae"] = module
    spec.loader.exec_module(module)
    BVVecMAE = module.BVVecMAE

    ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg   = ckpt["cfg"]
    model = BVVecMAE(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")

    # ── 임베딩 추출 (마스킹 없이 full forward)
    print("\n임베딩 추출 중...")
    embeddings = extract_embeddings(model, images, pad_masks, device, args.batch_size)
    print(f"  shape: {embeddings.shape}")

    # ── L2 정규화 (코사인 유사도 검색을 내적으로 근사)
    embeddings_norm = normalize(embeddings, norm="l2")

    # ── 좌표 매핑 테이블
    emb_map = build_embedding_map(parent_ids)

    # ── 저장
    np.save(EMB_DIR / "embeddings_raw.npy",  embeddings)
    np.save(EMB_DIR / "embeddings_norm.npy", embeddings_norm)
    emb_map.to_parquet(EMB_DIR / "embedding_map.parquet", index=False)
    print(f"\n저장 완료:")
    print(f"  embeddings_raw.npy   {embeddings.nbytes/1e6:.1f} MB")
    print(f"  embeddings_norm.npy  {embeddings_norm.nbytes/1e6:.1f} MB")
    print(f"  embedding_map.parquet")

    # ── K-Means 클러스터링
    clustered = cluster_embeddings(embeddings_norm, args.n_clusters, emb_map)
    cluster_path = EMB_DIR / f"clustered_k{args.n_clusters}.parquet"
    clustered.to_parquet(cluster_path, index=False)
    print(f"\n클러스터 분포 (k={args.n_clusters}):")
    for k, cnt in clustered["cluster"].value_counts().sort_index().items():
        bar = "=" * (cnt // max(1, len(clustered) // 40))
        print(f"  C{k:2d}: {cnt:5d}개  {bar}")

    # ── 정성 평가
    qualitative_eval(embeddings_norm, emb_map)

    print(f"\n[done] 03_extract_eval.py")
    print(f"웹 시각화: webs2vec/ 폴더 참고")


if __name__ == "__main__":
    main()
