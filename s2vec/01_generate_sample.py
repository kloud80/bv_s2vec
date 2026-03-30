"""
01_generate_sample.py
─────────────────────────────────────────────────────────────────────────────
BV-Vec Step 1: 샘플 학습 데이터 생성

실제 환경에서는 BigValue DB(tb_grid_total)에서 수도권 100m 격자 데이터를
직접 추출하지만, 이 스크립트는 동일한 구조를 가진 합성(synthetic) 데이터를
생성합니다. 전국 단위 실데이터가 있다면 이 스크립트 대신 DB 추출 코드를
사용하세요.

────────────────────────────────────────────────────────────────────────────
[데이터 구조 설명]

좌표 체계: UTM-K (EPSG:5179)
 - 한국 표준 단일 투영 좌표계 (단위: 미터)
 - 수도권 범위: X 748,400 ~ 1,026,800 / Y 1,816,000 ~ 2,047,600 (근사)

격자 계층:
 ┌─────────────────────────────────────┐
 │ 부모셀 (800m × 800m)                │
 │  ┌──┬──┬──┬──┬──┬──┬──┬──┐         │
 │  │  │  │  │  │  │  │  │  │  8×8   │
 │  ├──┼──┼──┼──┼──┼──┼──┼──┤  =64   │
 │  │  │  │  │  │  │  │  │  │  패치  │
 │  ...                    ...         │
 │  └──┴──┴──┴──┴──┴──┴──┴──┘         │
 │  각 패치 = 100m × 100m              │
 └─────────────────────────────────────┘

 parent_x = cent_x // 800  (부모셀 X 인덱스)
 parent_y = cent_y // 800  (부모셀 Y 인덱스)
 local_x  = (cent_x % 800) // 100  (패치 내 위치 0-7)
 local_y  = (cent_y % 800) // 100

입력 데이터: [N, 8, 8, 46] float32
 - N:  부모셀 수 (실제: ~11,000개 / 샘플: 200개)
 - 46: 채널 수 (인구/주거/면적/종사자/용도/교통/이주)

PAD 마스크: [N, 8, 8] bool
 - True = 해당 100m 격자에 데이터 없음 (수도권 경계 외곽, DB 미존재)
 - 학습 시 PAD 패치는 손실 계산에서 제외

실행:
    python 01_generate_sample.py
    python 01_generate_sample.py --n-cells 500

출력:
    data/sample_grids.npz        # images + pad_masks
    data/sample_grids_meta.pkl   # 채널명, 통계, 좌표 정보
"""

import argparse
import pickle
import numpy as np
from pathlib import Path

# ─── 경로 ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ─── 채널 정의 (46개) ─────────────────────────────────────────────────────────
# 실제 DB 컬럼 구조: BigValue tb_grid_total 테이블의 주요 지표
CHANNEL_NAMES = [
    # [인구] 0-7: 총 8개
    "pop_total",           # 총인구
    "pop_working_age",     # 생산가능인구 (15-64세)
    "pop_young",           # 청년인구 (15-34세)
    "pop_middle",          # 중장년인구 (35-54세)
    "pop_senior",          # 노년인구 (55세 이상)
    "pop_income_resident", # 거주 기준 소득인구
    "pop_income_worker",   # 직장 기준 소득인구
    "pop_salary_worker",   # 급여 근로인구

    # [주거] 8-13: 총 6개
    "house_total",         # 총세대수
    "house_apartment",     # 아파트 세대
    "house_officetel",     # 오피스텔 세대
    "house_villa",         # 빌라/연립 세대
    "house_alone",         # 1인가구
    "house_city_living",   # 도시형생활주택

    # [면적] 14-17: 총 4개 (전용면적 구간별 세대수)
    "area_under10",        # 10평 미만
    "area_10to19",         # 10~19평
    "area_20to29",         # 20~29평
    "area_over30",         # 30평 이상

    # [종사자] 18-28: 총 11개
    "worker_total",        # 총종사자수
    "worker_food",         # 음식업 종사자
    "worker_retail",       # 소매업 종사자
    "worker_finance",      # 금융업 종사자
    "worker_realestate",   # 부동산업 종사자
    "worker_science",      # 전문/과학 종사자
    "worker_education",    # 교육업 종사자
    "worker_health",       # 보건/의료 종사자
    "worker_manufacturing",# 제조업 종사자
    "worker_construction", # 건설업 종사자
    "worker_transport",    # 운수업 종사자

    # [용도지역] 29-36: 총 8개 (면적 비율 0~1)
    "zoning_commerce_general",  # 일반상업지역
    "zoning_living_general",    # 일반주거지역
    "zoning_living_private",    # 전용주거지역
    "zoning_living_sub",        # 준주거지역
    "zoning_industry_general",  # 일반공업지역
    "zoning_industry_private",  # 전용공업지역
    "zoning_greenbelt",         # 녹지지역
    "zoning_management",        # 관리지역

    # [교통] 37-42: 총 6개 (반경별 역/정류장 수)
    "transit_subway_200",  # 반경 200m 이내 지하철역
    "transit_subway_400",  # 반경 400m 이내 지하철역
    "transit_subway_600",  # 반경 600m 이내 지하철역
    "transit_bus_200",     # 반경 200m 이내 버스정류장
    "transit_bus_400",     # 반경 400m 이내 버스정류장
    "transit_bus_600",     # 반경 600m 이내 버스정류장

    # [이주] 43-45: 총 3개
    "movein_fresh",        # 입주 1년 미만 세대
    "movein_mid",          # 입주 1~3년 세대
    "movein_old",          # 입주 3년 이상 세대
]

# 채널별 현실적 분포 파라미터 [평균, 표준편차, 최솟값]
# 수도권 실제 데이터 통계를 참고한 근사값
CHANNEL_PARAMS = [
    (1200, 800, 0),    (700, 400, 0),   (250, 180, 0),    (280, 160, 0),
    (200, 140, 0),     (400, 300, 0),   (350, 280, 0),    (300, 250, 0),
    (500, 350, 0),     (300, 280, 0),   (30, 40, 0),      (80, 90, 0),
    (60, 50, 0),       (20, 30, 0),
    (15, 20, 0),       (40, 35, 0),     (80, 60, 0),      (120, 100, 0),
    (800, 600, 0),     (60, 80, 0),     (50, 70, 0),      (20, 40, 0),
    (15, 30, 0),       (25, 45, 0),     (30, 40, 0),      (20, 35, 0),
    (40, 80, 0),       (30, 60, 0),     (20, 40, 0),
    (0.05, 0.1, 0),    (0.3, 0.2, 0),   (0.1, 0.1, 0),    (0.05, 0.08, 0),
    (0.05, 0.1, 0),    (0.02, 0.05, 0), (0.2, 0.2, 0),    (0.1, 0.15, 0),
    (0.2, 0.4, 0),     (0.5, 0.5, 0),   (1.0, 0.8, 0),
    (2.0, 1.5, 0),     (3.5, 2.0, 0),   (5.0, 2.5, 0),
    (20, 30, 0),       (40, 40, 0),     (100, 80, 0),
]

assert len(CHANNEL_NAMES) == len(CHANNEL_PARAMS) == 46


def generate_sample_data(n_cells: int, seed: int = 42) -> tuple:
    """
    합성 격자 데이터 생성.

    실제 수도권 격자는 행정 통계 DB에서 추출되지만,
    이 함수는 동일 구조의 합성 데이터를 생성합니다.

    Returns:
        images    : [N, 8, 8, 46] float32 — 정규화된 피처
        pad_masks : [N, 8, 8] bool       — True=PAD(데이터 없음)
        parent_ids: [(px, py), ...]       — 부모셀 좌표 인덱스
        ch_mean   : [46] float32          — 채널별 원본 평균
        ch_std    : [46] float32          — 채널별 원본 표준편차
    """
    rng = np.random.default_rng(seed)
    GRID = 8

    ch_mean = np.array([p[0] for p in CHANNEL_PARAMS], dtype=np.float32)
    ch_std  = np.array([p[1] for p in CHANNEL_PARAMS], dtype=np.float32).clip(min=1e-8)

    # ── 가상 수도권 격자 좌표 배치
    # 실제 수도권: parent_x ≈ 935~1283, parent_y ≈ 2270~2560
    BASE_X, BASE_Y = 1060, 2428  # 서울 중심 근방
    parent_ids = []
    used = set()
    for _ in range(n_cells * 3):  # 겹치지 않도록 여유있게 시도
        px = BASE_X + int(rng.integers(-30, 30))
        py = BASE_Y + int(rng.integers(-30, 30))
        key = (px, py)
        if key not in used:
            used.add(key)
            parent_ids.append(key)
        if len(parent_ids) == n_cells:
            break

    N = len(parent_ids)
    images    = np.zeros((N, GRID, GRID, 46), dtype=np.float32)
    pad_masks = np.zeros((N, GRID, GRID), dtype=bool)

    for n in range(N):
        # 지역 유형 프로파일: 고밀 주거 / 상업 / 외곽 등을 랜덤 혼합
        region_factor = rng.standard_normal(46).astype(np.float32) * 0.4

        # 부모셀 내 유효 패치 수: 실제 데이터는 수도권 경계로 인해 불완전
        n_valid = int(rng.integers(16, GRID * GRID + 1))
        valid_pos = rng.choice(GRID * GRID, n_valid, replace=False)

        for pos in valid_pos:
            r, c = divmod(int(pos), GRID)
            for f, (mean, std, mn) in enumerate(CHANNEL_PARAMS):
                # 지역 프로파일 + 패치 내 미세 변이
                raw_val = mean + std * (region_factor[f] + rng.standard_normal() * 0.3)
                images[n, r, c, f] = float(max(mn, raw_val))

        # PAD 마스크: 데이터 없는 패치
        for pos in range(GRID * GRID):
            if pos not in set(int(p) for p in valid_pos):
                r, c = divmod(pos, GRID)
                pad_masks[n, r, c] = True

        # 채널 정규화: (x - mean) / std
        # 정규화 후 PAD 위치는 0으로 (모델이 PAD를 구분하도록)
        for f in range(46):
            images[n, :, :, f] = (images[n, :, :, f] - ch_mean[f]) / ch_std[f]
            images[n, :, :, f] *= (~pad_masks[n]).astype(float)

    return images, pad_masks, parent_ids, ch_mean, ch_std


def main():
    parser = argparse.ArgumentParser(description="BV-Vec 샘플 데이터 생성")
    parser.add_argument("--n-cells", type=int, default=200,
                        help="생성할 부모셀 수 (기본: 200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[01] 샘플 데이터 생성: {args.n_cells}개 부모셀")
    print(f"     격자: 8x8 패치 / 채널: 46개 / 좌표계: UTM-K EPSG:5179")

    images, pad_masks, parent_ids, ch_mean, ch_std = generate_sample_data(
        args.n_cells, args.seed
    )

    N = len(parent_ids)
    valid_ratio = (~pad_masks).sum() / (N * 64) * 100
    print(f"     생성 완료: {N}개 셀, 유효 패치 비율 {valid_ratio:.1f}%")

    # ── 저장
    npz_path  = DATA_DIR / "sample_grids.npz"
    meta_path = DATA_DIR / "sample_grids_meta.pkl"

    np.savez_compressed(
        npz_path,
        images=images,
        pad_masks=pad_masks,
    )

    meta = {
        "channel_names": CHANNEL_NAMES,
        "channel_mean":  ch_mean,
        "channel_std":   ch_std,
        "parent_ids":    parent_ids,
        "n_parents":     N,
        "grid_size":     8,
        "n_features":    46,
        "description":   "BV-Vec 합성 샘플 데이터. 실제 수도권 격자 통계를 참고해 생성.",
        "coord_system":  "UTM-K EPSG:5179",
        "cell_size_m":   800,
        "patch_size_m":  100,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    sz = npz_path.stat().st_size / 1e6
    print(f"\n  -> {npz_path}  ({sz:.2f} MB)")
    print(f"  -> {meta_path}")
    print(f"\n[done] 다음 단계: python 02_train_mae.py")


if __name__ == "__main__":
    main()
