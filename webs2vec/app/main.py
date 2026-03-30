"""
webs2vec — BV-Vec 격자 임베딩 시각화 서버
─────────────────────────────────────────────────────────────────────────────

수도권 약 11,000개 부모셀(800m×800m)의 256차원 임베딩을 Mapbox 지도 위에
시각화하고, 유사 지역 검색(코사인 유사도)을 제공하는 FastAPI 서버.

[데이터 파일 구조]
  data/embeddings/
    embeddings_norm_202603_part0.npy   # [N/2, 256] L2 정규화 임베딩 (분할)
    embeddings_norm_202603_part1.npy   # [N/2, 256]
    clustered_202603_k10.parquet       # K-Means k=10 클러스터 결과
    embedding_map_202603.parquet       # 인덱스 ↔ UTM 좌표 매핑
  data/meta/
    feat_norm_202603.npy               # [N, 46] 정규화 채널 피처
    feat_raw_202603.npy                # [N, 46] 원본(역정규화) 채널 피처
    capital_bvvec_202603_meta.pkl      # 채널명, 통계 정보

[API 엔드포인트]
  GET  /                    → index.html (Mapbox 프론트엔드)
  GET  /api/config          → 설정, 클러스터 정보, 모델 메타데이터
  GET  /api/cells.json      → 전체 셀 GeoJSON (Mapbox 렌더링용)
  GET  /api/cell/{emb_idx}  → 단일 셀 피처 상세
  GET  /api/cluster-profiles→ 클러스터별 해석 정보
  POST /api/similar         → 유사 지역 검색

실행:
  uvicorn app.main:app --host 0.0.0.0 --port 9030 --reload
  python app/main.py
"""

from contextlib import asynccontextmanager
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyproj import Transformer

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR  = DATA_DIR / "embeddings"
META_DIR = DATA_DIR / "meta"
STATIC   = Path(__file__).parent / "static"

# ─── Mapbox 토큰 ──────────────────────────────────────────────────────────────
# https://account.mapbox.com/ 에서 발급받아 아래에 입력하세요.
MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN_HERE"

# ─── 좌표 변환기 ──────────────────────────────────────────────────────────────
# UTM-K (EPSG:5179) ↔ WGS84 (EPSG:4326)
# always_xy=True: 모든 CRS에서 (경도, 위도) 순서 강제
T_FWD = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
T_INV = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

# ─── 클러스터 색상 (최대 10개) ────────────────────────────────────────────────
CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#8dd3c7", "#bebada", "#fb8072",
]

# ─── 채널 그룹 정의 ───────────────────────────────────────────────────────────
# 46개 채널을 7개 의미 그룹으로 분류
CHANNEL_GROUPS = {
    "Population": list(range(0, 8)),   # 인구 (8채널)
    "Housing":    list(range(8, 14)),  # 주거 (6채널)
    "Area":       list(range(14, 18)), # 면적 (4채널)
    "Workers":    list(range(18, 29)), # 종사자 (11채널)
    "Zoning":     list(range(29, 37)), # 용도지역 (8채널)
    "Transit":    list(range(37, 43)), # 교통 (6채널)
    "Move-in":    list(range(43, 46)), # 이주 (3채널)
}
GROUP_COLORS = {
    "Population": "#4e79a7",
    "Housing":    "#f28e2b",
    "Area":       "#e15759",
    "Workers":    "#76b7b2",
    "Zoning":     "#59a14f",
    "Transit":    "#edc948",
    "Move-in":    "#b07aa1",
}

# ─── 채널 한국어 라벨 ─────────────────────────────────────────────────────────
CHANNEL_LABELS_KR = {
    "pop_total":           "총인구",
    "pop_working_age":     "생산가능인구",
    "pop_young":           "청년인구(15-34)",
    "pop_middle":          "중장년인구(35-54)",
    "pop_senior":          "노년인구(55+)",
    "pop_income_resident": "거주 소득인구",
    "pop_income_worker":   "직장 소득인구",
    "pop_salary_worker":   "급여 근로인구",
    "house_total":         "총세대수",
    "house_apartment":     "아파트 세대",
    "house_officetel":     "오피스텔 세대",
    "house_villa":         "빌라/연립 세대",
    "house_alone":         "1인가구",
    "house_city_living":   "도시형생활주택",
    "area_under10":        "10평 미만",
    "area_10to19":         "10~19평",
    "area_20to29":         "20~29평",
    "area_over30":         "30평 이상",
    "worker_total":        "총종사자수",
    "worker_food":         "음식업 종사자",
    "worker_retail":       "소매업 종사자",
    "worker_finance":      "금융업 종사자",
    "worker_realestate":   "부동산업 종사자",
    "worker_science":      "전문/과학 종사자",
    "worker_education":    "교육업 종사자",
    "worker_health":       "보건/의료 종사자",
    "worker_manufacturing":"제조업 종사자",
    "worker_construction": "건설업 종사자",
    "worker_transport":    "운수업 종사자",
    "zoning_commerce_general":  "일반상업지역",
    "zoning_living_general":    "일반주거지역",
    "zoning_living_private":    "전용주거지역",
    "zoning_living_sub":        "준주거지역",
    "zoning_industry_general":  "일반공업지역",
    "zoning_industry_private":  "전용공업지역",
    "zoning_greenbelt":         "녹지지역",
    "zoning_management":        "관리지역",
    "transit_subway_200":  "지하철 200m",
    "transit_subway_400":  "지하철 400m",
    "transit_subway_600":  "지하철 600m",
    "transit_bus_200":     "버스 200m",
    "transit_bus_400":     "버스 400m",
    "transit_bus_600":     "버스 600m",
    "movein_fresh":        "입주 1년 미만",
    "movein_mid":          "입주 1~3년",
    "movein_old":          "입주 3년 이상",
}

# ─── 그룹별 한국어 해석 ───────────────────────────────────────────────────────
GROUP_INTERPRETATIONS = {
    "Population": (
        "고밀 주거지",
        "인구 밀도와 세대 구성이 두드러진 주거 특화 지역."
    ),
    "Housing": (
        "공동주택 밀집지",
        "아파트·공동주택 세대수 비중과 주거 밀도가 높은 지역."
    ),
    "Area": (
        "대형 평형 주거지",
        "전용면적 30평 이상 대형 주택 비중이 두드러진 고급 주거 지역."
    ),
    "Workers": (
        "도심 업무·상업지",
        "종사자 수와 사업체 밀도가 높은 업무·상업 중심지."
    ),
    "Zoning": (
        "복합 용도지구",
        "상업·준주거 등 혼합 용도 비율이 높은 지역."
    ),
    "Transit": (
        "교통 요충지 (역세권)",
        "지하철·버스 정류장 반경 400m 이내 접근성이 우수한 교통 중심 지역."
    ),
    "Move-in": (
        "신흥 주거지",
        "최근 1년 내 입주 세대 비율이 높은 신개발·재개발 지역."
    ),
}


# ─── 클러스터 이름 부여 ───────────────────────────────────────────────────────
def _name_cluster(group_z: dict) -> tuple:
    """
    7개 그룹의 z-score 패턴으로 클러스터 특성 이름 결정.

    z-score: 전체 셀 평균 대비 표준편차 단위 차이
      > +1.0: 매우 높음 / +0.5 ~ +1.0: 높음 / -0.5 ~ +0.5: 평균 수준
      < -0.5: 낮음
    """
    dominant = max(group_z, key=lambda k: group_z[k])
    max_z    = group_z[dominant]
    mean_z   = sum(group_z.values()) / len(group_z)

    # 전체적으로 낮은 경우
    if mean_z < -0.25:
        return ("저밀 외곽 지역", "전반적으로 모든 지표가 평균 이하인 도시 외곽 지역.")
    if mean_z < -0.1 and max_z < 0.2:
        return ("평균 이하 주변부", "대부분의 지표가 수도권 평균보다 낮은 중소 밀도 지역.")

    # 전체 평균 수준
    if abs(mean_z) < 0.08 and max_z < 0.15:
        return ("전형적 평균 지역", "모든 지표가 수도권 평균 수준에 해당하는 일반 주거지역.")

    # 약한 특성
    if max_z < 0.25:
        return ("저밀 혼합 지역", "지표 전반이 평균 이하이나 용도혼합 특성이 일부 나타나는 지역.")

    # 역세권 + 고밀
    if group_z.get("Transit", 0) > 1.5 and (
        group_z.get("Population", 0) > 1.0 or group_z.get("Housing", 0) > 1.0
    ):
        return ("초고밀 역세권", "지하철 접근성과 인구·주거 밀도가 수도권 최상위 수준인 초고밀 역세권.")

    if group_z.get("Transit", 0) > 0.5 and group_z.get("Workers", 0) > 0.4:
        return ("역세권 업무복합지", "지하철 접근성과 종사자 밀도가 높은 역세권 상업·업무 복합지역.")

    # 주거 고밀
    if group_z.get("Population", 0) > 0.8 and group_z.get("Housing", 0) > 0.8:
        return ("아파트 밀집 주거지", "인구·공동주택 세대수가 모두 높은 대단지 아파트 밀집 지역.")

    if group_z.get("Population", 0) > 0.4 and group_z.get("Housing", 0) > 0.4:
        return ("인구·주택 밀집지", "인구 밀도와 공동주택 세대수가 높은 도시 주거 밀집 지역.")

    # 도심 복합
    if (group_z.get("Workers", 0) + group_z.get("Zoning", 0) + group_z.get("Transit", 0)) > 1.5:
        return ("도심 복합상업지", "종사자·용도지역·교통이 복합적으로 높은 도심 상업 중심지.")

    return GROUP_INTERPRETATIONS.get(dominant, ("일반 지역", "특징 없음"))


def compute_cluster_profiles(
    feat_raw: np.ndarray,
    df: pd.DataFrame,
    channel_names: list,
) -> dict:
    """
    클러스터별 특징 프로파일 계산.

    각 클러스터의 채널별 평균을 전체 평균·표준편차로 z-score 변환해
    어떤 특성이 두드러지는지 정량화합니다.

    Returns:
        {
          "0": {
            "name": "초고밀 역세권",
            "desc": "...",
            "group_z": {"Population": 1.67, ...},
            "top_features": [{"name": "transit_subway_200", "z": 2.3}, ...],
            ...
          },
          ...
        }
    """
    global_mean = feat_raw.mean(axis=0)                           # [46]
    global_std  = feat_raw.std(axis=0).clip(min=1e-8)             # [46]

    profiles = {}
    for cid in sorted(df["cluster"].unique()):
        idxs         = df[df["cluster"] == cid]["emb_idx"].values
        cluster_mean = feat_raw[idxs].mean(axis=0)                # [46]
        z_scores     = (cluster_mean - global_mean) / global_std  # [46]

        # 그룹별 평균 z-score
        group_z = {
            gname: float(z_scores[indices].mean())
            for gname, indices in CHANNEL_GROUPS.items()
        }

        name, desc = _name_cluster(group_z)

        # 절대 z-score 상위 5개 채널
        top_idx = np.argsort(np.abs(z_scores))[::-1][:5]
        top_features = [
            {
                "name":  channel_names[i],
                "z":     round(float(z_scores[i]), 2),
                "mean":  round(float(cluster_mean[i]), 3),
                "group": next(
                    (g for g, il in CHANNEL_GROUPS.items() if i in il), "?"
                ),
            }
            for i in top_idx
        ]

        profiles[str(int(cid))] = {
            "name":           name,
            "desc":           desc,
            "group_z":        {k: round(v, 2) for k, v in group_z.items()},
            "dominant_group": max(group_z, key=lambda k: group_z[k]),
            "top_features":   top_features,
            "mean_raw":       [round(float(v), 3) for v in cluster_mean],
            "z_scores":       [round(float(v), 3) for v in z_scores],
            "count":          int(len(idxs)),
        }

    return profiles


# ─── GeoJSON 빌드 (벡터화) ────────────────────────────────────────────────────
def build_geojson(df: pd.DataFrame) -> str:
    """
    부모셀 DataFrame → Mapbox용 GeoJSON 문자열.

    각 부모셀(800m×800m)을 직사각형 폴리곤으로 변환.
    UTM-K 좌표에서 WGS84로 일괄 변환 (pyproj 벡터 연산).
    """
    half = 400  # 부모셀 반변 = 400m
    cx = df["center_x_utm"].values.astype(float)
    cy = df["center_y_utm"].values.astype(float)

    # 5개 꼭짓점 (SW → SE → NE → NW → SW 닫힘)
    px = np.stack([cx-half, cx+half, cx+half, cx-half, cx-half], axis=1)
    py = np.stack([cy-half, cy-half, cy+half, cy+half, cy-half], axis=1)

    # 일괄 좌표 변환: UTM-K → WGS84
    lons, lats = T_FWD.transform(px.ravel(), py.ravel())
    lons = lons.reshape(-1, 5)
    lats = lats.reshape(-1, 5)

    features = []
    records = df[["emb_idx", "cluster", "center_x_utm", "center_y_utm",
                  "parent_x", "parent_y"]].to_dict("records")

    for i, rec in enumerate(records):
        coords = [[round(lons[i, j], 6), round(lats[i, j], 6)] for j in range(5)]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "id":      int(rec["emb_idx"]),
                "cluster": int(rec["cluster"]),
                "cx":      int(rec["center_x_utm"]),
                "cy":      int(rec["center_y_utm"]),
                "px":      int(rec["parent_x"]),
                "py":      int(rec["parent_y"]),
            },
        })

    return json.dumps({"type": "FeatureCollection", "features": features})


# ─── 전역 상태 (서버 수명과 동일) ─────────────────────────────────────────────
state: dict = {}


# ─── 서버 시작 시 데이터 로드 ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan: 서버 시작/종료 시 실행.

    대용량 데이터(임베딩, GeoJSON 등)를 메모리에 올려두고
    API 요청마다 재로드하지 않도록 state dict에 캐싱.
    """
    print("[startup] 데이터 로딩 중...")

    # ── 임베딩: 10MB 제한으로 분할된 파일 재결합
    part_files = sorted(EMB_DIR.glob("embeddings_norm_*_part*.npy"))
    if part_files:
        emb_norm = np.concatenate([np.load(p) for p in part_files], axis=0)
        print(f"  임베딩 {len(part_files)}개 파일 로드: {emb_norm.shape}")
    else:
        # 단일 파일인 경우 (직접 생성 시)
        emb_norm = np.load(EMB_DIR / "embeddings_norm.npy")

    # ── 클러스터링 결과 + 좌표 매핑
    clustered = pd.read_parquet(EMB_DIR / "clustered_202603_k10.parquet")

    # ── 채널 메타데이터
    with open(META_DIR / "capital_bvvec_202603_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    # ── 사전 계산된 채널 피처 로드 (원본 NPZ 불필요)
    feat_norm = np.load(META_DIR / "feat_norm_202603.npy")  # [N, 46] 정규화
    feat_raw  = np.load(META_DIR / "feat_raw_202603.npy")   # [N, 46] 원본값

    # ── WGS84 좌표 추가 (Mapbox 렌더링용)
    lons, lats = T_FWD.transform(
        clustered["center_x_utm"].values,
        clustered["center_y_utm"].values,
    )
    clustered = clustered.copy()
    clustered["lon"] = lons
    clustered["lat"] = lats

    # ── GeoJSON 사전 빌드 (요청마다 생성하지 않도록)
    print("[startup] GeoJSON 빌드 중...")
    geojson_str = build_geojson(clustered)

    # ── 클러스터별 평균 임베딩 (256dim)
    cluster_emb = {}
    for cid in sorted(clustered["cluster"].unique()):
        idxs = clustered[clustered["cluster"] == cid]["emb_idx"].values
        cluster_emb[int(cid)] = emb_norm[idxs].mean(axis=0).tolist()

    # ── 클러스터 해석 프로파일 계산
    print("[startup] 클러스터 프로파일 계산 중...")
    channel_names    = meta["channel_names"]
    cluster_profiles = compute_cluster_profiles(feat_raw, clustered, channel_names)

    state.update({
        "emb_norm":         emb_norm,
        "df":               clustered,
        "feat_norm":        feat_norm,
        "feat_raw":         feat_raw,
        "meta":             meta,
        "channel_names":    channel_names,
        "geojson_str":      geojson_str,
        "cluster_emb":      cluster_emb,
        "n_clusters":       int(clustered["cluster"].nunique()),
        "cluster_profiles": cluster_profiles,
    })

    cnt = len(clustered)
    print(f"[startup] 준비 완료: {cnt:,}개 셀, {state['n_clusters']}개 클러스터")
    yield
    # 서버 종료 시 정리 (필요 시 추가)


# ─── FastAPI 앱 초기화 ────────────────────────────────────────────────────────
app = FastAPI(
    title="BV-Vec Visualization",
    description="수도권 도시 격자 임베딩 시각화 API",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ─── 라우터 ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """메인 페이지 (Mapbox 프론트엔드)."""
    return FileResponse(STATIC / "index.html")


@app.get("/api/cells.json", summary="전체 셀 GeoJSON")
async def get_cells():
    """
    Mapbox fill 레이어용 GeoJSON.

    각 부모셀(800m×800m)을 폴리곤으로 표현.
    서버 시작 시 사전 빌드되어 캐싱됨.
    """
    return JSONResponse(content=json.loads(state["geojson_str"]))


@app.get("/api/config", summary="앱 설정 + 클러스터 정보")
async def get_config():
    """
    프론트엔드 초기화에 필요한 모든 설정값.

    - 클러스터 색상, 카운트, 프로파일
    - 채널 이름 (영문/한국어)
    - 모델 메타데이터 (학습 설정, S2Vec 비교)
    """
    df = state["df"]
    cluster_counts = df.groupby("cluster").size().to_dict()
    ch_names       = state["channel_names"]

    return {
        "n_cells":          len(df),
        "n_clusters":       state["n_clusters"],
        "cluster_colors":   CLUSTER_COLORS[: state["n_clusters"]],
        "cluster_counts":   {str(k): int(v) for k, v in cluster_counts.items()},
        "channel_names":    ch_names,
        "channel_names_kr": [CHANNEL_LABELS_KR.get(n, n) for n in ch_names],
        "channel_groups":   {
            k: {"indices": v, "color": GROUP_COLORS[k]}
            for k, v in CHANNEL_GROUPS.items()
        },
        "group_colors":     GROUP_COLORS,
        "mapbox_token":     MAPBOX_TOKEN,
        "cluster_profiles": state["cluster_profiles"],
        "model_info": {
            "dataset":      "BigValue tb_grid_total (수도권 100m 격자)",
            "timepoint":    "2026년 3월 (202603)",
            "n_features":   46,
            "grid_size":    "8x8 (800m 부모셀)",
            "embed_dim":    256,
            "encoder":      "ViT-Small (depth=6, heads=8)",
            "decoder":      "Transformer (depth=2, dim=128)",
            "mask_ratio":   "75%",
            "optimizer":    "AdamW (beta1=0.9, beta2=0.95)",
            "lr":           "1e-4 (코사인 감쇠, warmup 20 epoch)",
            "min_lr":       "1e-6",
            "weight_decay": "0.05",
            "batch_size":   128,
            "epochs":       200,
            "best_epoch":   171,
            "val_loss":     0.5382,
            "reference":    "S2Vec (Google, arXiv:2504.16942)",
            "s2vec_lr":     "1.5e-4 (base_lr x batch/256 scaled)",
            "s2vec_wd":     "0.05",
            "s2vec_epochs": "200",
            "s2vec_mask":   "75% MAE",
            "warmup_epochs": 20,
        },
    }


@app.get("/api/cell/{emb_idx}", summary="단일 셀 피처 상세")
async def get_cell(emb_idx: int):
    """
    특정 셀의 46채널 피처값 반환.

    feat_norm: 정규화된 값 (0 근방 분포, 상대적 비교용)
    feat_raw:  역정규화 원본값 (실제 인구수, 세대수 등)
    """
    df        = state["df"]
    feat_norm = state["feat_norm"]
    feat_raw  = state["feat_raw"]
    ch_names  = state["channel_names"]

    row = df[df["emb_idx"] == emb_idx]
    if row.empty:
        return {"error": "not found"}
    row = row.iloc[0]

    return {
        "emb_idx":   emb_idx,
        "cluster":   int(row["cluster"]),
        "lon":       round(float(row["lon"]), 6),
        "lat":       round(float(row["lat"]), 6),
        "cx":        int(row["center_x_utm"]),
        "cy":        int(row["center_y_utm"]),
        "feat_norm": [round(float(v), 4) for v in feat_norm[emb_idx]],
        "feat_raw":  [round(float(v), 2) for v in feat_raw[emb_idx]],
        "ch_names":  ch_names,
    }


@app.get("/api/cluster-profiles", summary="클러스터 해석 프로파일")
async def get_cluster_profiles():
    """
    각 클러스터의 특성 분석 결과.

    - name: 규칙 기반 클러스터 이름 (예: "초고밀 역세권")
    - desc: 특성 설명
    - group_z: 7개 그룹별 z-score
    - top_features: 절대 z-score 상위 5개 채널
    """
    return state["cluster_profiles"]


class SimilarReq(BaseModel):
    """유사 지역 검색 요청 파라미터."""
    lon:   float         # 기준점 경도 (WGS84)
    lat:   float         # 기준점 위도 (WGS84)
    top_k: int = 20      # 반환할 유사 셀 수


@app.post("/api/similar", summary="유사 지역 검색")
async def find_similar(req: SimilarReq):
    """
    클릭한 위치와 가장 유사한 셀 top-k 반환.

    [검색 알고리즘]
    1. WGS84 좌표 → UTM-K 변환
    2. 가장 가까운 부모셀 찾기 (유클리드 거리 최소화)
    3. 해당 셀의 L2 정규화 임베딩 추출
    4. 전체 셀과 코사인 유사도 계산: emb_norm @ query (O(N·D))
    5. 기준 셀 제외 후 상위 top_k 반환

    코사인 유사도 범위: [-1, 1] (1에 가까울수록 유사한 지역 특성)
    """
    # WGS84 → UTM-K 좌표 변환
    cx, cy = T_INV.transform(req.lon, req.lat)

    df  = state["df"]
    emb = state["emb_norm"]

    # 가장 가까운 단일 셀 (유클리드 거리 최소)
    dx  = df["center_x_utm"].values - cx
    dy  = df["center_y_utm"].values - cy
    idx = int((dx**2 + dy**2).argmin())
    matches = df.iloc[[idx]]

    if matches.empty:
        return {"results": [], "query": []}

    # 쿼리 임베딩 준비 (L2 정규화)
    q_idx  = matches["emb_idx"].values
    q_emb  = emb[q_idx].mean(axis=0)
    q_emb /= np.linalg.norm(q_emb) + 1e-8

    # 전체 셀과 코사인 유사도 (행렬 곱)
    sims        = (emb @ q_emb).copy()
    sims[q_idx] = -999.0  # 기준 셀 자신 제외

    top_idx = np.argsort(sims)[::-1][: req.top_k]

    # 빠른 lookup
    df_indexed = df.set_index("emb_idx")
    results = []
    for rank, eidx in enumerate(top_idx):
        r = df_indexed.loc[int(eidx)]
        results.append({
            "rank":    rank + 1,
            "emb_idx": int(eidx),
            "sim":     round(float(sims[eidx]), 4),
            "cluster": int(r["cluster"]),
            "lon":     round(float(r["lon"]), 6),
            "lat":     round(float(r["lat"]), 6),
            "cx":      int(r["center_x_utm"]),
            "cy":      int(r["center_y_utm"]),
        })

    query = [
        {
            "emb_idx": int(r.emb_idx),
            "lon":     round(float(r.lon), 6),
            "lat":     round(float(r.lat), 6),
        }
        for r in matches.itertuples()
    ]

    return {"results": results, "query": query}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=9030, reload=True)
