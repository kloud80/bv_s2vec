# webs2vec — BV-Vec 격자 임베딩 시각화

수도권 약 11,000개 부모셀(800m×800m)의 256차원 임베딩을 Mapbox 지도 위에
시각화하고 유사 지역을 검색하는 FastAPI 웹 서버.

## 빠른 시작

```bash
pip install -r requirements.txt

# Mapbox 토큰 설정 (app/main.py의 MAPBOX_TOKEN 변수)
# https://account.mapbox.com/ 에서 무료 발급 가능

uvicorn app.main:app --host 0.0.0.0 --port 9030 --reload
# → http://localhost:9030
```

## 데이터 파일

`data/` 폴더에 이미 분석 완료된 임베딩 데이터가 포함됩니다.

```
data/
├── embeddings/
│   ├── embeddings_norm_202603_part0.npy  # L2 정규화 임베딩 (분할 1/2, ~5.7MB)
│   ├── embeddings_norm_202603_part1.npy  # L2 정규화 임베딩 (분할 2/2, ~5.7MB)
│   ├── clustered_202603_k10.parquet      # K-Means k=10 결과
│   └── embedding_map_202603.parquet      # 인덱스 ↔ UTM 좌표
└── meta/
    ├── feat_norm_202603.npy              # 채널별 정규화 피처 [N, 46]
    ├── feat_raw_202603.npy               # 채널별 원본 피처 [N, 46]
    └── capital_bvvec_202603_meta.pkl     # 채널명, 통계
```

임베딩 파일은 GitHub 10MB 제한으로 2분할 저장.
서버 시작 시 자동으로 재결합됩니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| 클러스터 맵 | K-Means k=10 결과를 색상으로 구분, 토글 ON/OFF |
| 클러스터 해석 | 더블클릭으로 그룹별 z-score, 주요 채널 표시 |
| 유사지역 검색 | 클릭 → 팝업 → 코사인 유사도 top-20 |
| 결과 이동 | 검색 결과 리스트 클릭 시 해당 위치로 지도 이동 |
| 지도 스타일 | Dark/Light 토글 |
| 모바일 지원 | 하단 슬라이드업 시트 |

## API

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /api/config` | 설정, 클러스터 정보, 모델 메타 |
| `GET /api/cells.json` | 전체 셀 GeoJSON |
| `GET /api/cell/{id}` | 단일 셀 46채널 피처 |
| `GET /api/cluster-profiles` | 클러스터 해석 프로파일 |
| `POST /api/similar` | `{lon, lat, top_k}` → 유사 셀 목록 |
