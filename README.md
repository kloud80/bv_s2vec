# BV-Vec: 도시 격자 임베딩 오픈소스

> S2Vec 방법론을 한국 도시 데이터(BigValue `tb_grid_total`)에 적용한 지리공간 임베딩 프레임워크

---

## S2Vec — 도시의 언어를 학습하는 지리공간 임베딩

**출처:** Google Research Blog (2026.03.24) | Google Earth AI 프로젝트
**논문:** ACM 게재 → https://dl.acm.org/doi/10.1145/3787217
**블로그:** https://research.google/blog/mapping-the-modern-world-how-s2vec-learns-the-language-of-our-cities/

---

### 핵심 요약

AI가 도시의 "성격"을 사람처럼 이해하도록 만드는 프레임워크.
건물, 도로, 상점, 인프라 분포 패턴을 학습해서 어떤 동네인지 수치로 표현.

---

### 작동 원리

#### 1단계 — S2 Geometry로 지표면 분할

- Google의 [S2 Geometry 라이브러리](https://s2geometry.io/)로 지구 표면을 계층적 셀로 분할
- 나라 단위 → 몇 ㎡ 단위까지 해상도 자유 전환
- 각 셀 안의 피처(건물, 카페, 버스정류장, 공원 등)를 카운팅

#### 2단계 — Feature Rasterization → 이미지 변환

- 셀 안의 피처를 다층 이미지(multi-layered image)로 변환
- 예: 커피숍 3개 + 공원 1개 → 이미지의 "색상값"으로 인코딩
- 지리 데이터를 컴퓨터 비전이 처리 가능한 형태로 전환

#### 3단계 — Masked Autoencoding (MAE) 자기지도학습

- 지역의 일부를 가리고(masking) → 나머지를 보고 빈 부분 복원
- 예: 고층 주거 빌딩 + 지하철역 주변 → "식료품점이 있을 것" 예측
- 라벨 없이 전 세계 수백만 번 반복 학습
- **결과물:** 모든 위치의 범용 임베딩 벡터 (그 지역 특성의 수학적 축약)

---

## 모델 아키텍처 상세

### 입력 데이터 구조

```
[부모셀 1개]
┌────────────────────────────────────────────────────┐
│  800m × 800m  (UTM-K 격자 단위)                     │
│                                                    │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐                        │
│  │  │  │  │  │  │  │  │  │  ← 각 칸 = 100m 패치   │
│  ├──┼──┼──┼──┼──┼──┼──┼──┤                        │
│  │  │  │  │PAD│  │  │  │  │  ← PAD = 데이터 없음   │
│  ├──┼──...                 │                       │
│  └──┴──┴──┴──┴──┴──┴──┴──┘                        │
│       8×8 = 64 패치                                 │
│       각 패치: 46채널 수치                           │
└────────────────────────────────────────────────────┘
```

**텐서 형태:** `[N, 8, 8, 46]`
- N: 부모셀 수 (~11,000개 / 샘플: 200개)
- 8×8: 공간 격자 (64 패치)
- 46: 입력 채널 수 (인구·주거·종사자·용도·교통·이주)

### 46채널 → 256차원 압축 흐름

```
입력 피처
[N, 64, 46]          46차원 행정 통계
    │
    │  patch_embed (Linear: 46 → 256)
    │  + 2D sincos 위치 임베딩
    ▼
[N, 64, 256]         패치 임베딩 공간
    │
    │  마스킹 (유효 패치의 75% → MASK 토큰)
    │  PAD 패치 → PAD 토큰 (어텐션 제외)
    ▼
[N, 64, 256]         마스킹 적용
    │
    │  Encoder: TransformerEncoder
    │  (depth=6, heads=8, FFN=1024, Pre-LN)
    ▼
[N, 64, 256]         컨텍스트 인코딩
    │
    │  PAD 제외 평균 풀링 (추출 시)
    ▼
[N, 256]             ★ 지역 임베딩 벡터 ★
                     한 지역의 도시 특성을
                     256개 숫자로 압축
```

> **왜 46 → 256인가?**
> 입력 채널(46)보다 임베딩 차원(256)이 더 크지만, 이는 의도된 설계입니다.
> 46채널은 64개 패치에 걸쳐 공간 분포 패턴을 형성하며,
> Transformer가 패치 간 상호작용(예: 지하철역 인근 상권 집중 등)을 학습해
> 단순 채널값 이상의 **공간 맥락 정보**를 256차원에 담습니다.
> (46채널 × 64패치 = 2,944차원의 원시 정보를 256으로 압축)

### MAE 학습 구조 (train 시)

```
같은 인코더 출력
    │
    │  dec_proj (Linear: 256 → 128)
    ▼
[N, 64, 128]         디코더 입력 (차원 축소)
    │
    │  Decoder: TransformerEncoder
    │  (depth=2, heads=4, FFN=512)
    ▼
[N, 64, 128]
    │
    │  dec_pred (Linear: 128 → 46)
    ▼
[N, 64, 46]          복원 예측값
    │
    │  MSE Loss (MASK된 유효 패치만)
    ▼
  손실값              역전파 → 인코더 학습
```

> **디코더는 학습 전용입니다.**
> 추론(임베딩 추출) 시에는 인코더만 사용하며,
> 디코더는 인코더가 의미있는 표현을 학습하도록 유도하는 역할만 합니다.

### 마스킹 전략

```python
# 유효 패치(PAD 아닌 것)의 75%를 랜덤 선택
valid_counts = (~pad_mask).sum(dim=1)        # 셀마다 유효 패치 수가 다름
n_mask = (valid_counts * 0.75).long()        # 마스킹할 수

# PAD는 절대 마스킹 대상 포함 안됨 (score=inf 처리)
rand = torch.rand(N, 64)
rand = rand.masked_fill(pad_mask, float("inf"))
```

수도권 경계부 셀은 64패치 중 일부만 유효합니다.
PAD 패치를 마스킹 대상에서 제외해야 실제 학습 신호가 보존됩니다.

---

## BV-Vec: 한국 적용 시 변경사항

| 항목 | S2Vec (Google) | BV-Vec (BigValue) |
|------|---------------|-------------------|
| 격자 체계 | S2 Geometry | UTM-K EPSG:5179 100m 격자 |
| 입력 데이터 | OSM, 위성, 이동 패턴 | 행정 통계 46채널 (인구/주거/종사자/용도/교통/이주) |
| 패치 구조 | 가변 | 8×8 고정 (800m 부모셀 → 100m 패치) |
| 임베딩 차원 | 512 | 256 (ViT-Small) |
| 학습률 | 1.5e-4 (linear scaling) | 1e-4 (batch=128 고정) |
| 마스킹 비율 | 75% | 75% (동일) |
| PAD 처리 | 없음 (전 지구 균일 격자) | PAD 토큰 별도 처리 (수도권 경계 불규칙) |
| 학습 결과 | - | val_loss 0.5382 (epoch 171/200) |

---

## 저장소 구조

```
opensource/
├── README.md               ← 이 파일
├── s2vec/                  ← MAE 학습 코드 + 샘플 데이터
│   ├── README.md
│   ├── requirements.txt
│   ├── 01_generate_sample.py   # 샘플 데이터 생성
│   ├── 02_train_mae.py         # ViT-MAE 학습
│   ├── 03_extract_eval.py      # 임베딩 추출 + 클러스터링
│   └── data/
│       ├── sample_grids.npz        # 합성 샘플 (200셀)
│       └── sample_grids_meta.pkl
└── webs2vec/               ← FastAPI 시각화 서버
    ├── README.md
    ├── requirements.txt
    ├── app/
    │   ├── main.py             # FastAPI 서버
    │   └── static/
    │       └── index.html      # Mapbox 프론트엔드
    └── data/
        ├── embeddings/
        │   ├── embeddings_norm_202603_part0.npy  # 임베딩 (분할)
        │   ├── embeddings_norm_202603_part1.npy
        │   ├── clustered_202603_k10.parquet      # K-Means 결과
        │   └── embedding_map_202603.parquet      # 좌표 매핑
        └── meta/
            ├── feat_norm_202603.npy              # 채널별 정규화 피처
            ├── feat_raw_202603.npy               # 채널별 원본 피처
            └── capital_bvvec_202603_meta.pkl     # 채널명, 통계
```

---

## 라이선스

- **코드**: MIT License
- **데이터 (webs2vec/data/)**: BigValue 제공 수도권 격자 통계 기반. 비상업적 연구/교육 목적 사용 가능.
- **원본 데이터 (raw)**: 비공개 (BigValue 내부 DB)
