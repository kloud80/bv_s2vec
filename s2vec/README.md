# s2vec — 도시 격자 MAE 학습 코드

BV-Vec의 ViT-MAE 학습 파이프라인. 전국 단위 격자 데이터가 있다는 전제 하에
샘플 합성 데이터로 전체 흐름을 재현할 수 있습니다.

## 빠른 시작

```bash
pip install -r requirements.txt

# 1. 샘플 데이터 생성 (200개 합성 부모셀)
python 01_generate_sample.py

# 2. MAE 학습
python 02_train_mae.py --epochs 100

# 3. 임베딩 추출 + 클러스터링
python 03_extract_eval.py
```

## 실제 데이터로 실행하는 경우

`01_generate_sample.py`를 자체 격자 DB 추출 코드로 대체하고,
동일한 구조의 NPZ 파일을 `data/` 에 저장하면 됩니다.

```
data/
├── sample_grids.npz          # images [N, 8, 8, 46] + pad_masks [N, 8, 8]
└── sample_grids_meta.pkl     # channel_names, channel_mean/std, parent_ids
```

## 모델 하이퍼파라미터

| 항목 | 값 | S2Vec 비교 |
|------|-----|-----------|
| 아키텍처 | ViT-Small | ViT-Base |
| EMBED_DIM | 256 | 512 |
| ENCODER_DEPTH | 6 | 12 |
| N_HEADS | 8 | 16 |
| 마스크 비율 | 75% | 75% |
| 학습률 | 1e-4 | 1.5e-4 (scaled) |
| WD | 0.05 | 0.05 |
| Optimizer | AdamW β=(0.9, 0.95) | AdamW |
| 스케줄 | Cosine warmup 20ep | Cosine warmup |
