# CTR / CVR Prediction Model Comparison

## 프로젝트 개요

광고 시스템의 **랭킹 단계** 모델 비교 프로젝트.

- **Part 1**: Criteo Display Advertising Challenge 데이터셋(45M rows)으로 CTR 예측 모델 5종 비교 (LR, FM, DeepFM, DCN v2, AutoInt)
- **Part 2**: Ali-CCP (IJCAI-18) 데이터셋(85M rows)으로 ESMM 멀티태스크 학습 (CTR + CVR + CTCVR)

---

## Part 1. CTR Prediction (Criteo)

### 데이터셋

- **출처**: Criteo Display Advertising Challenge (Kaggle)
- **크기**: 45,840,617 rows
- **Features**: 13 numerical (I1~I13) + 26 categorical (C1~C26) + label
- **결측값**: I12(76.5%), I1/I10(45.4%), C22(76.3%), C19/C20/C25/C26(44%) 등

### 전처리

- **Numerical**: `fillna(0)` → `clip(lower=0)` → `log1p` → `float32`
- **Categorical**: `min_freq=10` 미만 → `unknown(0)` 처리 → vocab 기반 정수 인덱스
- **Split**: 시간순 정렬 기준 앞 40M → train / 뒤 5.8M → val (data leakage 방지)

### Categorical Vocab 크기

high-cardinality feature 다수 존재 (C3: 10M, C12: 8.3M, C21: 7M, C16: 5.4M, C4: 2.2M). min_freq=10 cutoff로 long-tail 제거 후 임베딩.

### 모델

| Model | 핵심 아이디어 | 논문 |
|---|---|---|
| LR | 선형 baseline | McMahan et al. 2013 |
| FM | 2nd-order feature interaction | Rendle 2010 |
| DeepFM | FM + DNN 병렬 | Guo et al. 2017 |
| DCN v2 | Cross Network + DNN | Wang et al. 2021 |
| AutoInt | Self-Attention 기반 interaction | Song et al. 2019 |

### 실험 결과

| Model | AUC | Log Loss | Parameters |
|---|---|---|---|
| LR | 0.7546 | 0.5075 | 1,085,741 |
| FM | 0.7714 | 0.4912 | ~17M |
| DeepFM | 0.7829 | 0.4779 | 18,952,974 |
| DCN v2 | 0.7980 | 0.4563 | 18,259,872 |
| AutoInt | 0.7986 | 0.4556 | 17,719,122 |

학습 환경: GPU(CUDA 12.6), batch_size=4096, Adam lr=1e-3, 1 epoch (FM은 2 epoch)

### 주요 발견

**1. Feature Interaction 효과**
LR → FM → DeepFM 순으로 AUC 상승. feature 간 상호작용을 모델링할수록 성능 향상.

**2. FM 초기화 문제**
FM 2차 항 `0.5 * (||sum(vi)||² - sum(||vi||²))`이 기본 초기화(std=1)에서 logit 폭발(-54 ~ 40, 평균 -3.70) 발생.
→ embedding `std=0.01`로 초기화, `lr=1e-3` 유지하면서 LR scheduler(γ=0.1)로 안정화. 2 epoch에서 AUC 0.7714 달성.

**3. DCN v2 vs AutoInt**
Cross Network(DCN)와 Self-Attention(AutoInt) 모두 비슷한 성능(0.798x). AutoInt가 0.0006 높지만 추론 속도는 DCN이 빠름.

**4. 시간순 Split**
랜덤 split 대신 시간순 split 적용. 실제 서비스와 동일한 세팅으로 data leakage 방지. 논문(랜덤 split) 대비 AUC가 약간 낮게 나오는 원인.

### Embedding 시각화

FM으로 학습된 categorical embedding을 t-SNE로 2차원 시각화. 비슷한 클릭 패턴을 가진 카테고리값끼리 embedding 공간에서 군집화되는 경향 확인.

---

## Part 2. CTR + CVR Multi-task Learning (ESMM)

### 동기

Criteo는 **CTR label만** 제공하므로 CVR 학습 불가. 또한 feature semantic 비공개로 해석 한계. CVR까지 다루기 위해 **Ali-CCP (IJCAI-18)** 데이터셋으로 전환.

### CVR 단독 학습의 두 가지 문제

**Problem 1 — Sample Selection Bias (SSB)**

기존 CVR 예측 방식:
- 학습 데이터: 클릭된 샘플만 사용
- 추론 시점: 모든 노출에 대해 CVR 예측 필요

→ 학습은 **클릭 공간**, 추론은 **노출 공간**. 분포가 다름. 클릭 안 한 유저의 CVR을 예측해야 하는데, 모델은 그런 케이스를 본 적이 없음.

**Problem 2 — Data Sparsity**

- 노출 100만 → 클릭 1만 (CTR 1%) → 전환 100건 (CVR 1%)
- 전환 100건으로 CVR 모델 학습 → 데이터 부족으로 불안정

### ESMM 해결책

CVR을 직접 학습하지 않고, **CTR과 CTCVR을 동시에 학습**. CVR은 그 사이에서 유도.

```
pCTCVR = pCTR × pCVR

pCTR   = P(클릭 | 노출)        ← 노출 데이터 전체로 학습 가능
pCTCVR = P(클릭∧전환 | 노출)   ← 노출 데이터 전체로 학습 가능
pCVR   = P(전환 | 클릭)        ← 위 두 개로 유도 (직접 학습 X)
```

→ CTR/CTCVR 양쪽 모두 **노출 공간**에서 학습되므로 SSB 해소, 데이터 풍부.

### 데이터셋

- **출처**: Ali-CCP (Alibaba Click and Conversion Prediction), IJCAI-18
- **Split**: Train 42,299,905 / Val 21,508,307 / Test 21,508,307
- **Label 분포**:
  - Train CTR: 0.0389
  - Train CVR (clicked 기준): 0.0054
  - Train CTCVR: 0.000208

CTCVR positive 비율이 0.02% 수준으로 극단적 sparse.

### 모델

- Shared embedding + CTR tower + CVR tower
- Loss: `BCE(pCTR, y_click) + BCE(pCTR × pCVR, y_conversion)`
- Params: 27,369,298

### 실험 결과

| Epoch | Loss (CTR + CTCVR) | Val CTR AUC | Val CVR AUC | Val CTCVR AUC | Time |
|---|---|---|---|---|---|
| 1 | 0.1647 (0.1625 + 0.0022) | 0.6188 | 0.6566 | 0.6257 | 1439s |
| 2 | 0.1592 (0.1572 + 0.0020) | 0.6221 | 0.6712 | 0.6279 | 1460s |
| 3 | 0.1587 (0.1567 + 0.0020) | 0.6241 | **0.6797** | **0.6406** | 1466s |
| 4 | 0.1586 (0.1566 + 0.0020) | 0.6235 | 0.6746 | 0.6296 | 1475s |
| 5 | 0.1585 (0.1565 + 0.0020) | 0.6224 | 0.6700 | 0.6329 | 1458s |

**Test**: CTR AUC = 0.6240, CVR AUC = 0.6654, CTCVR AUC = 0.6326

→ Epoch 3에서 최고 성능, 이후 과적합 경향. CTR/CTCVR loss 비중 차이가 100배 이상이라 loss weighting 튜닝 여지 있음.

---

## 실제 광고 시스템 구조

본 프로젝트는 전체 광고 시스템 중 **랭킹(Ranking) 단계**에 해당.

| 단계 | 역할 | 기술 |
|---|---|---|
| 1. 후보 생성 (Candidate Generation) | 수백만 광고 → 수백 개 추려냄 | Two-Tower, ANN 검색 |
| 2. 랭킹 (Ranking) | 수백 개 → 수십 개 추려냄 | **DeepFM, DCN, ESMM ← 본 프로젝트** |
| 3. 재랭킹 (Re-ranking) | 다양성, 예산, 광고주 제약 고려 | MMR, 비즈니스 룰 |
| 4. 입찰 (Bidding) | 광고주 입찰가 결정 | pCTR × pCVR × bid price |
| 5. 예산 관리 (Budget Pacing) | 광고주 예산 고갈 방지 | 예산 제어 시스템 |

---

## 한계 및 향후 계획

- Criteo는 feature semantic 비공개로 해석 한계
- ESMM의 CTR AUC가 Criteo 대비 낮음 → Ali-CCP feature space 차이 + loss weight 튜닝 필요
- 시퀀스 기반 모델(DIN/DIEN/SASRec) 적용 예정
  - DIEN: GRU + Attention으로 유저 관심사 변화 모델링
  - SASRec: Self-Attention 기반 sequential recommendation

---

## 환경

- Python 3.12
- PyTorch 2.1 (CUDA 12.6)
- pandas, numpy, scikit-learn, matplotlib



![임베딩 시각화](./results/small_embedding.png)
