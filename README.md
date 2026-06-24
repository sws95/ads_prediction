# CTR / CVR Prediction & Bid Optimization (Criteo + Ali-CCP + iPinYou)

광고 시스템의 랭킹(예측) → 입찰(의사결정) 흐름을 공개 데이터로 재현.

- CTR 예측 — Criteo (45.8M rows), 모델 5종 비교
- CTR + CVR 멀티태스크 — Ali-CCP, ESMM (CTCVR = CTR × CVR)
- 입찰 최적화 — iPinYou RTB, pCTR → 입찰가 변환 (Linear / ORTB) + 예산 제약 평가

## Dataset

| Dataset | Task | 크기 | Split |
|---|---|---|---|
| Criteo (Kaggle) | CTR | 45,840,617 rows / 13 num + 26 cat | 시간순 앞 40M train / 뒤 5.8M val |
| Ali-CCP (IJCAI-18) | CTR + CVR | ~85M rows | train 42.3M / val·test 각 21.5M |
| iPinYou 2013 (1458) | Bidding | imp 3.08M / bid 14.7M rows | 시간순 6/6–6/11 train / 6/12 test |

- Criteo 결측: I12 76.5%, I1/I10 45.4%, C22 76.3%, C19/20/25/26 44%
- Ali-CCP label: CTR 0.0389 / CVR(clicked) 0.0054 / CTCVR 0.000208
- iPinYou 1458: CTR 0.0795%, PayingPrice 평균 68.9 / median 60. conversion 0건이라 CVR은 Ali-CCP에서 진행

## Tech Stack

- Language: Python 3.12, PyTorch 2.1 (CUDA 12.6)
- CTR/CVR: LR, FM, DeepFM, DCN v2, AutoInt, ESMM
- 입찰: Logistic Regression (pCTR, feature hashing 2^20) + Linear / ORTB1 / ORTB2
- bid landscape: Kaplan-Meier win rate model, 2nd-price 기대비용 기반 λ 이분탐색
- 전처리: num fillna(0)→clip(0)→log1p / cat min_freq=10 vocab 인덱싱

## 모델

| Model | 핵심 | 논문 |
|---|---|---|
| LR | 선형 baseline | McMahan 2013 |
| FM | 2nd-order interaction | Rendle 2010 |
| DeepFM | FM + DNN 병렬 | Guo 2017 |
| DCN v2 | Cross Network + DNN | Wang 2021 |
| AutoInt | Self-Attention interaction | Song 2019 |
| ESMM | CTR+CVR 멀티태스크 | Ma 2018 |
| Linear / ORTB | pCTR → 입찰가 변환 | Perlich 2012 / Zhang 2014 |

## Evaluation Results

### Criteo CTR (val)

| Model | AUC | Log Loss | Params |
|---|---|---|---|
| LR | 0.7546 | 0.5075 | 1.09M |
| FM | 0.7714 | 0.4912 | ~17M |
| DeepFM | 0.7829 | 0.4779 | 18.95M |
| DCN v2 | 0.7980 | 0.4563 | 18.26M |
| AutoInt | 0.7986 | 0.4556 | 17.72M |

## AutoInt vs DCN v2 추론 벤치마크 (GPU, warmup 20 / iter 200)

| batch | model | mean(ms) | std | p50 | p99 | throughput(/s) |
|-------|---------|---------:|------:|------:|------:|---------------:|
| 1     | AutoInt | 1.502 | 0.577 | 1.205 | 2.766 | 666 |
| 1     | DCN v2  | 0.595 | 0.123 | 0.563 | 1.216 | 1,682 |
| 256   | AutoInt | 1.374 | 0.497 | 1.162 | 2.405 | 186,358 |
| 256   | DCN v2  | 0.932 | 0.261 | 1.030 | 1.550 | 274,531 |
| 4096  | AutoInt | 6.522 | 0.371 | 6.465 | 7.505 | 628,000 |
| 4096  | DCN v2  | 1.163 | 0.203 | 1.101 | 1.739 | 3,521,204 |

**DCN v2 우위**: batch=1 단건 2.5배 / batch=4096 throughput 5.6배. p99·std 모두 DCN이 낮아 tail latency 안정적. AUC

batch 4096, Adam lr=1e-3, 1 epoch (FM 2 epoch). interaction 모델링할수록 AUC 상승.

DCN v2 ≈ AutoInt (AutoInt +0.0006, 추론은 DCN이 빠름). FM은 std=1 초기화에서 logit 폭발(-54~40) → std=0.01로 해결.

시간순 split이라 논문(랜덤 split)보다 약간 낮음 (leakage 방지).

### Ali-CCP ESMM

| Metric | Val (best, ep3) | Test |
|---|---|---|
| CTR AUC | 0.6241 | 0.6240 |
| CVR AUC | 0.6797 | 0.6654 |
| CTCVR AUC | 0.6406 | 0.6326 |

Criteo와 절대 AUC 직접 비교 불가 (익명 ID 위주, label 희소). 공개 ESMM 벤치마크도 0.62~0.68대.

CVR 단독 학습은 sample selection bias + data sparsity 발생 → ESMM은 pCTCVR = pCTR × pCVR로 유도해 동시 해결.

CTR loss(0.157) vs CTCVR loss(0.002) 약 100배 차이 → loss weighting 튜닝 여지. ep3 최고, 이후 과적합.

### iPinYou CTR 예측 (test 6/12, 1458 단일 캠페인)

| 지표 | 값 (tag 포함, 메인) |
|---|---|
| AUC | 0.9897 |
| 실제 CTR | 0.0796% |
| avg pCTR (raw → cal) | 0.4480% → 0.1168% |
| logloss (raw → cal) | 0.00531 → 0.00169 |

CTR 0.08% 불균형 → negative downsampling 10% 후 p/(p+(1-p)/w) 재보정 (He 2014).

AUC는 보정 무관, 입찰식은 pCTR 절대값을 쓰므로 보정 필수 (안 하면 과대입찰).

### iPinYou — AUC 0.99 이상 탐지 → 원인 규명

처음 전체 피처(UserTags 포함)로 학습했더니 AUC가 0.99로 비정상적으로 높게 나옴 → leakage 의심.

피처 ablation으로 원인을 UserTags로 특정 (CreativeID 혼입 / click 누수 / train-test 분리 오류는 차례로 배제).

| 설정 | AUC | 기여 |
|---|---|---|
| (A) tag 전부 포함 | 0.9897 | — |
| (B) 11278만 제거 | 0.8236 | 11278 단독 +0.166 |
| (C) tag 전부 제거 | 0.7092 | 나머지 태그 +0.114 |

11278(In-market/clothing): 보유자 CTR 34%(평균의 약 430배). 단 보유자 65.6%가 미클릭 → 미래 정보 누수가 아니라 사전 타게팅 신호로 판정 (누수라면 보유=클릭이어야 함).

이후 벤치마크 논문(Zhang 2014, Table 4)을 확인한 결과 1458 LR AUC = 0.9881 — 논문 역시 UserTags를 binary 피처로 포함해 동일하게 높은 AUC를 얻었음을 확인. 본 결과(0.9897)가 이를 재현하며, in-market 태그가 누수가 아닌 정상 신호임을 뒷받침.

→ tag 포함(0.99)을 CTR 메인 모델로 채택하고 입찰 시뮬레이션에도 동일 pCTR 사용. (C) tag 제거(0.71)는 in-market 태그의 기여를 분리한 ablation 결과이며, 아래 pCTR 품질 비교에만 사용.

### iPinYou — 예산별 입찰 전략 (획득 클릭 수)

ORTB λ 그리드: `[1e-4 … 5e-7]` → `logspace(-7,-4,40)` (동일 범위, 5배 조밀)

| 예산 | Const | Linear (clk/eCPC) | ORTB 8점 (clk/eCPC) | ORTB 40점 (clk/eCPC) |
|---|---|---|---|---|
| full/64 | 18 | 283 / 1673 | 334 / 559 | 335 / 1123 |
| full/32 | 25 | 340 / 1841 | 335 / 1464 | 340 / 2463 |
| full/16 | 41 | 341 / 2937 | 341 / 4139 | 342 / 5473 |
| full/8 | 59 | 344 / 8777 | 341 / 4139 | 343 / 9190 |
| full/4 | 109 | 347 / 20576 | 344 / 14918 | 348 / 21752 |
| full/2 | 208 | 355 / 38804 | 352 / 30035 | 353 / 38848 |

(available clicks = 356, full_cost = 30,297,100, avg market 67.7, c0 ≈ 54.9)

예산 = full_cost(전부 낙찰 비용)의 1/N. 예산별 파라미터 재탐색(Linear=base_bid, ORTB=λ).

**전 구간 Linear/ORTB ≫ Constant** (1/64에서 Const 대비 15배 이상).

λ 그리드 해상도 검증 (8점 → 40점):
- 클릭: 8점에서 여유 구간 Linear가 ORTB보다 높았으나, 40점으로 조밀화하니 ORTB가 따라잡거나 넘음 (1/32: 335→340, 1/4: 344→348) → 여유 구간 Linear 우위는 ORTB 성능 열위가 아니라 λ 탐색 해상도 부족.
- eCPC: 동시에 ORTB eCPC 상승해 Linear와 비슷해짐. 클릭 최대화 목표라 λ 후보가 많아지자 클릭 1개 더 따려고 공격적 λ(낮은 λ=높은 입찰가) 선택 (full/2: λ 1e-6→7e-7, 클릭 +1, 비용 약 314만 증가, eCPC 30035→38848).

낮은 eCPC는 별개 강점 → 8점 보수적 λ는 클릭 약간 적어도 eCPC 낮아 효율적, 40점 공격적 λ는 클릭 더 따되 eCPC 상승. 어느 쪽이 나은지는 광고주 KPI(클릭 수 vs 효율)에 달림 → λ는 클릭과 eCPC를 맞바꾸는 손잡이.

eCPC ↔ 입찰가 (2nd-price): eCPC 낮음 = λ 큼 = 입찰가 낮음 = 싸게 이길 노출 위주 클릭 획득. 실제 지불액은 입찰가가 아니라 시장가(2등 가격)이며, 입찰가 높이면 더 비싼 경매까지 이겨 지불액과 eCPC 함께 상승.

→ 예산이 충분하면 Linear ≈ ORTB. ORTB 명확한 우위는 가장 빡빡한 full/64 — 40점에서도 클릭(335 vs 283)과 eCPC(1123 vs 1673) 모두 앞섬. 빡빡할수록 오목 입찰 곡선의 배분 효과 큼.

![bid_results](./img/bid_results1.png)

bid landscape는 단순 모델 b/(b+55)이 시장가 급경사를 못 따라가는 한계를 보여줌 → KM으로 보완.
![bid_results](./img/bid_results1.png)

### iPinYou — pCTR 품질의 영향 (0.71 비교용)

같은 입찰 파이프라인에 tag 제거 모델(AUC 0.71)의 pCTR을 넣어 비교.

| 예산 | Constant | Linear | ORTB |
|---|---|---|---|
| full/64 | 18 | 22 | 20 |
| full/32 | 25 | 31 | 33 |
| full/16 | 41 | 58 | 64 |
| full/8 | 59 | 101 | 92 |
| full/4 | 109 | 174 | 172 |
| full/2 | 208 | 262 | 259 |

입찰식에 들어가는 것은 AUC가 아니라 pCTR 값. 0.99 모델은 클릭 날 노출에 높은 pCTR을 매겨 입찰을 정확한 노출에 집중, 0.71 모델은 pCTR을 잘못 측정해 입찰 분산 → 같은 예산에서 0.99 pCTR이 클릭 크게 더 획득 (full/64 Linear 22 vs 283).

Constant는 pCTR 미사용이라 두 설정에서 결과 동일 (18, 25, …) → 입찰 차이가 AUC 숫자가 아니라 pCTR 분별력에서 비롯됨을 확인.

### iPinYou — 정식 ORTB (KM 기반, 시장가 비관측)

| 항목 | 내용 |
|---|---|
| win rate model | W(b)=b/(b+c0), 시장가 급경사 못 따라감 → KM으로 대체 |
| λ 결정 | 2nd-price 기대비용 Σ W(bid)·E[market\|market≤bid] 기반 이분탐색 |
| 결과 | 시장가 비관측 KM-ORTB가 오라클 ORTB와 거의 일치 |
| ORTB1 vs ORTB2 | 성능 유사 |
| c 민감도 | λ 재탐색이 흡수, 변동 6~20% |

![ortb2_csens](./img/ortb2_csens.png)

### iPinYou — selection bias (bid 로그 기반 진짜 KM)

| bid | KM (진짜) | naive (이긴 것만) |
|---|---|---|
| 20 | 0.038 | 0.181 |
| 60 | 0.106 | 0.505 |
| 100 | 0.175 | 0.834 |
| 200 | 0.200 | 0.956 |

![km_real](./img/km_real.png)

bid 로그 win 20.9% / lose(censored) 79.1%. naive(이긴 것만)는 bid 100에서 win rate를 0.83으로 추정, 진짜 KM은 0.18 → 약 5배 과대추정.

실제 1458 입찰(300, win 20.9%)과 KM 200 근처 수렴값(0.2)이 일치 

→ 이긴 데이터만 보면 win rate 심하게 과대추정, censored 반영해야 실제 시장에 가까움.

1458 입찰가 300 단일 상수라 입찰가별 win rate 곡선 추정 자체는 불가 

→ 탐색 없는 단일 정책의 selection bias. bid landscape forecasting은 입찰가 탐색 데이터 필요.

## 실제 광고 시스템 구조

| 단계 | 역할 | 기술 | 본 프로젝트 |
|---|---|---|---|
| 1. Candidate Generation | 수백만 → 수백 | Two-Tower, ANN | - |
| 2. Ranking | 수백 → 수십 | DeepFM, DCN, ESMM | Criteo / Ali-CCP |
| 3. Re-ranking | 다양성·제약 | MMR, 룰 | - |
| 4. Bidding | 입찰가 결정 | Linear / ORTB, KM win rate | iPinYou |
| 5. Budget Pacing | 예산 고갈 방지 | 예산 제어 | 예산 제약 시뮬레이션 |

## Roadmap

- [x] Criteo CTR 5종 비교 (LR/FM/DeepFM/DCN v2/AutoInt)
- [x] Ali-CCP ESMM 멀티태스크 (CTCVR = pCTR × pCVR)
- [x] iPinYou CTR + leakage 4단계 검증 (tag 원인 규명)
- [x] 입찰 전략 Constant/Linear/ORTB 예산별 비교
- [x] KM bid landscape + 정식 ORTB (오라클 근접) + c 민감도
- [ ] 시퀀스 모델 (DIN/DIEN/SASRec)

## 임베딩 시각화

![임베딩 시각화](./img/small_embedding.png)
