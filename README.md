# CTR / CVR Prediction & Bid Optimization (Criteo + Ali-CCP + iPinYou)
 
## 프로젝트 개요
 
광고 시스템의 **랭킹(예측) → 입찰(의사결정)** 흐름을 공개 데이터로 재현하는 프로젝트.
 
- **CTR 예측** — Criteo Display Advertising Challenge (45.8M rows)로 모델 5종 비교
- **CTR + CVR 멀티태스크** — Ali-CCP (IJCAI-18)로 ESMM 학습 (CTCVR = CTR × CVR)
- **입찰 최적화** *(진행 예정)* — iPinYou RTB 데이터로 Linear bidding / ORTB 실험
## 데이터셋
 
### Criteo (CTR)
 
- 출처: Criteo Display Advertising Challenge (Kaggle)
- 크기: 45,840,617 rows
- Features: 13 numerical (I1\~I13) + 26 categorical (C1\~C26) + label
- 결측값: I12(76.5%), I1/I10(45.4%), C22(76.3%), C19/C20/C25/C26(44%) 등
### Ali-CCP (CTR + CVR)
 
- 출처: Ali-CCP (IJCAI-18)
- 크기: 약 85M rows (Train 42.3M / Val 21.5M / Test 21.5M)
- Label: CTR 0.0389, CVR(clicked 기준) 0.0054, CTCVR 0.000208 *(train 기준)*
### iPinYou (Bidding) — 진행 예정
 
- 출처: iPinYou Global RTB Bidding Algorithm Competition (2013)
- 특징: 낙찰가(winning price)가 포함된 거의 유일한 공개 RTB 데이터 → 입찰 의사결정 실험 가능
- 용도: pCTR/pCVR 예측 결과를 입찰가로 변환하는 단계(Linear bidding, ORTB) 실험
## 전처리
 
### 공통 (Criteo / Ali-CCP)
 
- Numerical: fillna(0) → clip(lower=0) → log1p → float32
- Categorical: min_freq=10 미만 → unknown(0) 처리 → vocab 기반 정수 인덱스
### Split
 
- **Criteo**: 시간순 정렬 기준 앞 40M → train / 뒤 5.8M → val (data leakage 방지)
- **Ali-CCP**: 데이터셋 제공 기준 Train / Val / Test split 그대로 사용
## 모델
 
| Model | 핵심 아이디어 | 논문 |
|---|---|---|
| LR | 선형 baseline | McMahan et al. 2013 |
| FM | 2nd-order feature interaction | Rendle 2010 |
| DeepFM | FM + DNN 병렬 | Guo et al. 2017 |
| DCN v2 | Cross Network + DNN | Wang et al. 2021 |
| AutoInt | Self-Attention 기반 interaction | Song et al. 2019 |
| ESMM | CTR + CVR 멀티태스크 (CTCVR = CTR × CVR) | Ma et al. 2018 |
 
## 실험 결과
 
### Criteo CTR
 
| Model | AUC | Log Loss | Parameters |
|---|---|---|---|
| LR | 0.7546 | 0.5075 | 1,085,741 |
| FM | 0.7714 | 0.4912 | ~17M |
| DeepFM | 0.7829 | 0.4779 | 18,952,974 |
| DCN v2 | 0.7980 | 0.4563 | 18,259,872 |
| AutoInt | 0.7986 | 0.4556 | 17,719,122 |
 
학습 환경: GPU(CUDA 12.6), batch_size=4096, Adam lr=1e-3, 1 epoch (FM은 2 epoch)
 
### Ali-CCP ESMM
 
| Metric | Val (best, epoch 3) | Test |
|---|---|---|
| CTR AUC | 0.6241 | 0.6240 |
| CVR AUC | 0.6797 | 0.6654 |
| CTCVR AUC | 0.6406 | 0.6326 |
 
학습 환경: GPU(CUDA 12.6), 5 epochs, params 27,369,298
 
> **참고 — Criteo와 AUC를 직접 비교하지 말 것**
> Ali-CCP는 Criteo와 데이터·태스크가 달라 절대 AUC 수치를 직접 비교하는 의미가 없다.
> Ali-CCP는 feature가 익명 ID 위주로 dense signal이 적고, CVR/CTCVR label이 극도로 희소(0.0054 / 0.0002)해
> 공개 ESMM 벤치마크에서도 AUC가 0.62~0.68 대로 보고된다. 본 결과도 그 범위 안에 있다.
 
## 주요 발견
 
### 1. Feature Interaction 효과
 
LR → FM → DeepFM 순으로 AUC 상승. feature 간 상호작용을 모델링할수록 성능 향상.
 
### 2. FM 초기화 문제
 
FM 2차 항(0.5 * (||sum(vi)||² - sum(||vi||²)))이 기본 초기화(std=1)에서 logit 폭발(-54 ~ 40) 발생. std=0.01로 초기화 후 안정적 학습 가능.
 
### 3. DCN v2 vs AutoInt
 
Cross Network(DCN)와 Self-Attention(AutoInt) 모두 비슷한 성능(0.798x). AutoInt가 0.0006 높지만 추론 속도는 DCN이 빠름.
 
### 4. 시간순 Split (Criteo)
 
랜덤 split 대신 시간순 split 적용. 실제 서비스와 동일한 세팅으로 data leakage 방지. 논문(랜덤 split) 대비 AUC가 약간 낮게 나오는 원인.
 
### 5. ESMM - CVR 단독 학습의 문제
 
기존 CVR 모델은 클릭된 샘플만 학습 데이터로 쓰지만 추론은 모든 노출에 대해 해야 함 → Sample Selection Bias 발생. 또한 전환 데이터 자체가 너무 적어 학습 불안정(Data Sparsity). ESMM은 CVR을 직접 학습하지 않고 pCTCVR = pCTR × pCVR 관계로 유도하여 두 문제 동시 해결.
 
### 6. ESMM 학습 양상
 
CTR loss(0.1565)와 CTCVR loss(0.0020) 간 약 100배 차이로 loss weighting 튜닝 여지 있음 (CTCVR loss에 가중치 α를 키우는 방향). Epoch 3에서 최고 성능, 이후 과적합 경향.
 
## Embedding 시각화
 
FM으로 학습된 categorical embedding을 t-SNE로 2차원 시각화. 비슷한 클릭 패턴을 가진 카테고리값끼리 embedding 공간에서 군집화되는 경향 확인.
 
## 실제 광고 시스템 구조
 
본 프로젝트는 전체 광고 시스템 중 **랭킹(Ranking)** 단계를 다루며, **입찰(Bidding)** 단계로 확장 진행 중이다.
 
| 단계 | 역할 | 기술 | 본 프로젝트 |
|---|---|---|---|
| 1. 후보 생성 (Candidate Generation) | 수백만 광고 → 수백 개 추려냄 | Two-Tower, ANN 검색 | - |
| 2. 랭킹 (Ranking) | 수백 개 → 수십 개 추려냄 | DeepFM, DCN, ESMM | ✅ Criteo / Ali-CCP |
| 3. 재랭킹 (Re-ranking) | 다양성, 예산, 광고주 제약 고려 | MMR, 비즈니스 룰 | - |
| 4. 입찰 (Bidding) | 광고주 입찰가 결정 | pCTR × pCVR × bid price | 🔜 iPinYou |
| 5. 예산 관리 (Budget Pacing) | 광고주 예산 고갈 방지 | 예산 제어 시스템 | - |
 
## 한계 및 향후 계획
 
- Criteo는 feature semantic 비공개로 해석 한계
- Criteo는 CVR label 없어 CTR만 가능 → Ali-CCP로 CTR + CVR 멀티태스크(ESMM) 진행 (완료)
- **입찰 최적화 확장**: iPinYou RTB 데이터로 pCTR/pCVR → 입찰가 변환(Linear bidding, ORTB) 실험 예정. 예측(랭킹)에서 의사결정(입찰)까지 end-to-end 광고 ML 파이프라인 완성이 목표.
- ESMM loss weighting 튜닝 (CTCVR loss 가중치 α 조정)
- 시퀀스 기반 모델(DIN/DIEN/SASRec) 적용 예정
## 환경
 
- Python 3.12
- PyTorch 2.1 (CUDA 12.6)
- pandas, numpy, scikit-learn, matplotlib
## 임베딩 시각화
 
![임베딩 시각화](./results/small_embedding.png)
