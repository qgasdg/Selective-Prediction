# 1. 문제 정의

## 1.1 동기

당뇨병은 허혈성 심장질환, 신부전 등 심각한 동반 질환을 유발하여 전체 30일 내 재입원의 최대 25%를 차지하는 핵심 위험 질환이다. 특히 '퇴원 후 30일 이내 재입원'은 환자에 대한 퇴원 계획의 실패를 의미하는 핵심 의료 질 지표이자, 미국의 병원 재입원 감소 프로그램(HRRP) 등에서 높은 재입원율을 기록하는 병원에게 재정적 페널티를 부과하는 주요 기준이 된다. 따라서 사전에 머신러닝을 통해 당뇨 환자의 재입원 고위험군을 선별할 수 있다면, 환자의 생명을 보호하는 예방 의학적 가치뿐만 아니라 병원의 재정적 손실을 막고 한정된 의료 자원을 최적화할 수 있다.

그러나 이러한 예측 모델이 실제 의료 현장에 배포될 때, 가장 중요한 질문 중 하나는 "모델이 틀릴 가능성이 높은 상황에서 어떻게 행동해야 하는가"이다. 대부분의 분류 모델은 입력이 주어지면 확신도와 무관하게 항상 예측을 출력하도록 설계된다. 확신도가 낮은 샘플에 대해서도 강제로 결정을 내리는 것은 의료처럼 오류 비용이 비대칭적이고 치명적인 도메인에서 심각한 문제를 초래할 수 있다.

이에 본 프로젝트는 Selective Prediction(선택적 예측, Abstention Learning)을 핵심 방법론으로 채택한다. 이는 모델이 스스로의 불확실성을 추정하여, 확신도가 충분히 높을 때만 예측을 출력하고 그렇지 않을 때는 예측을 거부(abstain)하는 프레임워크다. 예측을 거부한 샘플은 전문가(의료진)에게 넘겨져 직접 판단을 받는다. 이는 모델 성능에 제한이 있는 경우, 이를 무리하게 자동화하지 않고 시스템적 설계로 보완하려는 의도를 담고 있다. 즉, 기계가 잘하는 것(명확한 케이스의 빠른 대량 처리)과 인간이 잘하는 것(불확실한 케이스의 임상적 종합 판단)을 분리함으로써, 전체 의료 시스템의 안전성과 자원 분배 효율을 극대화하는 'Human-in-the-loop (인간 참여형 AI)' 설계이다.

## 1.2 문제 설정

본 프로젝트에서 다루는 구체적 문제는 다음과 같다.

당뇨 환자의 30일 내 재입원 여부를 예측하되, 모델이 불확실한 케이스에 대해서는 예측을 거부하고 의료진에게 검토를 요청하는 시스템을 구축한다.

이를 수식으로 표현하면, 모델은 입력 $x$에 대해 예측값 $\hat{y}$ 혹은 거부 신호 $\perp$를 출력한다.

$$f(x) = \begin{cases} \hat{y} & \text{if } \text{conf}(x) > \tau \\ \perp & \text{otherwise} \end{cases}$$

여기서 $\tau$는 사전에 설정하는 confidence threshold이며, $\text{conf}(x) = \max(P(Y=1 \mid x),\ P(Y=0 \mid x))$로 정의한다.

## 1.3 기대 효과 및 수혜자

* 병원 임상 워크플로우 및 자원 분배 최적화: 모든 환자의 재입원을 완벽히 예측하는 것은 불가능하지만, 병원의 가용 인력(예: 퇴원 관리 간호사)이 가장 먼저 집중해야 할 고위험군을 선별하는 1차 필터(Triage)로서 기능한다. 모델이 확신하는 케이스만 자동 처리함으로써 제한된 인력을 가장 필요한 곳에 효율적으로 배분할 수 있다.

* 후속 조치 비용 절감 (향상도 극대화): 자연 상태의 재입원율(기저율 약 9%)을 상회하는 밀도 높은 위험군을 솎아냄으로써, 정상 환자에게 불필요한 관리가 들어가는 오진(False Positive) 리스크와 리소스 낭비를 최소화한다.

* 의사결정 투명성: "모델이 예측한 것"과 "모델이 거부한 것"을 명확히 구분함으로써, 모델의 한계를 가시화하고 신뢰 가능한 배포(trustworthy deployment)를 지원한다.

---

# 2. 데이터셋

## 2.1 선정 과정

데이터셋 선정 시 다음 기준을 적용했다.

1. Selective Prediction의 타당성: 예측 거부가 실세계에서 의미를 가지는 도메인이어야 한다.
2. 적절한 클래스 불균형: 양성 비율이 극단적(< 5%)이면 confidence 추정이 왜곡된다.
3. Feature 다양성: EDA 및 feature importance 분석이 의미 있을 만큼 feature가 충분해야 한다.
4. 고전적 머신러닝 적합성: 이미지·텍스트가 아닌 tabular 데이터여야 한다.

이 기준에 따라 몇 가지 후보를 검토했다.

| 데이터셋 | 탈락 이유 |
| --- | --- |
| Wisconsin Breast Cancer | 유명한 benchmark dataset, novelty 부족 |
| Stroke Prediction (Kaggle) | 양성 비율 ~5%로 지나치게 불균형, Kaggle notebook 다수 존재 |
| Heart Disease (UCI) | 샘플 수(303개) 너무 적음 |

최종적으로 Diabetes 130-US Hospitals for Years 1999–2008 (UCI, ID: 296) 를 선택했다.

## 2.2 데이터셋 개요

| 항목 | 내용 |
| --- | --- |
| 출처 | UCI Machine Learning Repository |
| 수집 기간 | 1999 – 2008년 |
| 수집 기관 | 미국 130개 병원 |
| 원본 크기 | 101,766 rows × 50 columns |
| 정제 후 크기 | 69,990 rows × 44 columns |
| 타깃 변수 | `readmitted`: 30일 내 재입원 여부 (이진화) |
| 양성 비율 | ~9.0% (6,285 / 69,990) |
| 라이선스 | CC BY 4.0 (학술 사용 허가) |

주요 Feature 그룹

* 인구통계: 나이(10세 단위 구간), 성별, 인종
* 입원 정보: 입원 기간, 검사 횟수, 시술 횟수, 처방 약물 수, 진단 수
* 임상 검사: 당화혈색소(A1Cresult), 혈당(max_glu_serum)
* 진단 코드: ICD-9 기반 주진단/부진단 3개 (diag_1, diag_2, diag_3)
* 당뇨 약물: 24개 약물 컬럼 (용량 변화: Up / Down / Steady / No)
* 이전 방문 이력: 외래·응급·입원 방문 횟수

## 2.3 데이터 정제

원본 데이터에는 분석 전 반드시 처리해야 할 세 가지 구조적 문제가 있다.

① 재입원 불가 환자 제거

`discharge_disposition_id`가 {11, 13, 14, 19, 20, 21}인 환자는 퇴원이 아닌 사망 또는 호스피스 이송에 해당한다. 이들은 정의상 재입원이 불가능하므로 타깃 변수가 항상 0이 되어 모델을 오도한다. 해당 케이스 2,423건(2.38%)을 제거했다.

② 환자별 첫 입원만 유지

원본 데이터에는 같은 환자(`patient_nbr`)의 여러 입원 기록이 중복 포함되어 있다(30,248건). 동일 환자의 이후 입원 기록을 훈련에 포함하면, 미래 정보가 과거 예측에 암묵적으로 누출(data leakage)될 수 있다. 따라서 가장 이른 `encounter_id`의 기록만 유지했다.

③ 고결측 컬럼 처리

`weight`(96.9% 결측)는 제거했다. `medical_specialty`(49.1% 결측)와 `payer_code`(39.6% 결측)는 결측 자체가 "해당 정보 없음"이라는 임상적 의미를 가질 수 있으므로, 결측을 별도 카테고리("Unknown")로 처리하거나 제거 여부를 실험에서 비교한다.

## 2.4 Selective Prediction 타당성

이 데이터셋에서 예측 거부가 실질적 의미를 갖는 이유는 다음과 같다.

첫째, 오류 비용이 비대칭적이다. 재입원 고위험 환자를 저위험으로 예측(False Negative)하면 적절한 퇴원 처치가 이루어지지 않아 환자 상태가 악화될 수 있다. 반면 저위험 환자를 고위험으로 예측(False Positive)하면 불필요한 의료 자원이 낭비된다.

둘째, 불확실한 환자는 의료진이 판단하는 것이 합리적이다. 모델이 확신하기 어려운 경계선상의 환자는 임상적으로도 판단이 어려운 케이스일 가능성이 높다. 이런 환자를 사람에게 넘기는 것은 모델의 실패가 아니라 설계 의도다.

셋째, Coverage-Accuracy trade-off를 정량화할 수 있다. 병원 운영 관점에서 "전체 환자의 70%를 자동 처리하면서 그 중 90% 이상의 정확도를 달성한다"는 식의 구체적 운영 시나리오를 제시할 수 있다. 이는 모델의 실용성을 설득력 있게 전달하는 분석 방식이다.

---

# 3. 방법론

## 3.1 실험 설계 개요

본 프로젝트는 다음 세 단계로 실험을 구성한다.

1. 전처리 및 피처 엔지니어링: 섹션 2.3의 정제 규칙을 적용한 뒤, 범주형 변수 인코딩과 ICD-9 진단 코드 그룹핑을 수행한다.
2. 모델 학습 및 하이퍼파라미터 튜닝: 베이스라인(Logistic Regression)과 두 개의 주력 모델(Random Forest, XGBoost)을 학습하고, 주력 모델은 RandomizedSearchCV로 최적 파라미터를 탐색한다.
3. Selective Prediction 분석: 각 모델의 confidence threshold τ를 0.50부터 0.95까지 스윕하며 Coverage–Accuracy 트레이드오프를 측정한다.

데이터는 70 / 15 / 15 비율로 층화 분할(stratified split)하였다. 클래스 비율을 유지하면서 분할함으로써 양성 비율(~9%)이 세 분할에서 동일하게 유지된다.

| 분할 | 행 수 | 양성 비율 |
| --- | --- | --- |
| Train | 48,990 | 9.0% |
| Validation | 10,498 | 9.0% |
| Test | 10,499 | 9.0% |

모든 최종 평가는 Test 세트에서만 수행하며, Validation 세트는 하이퍼파라미터 탐색 중 조기 종료 기준으로만 활용한다.

## 3.2 평가 지표

클래스 불균형(양성 9%)이 심한 이진 분류 문제이므로, 단일 지표로 모델을 평가하면 오해를 유발할 수 있다. 본 프로젝트에서는 다음 지표를 종합적으로 사용한다.

표준 지표

* ROC-AUC: 임계값에 무관한 모델의 판별력. 불균형 데이터에서 Accuracy보다 신뢰도가 높다.
* F1 (binary): 양성 클래스에 대한 Precision과 Recall의 조화평균. 재입원 환자 탐지 능력을 직접 반영한다.
* Recall (Sensitivity): 실제 재입원 환자 중 모델이 잡아낸 비율. 의료 도메인에서 False Negative 비용이 크므로 핵심 지표로 취급한다.

Selective Prediction 특화 지표

* Coverage: $\frac{|\{x : \text{conf}(x) \geq \tau\}|}{|X|}$ — 전체 환자 중 모델이 예측을 출력한 비율.
* Selective Accuracy: Coverage된 샘플에서만 계산한 Accuracy.
* Coverage–Accuracy Curve: τ를 0.50에서 0.99까지 스윕하며 (Coverage, Selective Accuracy) 궤적을 그린 곡선. 두 지표 간 트레이드오프를 시각화한다.

## 3.3 모델 선택

### 베이스라인: Logistic Regression

선형 모델을 베이스라인으로 채택한 이유는 두 가지다. 첫째, 해석이 용이하여 피처 중요도를 계수(coefficient)로 직접 읽을 수 있다. 둘째, 모든 주력 모델의 성능이 이를 상회해야 비로소 복잡도 증가가 정당화된다.

클래스 불균형 보정을 위해 `class_weight='balanced'`를 적용하였다. 이 옵션은 각 클래스의 샘플 수에 반비례하는 가중치를 손실 함수에 부여하여, 소수 클래스(재입원)를 더 강하게 학습하도록 유도한다.

### 주력 모델 ①: Random Forest

앙상블 기반 비선형 모델로, 다음 특성이 이 문제에 적합하다.

* 피처 간 비선형 상호작용을 자동으로 포착한다.
* 트리 기반 모델이므로 결측값·이상치에 상대적으로 강건하다.
* `predict_proba`가 각 트리의 투표 비율로 정의되어 confidence 추정에 자연스럽게 활용된다.

`class_weight='balanced'`를 적용하고, 탐색 공간을 아래와 같이 설정하였다.

| 하이퍼파라미터 | 탐색 범위 |
| --- | --- |
| `n_estimators` | {100, 200, 300, 500} |
| `max_depth` | {None, 10, 20, 30} |
| `min_samples_split` | {2, 5, 10} |
| `min_samples_leaf` | {1, 2, 4} |
| `max_features` | {√p, log₂p} |

### 주력 모델 ②: XGBoost

Gradient Boosting 계열 모델로, 순차적 잔차 학습을 통해 앙상블을 구성한다.

* 일반적으로 tabular 데이터에서 최고 수준의 성능을 보인다.
* `scale_pos_weight` 파라미터로 클래스 불균형을 직접 보정할 수 있다. 본 실험에서 음성/양성 비율은 10.14:1이므로 이 값을 `scale_pos_weight`로 설정하였다.

| 하이퍼파라미터 | 탐색 범위 |
| --- | --- |
| `n_estimators` | {100, 200, 300, 500} |
| `max_depth` | {3, 4, 5, 6, 7} |
| `learning_rate` | {0.01, 0.05, 0.1, 0.2} |
| `subsample` | {0.6, 0.7, 0.8, 1.0} |
| `colsample_bytree` | {0.6, 0.7, 0.8, 1.0} |
| `min_child_weight` | {1, 3, 5, 7} |
| `gamma` | {0, 0.1, 0.2} |

## 3.4 Selective Prediction 구현

본 프로젝트의 Selective Predictor는 다음 수식에 따라 작동한다.

$$f(x) = \begin{cases} \hat{y} & \text{if } \text{conf}(x) \geq \tau \\ \perp & \text{otherwise} \end{cases}$$

$$\text{conf}(x) = \max\bigl(P(Y=1 \mid x),\ P(Y=0 \mid x)\bigr)$$

구현 관점에서 `SelectivePredictor`는 임의의 sklearn 호환 분류기를 감싸는 래퍼 클래스로, `predict_proba`의 출력 중 최댓값을 confidence로 사용한다. τ는 고정 값이 아니라 사후 튜닝 가능한 파라미터로 설계되어, 모델을 재학습하지 않고도 운영 요구사항(Coverage 목표 등)에 맞게 조정할 수 있다.

---

# 4. 실험 결과

## 4.1 피처 엔지니어링 결과

원본 50개 컬럼에서 정제·인코딩을 거쳐 최종 169개 피처가 생성되었다. 주요 변환 내역은 다음과 같다.

| 변환 | 설명 |
| --- | --- |
| ICD-9 그룹핑 | diag_1/2/3을 9개 질병 카테고리로 매핑 (Diabetes, Circulatory, Respiratory 등) |
| 나이 순서 인코딩 | '[0-10)' → 0, '[10-20)' → 1, … '[90-100)' → 9 |
| 약물 순서 인코딩 | No=0, Steady=1, Up=2, Down=3 (21개 약물 컬럼) |
| 임상검사 순서 인코딩 | A1Cresult, max_glu_serum 각 0–3 스케일 |
| One-Hot Encoding | race, medical_specialty, payer_code, diag_1/2/3 등 |
| 결측 처리 | medical_specialty·payer_code → "Unknown" 카테고리 |

## 4.2 전체 예측 모드 성능 비교

모든 모델을 θ=0.5 (threshold 없이 전체 예측) 상태로 Test 세트에서 평가한 결과는 다음과 같다.

| 모델 | Accuracy | Precision | Recall | F1 (binary) | F1 (macro) | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6268 | 0.1288 | 0.5472 | 0.2085 | 0.4822 | 0.6297 |
| Random Forest | 0.9044 | 0.3057 | 0.0509 | 0.0873 | 0.5184 | 0.6510 |
| XGBoost | 0.6826 | 0.1512 | 0.5493 | 0.2372 | 0.5184 | 0.6681 |

XGBoost가 ROC-AUC(0.668)와 F1 binary(0.237) 모두에서 최고 성능을 기록하였다. Random Forest는 Accuracy(0.904)가 높지만, 이는 다수 클래스(음성)를 대부분 올바르게 예측한 결과로, Recall이 0.051에 불과해 실제 재입원 환자를 거의 탐지하지 못한다.

## 4.3 하이퍼파라미터 튜닝 결과

RandomizedSearchCV(30회 반복, 3-fold CV, 최적화 기준: ROC-AUC)로 탐색한 최적 파라미터는 다음과 같다.

Random Forest 최적 파라미터

| 파라미터 | 최적값 |
| --- | --- |
| `n_estimators` | 300 |
| `max_depth` | 30 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 4 |
| `max_features` | sqrt |

XGBoost 최적 파라미터

| 파라미터 | 최적값 |
| --- | --- |
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.6 |
| `min_child_weight` | 3 |
| `gamma` | 0.1 |

XGBoost의 최적 `max_depth`가 4로 얕게 결정된 것은 과적합 방지를 위한 정규화 효과로 해석된다. 반면 Random Forest는 `max_depth=30`으로 깊은 트리를 허용하되 `min_samples_leaf=4`로 리프 노드의 최소 샘플 수를 제한하여 분산을 통제하였다.

## 4.4 Selective Prediction 성능

### Confidence Threshold Sweep — XGBoost

XGBoost의 τ 스윕 결과는 다음과 같다. τ를 높일수록 Coverage는 줄어들지만, 예측된 샘플에서의 정확도와 F1이 지속적으로 향상된다.

| τ | Coverage | Accuracy | Precision | F1 (binary) | Recall | 기권율 |
|---|---|---|---|---|---|---|
| 0.50 | 100.0% | 0.6826 | 0.1512 | 0.2372 | 0.5493 | 0.0% |
| 0.55 | 72.5% | 0.7266 | 0.1728 | 0.2649 | 0.5665 | 27.5% |
| 0.60 | 46.3% | 0.7625 | 0.2048 | 0.3057 | 0.6019 | 53.7% |
| 0.65 | 25.4% | 0.7766 | 0.2430 | 0.3591 | 0.6872 | 74.6% |
| 0.70 | 11.9% | 0.7775 | 0.2969 | 0.4335 | 0.8030 | 88.1% |
| 0.75 | 4.8% | 0.7874 | 0.3584 | 0.5135 | 0.9048 | 95.2% |
| 0.80 | 1.6% | 0.8713 | 0.4285 | 0.5769 | 0.8824 | 98.4% |
| 0.85 | 0.7% | 0.9452 | 0.7272 | 0.8000 | 0.8889 | 99.3% |

τ=0.70 지점에서 XGBoost는 전체 환자의 11.9%(약 1,245명/10,499명)에 대해 예측을 출력한다. 이때 선별된 환자군 내의 실제 정답 비율(Precision)은 약 29.7%로, 기존 기저율(9%) 대비 3배 이상 높은 밀도로 고위험군을 타겟팅하며 Recall 80.3%, F1 0.434를 달성한다. 나머지 88.1%의 불확실한 케이스는 의료진에게 검토를 넘긴다.

### Confidence Threshold Sweep — Random Forest

| τ | Coverage | Accuracy | Precision | F1 (binary) | Recall | 기권율 |
|---|---|---|---|---|---|---|
| 0.50 | 100.0% | 0.9044 | 0.3057 | 0.0873 | 0.0509 | 0.0% |
| 0.60 | 89.4% | 0.9225 | 0.9112 | 0.0268 | 0.0136 | 10.6% |
| 0.70 | 59.1% | 0.9402 | 1.0000 | 0.0054 | 0.0027 | 40.9% |
| 0.80 | 16.9% | 0.9605 | - | 0.0000 | 0.0000 | 83.1% |
| 0.90 | 0.8% | 1.0000 | - | 0.0000 | 0.0000 | 99.3% |

Random Forest는 τ=0.70에서 Coverage 59.1%, Accuracy 94.0%를 달성하지만 F1이 0.005로 사실상 재입원 환자를 탐지하지 못한다. 높은 confidence로 예측하는 샘플 대부분이 음성(비재입원)이기 때문이다.

---

# 5. 결과 분석 및 토론

## 5.1 최고 성능 모델 분석

ROC-AUC와 F1 binary를 종합하면 XGBoost가 가장 우수한 모델이다. 단순히 성능 지표가 높을 뿐만 아니라, Selective Prediction 맥락에서 두 모델의 질적 차이는 극명하게 드러난다.

Random Forest의 90% Accuracy는 극심한 클래스 불균형이 만든 '허영 지표(Vanity Metric)'에 불과하다. 특히 모델이 스스로 높은 확률로 확신하는 구간에서조차 실제 재입원 환자를 사실상 전원 놓치는 '과확신(Overconfidence)' 현상을 보였다. RF를 배포하면 모델이 확신한다고 표시한 환자 중 재입원 고위험자가 거의 포함되지 않아 시스템 자체가 무력화된다.

반면 XGBoost는 τ=0.70에서 Coverage 11.9% 내에 실제 재입원 환자의 80% 이상을 포착해 낸다. 즉, 자신이 아는 것과 모르는 것을 정확히 구분하는 '신뢰할 수 있는(Trustworthy)' 모델임을 증명했다. 이는 전체 AUC만 보고 모델을 선택하면 실제 운영에서 예상과 전혀 다른 결과가 나올 수 있음을 시사한다.

## 5.2 클래스 불균형이 학습을 어렵게 만드는 이유

이 데이터셋에서 가장 큰 도전은 극심한 클래스 불균형(9% vs. 91%) 이다.

* Accuracy의 함정: 모든 환자를 "재입원 없음"으로 예측하는 trivial classifier도 Accuracy 91%를 달성한다. Random Forest의 0.904 Accuracy는 이 trivial 수준과 거의 동일하다.
* Confidence 왜곡: 불균형 데이터에서 트리 앙상블은 다수 클래스에 높은 확률을 부여하는 경향이 있다. RF에서 confidence가 높은 샘플이 대부분 음성인 이유가 여기에 있다.
* Recall–Precision 긴장: 재입원 환자를 더 잡으려 할수록(Recall↑) 정상 환자를 잘못 분류하는 비율도 높아진다(Precision↓). 의료 도메인에서 이 트레이드오프는 명확한 운영 정책으로 해결해야 한다.

## 5.3 실패 케이스 분석

False Negative (재입원 → 정상 예측)의 특성

모든 모델에서 재입원 환자를 놓치는 경우가 가장 많다. 이러한 환자는 다음과 같은 특성을 가지는 경향이 있다.

* `number_inpatient`(이전 입원 횟수)가 낮거나 0인 환자: 이전 방문 이력이 없으면 모델이 위험 신호를 학습할 근거가 없다.
* 진단 코드가 순환기계(Circulatory)가 아닌 "Other"에 해당하는 환자: 모델이 학습한 패턴에서 벗어난 희귀 진단의 경우 예측 불확실성이 높다.
* confidence가 0.50~0.55 구간에 집중된 경계선 샘플: 양성·음성 사이에서 모델이 결정을 내리지 못하는 케이스다.

Selective Prediction의 자가 수정

τ를 높이면 이러한 경계선 샘플들이 기권 대상이 되어 의료진에게 넘어간다. τ=0.70에서 XGBoost가 기권하는 88.1%의 환자 중 상당수가 바로 이 경계선 케이스이며, 이는 Selective Prediction이 모델의 가장 취약한 구간을 정확히 식별하고 있음을 시사한다.

## 5.4 Selective Prediction의 임상적 해석

τ=0.70에서 XGBoost의 운영 시나리오를 실제 병원 맥락에서 해석하면 다음과 같다.
"하루 퇴원 환자 100명 중 12명에 대해서는 모델이 자동으로 판정하며, 그 12명 안에서 재입원 고위험자의 80%를 식별한다. 나머지 88명은 의료진이 직접 검토한다."

이는 완전 자동화보다는 의료진의 의사결정을 선별적으로 보조하는 방식으로, 임상 현장의 실용성에 부합한다. 완전 자동화(τ=0.50)와 비교하면:

| 시나리오 | 자동 처리율 | Recall | 의료진 검토 부담 |
| --- | --- | --- | --- |
| 완전 자동화 (τ=0.50) | 100% | 54.9% | 없음 |
| SP τ=0.70 | 11.9% | 80.3% (자동화 구간) | 88.1% |
| SP τ=0.85 | 0.7% | 88.9% (자동화 구간) | 99.3% |

이 시스템의 가장 큰 실무적 장점은 병원의 가용 리소스에 맞춰 Coverage-Precision Trade-off를 동적으로 운영할 수 있다는 점이다.

인력이 부족할 때 (타율 극대화): 퇴원 관리 인력이 부족한 성수기에는 τ=0.85로 높여 자동 처리율을 0.7%로 대폭 줄이는 대신, 선별된 환자 내의 타율(Precision)을 72% 이상으로 끌어올려 가장 위험한 환자에게만 확실하게 자원을 집중할 수 있다.

인력이 충분할 때 (방어율 극대화): 의료진 인력이 충분하다면 τ=0.65로 낮춰 타율은 다소 희생하더라도 더 넓은 범위의 재입원 위험 환자를 선제적으로 방어할 수 있다.

## 5.5 추가 개선 방향

시간과 자원이 더 있다면 다음을 시도할 수 있다.

1. Conformal Prediction 적용: 단순 max(P) 대신 통계적으로 보장된 prediction set을 구성하면, 특정 오류율 보증 하에 coverage를 제어할 수 있다.
2. SMOTE / Class-weighted Focal Loss: 클래스 불균형 보정 기법을 추가 적용하여 Recall 개선을 시도한다.
3. 의료 전문 피처 추가: HbA1c 변화 추이, 퇴원 후 처방 변경 여부 등 임상적으로 의미 있는 파생 피처를 생성한다.
4. SHAP 분석: 개별 예측에 대한 설명 가능성을 확보하여 의료진이 모델 판단을 신뢰할 근거를 제공한다.
5. 단순 확률을 대체할 정교한 불확실성(Uncertainty) 지표 도입
    현재의 선택적 예측(Selective Prediction)은 모델이 출력하는 단순 확률값(predict_proba)의 최댓값을 확신도(Confidence)로 사용하고 있다. 그러나 트리 기반 앙상블 모델은 클래스 불균형 데이터에서 극단적인 확률값을 출력하며 틀리는 '과확신(Overconfidence)' 현상을 보일 위험이 크다. 이를 근본적으로 해결하기 위해 다음의 기법 도입을 고려할 수 있다.

    * 앙상블 분산(Ensemble Variance) 활용: 단순 다수결 비율이 아닌, 앙상블을 구성하는 개별 트리들 간의 예측 의견 충돌(분산) 정도를 불확실성 지표로 사용한다.

    * 확률 보정(Probability Calibration): Platt Scaling이나 Isotonic Regression을 적용하여, 다수 클래스 쪽으로 쏠려 있는 모델의 출력 확률을 실제 발생 확률과 일치하도록 교정(Calibrate)한 뒤 임계값(Threshold)을 적용한다.

---

# 6. 결론

본 프로젝트는 당뇨 환자의 30일 내 재입원 예측에 Selective Prediction(선택적 예측) 프레임워크를 적용하였다. 핵심 아이디어는 모델이 스스로의 불확실성을 추정하여, 확신도가 낮은 케이스에 대해서는 예측을 거부하고 의료진에게 위임하는 것이다.

세 모델의 전체 예측 ROC-AUC는 0.630–0.668로 비슷한 수준이다. 그러나 이 보고서의 핵심 발견은 AUC가 유사한 모델도 불확실성 추정의 질은 크게 다를 수 있다는 점이다. XGBoost는 confidence가 높은 구간에서 실제 재입원 환자를 80% 이상 포착하는 반면, Random Forest는 같은 구간에서 과확신(Overconfidence)의 오류를 범하며 재입원 환자를 사실상 전원 놓쳤다. 단일 지표로는 드러나지 않는 이 치명적인 차이가 Selective Prediction 분석을 통해 비로소 가시화되었다.

이 결과는 의료 AI 배포에서 단일 성능 지표의 한계를 명확히 보여준다. 본 연구는 의료 AI의 목표가 단순히 '단일 성능 지표(AUC, Accuracy)의 극대화'에 머물러서는 안 됨을 시사한다. 맹목적인 예측 강요가 아닌, 모델 스스로 불확실성을 인정하고 신뢰할 수 있는 기권(Abstain)을 통해 전문가(의료진)와 협업하는 시스템을 구축할 때 비로소 머신러닝이 실제 임상 워크플로우에 안전하고 효과적으로 안착할 수 있을 것이다.

---

# 참고문헌

1. Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. BioMed Research International.
2. Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. Advances in Neural Information Processing Systems (NeurIPS).
3. Dua, D., & Graff, C. (2019). UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008. University of California, Irvine. [https://archive.ics.uci.edu/dataset/296](https://archive.ics.uci.edu/dataset/296)
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
5. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

---

본 프로젝트에서는 Claude(Anthropic)를 보고서 초안 작성 및 문장 다듬기에 활용하였다. 실험 설계, 코드 구현, 결과 해석은 직접 수행하였으며, AI 출력물은 검토 후 수정하여 반영하였다.