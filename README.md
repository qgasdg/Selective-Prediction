# Selective Prediction for Diabetes Readmission

당뇨 환자의 30일 내 재입원 예측에 **Selective Prediction(선택적 예측)** 프레임워크를 적용한 머신러닝 텀 프로젝트입니다.

모델은 예측 확신도가 낮은 케이스에 대해 예측을 거부(abstain)하고 의료진에게 위임합니다.

$$f(x) = \begin{cases} \hat{y} & \text{if } \text{conf}(x) \geq \tau \\ \perp & \text{otherwise} \end{cases}$$

## Dataset

**Diabetes 130-US Hospitals (UCI, ID: 296)** — 1999~2008년 미국 130개 병원 데이터

| 항목 | 내용 |
|---|---|
| 원본 크기 | 101,766 rows × 50 columns |
| 정제 후 크기 | 69,990 rows × 44 columns |
| 타깃 변수 | 30일 내 재입원 여부 (이진) |
| 양성 비율 | ~9.0% |

## Models

| 모델 | 역할 |
|---|---|
| Logistic Regression | 베이스라인 |
| Random Forest | 주력 모델 ① |
| XGBoost | 주력 모델 ② |

하이퍼파라미터 탐색은 `RandomizedSearchCV` (30회 반복, 3-fold CV, ROC-AUC 최적화)로 수행했습니다.

## Key Results

전체 예측 모드 (τ = 0.50):

| 모델 | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.627 | 0.547 | 0.209 | 0.630 |
| Random Forest | 0.904 | 0.051 | 0.087 | 0.651 |
| XGBoost | 0.683 | 0.549 | 0.237 | **0.668** |

XGBoost Selective Prediction (threshold sweep):

| τ | Coverage | Recall | F1 |
|---|---|---|---|
| 0.50 | 100.0% | 0.549 | 0.237 |
| 0.60 | 46.3% | 0.602 | 0.306 |
| 0.70 | 11.9% | 0.803 | 0.434 |
| 0.85 | 0.7% | 0.889 | 0.800 |

τ=0.70에서 전체 환자의 11.9%에 대해 예측을 출력하며, 그 구간에서 실제 재입원 환자의 80%를 포착합니다.

## Project Structure

```
.
├── main.py                  # 학습 파이프라인 진입점
├── requirements.txt
├── data/
│   └── diabetic_data.csv
├── src/
│   ├── preprocess.py        # 데이터 정제 및 피처 엔지니어링
│   ├── models.py            # SelectivePredictor, RF/XGB 튜닝
│   ├── evaluate.py          # 평가 지표 및 시각화
│   └── eda.py               # EDA 플롯
└── results/                 # 실험 결과 (이미지, CSV, JSON)
```

## Reproduction

### 1. 데이터 준비

[UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)에서 `diabetic_data.csv`를 다운로드한 후 `data/` 디렉토리에 위치시킵니다.

```
data/
└── diabetic_data.csv   # 101,766 rows × 50 columns
```

### 2. 환경 설정

Python 3.8 이상 환경에서:

```bash
pip3 install -r requirements.txt
```

### 3. 표 재현

report의 표를 그대로 재현하려면 `reproduce.py`를 사용합니다. 하이퍼파라미터 탐색 없이 report에서 찾은 best params와 고정 시드(`SEED=42`)로 학습합니다.

```bash
python reproduce.py
```

출력되는 표:

- **Table 1** — 전체 모델 비교 (Accuracy / Precision / Recall / F1 / ROC-AUC)
- **Table 2** — XGBoost Selective Prediction threshold sweep (τ = 0.50 ~ 0.85)
- **Table 3** — Random Forest Selective Prediction threshold sweep

### 4. 전체 파이프라인 재실행 (느림)

하이퍼파라미터 탐색부터 EDA·시각화까지 전 과정을 다시 돌리려면 `main.py`를 사용합니다.

```bash
python main.py
```

실행 순서:

1. **EDA** — 타깃 분포, 주요 피처 분포 시각화 (`results/eda/`)
2. **전처리** — 사망/호스피스 환자 제거, 환자별 첫 입원만 유지, ICD-9 그룹핑, One-Hot 인코딩 (원본 50 → 169 피처)
3. **데이터 분할** — 70/15/15 층화 분할 (Train 48,990 / Val 10,498 / Test 10,499)
4. **모델 학습** — RF·XGB는 `RandomizedSearchCV` (30회 반복, 3-fold CV, ROC-AUC 최적화)
5. **Selective Prediction** — τ를 0.50~0.99 스윕하며 Coverage–Accuracy 트레이드오프 측정

결과는 `results/`에 저장됩니다 (이미지, CSV, `best_hyperparams.json`).

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data` | `data/diabetic_data.csv` | 데이터 경로 |
| `--n_iter` | `30` | RandomizedSearchCV 반복 횟수 |
| `--tau` | `0.70` | Selective Predictor confidence threshold |

## References

1. Strack et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International.*
2. Geifman & El-Yaniv (2017). Selective Classification for Deep Neural Networks. *NeurIPS.*
3. UCI ML Repository — Diabetes 130-US Hospitals (ID: 296)
4. Chen & Guestrin (2016). XGBoost. *KDD.*
5. Breiman (2001). Random Forests. *Machine Learning.*
