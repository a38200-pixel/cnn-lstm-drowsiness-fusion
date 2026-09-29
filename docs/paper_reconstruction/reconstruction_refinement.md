# Paper Reconstruction Refinement Plan

## 1. 단계와 목적

현재 프로젝트 단계는 다음과 같다.

```text
Stage 1 — INITIAL PAPER-INFORMED RECONSTRUCTION BASELINE: COMPLETED
Stage 2 — PAPER RECONSTRUCTION REFINEMENT: PLANNED / NEXT
Stage 3 — PROPOSED CONTEXT MODEL IMPROVEMENT: NOT STARTED
```

Stage 1에서는 원논문에서 확인되는 구조와 parameter count로부터 추론한 구조를 바탕으로 VGG19/VGG16 + LSTM baseline을 구현하고 4-fold 실험을 완료했다. 그러나 현재 결과와 원논문 Table III 사이에는 상당한 차이가 있으므로, Paper Reconstruction 전체가 완료되었다고 표현하지 않는다.

Reconstruction Refinement의 목적은 원논문 성능 숫자에 맞추는 것이 아니다. 원논문에서 공개되지 않았지만 결과에 큰 영향을 줄 수 있는 plausible reconstruction conditions를 사전에 제한해 정의하고, 핵심 구조와 평가 프로토콜을 유지하면서 조건별 sensitivity를 체계적으로 분석하는 것이다.

이 단계는 `reconstruction refinement`, `sensitivity analysis for unreported conditions`, `plausible reconstruction conditions`로 표현한다. `original settings recovery`, `exact reproduction`, `tuning until paper accuracy is reached`로 표현하지 않는다.

## 2. 현재 Baseline과 성능 차이

현재 official baseline은 fixed 30 epochs의 final epoch 결과다.

| Source | Model | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Current R0 | VGG19 + LSTM | 79.94% | 80.88% | 75.28% | 77.94% |
| Current R0 | VGG16 + LSTM | 80.04% | 81.90% | 74.15% | 77.72% |
| Paper Table III | VGG19 + LSTM | 90.53% | 91.74% | 91.28% | 91.50% |
| Paper Table III | VGG16 + LSTM | 89.39% | 91.81% | 89.09% | 90.42% |

VGG19 F1 차이는 `91.50 - 77.94 = 13.56 percentage points`다. 원논문 본문의 VGG19 F1 91.46%와 Table III의 91.50%도 서로 다르므로 Table III의 91.50%를 reference 주값으로 사용하고 불일치를 함께 기록한다.

현재 결과는 paper-informed reconstruction baseline이며, 원논문과 동일한 조건의 reproduction이라고 주장하지 않는다.

## 3. 조건 분류

### PAPER-REPORTED

- Dataset: SUST-DDD, 2,074 videos
- Class distribution: drowsy 975 / not drowsy 1,099
- Video unit: 10 seconds
- Input: 224×224×3, 20 frames/video
- VGG16/VGG19 사용, CNN output을 LSTM input으로 사용
- One LSTM layer, two fully connected layers
- ReLU/Sigmoid activation과 final Softmax 언급
- 75% training / 25% test
- 4-fold cross-validation, k=4

### STRUCTURALLY INFERRED

Fig.1의 LSTM parameter count `9,439,232`와 hidden size `512`를 single-bias LSTM 식에 대입하면 input size는 4096으로 역산된다.

```text
9,439,232 = 4 × 512 × (input_size + 512 + 1)
input_size = 4,096
```

따라서 다음 조건은 refinement 동안 변경하지 않는다.

- LSTM input size 4096
- LSTM hidden size 512
- Dense 512 → Dense 64 → Dense 2

### RECONSTRUCTION ASSUMPTION

Stage 1에서 실행을 위해 선택한 조건이다. ImageNet pretrained/frozen VGG, `fc1` after ReLU, uniform sampling, ImageNet normalization, Adam과 세부 hyperparameter, fixed 30 epochs, validation/checkpoint 정책 등이 여기에 해당한다.

### REFINEMENT CANDIDATE

원논문에 공개되지 않았고 성능 민감도가 클 가능성이 있어 비교를 검토하는 조건이다. 후보로 문서화된 값은 실행 조건으로 확정된 값이 아니다. 각 실험 전 condition ID, 변경 변수, fixed epoch 및 config를 사전 확정한다.

## 4. Refinement 동안 고정할 핵심 구조

```text
20-frame sequence
→ VGG19 4096D feature
→ LSTM input_size 4096
→ LSTM hidden_size 512
→ Dense 512
→ ReLU
→ Dense 64
→ Sigmoid
→ Dense 2
→ classification
```

4096D feature와 LSTM512 사이에 projection layer를 추가하지 않는다.

다음 변경은 Reconstruction Refinement 범위가 아니라 Stage 3 Proposed Context Model Improvement로 미룬다.

- BiLSTM, Attention, Temporal Pooling
- Feature bottleneck 또는 4096D→LSTM512 사이 projection
- 새로운 CNN backbone
- 새로운 fusion architecture
- EAR/MAR/Head Pose 결합

## 5. Refinement Conditions

### R0 — Current Baseline Reference

R0는 기존 결과를 그대로 사용하는 reference다. 재학습하거나 산출물을 다시 생성하지 않는다.

- ImageNet pretrained VGG19, frozen backbone
- Offline feature cache
- First FC after ReLU인 `fc1` 4096D feature
- ImageNet normalization
- Deterministic uniform 20-frame sampling
- Adam, learning rate 0.0001, weight decay 0.00001
- Batch size 16, fixed 30 epochs, seed 42
- Scheduler/validation/early stopping/best checkpoint selection 없음
- Official metric은 final epoch

### R1 — VGG Feature Extraction Position

목적은 논문이 공개하지 않은 VGG feature extraction 위치의 민감도를 확인하는 것이다.

현재 extractor에서 torchvision VGG classifier는 다음과 같이 해석된다.

| Name | `classifier` 실행 범위 | 4096D representation |
|---|---|---|
| `fc1` current | indices 0–1 | First Linear → ReLU |
| `fc2` candidate | indices 0–4 | First Linear → ReLU → Dropout → Second Linear → ReLU |

Extractor는 숫자 slicing 대신 `fc1`, `relu1`, `dropout1`, `fc2`, `relu2`를 명시적으로 실행한다. Evaluation mode에서는 Dropout이 비활성화된다.

R1은 feature extraction point만 `fc1`에서 `fc2`로 변경하고 LSTM 구조와 학습 조건은 R0와 동일하게 유지해 4-fold 실험을 완료했다. R1 전용 config는 `configs/paper_reconstruction/vgg19_lstm_fc2.yaml`이다.

- Feature cache: `data/features/paper_reconstruction/refinement/R1/vgg19/fc2/`
- Manifest: `data/features/paper_reconstruction/refinement/R1/vgg19/fc2_manifest.csv`
- Output: `outputs/paper_reconstruction/refinement/R1_vgg19_fc2/`
- MLflow parent run: `R1_vgg19_fc2`

기존 R0 `fc1` cache, output, history, metrics와 checkpoint를 수정하거나 덮어쓰지 않았다.

### R2 — CNN Training Policy

원논문은 CNN을 training한 뒤 output을 LSTM에 입력한 것으로 읽을 수 있는 문장을 포함하지만 pretrained/frozen/fine-tuned 정책은 공개하지 않는다.

검토 범위는 다음과 같다.

- Frozen ImageNet-pretrained VGG19: R0 reference
- Partially fine-tuned VGG19: refinement candidate
- Fully fine-tuned VGG19: refinement candidate

Fine-tuning은 offline frozen feature cache 방식과 학습 pipeline을 바꿀 수 있는 큰 변경이다. 계산량, GPU memory, dataset 규모 및 논문 문구의 plausibility를 먼저 분석하고, 실제 실행 전 R2 조건 하나만 사전 확정한다. 위 후보 전체를 자동으로 실행하는 계획은 아니다.

### R3 — Optimizer / Learning Rate

원논문은 optimizer와 learning rate를 공개하지 않았다. R0는 Adam, learning rate 0.0001, weight decay 0.00001이다.

제한된 refinement candidate로 다음을 검토할 수 있다.

- Adam의 사전 정의된 제한적 learning-rate 대안
- SGD + momentum과 그에 맞는 사전 정의 learning rate

정확한 후보 수와 값은 실행 전 별도 decision/config에서 고정한다. Test 결과를 본 뒤 learning rate를 계속 추가하지 않으며, optimizer와 learning rate를 동시에 바꿀 경우 하나의 명시적인 policy condition으로 취급한다.

### R4 — Fixed Training Duration

원논문은 epoch 수를 공개하지 않았다. R0는 fixed 30 epochs다. 기존 history의 held-out 최고 성능 epoch는 fold마다 달랐고 후반 과적합 경향도 있었지만, 이는 post-hoc diagnostic only이다.

R4를 실행한다면 fixed epoch 수를 학습 전에 사전 정의한다. Test metric으로 best epoch를 선택하지 않고, 사전 정의된 final epoch만 official metric으로 사용한다.

### R5 — Preprocessing / Frame Sampling

낮은 우선순위 후보이며 R1–R4 이후에도 큰 차이가 남을 때 검토한다.

R0 조건은 deterministic uniform 20-frame sampling, OpenCV BGR→RGB, bilinear 224×224 resize, ImageNet normalization, augmentation 없음이다. 원논문이 공개하지 않은 sampling, interpolation, normalization, augmentation의 plausible alternative를 검토하되, 실행 시 한 condition에서 가능한 한 하나의 assumption만 변경한다.

## 6. 우선순위와 의사결정 Gate

```text
R0 existing reference
→ R1 feature extraction point
→ R2 CNN training policy
→ R3 optimizer / learning-rate policy
→ R4 fixed training duration
→ R5 preprocessing / sampling, if still warranted
```

R1은 기존 extractor가 이미 `fc2`를 지원하고 별도 cache로 격리할 수 있어 첫 후보로 둔다. R2는 pipeline과 계산량 변화가 커서 구현 가능성 검토와 사전 조건 확정 후 진행한다. 실제 결과, 비용과 해석 가능성에 따라 후속 순서를 조정할 수 있지만, test 성능에 맞추기 위한 무제한 탐색은 하지 않는다.

## 7. 실험 원칙

- 가능한 한 한 번에 하나의 reconstruction assumption만 변경한다.
- Condition ID, 목적, 변경 변수, 불변 변수, fixed epoch를 실행 전에 기록한다.
- R0→R1에서는 feature extraction point만 변경한다.
- R0→R3에서는 사전 정의한 optimizer 또는 LR policy만 변경한다.
- 모든 결과를 보존하며 성능이 하락한 condition도 삭제하지 않는다.
- Paper score에 가까워진 조건만 선택적으로 보고하지 않는다.
- Test 결과를 보고 후보를 반복 추가하지 않는다.

분석에는 absolute metric 변화, VGG19/VGG16 상대 관계, fold variability, precision/recall balance, training behavior와 overfitting tendency를 포함한다. 원논문의 `VGG19 > VGG16` 경향은 참고하되 그 관계를 맞추기 위해 test에 조건을 조정하지 않는다.

## 8. Evaluation Protocol

모든 refinement condition은 다음 protocol을 유지한다.

- Label-stratified 4-fold
- Fold마다 approximately 75% training / 25% held-out test
- Separate validation 없음
- Early stopping 없음
- Best checkpoint selection 없음
- Test 기반 best-epoch selection 없음
- Condition별 사전 정의 fixed epoch의 final metric 사용
- Drowsy를 positive class로 평가
- Fold mean과 population standard deviation (`ddof=0`) 보고

Held-out test metric은 각 fold의 최종 성능 평가에 사용한다. Per-epoch test history를 기록하더라도 best epoch 값은 post-hoc diagnostic only이며 official result로 사용하지 않는다. 원논문은 validation, early stopping, epoch/checkpoint selection 정책을 공개하지 않았다.

## 9. VGG16의 역할

Primary refinement backbone은 VGG19다. 원논문 Table III의 최고 모델이며 현재 프로젝트에서도 primary baseline으로 사용했기 때문이다.

모든 condition을 처음부터 VGG16에서 반복하지 않는다. VGG19에서 해석 가치가 있는 plausible condition이 확인된 뒤 필요하면 동일 조건을 VGG16에 적용하여 backbone 간 상대적 경향을 비교한다.

## 10. 결과 기록 계획

Condition별로 최소한 다음을 표에 누적한다.

| Condition | Backbone | Accuracy | Precision | Recall | F1 | Fold F1 std | Status |
|---|---|---:|---:|---:|---:|---:|---|
| R0 | VGG19 | 79.94% | 80.88% | 75.28% | 77.94% | existing summary 참조 | Completed |
| R0 | VGG16 | 80.04% | 81.90% | 74.15% | 77.72% | existing summary 참조 | Completed |
| R1 | VGG19 | 79.51% | 81.45% | 73.24% | 76.91% | 3.99%p | Completed |
| R2 | VGG19 | — | — | — | — | — | Candidate not fixed |
| R3 | VGG19 | — | — | — | — | — | Candidate not fixed |
| R4 | VGG19 | — | — | — | — | — | Candidate not fixed |
| R5 | VGG19 | — | — | — | — | — | Deferred candidate |

## MLflow Tracking Plan

새로운 refinement experiment부터 MLflow로 기록한다. 기존 R0 결과를 재학습하거나 기존 MLflow 산출물로 소급 변환하지 않는다.

- Recommended experiment name: `paper-reconstruction-refinement`
- Condition별 parent run: `R1_vgg19_fc2`
- Fold별 nested child run: `fold_1` … `fold_4`

```text
R1_vgg19_fc2
├── fold_1
├── fold_2
├── fold_3
└── fold_4
```

Parent run에는 CV mean/std, reconstruction condition, experiment purpose와 condition description을 기록한다. Child run에는 fold, train/test size, optimizer, learning rate, weight decay, batch size, epochs, seed, backbone, feature extraction point, frozen/fine-tuned policy와 final accuracy/precision/recall/F1/loss를 기록한다. Per-epoch metric은 `step=epoch`으로 기록할 수 있다.

### Required tags

```text
stage = reconstruction_refinement
condition_id = R1
backbone = vgg19
feature_dim = 4096
evaluation_protocol = stratified_4fold_75_25
validation = none
early_stopping = false
checkpoint_selection = final_epoch
result_role = official
paper_condition_type = paper_reported | structurally_inferred | reconstruction_assumption
```

필요하면 `feature_extraction_point`, `cnn_training_policy`, `parent_condition`, `code_revision` tag를 추가한다.

### Artifacts

- 실행에 사용한 YAML config
- Fold별 `metrics.json`, `history.json`
- `cv_summary.json`
- Reconstruction condition description
- 필요하면 confusion matrix
- 필요성과 용량을 검토한 model checkpoint

Dataset과 대용량 feature cache는 MLflow artifact로 업로드하지 않는다. Feature cache는 기존 cache와 분리된 로컬 경로에서 관리한다.

## R0 vs R1 결과 비교

R0와 R1의 유일한 reconstruction condition 차이는 VGG19의 4096D feature extraction point다. R0는 `fc1 after ReLU`, R1은 `fc2 after ReLU`를 사용하며, LSTM 구조와 학습·평가 조건은 동일하다.

### REFINEMENT RESULT — Official Final-Epoch Comparison

Official result는 사전에 고정한 30 epochs의 final epoch metric이다. Test fold의 best epoch를 선택하지 않았다.

| Condition | Feature point | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| R0 | fc1 after ReLU | 79.94% | 80.88% | 75.28% | 77.94% |
| R1 | fc2 after ReLU | 79.51% | 81.45% | 73.24% | 76.91% |
| R1 − R0 | — | -0.43%p | +0.57%p | -2.04%p | -1.03%p |

R1 fold별 final-epoch 결과는 다음과 같다.

| Fold | Final F1 |
|---:|---:|
| 1 | 71.07% |
| 2 | 76.47% |
| 3 | 77.87% |
| 4 | 82.23% |

R1 official F1의 population std는 `3.99%p`, Recall의 population std는 `7.05%p`다.

### POST-HOC DIAGNOSTIC — Best-Epoch Comparison

> **POST-HOC DIAGNOSTIC ONLY — official result가 아니며 model/checkpoint selection에 사용하지 않는다.**

아래 값은 각 held-out test fold에서 test F1이 가장 높았던 epoch를 실험 종료 후 선택해 평균한 진단값이다.

| Condition | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| R0 fc1 | 81.97% | 81.19% | 80.31% | 80.70% |
| R1 fc2 | 80.86% | 78.49% | 82.56% | 80.28% |
| R1 − R0 | -1.11%p | -2.70%p | +2.25%p | -0.42%p |

R1 fold별 post-hoc best epoch는 다음과 같다.

| Fold | Best epoch | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 77.84% | 72.63% | 84.84% | 78.26% |
| 2 | 9 | 82.27% | 83.63% | 77.46% | 80.43% |
| 3 | 18 | 79.19% | 74.29% | 85.25% | 79.39% |
| 4 | 11 | 84.14% | 83.40% | 82.72% | 83.06% |

R1 post-hoc best F1 mean은 `80.28%`, population std는 약 `1.78%p`다. Best epoch가 `9 / 11 / 18 / 20`으로 달라 하나의 공통 optimal epoch가 관찰되었다고 볼 수 없다.

### Final vs Best Gap

Aggregate gap의 양수 값은 post-hoc best F1이 official final F1보다 높다는 뜻이다.

| Condition | Official final F1 | Post-hoc best F1 | Best − Final |
|---|---:|---:|---:|
| R0 fc1 | 77.94% | 80.70% | +2.76%p |
| R1 fc2 | 76.91% | 80.28% | +3.37%p |

R1 fold별로는 final 시점의 감소 폭을 `Final − Best`로 표시한다.

| Fold | Best F1 | Final F1 | Final − Best |
|---:|---:|---:|---:|
| 1 | 78.26% | 71.07% | -7.19%p |
| 2 | 80.43% | 76.47% | -3.96%p |
| 3 | 79.39% | 77.87% | -1.52%p |
| 4 | 83.06% | 82.23% | -0.82%p |

### 해석과 R1 결론

1. Official final 기준에서 R1 fc2는 R0 fc1보다 F1이 `1.03%p` 낮았다. 현재 기준에서는 fc2가 더 나은 reconstruction condition이라고 볼 근거가 없다.
2. Post-hoc best 기준 F1 차이는 `-0.42%p`로 작다. Feature extraction point만으로 원논문과 현재 reconstruction 사이의 큰 성능 차이를 설명하기 어렵다.
3. Post-hoc 진단에서 R1은 R0보다 Precision이 `2.70%p` 낮고 Recall이 `2.25%p` 높았다. 이 실험 범위에서는 fc2 representation이 상대적으로 recall-heavy한 prediction behavior를 보였지만 일반화하지 않는다.
4. R0와 R1 모두 final epoch보다 중간 epoch의 test F1이 높았다. Fixed 30-epoch duration이 현재 reconstruction의 과적합 또는 후반 성능 저하와 관련될 가능성이 있으나, test best epoch는 official 결과 선택에 사용할 수 없다.

R1의 fc2 feature extraction은 동일한 fixed training protocol에서 R0 fc1보다 official F1이 낮았으며, post-hoc best-epoch 기준에서도 두 조건의 차이는 매우 작았다. 따라서 feature extraction point는 원논문과의 성능 차이를 설명하는 주요 요인으로 보기 어렵다.

현재 reconstruction reference는 R0 fc1을 유지한다. 이는 원논문이 fc1을 사용했다거나 fc1이 통계적으로 우월하다는 의미가 아니다.

다음 refinement stage는 R2 CNN training policy다. ImageNet pretrained frozen VGG19와 fine-tuning policy의 plausibility를 검토하되, 이 문서에서는 R2 조건을 확정하거나 구현하지 않는다. R0/R1의 final-vs-best gap 때문에 training duration도 중요한 후속 refinement 항목으로 남기지만 R4를 R2보다 먼저 실행한다고 확정하지 않는다.

## 12. 보존 및 실행 경계

현재 VGG19/VGG16 baseline의 feature cache, history, metrics, `cv_summary.json`, checkpoint를 수정하거나 덮어쓰지 않는다. R0는 현재 baseline reference로 유지한다.

R1 feature cache와 4-fold training은 완료되었으며 결과는 위 비교에 기록했다. 기존 R0/R1 cache, output, history, metrics, checkpoint와 MLflow logs는 결과 문서화를 위해 읽기만 하고 수정하지 않는다.

## 13. 현재 상태

```text
INITIAL PAPER-INFORMED BASELINE: COMPLETED
RECONSTRUCTION REFINEMENT: PLANNED
PRIMARY BACKBONE: VGG19
R0: COMPLETED
R1: COMPLETED
R1 FEATURE CACHE: COMPLETED
R1 4-FOLD TRAINING: COMPLETED
R1 OFFICIAL F1: 76.91%
R1 POST-HOC BEST F1: 80.28%
R1 INTERPRETATION: fc2 did not improve reconstruction performance
REFERENCE FEATURE POINT: fc1
MLFLOW: READY
NEXT: R2 CNN TRAINING POLICY
PROPOSED CONTEXT MODEL IMPROVEMENT: NOT STARTED
```
