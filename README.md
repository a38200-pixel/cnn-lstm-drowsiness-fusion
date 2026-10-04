# CNN-LSTM Drowsiness Fusion

SUST-DDD의 VGG19/VGG16+LSTM baseline을 paper-informed reconstruction으로 구현하고, reconstruction refinement를 거쳐 Context Model Improvement, 행동 규칙, fusion detection으로 확장하는 프로젝트입니다.

## 현재 상태

- `DATA PREPARATION: COMPLETED`
- `VGG19 / VGG16 FEATURE EXTRACTION: COMPLETED`
- `VGG19 / VGG16 4-FOLD TRAINING: COMPLETED`
- `INITIAL PAPER-INFORMED BASELINE: COMPLETED`
- `VGG19 BASELINE F1: 77.94%`
- `VGG16 BASELINE F1: 77.72%`
- `BEST-EPOCH ANALYSIS: POST-HOC DIAGNOSTIC ONLY`
- `VALIDATION: NOT USED`
- `EARLY STOPPING: NOT USED`
- `BEST CHECKPOINT SELECTION: NOT USED`
- `OFFICIAL RESULT: FIXED EPOCH / FINAL EPOCH`
- `RECONSTRUCTION REFINEMENT: PLANNED`
- `R1 VGG19 FC2: COMPLETED`
- `R1 FEATURE CACHE / 4-FOLD TRAINING: COMPLETED`
- `R1 OFFICIAL F1: 76.91%`
- `REFERENCE FEATURE POINT: FC1 (OFFICIAL F1 77.94%)`
- `R2 VGG19 LAST-BLOCK FINE-TUNING: IMPLEMENTED / READY FOR OFFICIAL FOLD1`
- `R2 PREVIOUS FOLD1: INTERRUPTED AT EPOCH 25 / NO CHECKPOINT / NOT OFFICIAL`
- `R2 EXECUTION: CUDA REQUIRED / WORKERS=8 / CHECKPOINT-RESUME ENABLED`
- `R2 PROFILING: COMPLETED / DATALOADER EXECUTION SETTING SELECTED`
- `R2 NEW FULL TRAINING / OFFICIAL RESULT: NOT RUN / NOT AVAILABLE`
- `NEXT: NEW R2 OFFICIAL FOLD1 RUN`
- `MLFLOW TRACKING: READY FOR NEW REFINEMENT EXPERIMENTS`
- `PROPOSED CONTEXT MODEL IMPROVEMENT: DEFERRED UNTIL RECONSTRUCTION REFINEMENT`

## Paper Reconstruction Baseline

### Goal

원논문에서 직접 확인되는 조건, Figure와 parameter count로부터 구조적으로 역산되는 조건, 공개되지 않아 이번 구현에서 선택한 가정을 분리해 재구성합니다. 이 구현을 원논문과 완전히 동일한 실험이라고 주장하지 않습니다.

### A. PAPER-REPORTED

원논문에서 직접 확인되는 조건입니다.

- Dataset: SUST-DDD, 2,074 videos
- Class distribution: drowsy 975 / not drowsy 1,099
- Video unit: 10 sec
- Input: 224×224×3, video당 20 frames
- Evaluation: 75% training / 25% test, k-fold cross-validation, k=4
- CNN backbones: VGG16, VGG19, AlexNet, VGGFaceNet
- CNN에서 ReLU activation과 2×2 pooling 사용
- 1 LSTM layer, 2 fully connected layers, ReLU/Sigmoid activation, final Softmax

이 가운데 VGG19를 primary, VGG16을 comparison baseline으로 선택한 것은 현재 reconstruction의 실험 범위다.

원논문 Table III:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| VGG19 + LSTM | 90.53% | 91.74% | 91.28% | 91.50% |
| VGG16 + LSTM | 89.39% | 91.81% | 89.09% | 90.42% |

VGG19 F1은 본문 일부에서 91.46%, Table III에서 91.50%로 서로 다릅니다. 이 프로젝트는 Table III의 91.50%를 reference 주값으로 사용합니다.

### B. STRUCTURALLY INFERRED

4096D는 실험적으로 선택한 feature dimension이 아닙니다. Fig.1의 LSTM parameter count로부터 구조적으로 역산했습니다.

```text
LSTM parameters = 4 × hidden_size × (input_size + hidden_size + 1)
9,439,232 = 4 × 512 × (input_size + 512 + 1)
input_size = 4,096
```

> The 4096-dimensional LSTM input was structurally inferred from the parameter count shown in Fig.1.

구현 대상 구조와 parameter count:

| Component | Shape | Parameters |
|---|---|---:|
| Single-bias LSTM | 4096 → 512 | 9,439,232 |
| Dense 1 | 512 → 512 | 262,656 |
| Dense 2 | 512 → 64 | 32,832 |
| Output | 64 → 2 | 130 |
| Total | LSTM + dense classifier | 9,734,850 |

따라서 baseline 구조는 20×4096 input, hidden size 512, one LSTM layer, Dense 512→64→2로 고정했습니다.

### C. RECONSTRUCTION ASSUMPTIONS

다음 조건은 원논문에 충분히 공개되지 않아 이번 baseline에서 명시적으로 선택했습니다.

#### Data / Sampling

- Label-stratified 4-fold, seed 42, subject-wise information unavailable
- 전체 유효 frame 범위에서 deterministic uniform 20-frame sampling
- `np.linspace` 후 `np.rint`; 정확한 `.5`는 ties-to-even
- Short video는 중복 index를 허용하고 anomaly로 보고
- 실제 SUST-DDD 결과: short 0, invalid 0, duplicate index 0
- OpenCV frame load, BGR→RGB, bilinear 224×224 resize
- ImageNet normalization, data augmentation 없음

#### VGG Feature Extraction

- torchvision ImageNet pretrained VGG16/VGG19
- Frozen backbone을 offline feature extractor로 사용
- Backbone별 별도 video-level `[20, 4096]` feature cache
- 첫 번째 fully connected layer 뒤 ReLU 출력(`fc1`)을 4096D feature로 사용

원논문은 정확한 VGG feature extraction 위치를 공개하지 않았습니다. 즉, “논문이 fc1을 사용했다”는 의미가 아니라 현재 reconstruction의 선택입니다.

> The paper does not specify the exact VGG feature extraction location. This reconstruction uses the 4096D first fully connected representation after ReLU.

#### LSTM Classifier

```text
[B, 20, 4096]
→ single-bias LSTM512
→ final hidden state
→ Dense512 → ReLU
→ Dense64 → Sigmoid
→ Dense2 → logits
```

- ReLU와 Sigmoid는 모두 실제로 사용했습니다.
- 논문은 두 activation의 정확한 적용 위치를 충분히 설명하지 않아 Dense512 뒤 ReLU, Dense64 뒤 Sigmoid로 해석했습니다.
- Attention, Dropout, BatchNorm은 사용하지 않았습니다.
- 마지막 LSTM hidden state를 sequence representation으로 사용했습니다.

일반 `torch.nn.LSTM`은 `bias_ih`와 `bias_hh` 두 bias를 사용해 Fig.1보다 parameter가 2,048개 많습니다. 따라서 Fig.1의 `9,439,232`와 맞도록 단일 bias vector를 갖는 custom LSTM을 구현했습니다. Input weight, recurrent weight, bias는 모두 global seed를 따르는 `[-1/√512, +1/√512]` uniform initialization을 사용하고 initial hidden/cell state는 0입니다. 이 initialization이 Keras default와 같다고 가정하지 않습니다.

논문은 final Softmax를 명시하지만 현재 PyTorch model은 Dense2의 raw logits를 반환합니다. `CrossEntropyLoss`가 내부에서 log-softmax와 negative log likelihood를 처리하므로 training graph에 explicit Softmax를 중복 적용하지 않습니다. Probability가 필요한 inference에서만 Softmax를 적용하며, 이는 논문의 Softmax classification과 기능적으로 대응하는 구현입니다.

#### Training Hyperparameters

| Parameter | Value | Classification |
|---|---|---|
| Optimizer | Adam | Reconstruction assumption |
| Learning rate | 0.0001 | Reconstruction assumption |
| Weight decay | 0.00001 | Reconstruction assumption |
| Batch size | 16 | Reconstruction assumption |
| Epochs | 30 | Reconstruction assumption |
| Seed | 42 | Reconstruction assumption |
| Scheduler | None | Reconstruction assumption |
| Early stopping | None | Reconstruction assumption |
| Validation split | None | Reconstruction assumption |
| Loss | CrossEntropyLoss | Reconstruction assumption |
| Positive class | drowsy | Evaluation policy |
| Fold count | 4 | Paper-reported |
| Train/Test ratio | 75/25 | Paper-reported |

Optimizer, learning rate, weight decay, batch size, epoch 수는 원논문에서 확인되지 않습니다.

#### Evaluation Policy

- 각 fold의 약 75%를 train, 약 25%를 training에서 제외된 held-out test fold로 해석
- 현재 reconstruction에는 별도 validation split 없음
- Fixed 30 epochs, epoch 30 final metric을 official result로 사용
- Early stopping과 best checkpoint selection 없음
- 4-fold mean과 population std(`ddof=0`) 사용
- Drowsy를 positive class로 평가

원논문은 validation 사용 여부, epoch selection, early stopping, checkpoint selection, test evaluation frequency와 fold aggregation 방식을 공개하지 않았습니다. 위 정책은 원논문의 정책에 대한 주장이 아니라 이번 reconstruction의 선택입니다.

### Baseline Results

Official final-epoch 4-fold mean:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| VGG19 + LSTM | 79.94% | 80.88% | 75.28% | 77.94% |
| VGG16 + LSTM | 80.04% | 81.90% | 74.15% | 77.72% |

VGG16은 Accuracy/Precision이, VGG19는 Recall/F1이 소폭 높지만 차이는 작습니다. 특정 backbone의 명확한 우위를 주장하지 않습니다. 원논문과의 상세 비교는 [results.md](docs/paper_reconstruction/results.md)에 있습니다.

### Known Unknowns / Limitations

원논문에서 공개되지 않은 조건과 이번 선택의 전체 대응표는 [baseline_spec.md](docs/paper_reconstruction/baseline_spec.md#known-unknowns)에 정리했습니다. 핵심 unknown은 pretrained/freeze 정책, FC feature 위치, sampling/preprocessing, optimizer와 hyperparameter, validation/epoch/checkpoint selection, initialization, fold 생성 및 aggregation 방식입니다.

### Post-hoc Diagnostic Analysis

History에서 held-out test F1이 가장 높은 epoch를 사후 분석했습니다.

- VGG19: fold별 epoch `4 / 9 / 25 / 26`
- VGG16: fold별 epoch `28 / 22 / 5 / 7`

Held-out test fold를 기준으로 epoch를 선택하면 test가 validation 역할을 하므로 이 값은 `POST-HOC DIAGNOSTIC ONLY`이며 official metric이 아닙니다. 상세 수치는 [results.md](docs/paper_reconstruction/results.md#6-post-hoc-best-epoch-diagnostic)에 있습니다.

## 문서 안내

- [baseline_spec.md](docs/paper_reconstruction/baseline_spec.md): 세 분류와 상세 구현 명세, Known Unknowns
- [data_preparation.md](docs/paper_reconstruction/data_preparation.md): metadata, split, sampling 정책
- [model_pipeline.md](docs/paper_reconstruction/model_pipeline.md): frame→VGG→LSTM pipeline
- [results.md](docs/paper_reconstruction/results.md): official 결과와 diagnostic 분석
- [reconstruction_refinement.md](docs/paper_reconstruction/reconstruction_refinement.md): 미공개 조건의 sensitivity analysis와 R0–R5 실험 계획

## 다음 단계

Proposed Context Model Improvement에 앞서 Paper Reconstruction Refinement를 진행합니다. 원논문 핵심 구조와 기존 4-fold 75/25 train/test, fixed-epoch 평가 정책을 유지하면서 feature extraction 위치, CNN training policy, optimizer/learning rate, training duration과 preprocessing/sampling의 민감도를 제한적으로 검토합니다.

상세 계획은 [reconstruction_refinement.md](docs/paper_reconstruction/reconstruction_refinement.md)에 있습니다.
