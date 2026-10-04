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

원논문은 CNN을 training한 뒤 output을 LSTM에 입력한 것으로 읽을 수 있는 문장을 포함하지만 pretrained/frozen/fine-tuned 정책은 공개하지 않는다. R2의 last-block fine-tuning은 원논문 조건이 아니라 영향을 검증하기 위해 사전 확정한 `REFINEMENT CONDITION`이다.

#### R2 목적과 비교

핵심 질문은 frozen ImageNet VGG19 feature를 그대로 사용하는 R0보다 마지막 convolution block만 fine-tuning해 고수준 representation을 SUST-DDD domain에 적응시키는 것이 reconstruction performance에 영향을 주는가이다.

```text
R0: VGG19 all frozen → frozen fc1 4096D → LSTM512
R2: VGG19 block1~4 frozen → block5 trainable → frozen fc1 4096D → LSTM512
```

초기 convolution layer의 일반적인 edge/texture representation은 유지하면서 task-specific 성격이 더 강한 후반 representation만 적응시키기 위해 block5를 선택했다. 2,074-video 규모에서 full VGG19 fine-tuning은 과적합 위험과 reconstruction 해석 범위가 크며, fc1-only fine-tuning도 매우 많은 parameter를 업데이트하므로 첫 partial fine-tuning 조건으로 선택하지 않았다.

#### 확정 구조와 gradient 정책

- Backbone: ImageNet-pretrained VGG19
- Frozen CNN: block1, block2, block3, block4
- Trainable CNN: block5
- Feature point: frozen `fc1 after ReLU`, 4096D
- Trainable temporal model: single-bias LSTM, input 4096, hidden 512, one layer
- Trainable classifier: Dense512→ReLU→Dense64→Sigmoid→Dense2 logits
- 사용하지 않음: fc2, torchvision final classifier, offline feature cache

torchvision VGG19 `features`의 block 경계는 구현에서 다음과 같이 명시했다.

| Block | Feature layer range | 의미 |
|---|---:|---|
| Block 1 | `0:5` | conv1_1–conv1_2 + pool1 |
| Block 2 | `5:10` | conv2_1–conv2_2 + pool2 |
| Block 3 | `10:19` | conv3_1–conv3_4 + pool3 |
| Block 4 | `19:28` | conv4_1–conv4_4 + pool4 |
| Block 5 | `28:37` | conv5_1–conv5_4 + pool5 |

Block1~4는 frozen parameter와 activation graph가 필요하지 않아 `no_grad`로 통과한다. Block5 output은 frozen fc1을 통과하지만 block5까지의 input gradient는 유지된다. Fc1 parameter 자체는 optimizer에 포함하지 않는다.

#### Input과 end-to-end 흐름

```text
[B, 20, 3, 224, 224]
→ reshape [B×20, 3, 224, 224]
→ frozen VGG19 blocks 1~4
→ trainable VGG19 block 5
→ avgpool → flatten
→ frozen fc1 → ReLU
→ [B×20, 4096]
→ reshape [B, 20, 4096]
→ LSTM512
→ Dense classifier
→ [B, 2] logits
```

Block5 weight가 매 step 갱신되므로 R2는 R0/R1 offline feature cache를 사용할 수 없다. 기존 20-frame metadata, fold assignment, label, BGR→RGB, bilinear resize와 ImageNet normalization은 그대로 사용한다.

#### 사전 확정 학습 조건

| Item | R2 value |
|---|---|
| Optimizer | Adam |
| Block5 learning rate | 0.00001 |
| LSTM/head learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Batch size | 16 |
| Epochs | 30 fixed |
| Seed | 42 |
| Scheduler | None |
| Validation | None |
| Early stopping | None |
| Checkpoint selection | Final epoch |
| Evaluation | Stratified 4-fold, 75% training / 25% held-out test |
| Aggregation | Mean + population std (`ddof=0`) |

Optimizer는 block5와 LSTM/head를 서로 중복 없는 두 parameter group으로 구성한다. Block1~4와 fc1은 어느 group에도 포함하지 않는다. Batch size 16을 우선 유지하며 전체 학습 전에 전용 CUDA smoke CLI로 한 batch forward/backward와 OOM 여부를 확인한다. OOM이면 batch size를 자동 변경하지 않고 사용자에게 보고한다.

#### MLflow와 outputs

- Experiment: `paper-reconstruction-refinement`
- Parent: `R2_vgg19_lastblock_finetune`
- Children: `fold_1` … `fold_4`
- Output: use a NEW directory via --output-root; currently recommended: outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers8/.
- Artifacts: config, fold history/metrics/training metadata, CV summary
- 제외: dataset, frames, feature cache

#### 결과 해석 계획

Official result는 fixed epoch의 final metric이며 post-hoc best test F1은 diagnostic only다. R2 완료 후 R0 frozen VGG19 fc1과 Accuracy, Precision, Recall, F1, fold std를 비교한다.

- R2가 개선되면: `CNN training policy may contribute to the reconstruction gap.`
- R2가 개선되지 않으면: `Last-block adaptation alone does not explain the reconstruction gap.`

어느 경우에도 원논문의 실제 CNN training policy를 확인했다고 단정하지 않는다.

#### 사용자 실행 순서

Repo root에서 다음 순서로 실행한다.

```powershell
# 1. MLflow server
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000

# 2. CUDA 확인
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"

# 3. 실제 fold 1 첫 batch를 사용하는 batch16 CUDA smoke test
python scripts/smoke_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml

# 4. Smoke 통과 후 fold 1 전체 실행
python scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --output-root outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers8

# 5. Fold 1 output/MLflow 확인 후 전체 4-fold 실행
python scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml
```

Batch16 smoke에서 CUDA OOM이 발생하면 script는 batch size를 변경하지 않고 실패한다. 전체 학습 전에 결과를 검토하고 batch 8 전환 여부를 별도로 결정한다.

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
| R2 | VGG19 | — | — | — | — | — | Implemented; user execution required |
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

현재 실행 대상은 R2 CNN training policy다. R0/R1의 final-vs-best gap 때문에 training duration도 중요한 후속 refinement 항목으로 남지만 R4를 R2보다 먼저 실행한다고 확정하지 않는다.

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
R2: VGG19 LAST BLOCK FINE-TUNING — IMPLEMENTED / USER EXECUTION REQUIRED
R2 FEATURE POINT: fc1 4096D
R2 BLOCK1~4: FROZEN
R2 BLOCK5: TRAINABLE
R2 FC1: FROZEN
R2 LSTM/HEAD: TRAINABLE
R2 PREVIOUS FOLD1: INTERRUPTED / EPOCH 25 / NO CHECKPOINT / NOT OFFICIAL
R2 NEW FULL TRAINING: NOT RUN
R2 CUDA: REQUIRED / CPU FALLBACK DISABLED
R2 CHECKPOINT/RESUME: READY
R2 PROFILING: COMPLETED
R2 DATALOADER: SELECTED / WORKERS=8 / PIN_MEMORY=TRUE / PERSISTENT_WORKERS=TRUE / PREFETCH_FACTOR=2
R2 OFFICIAL FOLD1 / OFFICIAL 4-FOLD: NOT RUN
R2 RESULT: NOT AVAILABLE
MLFLOW: READY
NEXT: NEW R2 OFFICIAL FOLD1 RUN
PROPOSED CONTEXT MODEL IMPROVEMENT: NOT STARTED
```


## R2 execution reliability and timing (2026-10-02 historical snapshot)

이 절과 아래 10월 2일 benchmark는 당시 기록이다. 현재 설정과 공식 Fold1 실행 명령은 마지막의 **R2 selected execution setting (2026-10-04)** 절을 따른다.

R2 full training requires CUDA; CPU fallback is disabled. The previous Fold 1
MLflow run `2211769a4b404b90b58d138712834f9d` is treated as **INTERRUPTED / NO
CHECKPOINT AVAILABLE / NOT OFFICIAL**. Its metrics through epoch 25 and all R0/R1
artifacts are preserved; the database run status is left untouched. It cannot
resume at epoch 26. New training always uses a new output directory and MLflow run.

Each fold writes `experiment_environment.json` before model construction: device,
GPU/name/memory, PyTorch/torchvision/CUDA/cuDNN/Python versions, precision, batch
size, worker/pinning/persistence/prefetch settings, fold/condition/seed. These are
also logged as MLflow parameters. AMP remains disabled. Research conditions,
splits, preprocessing, sampling, optimizer and fixed final-epoch evaluation are unchanged.

After every completed train+test epoch, `last_checkpoint.pt` is atomically replaced
with model/optimizer state, Python/NumPy/CPU/CUDA RNG, dedicated shuffle-generator
state, config snapshot, fold, epoch and history. No scheduler is used.
`history.json`, `intermediate_metrics.json` and GPU memory are updated each epoch.
A crash during saving leaves the previous complete checkpoint intact; the
checkpoint history is authoritative if local JSON updates were interrupted.
Resume loads a trusted local checkpoint into a new output/run, validates fold and
exact config, restores optimizer/RNG and starts at saved epoch + 1. An interrupted
partial epoch is repeated from its preceding completed epoch. With the unchanged
num_workers=0 setting, the shuffle RNG is restored for reproducible continuation.

Full training records `epoch_train_seconds`, `epoch_test_seconds`,
`epoch_total_seconds`, `avg_train_batch_seconds` and checkpoint-save time in
MLflow. Epoch total includes checkpoint serialization but excludes the subsequent
JSON/MLflow writes. Normal training synchronizes only at epoch phase boundaries.
The separate profile CLI synchronizes at batch phase boundaries and measures
loader wait (decode/preprocessing for num_workers=0), H2D, forward+loss,
backward, optimizer and total; first-batch warmup/startup is included. These
synchronized diagnostic times are not a measured improvement in full-epoch time.
Profile output contains per-batch details, averages, percentages and GPU memory.
It changes only its temporary diagnostic model; no official run/checkpoint is made.
Train batches are capped at 30 and test batches at 5; only the selected video subset
is decoded. Reading the existing sequence/split CSV is not a dataset video scan.

October 2 historical DataLoader: num_workers=0, pin_memory=true, persistent_workers=false, prefetch_factor=none.
prefetch_factor=none. Optional profile-only `--num-workers 2`, `4` or `8` permits
bounded comparisons; the official worker setting is not changed. Multi-worker
prefetch can decode ahead of measured batches. Frozen block1~4 already use
no_grad; frozen FC1 stays outside no_grad so gradients reach block5. Tests cover
this path. AMP or worker changes may be assessed later using profile evidence.

PowerShell commands, from the repository with `.venv` activated:

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
python scripts/profile_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --batches 20 --test-batches 2 --output outputs/r2_profile_fold1.json
nvidia-smi -l 1
python scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --output-root outputs/paper_reconstruction/refinement/R2_cuda_fold1_new
python scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --resume-from outputs/paper_reconstruction/refinement/R2_cuda_fold1_new/fold_1/last_checkpoint.pt --output-root outputs/paper_reconstruction/refinement/R2_cuda_fold1_resumed
```

Run monitoring in a separate terminal. Review bounded profiling before starting
full training. Output paths must be new; choose another name on repeated runs.
Resume applies only to new checkpoints, never the old interrupted Fold 1.


### Bounded CUDA smoke evidence

2026-10-02: one real train batch and one real test batch, batch16, no AMP,
RTX 3080, workers=0. Train: loader 40.99s (72.41%), H2D 0.100s,
forward 15.03s, backward 0.359s, optimizer 0.127s, total 56.61s.
Test: loader 46.71s (73.60%), forward 16.67s, total 63.47s.
All three gradient checks passed (block5, temporal, frozen gradient absence).
Memory: allocated 0.726GiB at end, reserved 16.018GiB, peak allocated
12.390GiB versus physical VRAM 10.000GiB. These are PyTorch allocator counters,
not direct physical-VRAM residency. Shared-memory/spill behavior is a hypothesis
requiring monitoring, not a confirmed cause. Loader/decode/preprocessing was the
largest measured component in this cold-start sample; no full-epoch improvement
was measured. Detailed artifact: `outputs/r2_profile_smoke_20261002.json`.
The existing deterministic warn-only setting emitted an adaptive pooling backward
warning; it was not changed, so bitwise CUDA determinism is not guaranteed.


## R2 DataLoader tuning evidence (2026-10-02)

The original 20-train/2-test-batch profile (`outputs/r2_profile_fold1.json`)
measured loader shares of 83.12% train and 85.12% test. Train loading was
51.765s/batch and train total 62.279s/batch; test loading was 53.088s and
test total 62.371s. Loader wait combines disk access, video decode, resize,
tensor conversion, normalization and collation; these components were not
separately instrumented.

Added `scripts/benchmark_r2_dataloader.py`: real Fold 1 original-frame data,
shuffle=True / seeded RandomSampler / drop_last=False / default collation / batch16.
All candidates use the same bounded subset and seed and measured sequence/label
order is checked. Modes are `loader_only` and `loader_plus_h2d` (default); no
model or optimizer runs. Warm-up defaults to 3 and measured batches are capped
at 20. Windows main guard and module-level dataset functions support spawn.
Workers>0 use pinning, persistent workers and prefetch=2. Optional prefetch=4 is
limited to at most two selected worker counts via `--prefetch-workers`.
Reading metadata CSV is permitted; no full video dataset scan is performed.

The executed comparison used warm-up 3 + measured 3 batches per candidate,
loader_plus_h2d, the same 352-sequence subset (96 sequences consumed per
candidate, plus bounded worker prefetch). Mean/median/p95 below are loader
wait in seconds. Throughput uses measured wall time including H2D and reporting.

| Workers | Mean loader s | Median s | p95 s | Sequences/s | Frames/s |
|---:|---:|---:|---:|---:|---:|
| 0 | 50.839 | 49.403 | 55.651 | 0.314 | 6.285 |
| 2 | 32.119 | 0.457 | 86.345 | 0.497 | 9.940 |
| 4 | 50.358 | 0.005 | 135.961 | 0.317 | 6.345 |
| 8 | 63.676 | 0.003 | 171.922 | 0.251 | 5.019 |

**RECOMMENDATION ONLY**: workers=2, pin_memory=true, persistent_workers=true,
prefetch_factor=2 is the first candidate to recheck. Mean loading decreased
36.82% and observed sequences/s increased about 1.58x versus workers=0,
but p95 increased from 55.651s to 86.345s. The measured window (3 batches)
is shorter than the prefetch queue, so these are not steady-state throughput
claims. A separate `early-drowsiness-repro` blink extraction process with
`--workers 6` was observed running concurrently. It was left untouched.
Sequential cache effects and this CPU/I/O contention prevent a confident optimum.
Workers=4/8 did not establish an advantage. Prefetch=4 was deferred because
no stable winner was established under the limited, contended trials.
Benchmark time including warm-up and worker cleanup, excluding CLI imports and
approval waits: 1765.2s (29.4 minutes).
First sandbox attempt could not spawn workers; those failed records remain in
the original artifact. Multiprocess candidates were rerun outside sandbox.
The summary includes only successful measurements, with both source artifacts retained.

October 2 historical config: num_workers=0, pin_memory=true, persistent_workers=false,
prefetch_factor=null. Shared `r2_dataloader.loader_settings` is used by training,
profiling and environment logging, so any reviewed config change is actually used.
Zero workers always disable persistence/prefetch. Explicit execution defaults
normalize for checkpoint config matching; research/config mismatches still fail.
No new dependency was added. CPU model/logical count/parent and worker torch thread
counts are logged; physical core count is left unknown when not easily available.
The benchmark CLI now also samples system available RAM; the first comparison's
RAM snapshot was collected separately, not as a per-condition peak measurement.
This does not establish process peak RAM use. PyTorch allocator counters are not
physical VRAM residency; batch16 and AMP=false remain unchanged.

No optimized GPU pipeline profile or improved epoch time was measured in this task.
The profile CLI now reports **ESTIMATE ONLY** using fold batch counts computed by
DataLoader len with drop_last=False (Fold 1: train98/test33), bounded measured
pipeline averages, and explicit cold-start/queue/final-batch/checkpoint limitations.
Loader-only throughput is never extrapolated as a full training epoch time.
Tuning is execution optimization; split, sampling, preprocessing, architecture,
freeze policy, optimizer, learning rates, batch size, epochs and evaluation are unchanged.

PowerShell, with the project .venv activated, after the other CPU/video work finishes:

```powershell
python scripts/benchmark_r2_dataloader.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --warmup-batches 3 --batches 20 --workers 0 2 4 8 --output outputs/r2_dataloader_benchmark_recheck.json
Get-Content -Raw outputs/r2_dataloader_benchmark_recheck.json | ConvertFrom-Json | Select-Object -ExpandProperty recommendation
python scripts/benchmark_r2_dataloader.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --warmup-batches 3 --batches 20 --workers 2 --prefetch-workers 2 --output outputs/r2_dataloader_prefetch_recheck.json
python scripts/profile_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --batches 20 --test-batches 2 --num-workers 2 --output outputs/r2_profile_fold1_workers2.json
python scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --output-root outputs/paper_reconstruction/refinement/R2_cuda_fold1_loader_reviewed
```

`--num-workers 2` temporarily profiles the recommendation (persistent=true/prefetch=2),
without mutating config. Review mean/tails/throughput and RAM before editing only
training.num_workers/pin_memory/persistent_workers/prefetch_factor in YAML.
Then run profile without the override to verify the selected config. The training
command uses the actual YAML; at the end of the 2026-10-02 task it used workers=0.
This recommendation was superseded by the selected workers=8 setting on 2026-10-04.
Choose new output names each time; existing R0/R1 and interrupted R2 records are preserved.

Artifacts: `outputs/r2_dataloader_benchmark_summary_20261002.json`, plus original
baseline and worker rerun JSONs. Targeted spawn tests verify settings and identical
batch shape, sequence identity, labels and deterministic mock sample tensors.


## R2 selected execution setting (2026-10-04)

### 현재 상태와 CUDA 환경

```text
R2 IMPLEMENTATION: COMPLETED / READY FOR OFFICIAL FOLD1
CUDA POLICY: REQUIRED / CPU FALLBACK DISABLED
CHECKPOINT / RESUME: READY
PROFILING: COMPLETED (DIAGNOSTIC ONLY)
DATALOADER SETTING: SELECTED
NUM_WORKERS: 8
PIN_MEMORY: TRUE
PERSISTENT_WORKERS: TRUE
PREFETCH_FACTOR: 2
OLD FOLD1: INTERRUPTED / NO CHECKPOINT / RESUME NOT POSSIBLE / NOT OFFICIAL
NEW OFFICIAL FOLD1 / OFFICIAL 4-FOLD: NOT RUN
OFFICIAL R2 RESULT: NOT AVAILABLE
```

R2 requires CUDA. CPU fallback is disabled. Full training CLI는 CUDA 미지원 시 학습
시작 전에 RuntimeError로 종료한다. 아래 환경은 완료된 workers=8 profile에서 확인했다.

| 항목 | 확인된 환경 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080, physical VRAM 약 10 GiB |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA build / cuDNN | 12.8 / 91900 |
| Python | 3.12.12 |
| Mixed precision | false |

기존 Fold1 MLflow child run `2211769a4b404b90b58d138712834f9d`에는 epoch 25까지의
지표가 남아 있지만 당시 중간 checkpoint 기능이 없었다. **INTERRUPTED / NO CHECKPOINT
AVAILABLE / DIAGNOSTIC / NOT OFFICIAL**로 분류하며 epoch 26부터 resume할 수 없다.
DB에 남은 RUNNING 표시는 현재 실행 중이거나 학습을 완료했다는 근거가 아니다.
기존 MLflow/outputs는 보존하고 새 공식 run은 별도 output과 MLflow run으로 시작한다.

### 초기 병목과 DataLoader 재검증

초기 workers=0, pin_memory=true, persistent_workers=false, prefetch_factor=null의
20-train/2-test-batch CUDA profile에서 data loading share는 train **83.12%**, test
**85.12%**였다. Original-frame loading/decode/preprocessing/DataLoader 공급 지연이
가장 큰 **observed bottleneck**이었다. GPU가 사용되지 않았다거나 GPU 연산이 항상
병목이 아니라고 단정하지 않는다. I/O, resize, normalization 시간은 분리 측정하지 않았다.

첫 10월 2일 workers 비교에는 다른 repo의 `extract_blinks_batch.py --workers 6`가
동시에 실행되었다. **CONCURRENT WORKLOAD PRESENT / DIAGNOSTIC ONLY**로 분류하며,
그 결과만으로 최종 workers를 선정하지 않았다. 당시 workers=2 추천과 config=0 유지는
역사적 판단이며 현재 실행 설정과 구분한다.

사용자가 다른 영상 처리 작업 종료 후 실행한 독립 재검증은
`outputs/r2_dataloader_benchmark_recheck.json`에 보존되어 있다. 후보별 warm-up 3,
measured 20 batches, batch16, pin_memory=true이고 workers>0은 persistent=true,
prefetch=2였다. JSON recommendation status는 RECOMMENDATION_ONLY이며 실제 GPU
profile 완료 및 실행 안정성 관찰을 함께 고려해 현재 설정을 선정했다.

| Workers | 평균 loader s | 중앙값 ms | p95 s | Sequences/s |
|---:|---:|---:|---:|---:|
| 0 | 42.97 | 43574.94 | 54.41 | 0.372 |
| 2 | 31.63 | 31714.65 | 65.46 | 0.505 |
| 4 | 31.67 | 2247.84 | 122.25 | 0.504 |
| 8 | 25.45 | 1.03 | 138.02 | 0.626 |

workers=8 recommendation의 평균 loader wait는 **25445.98 ms**, 중앙값 **1.03 ms**,
p95 **138023.82 ms**, 처리량 **0.626 sequences/s**였다. 준비된 prefetch batch는 거의
즉시 공급되지만 간헐적 long stall이 있다. 일정한 25초 loading이나 완전히 안정적인
throughput으로 해석하지 않는다. 재검증 workers=0 대비 평균 loader wait/throughput은
개선되었지만 tail latency는 더 컸다.

### 실제 VGG19 GPU pipeline 비교

Source: `outputs/r2_profile_fold1.json`, `outputs/r2_profile_fold1_workers8.json`.
각 train 20 / test 2 batches의 diagnostic profile이며 warm-up은 제외하지 않았다.
아래 감소율은 진단 평균의 비교이며 공식 epoch 개선율이 아니다.

| 경로 | 구간 | workers=0 s/batch | workers=8 s/batch | 감소율 |
|---|---|---:|---:|---:|
| Train | Data loading | 51.77 | 31.95 | 38.29% |
| Train | Forward | 10.22 | 10.41 | -1.82% |
| Train | Batch total | 62.28 | 42.80 | 31.28% |
| Test | Data loading | 53.09 | 45.50 | 14.29% |
| Test | Forward | 9.21 | 9.09 | 1.26% |
| Test | Batch total | 62.37 | 54.64 | 12.39% |

workers=8의 loader share는 train **74.64%**, test **83.27%**였다. Train batch total은
약 **31.28% 감소**했다. Prefetch 효과와 평균 처리량 개선을 관찰했지만 공급 편차는 남아 있다.
9번째 train batch의 loader wait는 **120.22초**, 10번째는 **125.23초**였다.
첫 batch는 **282.28초**로 기록되었고 시작/초기 로딩 비용을 포함한다. 약 125초의 중간
stall과 최대 282.28초의 첫-batch 대기를 함께 기록하며 상시 속도나 원인을 단정하지 않는다.

Profile의 epoch 추정은 train98/test33 batches를 이용한 **ESTIMATE ONLY**이다.
해당 artifact의 추정 합계는 약 **99.96분**이지만 실제 epoch 실행 시간은 확인되지 않았다.
첫-batch 비용, prefetch 상태, 마지막 작은 batch, checkpoint/JSON/MLflow 비용 때문에
정확한 epoch time이나 30-epoch 총시간으로 취급하지 않는다. PyTorch allocated/reserved/
peak 메모리는 allocator counter이며 실제 물리 VRAM 점유와 구분한다.

### workers=2 제외와 실행 설정 선정

workers=2는 본질적으로 느려서가 아니라 현재 Windows/current pipeline에서
**EXECUTION STABILITY ISSUE**를 반복 관찰해 full training 후보에서 제외했다.
장시간 20-batch profile은 결과 JSON이 생성되지 않았고, 낮은 GPU utilization에서
대기 상태였으며 DataLoader 자식 worker가 보이지 않았다. 사용자는 short profile에서도
멈춤과 Ctrl+C 처리 실패를 보고했다. 후속 2-train/1-test-batch profile의 PID 3344와
자식 PID 32620은 2026-10-04 실행 명령 확인 후 process-tree termination으로 종료했다.
정확한 hang 지점/원인은 확보된 로그로 확인되지 않았으므로 구조적 성능 실패로 단정하지 않는다.
Unit spawn test 통과와 별개로 실제 pipeline에서 repeatable execution stability를
확보하지 못한 상태다. 결과 JSON이 없는 실패는 이 실행 이력으로 남긴다.

**Selected execution setting / empirically selected DataLoader configuration**:

```yaml
num_workers: 8
pin_memory: true
persistent_workers: true
prefetch_factor: 2
```

workers=0은 완료된 profile을 확보했지만 평균 공급이 느렸고 workers=2는 반복 대기 문제가
있었다. workers=8은 실제 20-batch CUDA train profile와 소수 test batches를 정상 완료하며
workers=0 대비 약 31% 낮은 train batch total을 관찰한 병렬 설정이다. 이 실행 완료 증거와
평균 처리량 개선으로 선정했으며, 이론적 최적값/항상 가장 빠른 설정/long stall 해결을
증명한 것은 아니다. Prefetch=4의 추가 benchmark는 이번 작업에서 실행하지 않았다.

DataLoader worker/persistence/prefetch는 execution/data supply optimization setting이다.
Config에서는 이 실행 설정만 반영했다. 다음 research condition은 동일하다:
ImageNet-pretrained VGG19, block1~4 frozen, block5 trainable, FC1 frozen after ReLU,
4096D, LSTM512와 기존 dense classifier, Adam, CNN LR=0.00001,
LSTM/head LR=0.0001, weight decay=0.00001, batch16, epoch30, seed42, AMP=false,
동일 fold/sampling/RGB resize/ImageNet normalization, no validation, no early stopping,
official metric=final epoch.

### Checkpoint / resume 및 timing

새 pipeline은 매 train+held-out-test epoch 완료 후 `last_checkpoint.pt`를 atomic replace로
저장한다. Model/optimizer/epoch, Python/NumPy/torch CPU·CUDA RNG, shuffle generator,
config snapshot, condition/fold와 history를 포함하며 scheduler는 사용하지 않는다.
`history.json`과 `intermediate_metrics.json`도 epoch마다 갱신한다. 완료 전 중단된 epoch는
마지막 저장 epoch 다음부터 다시 실행하며 checkpoint epoch=25이면 start_epoch=26이다.
Resume은 config/condition/fold를 검증하고 model·optimizer·RNG를 복원해 새 output/run에 기록한다.
이는 새 checkpoint에만 적용되며 이전 interrupted Fold1의 복구를 의미하지 않는다.

시작 console, `experiment_environment.json`, MLflow에는 device/GPU/VRAM,
torch/torchvision/CUDA/cuDNN/Python, precision, batch size, worker/pin/persistence/prefetch,
condition/fold/seed를 기록한다. Full run의 정확한 MLflow metric 이름은
`epoch_train_seconds`, `epoch_test_seconds`, `epoch_total_seconds`,
`avg_train_batch_seconds`, `checkpoint_save_seconds`다. Epoch total은 checkpoint
직렬화를 포함하지만 이후 JSON/MLflow 쓰기는 제외한다. 별도 profile은 세부
data/H2D/forward/backward/optimizer 시간을 동기화하여 측정한다.

### 새 공식 Fold1 실행 조건과 명령

Repo root의 프로젝트 .venv를 사용하고 MLflow server가 실행 중이어야 한다.
출력 경로가 이미 존재하면 CLI가 실패하므로 다른 새 이름을 선택한다.
Workers=8 config와 저장 환경 metadata를 확인하며 diagnostic 결과를 official metric과 섞지 않는다.
다음 명령은 사용자 실행용이며 이번 작업에서 실행하지 않았다.

```powershell
.\.venv\Scripts\Activate.ps1
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

서버가 이미 실행 중이면 기존 서버를 사용하고 별도 터미널에서:

```powershell
.\.venv\Scripts\python.exe scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --output-root outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers8
```

새 run의 완료된 epoch checkpoint가 있는 경우에만:

```powershell
.\.venv\Scripts\python.exe scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --resume-from outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers8/fold_1/last_checkpoint.pt --output-root outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers8_resumed
```

기존 R0/R1, interrupted Fold1 MLflow/outputs, 초기/재검증 benchmark JSON과 모든 profile
artifact는 보존했다. 새 training/benchmark, 전체 pytest 또는 전체 dataset scan은 실행하지 않았다.
