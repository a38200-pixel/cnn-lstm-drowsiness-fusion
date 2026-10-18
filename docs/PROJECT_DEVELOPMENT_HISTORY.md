# SUST-DDD Drowsiness Detection Project Development History

[확실] **Last updated: 2026-10-18 KST — Stage2 scope closed / Stage3 DESIGN ONLY.** R2 training/official aggregate/해석은 COMPLETED, official F1은80.24% ± 2.22%p다. 날짜별 과거 snapshot은 보존하며 최신 상태는 status matrix와 Stage3 계획을 따른다.

[확실] 이 문서는 프로젝트의 누적 연구·개발·실험·문제 해결 기록이다. 프로젝트는 **exact paper reproduction이 아니라 paper-informed reconstruction**이며, 실험하지 않은 결과를 만들지 않는다. **Official final result와 diagnostic result를 구분**하고, 확인되지 않은 정보는 미확인/미검증 또는 `NOT AVAILABLE`로 표시한다.

[확실] 근거는 기존 README, `docs/paper_reconstruction/`, 코드/config, 작은 output JSON/history, 읽기 전용 MLflow DB 조회다. 최신 실행 산출물과 문서가 충돌하면 산출물을 우선하고 불일치를 남긴다. 원본 영상 scan, 학습/profile/benchmark/feature extraction, checkpoint 역직렬화는 이 문서 작성 과정에서 실행하지 않았다.

# 1. Project Overview

[확실] 프로젝트는 SUST-DDD의 영상 기반 졸음 분류를 출발점으로, 공개된 CNN-LSTM 구조를 재구성하고 이후 Context Model Improvement, 행동 검출, Context+Behavior Fusion으로 확장한다. 출처: `README.md`, `docs/paper_reconstruction/reconstruction_refinement.md`.

```text
SUST-DDD 분석
→ Paper-Informed Reconstruction / R0 Baseline
→ Paper Reconstruction Refinement
→ R1 VGG Feature Extraction Position
→ R2 CNN Training Policy / Representation Adaptation
→ R3~R5 후속 Gate
→ Proposed Context Model Improvement
→ EAR / MAR / Head Pose Behavior Detection
→ Context + Behavior Fusion
→ Final Comparison (세부 ablation 설계는 NOT AVAILABLE)
```

[확실] Stage1/R0/R1/R2와 R4 diagnostic을 마치고 Stage2를 COMPLETED로 종료했다. R3 OPTIONAL/NOT RUN, R4 MIXED/NO NEW TRAINING RECOMMENDED, R5 DEFER/NOT RUN을 남긴 범위 종료다. Stage3는 DESIGN ONLY이며 첫 후보는 LSTM hidden-state mean pooling이다. Stage3/Behavior/Fusion 구현은 NOT STARTED다.

# 2. Research Question and Scope

[확실] 초기 질문은 공개 정보로 재구성한 VGG19/VGG16+LSTM이 어떤 성능을 보이는가였다. 두 backbone의 4-fold baseline을 얻은 뒤, 원논문 reference와의 차이를 설명할 수 있는 **미공개 reconstruction assumption의 sensitivity**를 제한적으로 검토하게 되었다.

[확실] Refinement는 paper score에 도달할 때까지 무제한 tuning하는 단계가 아니다. 핵심 20-frame/4096D/LSTM512/Dense 구조와 평가 정책을 유지하고, 실험마다 변경 변수와 fixed epoch를 사전 정의한다. 성능이 하락한 조건과 실패 산출물도 보존한다.

[확실] VGG19는 primary refinement backbone이고 VGG16은 comparison baseline이다. 모든 refinement를 VGG16에 반복하는 것은 확정 계획이 아니며, 필요하면 해석 가치가 있는 조건을 적용한다.

# 3. Paper Analysis and Reconstruction Basis

[확실] 다음 분류는 `baseline_spec.md`와 `model_pipeline.md`의 근거 분류를 통합한 것이다. 이번 작성에서 원논문 원문을 새로 검증하지 않았으므로 논문 정보는 저장소의 문헌 분석 기록에 근거한다.

| Classification | 근거/현재 사용 내용 |
|---|---|
| PAPER-REPORTED | SUST-DDD 2,074 videos; 20 frames/video; 224×224×3; VGG16/VGG19; one LSTM; two FC; ReLU/Sigmoid 언급; final Softmax; 4-fold; 75% train/25% test |
| STRUCTURALLY INFERRED | Fig.1 dimension/parameter count에 대응하는 4096D, hidden512, Dense512, Dense64, Dense2 |
| RECONSTRUCTION ASSUMPTION | ImageNet pretrained, freeze 정책, 정확한 FC 위치, sampling/preprocessing, initialization, final hidden state, activation 위치, optimizer/LR/weight decay/batch/epochs |
| PROJECT POLICY | final epoch official; no separate validation; best epoch diagnostic only; population std; drowsy positive; 기존 결과 보존 |

[확실] ReLU/Sigmoid의 존재와 **각 activation의 정확한 layer 대응**은 구분한다. Dense512 뒤 ReLU, Dense64 뒤 Sigmoid는 현재 구현의 해석이다. 정확한 VGG FC extraction 위치도 원논문 사실로 단정하지 않는다.

[확실] 저장소가 기록한 Table III reference는 다음과 같다. 원논문 VGG19 본문 F1 91.46%와 Table III 91.50%의 불일치를 보존하고 **Table III 91.50%를 주값**으로 사용한다.

| Paper Table III | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| VGG19+LSTM | 90.53 | 91.74 | 91.28 | 91.50 |
| VGG16+LSTM | 89.39 | 91.81 | 89.09 | 90.42 |

# 4. Dataset Preparation

[확실] `data_preparation.md`와 기존 summary에 따르면 filename label parsing, metadata, stratified split, sequence metadata 및 anomaly 검증을 준비했다. 기대 universe와 실제 생성 결과를 확인하고, 오류 영상을 임의 제외하거나 가짜 frame으로 대체하지 않는 정책을 사용했다.

| 항목 | 기록된 값 |
|---|---|
| Videos | 2,074 |
| drowsy / not_drowsy | 975 / 1,099 |
| Label mapping | `d_*.mp4=drowsy(1)`, `n_*.mp4=not_drowsy(0)` |
| Frames per sequence / total entries | 20 / 41,480 |
| Sampling | endpoint 포함 `np.linspace` 후 `np.rint`, ties-to-even |
| Short / invalid / duplicate-index videos | 0 / 0 / 0 |
| Fold construction | 클래스별 ID 정렬 → seed42 shuffle → round-robin holdout |
| Subject-wise | not used; subject 정보 unavailable |
| Coverage | 각 영상이 정확히 한 번 holdout; train/test disjoint |

[확실] `data/splits/paper_reconstruction/split_summary.json`의 fold 수는 다음과 같다. 새 dataset scan으로 계산한 값이 아니다.

| Fold | Train | Test | Train drowsy / not_drowsy | Test drowsy / not_drowsy |
|---:|---:|---:|---:|---:|
| 1 | 1,555 | 519 | 731 / 824 | 244 / 275 |
| 2 | 1,555 | 519 | 731 / 824 | 244 / 275 |
| 3 | 1,555 | 519 | 731 / 824 | 244 / 275 |
| 4 | 1,557 | 517 | 732 / 825 | 243 / 274 |

[확실] Frame preprocessing은 OpenCV decode, BGR→RGB, bilinear 224×224 resize, 0..1 tensor, ImageNet normalization이다. Augmentation은 없다. Short video의 중복 index 허용 정책은 구현되어 있지만 실제 생성 기록에서는 short video가 0이었다.

# 5. Evaluation Policy

[확실] **Official result는 사전 고정한 30 epoch 중 epoch30 final metric**이다. 별도 validation, scheduler, early stopping, best checkpoint selection, test 기반 best-epoch selection을 사용하지 않는다.

[확실] 각 fold의 약 25%는 training에서 제외된 held-out test다. Accuracy/Precision/Recall/F1을 기록하고 drowsy를 positive class로 사용한다. 네 fold의 mean과 population std(`ddof=0`)를 보고한다.

[확실] Held-out test를 매 epoch 관찰하지만 최고 test F1은 **POST-HOC DIAGNOSTIC ONLY**다. 이를 official 결과나 checkpoint 선택에 사용하면 test가 selection에 관여하므로 final policy와 분리한다. 원논문의 validation/epoch/checkpoint selection/aggregation 정책은 미보고로 기록되어 있다.

# 6. Stage 1 — Initial Paper-Informed Reconstruction

## 6.1 Architecture Reconstruction

[확실] 공개 구조와 parameter count를 바탕으로 기존의 다른 sequence/feature 설정을 재사용하지 않고, 다음 baseline pipeline을 구성했다. 출처: `model_pipeline.md`, `paper_lstm.py`.

```text
원본 video → deterministic 20-frame indices → RGB preprocessing
→ Frozen ImageNet VGG19 또는 VGG16 → fc1 after ReLU
→ video별 [20,4096] offline feature cache
→ custom single-bias LSTM512 → final hidden state
→ Dense512 → ReLU → Dense64 → Sigmoid → Dense2 logits
→ CrossEntropyLoss → held-out test classification metrics
```

[확실] 논문은 Softmax를 명시하지만 모델은 raw logits를 반환한다. `CrossEntropyLoss` 내부의 log-softmax를 사용하므로 training graph에 Softmax를 중복 적용하지 않는다. Probability가 필요한 inference에서만 Softmax를 적용하는 구현 선택이다.

## 6.2 4096D Structural Inference

[확실] 4096D는 성능을 보고 선택한 feature dimension이 아니다. Fig.1의 LSTM parameter 수와 hidden size로 역산했다.

```text
LSTM parameters = 4 × hidden_size × (input_size + hidden_size + 1)
9,439,232 = 4 × 512 × (input_size + 512 + 1)
input_size = 4,096
```

| Component | Parameters |
|---|---:|
| Single-bias LSTM4096→512 | 9,439,232 |
| Dense512→512 | 262,656 |
| Dense512→64 | 32,832 |
| Dense64→2 | 130 |
| Temporal model + classifier | 9,734,850 |

## 6.3 VGG Feature Extraction

[확실] R0는 torchvision ImageNet-pretrained VGG를 frozen/eval mode로 사용한다. 첫 Linear 뒤 ReLU인 `fc1`까지 실행하고 영상별 `[20,4096]` cache를 저장했다. 정확한 FC 위치는 reconstruction assumption이다.

[확실] Backbone별 cache와 manifest를 분리했다. Cache shape와 NaN/Inf 검증이 있어 잘못된 representation을 묵인하지 않는다. 기존 cache를 재추출하거나 덮어쓰지 않고 reference로 보존한다.

## 6.4 Custom LSTM Reconstruction

[확실] 일반 `torch.nn.LSTM`은 두 bias로 Fig.1 count보다 2,048개 많은 parameter를 갖는다. 이에 **single-bias custom LSTM**을 구현해 9,439,232개와 대응시켰다.

[확실] Input/recurrent weight와 bias는 seed를 따르는 uniform `±1/√512`, initial hidden/cell은 0이다. 이는 Keras default initialization과 같다고 주장하지 않는다. Final hidden state를 classifier 입력으로 사용하고 Attention/Dropout/BatchNorm을 추가하지 않았다.

## 6.5 R0 VGG19

[확실] VGG19를 primary baseline으로 구현하고 feature extraction 및 네 fold의 30-epoch 학습을 완료했다. Config는 `configs/paper_reconstruction/vgg19_lstm.yaml`, output은 `outputs/paper_reconstruction/vgg19_lstm/`다.

[확실] Adam, LR0.0001, weight decay0.00001, batch16, epoch30, seed42를 사용했다. Feature cache 이후에는 LSTM/head만 학습하므로 반복 CNN decode/forward 비용을 피할 수 있었다.

## 6.6 R0 VGG16

[확실] VGG16은 같은 feature dimension, temporal model, hyperparameter, fold 및 평가 정책으로 comparison baseline을 완료했다. Config는 `configs/paper_reconstruction/vgg16_lstm.yaml`, output은 `outputs/paper_reconstruction/vgg16_lstm/`다.

## 6.7 R0 Results

[확실] **OFFICIAL FINAL-EPOCH RESULT.** 다음 값은 각 output의 `cv_summary.json`에 기록된 epoch30 결과이며 단위는 %다.

| Model | Fold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| VGG19 | 1 | 77.26 | 76.47 | 74.59 | 75.52 |
| VGG19 | 2 | 80.92 | 84.36 | 72.95 | 78.24 |
| VGG19 | 3 | 79.58 | 79.49 | 76.23 | 77.82 |
| VGG19 | 4 | 82.01 | 83.19 | 77.37 | 80.17 |
| VGG16 | 1 | 78.03 | 76.64 | 76.64 | 76.64 |
| VGG16 | 2 | 81.89 | 82.89 | 77.46 | 80.08 |
| VGG16 | 3 | 81.12 | 83.80 | 74.18 | 78.70 |
| VGG16 | 4 | 79.11 | 84.26 | 68.31 | 75.45 |

[확실] 집계는 mean(%) ± population std(%p)다. 표는 소수 둘째 자리 반올림이며 원래 정밀 값은 JSON에 보존되어 있다.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| R0 VGG19 | 79.94 ± 1.77 | 80.88 ± 3.11 | 75.28 ± 1.67 | **77.94 ± 1.65** |
| R0 VGG16 | 80.04 ± 1.54 | 81.90 ± 3.08 | 74.15 ± 3.58 | **77.72 ± 1.79** |

[확실] VGG16은 Accuracy/Precision, VGG19는 Recall/F1이 소폭 높았다. 저장소는 차이가 작아 특정 backbone의 명확한 우위를 주장하지 않는다. 원논문과 조건이 동일하다고 확인되지 않아 수치 차이만으로 exact reproduction 성패를 단정하지 않는다.

## 6.8 Post-hoc Diagnostic Findings

[확실] **POST-HOC DIAGNOSTIC ONLY — official result와 분리.** History에서 각 fold의 최고 held-out test F1을 사후 선택했다.

| Model | Fold별 best epoch | Best F1 mean (%) | Population std (%p) |
|---|---|---:|---:|
| R0 VGG19 | 4 / 9 / 25 / 26 | 80.70 | 1.29 |
| R0 VGG16 | 28 / 22 / 5 / 7 | 81.00 | 1.31 |

[확실] 일부 fold에서 train loss가 계속 감소하는 동안 test loss가 증가했고 best 시점도 달랐다. 후반 성능 저하를 조사하는 근거로 남겼으며, 이 결과로 공통 optimal epoch를 확정하거나 official epoch를 교체하지 않았다.

# 7. Stage 2 — Paper Reconstruction Refinement

## 7.1 Refinement Strategy

[확실] 원논문의 미공개 조건 중 영향을 검토할 후보를 R1 feature point, R2 CNN training policy, R3 optimizer/LR, R4 fixed duration, 필요 시 R5 preprocessing/sampling으로 제한했다. 핵심 dimension과 평가 정책을 유지하면서 해석 가능한 변경을 비교한다.

[확실] R0는 재학습하지 않는 reference다. R1/R2는 별도 config/output으로 격리해 기존 결과를 보존했다. 후보 값은 실행 전에 정하며 test score를 보고 무제한 추가하지 않는 정책이다.

## 7.2 R1 — VGG19 FC2 Feature Point

[확실] 첫 sensitivity 실험은 **fc1 after ReLU → fc2 after ReLU** 변경이었다. FC2도 4096D이므로 LSTM 구조를 바꾸지 않고 representation 위치만 비교할 수 있었다.

[확실] CNN frozen, 4096D, LSTM/head, optimizer/LR, batch16/epoch30/seed42, fold/sampling/preprocessing/evaluation은 유지했다. Eval extraction에서 중간 Dropout은 비활성화된다. R1 cache는 `data/features/paper_reconstruction/refinement/R1/vgg19/fc2/`에 분리했다.

## 7.3 R1 Results

[확실] **OFFICIAL FINAL-EPOCH RESULT.** 출처는 `outputs/paper_reconstruction/refinement/R1_vgg19_fc2/cv_summary.json`다.

| Fold | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---:|---:|---:|---:|---:|
| 1 | 75.53 | 80.00 | 63.93 | 71.07 |
| 2 | 79.96 | 85.35 | 69.26 | 76.47 |
| 3 | 79.19 | 77.87 | 77.87 | 77.87 |
| 4 | 83.37 | 82.57 | 81.89 | 82.23 |
| Mean | **79.51** | **81.45** | **73.24** | **76.91** |
| Population std (%p) | 2.78 | 2.80 | 7.05 | 3.99 |

[확실] **POST-HOC DIAGNOSTIC ONLY.** R1 best epoch는 20/9/18/11이며, best F1 mean80.28%, population std 약1.78%p다. 이 값을 model/checkpoint selection에 사용하지 않는다.

## 7.4 R0 vs R1 Interpretation

[확실] Official R1−R0 VGG19 변화는 다음과 같다.

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| −0.43%p | +0.57%p | −2.04%p | **−1.03%p** |

[확실] **fc2 alone did not improve official reconstruction performance.** Diagnostic best F1 차이도 −0.42%p로 작았다. Feature point만으로 큰 reconstruction gap을 설명하기 어렵다는 해석을 기록하고 R0 fc1을 reference로 유지했다. 이는 원논문이 fc1을 사용했다는 뜻은 아니다.

# 8. R2 — VGG19 Last Block Fine-Tuning

## 8.1 Why R2 Was Introduced

[확실] R1의 결과를 바탕으로, fixed ImageNet representation의 domain adaptation 효과를 다음 질문으로 두었다. **Frozen CNN을 사용하는 R0보다 마지막 convolution block을 SUST-DDD task에 적응시키면 성능이 달라지는가?** 이 질문은 검증 대상이며 결과가 이미 개선됐다는 주장이 아니다.

[확실] 문서의 설계 의도는 lower block의 일반 representation을 보존하고 block5만 적응시키는 것이다. Full VGG fine-tuning이나 거대한 FC1-only update를 첫 조건으로 택하지 않고 범위와 비용을 제한했다. R2는 **REFINEMENT CONDITION**이며 원논문 training policy를 복원했다는 주장이 아니다.

## 8.2 R2 Architecture

```text
[B,20,3,224,224] → [B×20,3,224,224]
→ frozen VGG19 block1~4 → trainable block5
→ avgpool/flatten → frozen fc1 + ReLU → [B,20,4096]
→ single-bias LSTM512 → Dense512/ReLU → Dense64/Sigmoid
→ Dense2 logits
```

[확실] ImageNet-pretrained VGG19를 사용한다. FC2와 torchvision final classifier는 사용하지 않는다. Parameter 수는 현재 MLflow metadata와 모델 구조에 대응한다.

| Total | Trainable | Frozen |
|---:|---:|---:|
| 132,523,778 | 19,174,082 | 113,349,696 |

## 8.3 Gradient Policy

[확실] Block1~4는 parameter를 freeze하고 `no_grad`로 실행한다. Block5는 trainable이다. FC1 parameter는 frozen이지만 **FC1 forward를 no_grad로 감싸지 않아 block5까지 input gradient를 전달**한다. LSTM과 classifier도 trainable이다.

[확실] Smoke/profile의 gradient checks는 block5 gradient, temporal gradient, frozen gradient absence를 확인했다. Freeze와 gradient 전달을 별개로 검증해 frozen FC1이 block5의 학습 경로를 끊지 않도록 했다.

## 8.4 Optimizer Parameter Groups

[확실] Adam은 block5와 LSTM/head의 서로 중복 없는 parameter group으로 구성한다. Frozen blocks와 FC1은 optimizer에서 제외하고 trainable parameter coverage를 검사한다.

| Item | R2 fixed value |
|---|---|
| Block5 LR | 0.00001 |
| LSTM/head LR | 0.0001 |
| Weight decay | 0.00001 |
| Batch / epochs / seed | **16 / 30 / 42** |
| AMP | **false** |
| Scheduler / validation / early stopping | none |
| Official metric | epoch30 final |

## 8.5 CUDA Execution Policy

[확실] R2 full training은 CUDA를 강제한다. `require_cuda()`는 CUDA를 사용할 수 없으면 오류로 종료하며 CPU fallback을 허용하지 않는다. 이전 중단 실행의 정확한 device는 이번 작성에서 새로 검증하지 않아 현재 CUDA 실행과 혼동하지 않는다.

[확실] 현재 run의 `experiment_environment.json`은 다음 환경을 기록한다.

| Python | PyTorch | torchvision | CUDA build | cuDNN | GPU |
|---|---|---|---|---|---|
| 3.12.12 | 2.11.0+cu128 | 0.26.0+cu128 | 12.8 | 91900 | RTX3080, 약10 GiB |

## 8.6 Checkpoint and Resume

[확실] 초기 R2 Fold1은 epoch25까지 metric이 남았지만 중간 checkpoint가 없어 epoch26부터 이어갈 수 없었다. 이를 **INTERRUPTED / NO CHECKPOINT / NOT OFFICIAL**로 보존한 뒤 epoch별 checkpoint/resume 기능을 구현했다.

[확실] 매 train+test epoch 완료 후 `last_checkpoint.pt.tmp`를 저장하고 `os.replace`로 `last_checkpoint.pt`를 atomic replace한다. 저장 중 중단되면 이전 완성 checkpoint를 유지하는 구조다. JSON/history 갱신이 중단된 경우 checkpoint 내부 history가 기준이지만 이번 문서 작업에서는 역직렬화하지 않았다.

| Saved state | 내용 |
|---|---|
| Model / optimizer | state dictionaries |
| Progress | epoch, contiguous history |
| Identity | config snapshot, fold, condition |
| RNG | Python, NumPy, torch CPU/CUDA |
| Shuffle | dedicated generator state |
| Scheduler | None |

[확실] Resume은 config/condition/fold와 history 연속성을 검증하고 model/optimizer/RNG를 복원해 **saved epoch+1**부터 새 output/run에 기록한다. 진행 중 partial epoch의 batch 위치는 저장하지 않으므로 그 epoch를 다시 시작한다.

[확실] Config 비교에는 effective loader setting이 포함된다. 현재 workers4 checkpoint를 YAML 기본 workers8로 resume하면 불일치하므로 동일 CLI override를 유지해야 한다. Multiprocess resume의 bitwise 동일성과 현재 checkpoint의 실제 복원 성공은 미검증이다.

## 8.7 MLflow Logging

[확실] 새 refinement부터 MLflow experiment `paper-reconstruction-refinement`, condition별 parent, fold별 nested child를 사용한다. R0를 재학습하거나 소급 변환하지 않는 정책이다.

[확실] Config/environment/condition/fold/optimizer와 final/intermediate metrics를 기록한다. 매 epoch timing은 `epoch_train_seconds`, `epoch_test_seconds`, `epoch_total_seconds`, `avg_train_batch_seconds`, `checkpoint_save_seconds`다. Epoch total은 checkpoint serialization을 포함하고 이후 JSON/MLflow 쓰기는 제외한다.

[확실] `with parent_context`와 `with child_context` 내부 예외가 전파되면 MLflow context manager가 `FAILED`로 종료하는 흐름이다. 반면 강제 종료로 context cleanup이 실행되지 않으면 과거 run처럼 DB에 `RUNNING`이 남을 수 있으므로 DB 상태만으로 프로세스 생존을 단정하지 않는다.

# 9. R2 Performance Engineering and Troubleshooting

## 9.1 Initial Training Time Problem

[확실] R0/R1은 frozen offline cache를 한 번 만든 뒤 temporal model만 반복 학습했다. R2에서는 block5가 optimizer step마다 바뀌므로 고정 feature cache를 재사용할 수 없다. 매 batch 원본 영상 접근·frame decode·resize/normalization·VGG forward와 gradient 계산이 추가되어 실행 비용 구조가 달라졌다.

[확실] 초기 긴 실행과 checkpoint 부재를 겪은 뒤 CUDA 강제, environment 기록, bounded profile, epoch timing, checkpoint/resume을 준비했다. 성능 문제를 연구 모델 변경과 분리하고 batch16/AMP=false를 유지한 채 DataLoader 공급을 조사했다.

## 9.2 workers=0 Profiling

[확실] **DIAGNOSTIC ONLY.** `outputs/r2_profile_fold1.json`의 train20/test2 bounded profile에서 다음 평균을 확인했다.

| Path | Loading s/batch | Forward s/batch | Total s/batch | Loader share |
|---|---:|---:|---:|---:|
| Train | 51.765 | 10.222 | 62.279 | 83.12% |
| Test | 53.088 | 9.210 | 62.371 | 85.12% |

[확실] Profile은 batch phase별 CUDA synchronization을 포함하고 cold-start를 제외하지 않았다. 이 평균은 측정된 full-epoch runtime이 아니다.

## 9.3 DataLoader Bottleneck

[확실] 위 표본에서 가장 큰 **observed bottleneck**은 data loading이었다. Loader wait에는 disk I/O, video decode, resize, tensor conversion/normalization, collation이 포함되며 각 성분은 분리 계측하지 않았다. GPU 연산이 항상 병목이 아니라거나 사용되지 않았다고 단정하지 않는다.

## 9.4 First Worker Benchmark

[확실] 2026-10-02 문서의 첫 비교는 loader_plus_h2d, warm-up3+measured3 batches, 동일 bounded subset과 seed, batch16이었다. 모델/optimizer를 실행하지 않는 benchmark로 0/2/4/8 workers를 비교했다.

| Workers | Mean loader s | Median s | p95 s | Sequences/s |
|---:|---:|---:|---:|---:|
| 0 | 50.839 | 49.403 | 55.651 | 0.314 |
| 2 | 32.119 | 0.457 | 86.345 | 0.497 |
| 4 | 50.358 | 0.005 | 135.961 | 0.317 |
| 8 | 63.676 | 0.003 | 171.922 | 0.251 |

[확실] 당시 workers2는 재검증 후보였고 최종 선택이 아니었다. Window가 짧고 queue/cold-start 영향이 있어 `RECOMMENDATION ONLY / DIAGNOSTIC ONLY`로 남겼다. 초기 sandbox worker-spawn 실패 기록도 보존했다.

## 9.5 Concurrent Workload Limitation

[확실] 첫 비교 당시 다른 저장소의 blink extraction 작업이 workers6으로 동시에 실행되었다는 문서 기록이 있다. CPU/disk 자원 경쟁과 순차 cache 효과가 개입할 수 있어 독립 최적값을 확정하지 않았다. 실제 영향량은 미확인이다.

[확실] 이후 다른 영상 작업 종료 후 독립 재검증을 수행했다는 기록과 `outputs/r2_dataloader_benchmark_recheck.json`이 있다. Warm-up3+measured20, workers>0은 persistent=true/prefetch2였으며 문서의 요약은 다음과 같다.

| Workers | Mean loader s | Median ms | p95 s | Sequences/s |
|---:|---:|---:|---:|---:|
| 0 | 42.97 | 43,574.94 | 54.41 | 0.372 |
| 2 | 31.63 | 31,714.65 | 65.46 | 0.505 |
| 4 | 31.67 | 2,247.84 | 122.25 | 0.504 |
| 8 | 25.45 | 1.03 | 138.02 | 0.626 |

[확실] Workers8의 평균 공급은 개선되었지만 tail latency가 더 컸다. Benchmark recommendation을 full training 안정성으로 해석하지 않고 실제 GPU pipeline profile과 비교했다.

## 9.6 workers=2 Hang

[확실] 문서에는 workers2의 20-batch profile이 장시간 대기하며 결과 JSON을 생성하지 않았고, 짧은 재시도에서도 hang/Ctrl+C 처리 문제가 보고되었다고 기록되어 있다. 2026-10-04에는 사용자 요청에 따라 확인된 profile 프로세스 트리를 종료했다.

[확실] 정확한 hang 원인은 미확인이다. **PERFORMANCE FAILURE가 아니라 EXECUTION STABILITY ISSUE**로 분류하고 당시 full training 후보에서 제외했다. Unit spawn test 성공과 실제 video pipeline 안정성을 구분했다.

## 9.7 workers=8 Profiling

[확실] Workers8/prefetch2의 train20/test2 profile은 완료했다. 출처는 `outputs/r2_profile_fold1_workers8.json`이다.

| Path | Loading s/batch | Forward s/batch | Total s/batch | Loader share |
|---|---:|---:|---:|---:|
| Train | 31.945 | 10.409 | **42.799** | 74.64% |
| Test | 45.503 | 9.094 | 54.642 | 83.27% |

[확실] Train 평균 total은 workers0 대비 약31.28% 낮았다. 그러나 문서에는 첫 batch loading282.28초, 중간 loading120.22/125.23초의 stall이 기록되어 있다. 2026-10-04에는 이 profile 완료와 평균 개선을 근거로 workers8을 선정했지만 full-run 안정성은 미검증이었다.

## 9.8 workers=8 Full Training Failure

[확실] 2026-10-06 official attempt는 workers8/pin=true/persistent=true/prefetch2에서 DataLoader shared-memory 오류로 실패했다. 해당 폴더는 `outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers8_official/`이다.

[확실] 직접 파일 확인 결과 환경 JSON만 남아 있다. **완료 epoch0, history 없음, last_checkpoint 없음**이며 해당 MLflow parent/child는 `FAILED`다. 보고된 traceback은 train epoch 도중 default collate의 shared-storage 생성 단계다. 마지막 완료 epoch가 없어 resume할 수 없다.

[확실] Short profile 성공과 full training 실패를 함께 기록한다. 이후 current official 실행은 workers4로 변경되었으므로 **workers8은 현재 official 실행 설정에서 제외**되었다. YAML 기본값 자체는 여전히8이며 CLI override로 변경했다.

## 9.9 Windows Error 1455

[확실] 보고된 경로는 `default_collate → _typed_storage()._new_shared(...) → _new_using_filename_cpu(...)`다. Worker가 CPU tensor를 다른 프로세스에 전달하기 위한 shared storage 생성 단계에서 오류가 발생했다.

[확실] Windows error1455는 `ERROR_COMMITMENT_LIMIT`이다. 정의 출처: [Microsoft System Error Codes](https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--1300-1699-). CUDA OOM traceback과 구분되는 CPU 공유 메모리 allocation 실패다.

[확실] Float32 입력 `[16,20,3,224,224]`만 약183.75 MiB다. 단일 loader의 workers×prefetch 입력량 계산은 8×2 약2.87 GiB, 4×1 약0.72 GiB이며, worker 내부 tensor·pinned copy·기타 메모리를 포함한 실제 peak 값이 아니다.

[확실] Worker 수와 prefetch를 줄여 공유 배치 공급 부담을 낮추는 실행 변경을 선택했다. 실패 당시 commit/pagefile 사용량, pagefile 크기, 메모리 누수 여부와 다른 프로세스의 기여는 미확인이다. Pagefile 설정을 변경했다는 기록도 이 조사에서 확인되지 않았다.

## 9.10 workers=4 / prefetch=1 Validation

[확실] 2026-10-07의 두 JSON은 `diagnostic_only`이며 batch16, AMP=false, workers4/pin=true/persistent=true/prefetch1을 기록한다. Block5/temporal gradient와 frozen gradient absence checks는 모두 통과했다.

| Diagnostic | Train/Test batches | Train loading s | Train forward s | Train total s | Test total s |
|---|---|---:|---:|---:|---:|
| 1-batch smoke | 1 / 1 | 50.834 | 16.833 | 68.209 | 67.744 |
| 5-batch profile | 5 / 1 | 28.768 | 11.188 | **40.193** | 59.854 |

[확실] Sources는 `R2_workers4_prefetch1_smoke_20261007.json`, `R2_workers4_prefetch1_profile5_20261007.json`이며 모두 refinement output 아래에 있다. 성공한 파일과 별개로 2026-10-06에는 사용자 요청으로 중단한 workers4 short profile 시도도 있었고, 두 성공 JSON과 동일 실행으로 취급하지 않는다.

[확실] 5-batch profile의 첫 train batch total은146.14초이며 평균에 cold-start가 포함된다. 두 profile의 `epoch_time_estimate`는 `ESTIMATE ONLY`다. 표본 길이와 queue 상태가 달라 workers8보다 항상 빠르거나 최적이라고 확정하지 않는다.

## 9.11 Current Stable Execution Setting

[확실] **Workers4/prefetch1은 R2 4-FOLD FULL-RUN VALIDATED다.** 네 fold의 epoch30 final을 확보했으며 Fold3/4의 중단 후 resume을 포함한다. 무중단 완주나 globally optimal setting을 보장하지 않는다. Effective setting은 다음과 같다.

```yaml
num_workers: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 1
```

[확실] Train/profile CLI는 `--num-workers`, `--pin-memory`, `--persistent-workers`, `--prefetch-factor`를 지원하며 boolean에는 `--no-...`도 있다. Profile이 prefetch2로 강제하던 코드를 override 적용으로 바꾸어 임시 실행 설정을 사용하게 했다. Unnecessary helper/factory를 추가하지 않고 argparse에서 config의 execution 값만 변경한다.

[확실] 이 설정은 **execution optimization**이며 model/optimizer/LR/fold/sampling/preprocessing/seed/batch16/AMP=false/epoch30/final policy를 변경하지 않는다. 이전 CLI/checkpoint/DataLoader targeted test19개 통과 기록은 개발 과정의 검증이며 이 문서 작성에서 테스트를 재실행하지 않았다.

# 10. R2 Fold1 Completed

[확실] **Snapshot: 2026-10-09 17:02:21 KST.** Output은 `outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers4_prefetch1_official/`이며 2026-10-07 20:48 KST 시작한 MLflow parent/child는 2026-10-09 10:23:37 KST에 모두 `FINISHED`로 종료했다. History 마지막 행, metrics.json, MLflow final metric은 제공된 값과 일치한다.

| Item | Snapshot value |
|---|---|
| Status / completed epoch | **COMPLETED / 30 of 30** |
| History | `fold_1/history.json`, epoch1~30 연속 기록 |
| Checkpoint | `fold_1/last_checkpoint.pt` 존재, 683,546,733 bytes |
| Final output | `metrics.json`, `training_metadata.json`, `final_model.pt` 존재 |
| Official accuracy / precision | 0.7822736030828517 / 0.7673469387755102 (78.23% / 76.73%) |
| Official recall / F1 | 0.7704918032786885 / 0.7689161554192228 (77.05% / 76.89%) |
| Train loss / test loss | 0.012044325465234625 / 0.9001310154623379 |
| Confusion matrix | `[[218,57],[56,188]]`, drowsy positive |
| Epoch30 train / test seconds | 3267.979770800099 / 1103.5971713999752 |
| Avg train batch / checkpoint save seconds | 33.34673235510305 / 0.9703468999359757 |
| Epoch30 total seconds | 4372.547306400025, 약72.88분 |
| Parent/child MLflow | FINISHED / FINISHED |
| **R2 OFFICIAL FOLD1 RESULT** | **AVAILABLE / epoch30 final** |
| **R2 OFFICIAL OVERALL / 4-FOLD RESULT** | **NOT AVAILABLE** |

[확실] **POST-HOC DIAGNOSTIC ONLY:** 전체30 epoch history의 best test F1은 epoch6의 0.789272030651341 (78.93%)다. Best epoch를 official model selection에 사용하지 않고 epoch30 final을 유지한다.

[확실] Official Fold1 F1은 R0 VGG19 75.52%, R1 FC2 71.07%, R2 block5 fine-tuning 76.89%다. R2−R0는 +1.37%p, R2−R1은 +5.82%p다. **단일 Fold1 결과만으로 전체 우위를 결론내리지 않고 Fold2~4 완료 후 mean/population std로 판단한다.**

[추측] Epoch30의 큰 train/test loss 차이는 후반 overfitting 가능성을 시사하는 후속 분석 대상이다. Overfitting 확정으로 표현하지 않고 전체 epoch history와 함께 분석한다.

[확실] Confusion matrix는 TN218/FP57/FN56/TP188이다. Fold1 final에서 FP/FN 수는 거의 같아 오류 수가 한쪽으로 크게 치우치지 않았다. 이 관찰을 다른 fold에 일반화하지 않는다.

[확실] Checkpoint는 존재/크기만 확인했다. 내부 epoch/state 무결성과 실제 복원 성공은 역직렬화하지 않아 미검증이다.

[확실] **README status was stale at the 2026-10-08 snapshot.** 당시 workers8/new official NOT RUN 표기와 workers4 RUNNING 산출물이 달랐다. 2026-10-09에는 README/refinement/이 문서의 Fold1 상태를 COMPLETED로 갱신하고 과거 workers8 실패 이력을 유지했다.

[확실] 요청의 Fold2~4 NOT RUN 전제와 달리, 이번 조회에는 `R2_cuda_fold2_workers4_prefetch1_official/fold_2/experiment_environment.json`과 2026-10-09 16:58:37 KST 시작한 local MLflow Fold2 RUNNING 기록이 있다. 아직 epoch metric은 없으며 이 작업에서 Fold2를 실행하거나 변경하지 않았다. 이번 작성에서 프로세스 생존을 직접 조회하지 않았으므로 RUNNING은 MLflow에 기록된 상태다.

## 10.1 R2 Fold2 Completed (2026-10-11)

[확실] **Snapshot: 2026-10-11 18:00:52 KST.** Fold2 output은 `outputs/paper_reconstruction/refinement/R2_cuda_fold2_workers4_prefetch1_official/fold_2/`다. History의 epoch1~30, metrics.json, MLflow final metric을 사용자 제공값과 교차 확인했다. Fold2 parent/child는 2026-10-11 06:16:17 KST에 FINISHED로 종료했고 last_checkpoint.pt(683,546,669 bytes), final_model.pt, training_metadata.json이 존재한다. Checkpoint 내부는 역직렬화하지 않아 미검증이다.

### Official Fold2 epoch30 final

| Metric | Exact value | Display |
|---|---:|---:|
| Train loss | 0.012601864343011974 | 0.0126 |
| Test loss | 0.7646111181734384 | 0.7646 |
| Accuracy | 0.8111753371868978 | 81.12% |
| Precision | 0.8016528925619835 | 80.17% |
| Recall | 0.7950819672131147 | 79.51% |
| F1 | 0.7983539094650205 | **79.84%** |

[확실] Confusion matrix는 `[[227,48],[50,194]]`이며 drowsy positive, TN227/FP48/FN50/TP194다. Fold2 final의 FP/FN 수는 비슷하지만 이 오류 분포를 다른 fold나 전체 성능으로 일반화하지 않는다.

| Epoch30 timing | Seconds |
|---|---:|
| epoch_train_seconds | 3234.7271157000214 |
| epoch_test_seconds | 1117.4555496000685 |
| avg_train_batch_seconds | 33.0074195479594 |
| checkpoint_save_seconds | 1.028082299977541 |
| epoch_total_seconds | 4353.210771699902 (**약72.55분**) |

[확실] Timing 정의는 기존과 같아 epoch total은 checkpoint 저장까지 포함하고 이후 JSON/MLflow 쓰기는 제외한다.

### Completed official folds — 2 / 4

| Fold | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---:|---:|---:|---:|---:|
| 1 | 78.23 | 76.73 | 77.05 | 76.89 |
| 2 | 81.12 | 80.17 | 79.51 | 79.84 |

[확실] **R2 official overall/4-fold aggregate는 NOT AVAILABLE**이다. 두 fold의 interim 평균은 작성하지 않았다. Single-fold 또는 부분 결과만으로 R0/R1 대비 전체 우위를 결론내리지 않고 네 fold의 epoch30 mean/population std로 판단한다.

### Diagnostic and loss interpretation

[확실] **POST-HOC DIAGNOSTIC ONLY:** Fold2 best test F1은 epoch25의 0.8104838709677419 (**81.05%**)다. Best epoch는 official selection에 사용하지 않으며 epoch30 final과 분리한다.

[추측] Train loss0.0126/test loss0.7646 차이는 Fold1과 함께 possible late-stage overfitting의 후속 분석 대상이다. 확정으로 단정하지 않고 Fold1/2 전체 history와 Fold3/4 결과까지 함께 검토한다.

### Execution validation and remaining work

[확실] Fold2 환경 JSON도 workers4/pin_memory=true/persistent_workers=true/prefetch_factor=1, batch16, AMP=false, seed42를 기록한다. **Workers4/prefetch1: two full folds completed successfully.** 전체4-fold 안정성이나 optimal setting을 보장하지 않는다. 연구 조건은 변경하지 않았다.

[확실] 요청의 Fold3 NOT RUN 전제와 달리 2026-10-11 18:00:39 KST 시작한 Fold3 local MLflow child/parent가 RUNNING이며 별도 `R2_cuda_fold3_workers4_prefetch1_official/fold_3/experiment_environment.json`이 존재한다. 조회 시 환경 JSON만 있고 epoch metric은 없었다. RUNNING은 MLflow 기록이며 이번 작업에서는 프로세스 생존을 직접 조회하지 않았다. 학습을 실행하거나 기존 프로세스를 변경하지 않았다.

[확실] **NEXT:** 이미 시작된 Fold3 official completion → Fold4 → 4-fold aggregation → R0/R1/R2 comparison → R2 interpretation and R3~R5 refinement gate. Fold4는 NOT RUN이다. Stage1/R0/R1은 COMPLETED, Stage2는 IN PROGRESS, Stage3 Context는 NOT STARTED, Behavior/Fusion은 PLANNED / NOT STARTED다. 실패·중단 이력과 기존 Fold1 결과는 보존했다.


## 10.2 R2 Fold3/Fold4 Resume Completion and Official Aggregate (2026-10-18)

[확실] **Snapshot: 2026-10-18 15:04:38 KST.** 위 Fold1/2 항목은 각각의 날짜 snapshot이다. 현재는 Fold1~4 training, aggregate와 해석 모두 COMPLETED이며 아래 결과가 최신이다.

[확실] Fold3는 최초15 epoch 완료 후 중단, 첫 resume에서16 epoch 저장 후 사용자 요청 중단, 다음 resume에서17~30 epoch를 마쳤다. 최종 output은 `R2_cuda_fold3_resume_next/fold_3/`이며 final model 저장은2026-10-13 13:08:37 KST, MLflow는 FINISHED다.

[확실] Fold4는 최초21 epoch까지 저장 후 중단되었고2026-10-15에 epoch22부터 resume해30 epoch를 마쳤다. 최종 output은 `R2_cuda_fold4_resumed_20261015/fold_4/`이며 final model 저장 시각은2026-10-15 22:17:26 KST다. 두 경로는 `outputs/paper_reconstruction/refinement/` 기준이다. Resume 이전 부분 history를 별도 fold로 중복 집계하지 않았다.

[확실] **OFFICIAL: epoch30 final, mean ± population std (ddof=0).** 원래 float precision으로 계산하고 표시만 반올림했다. History는 각 fold에 epoch1~30이 연속이며 metrics.json의 final 값과 일치한다. 아래 표는 best epoch를 사용하지 않는다.

| Fold | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| 1 | 78.23 | 76.73 | 77.05 | 76.89 |
| 2 | 81.12 | 80.17 | 79.51 | 79.84 |
| 3 | 83.04 | 84.21 | 78.69 | 81.36 |
| 4 | 83.95 | 83.06 | 82.72 | 82.89 |
| **Mean** | **81.58** | **81.04** | **79.49** | **80.24** |
| **Population std (%p)** | **2.19** | **2.89** | **2.06** | **2.22** |

[확실] Official 조건 비교와 delta는 다음과 같다. %p delta도 반올림 전 mean끼리 계산했다.

| Condition | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| R0 VGG19 frozen FC1 | 79.94 | 80.88 | 75.28 | 77.94 |
| R1 VGG19 frozen FC2 | 79.51 | 81.45 | 73.24 | 76.91 |
| R2 Block5 fine-tuning | 81.58 | 81.04 | 79.49 | 80.24 |
| **R2−R0 (%p)** | **+1.64** | **+0.17** | **+4.21** | **+2.30** |
| **R2−R1 (%p)** | **+2.07** | **−0.41** | **+6.25** | **+3.33** |

[확실] R2는 R0 대비 official mean F1을 2.30%p 개선했다. 가장 큰 변화는 Recall +4.21%p이며 Precision +0.17%p는 작다. 이번 official 비교에서 R1 feature point 변경보다 R2 representation adaptation의 결과가 좋았다. 통계적 유의성, 원논문 policy 복원 또는 paper performance gap 해결을 입증한 것은 아니다.

[확실] F1/Recall population std는 R0 1.65/1.67%p, R1 3.99/7.05%p, R2 2.22/2.06%p다. R2 F1 range는6.00%p이며 R1보다 F1/Recall 편차가 작지만 R0보다 크다. 평균 개선과 fold variance 개선이 동시에 발생하지는 않았다.

[확실] Final train/test loss는 Fold1 0.012044/0.900131, Fold2 0.012602/0.764611, Fold3 0.123243/0.648642, Fold4 0.025707/0.601602다. 네 fold 모두 초반10보다 마지막10 평균 test loss가 높고 Fold3의 후반 train loss 반등도 있어 단조 감소로 설명하지 않는다.

[확실] **POST-HOC DIAGNOSTIC ONLY:** best epochs는6/25/18/18, F1은78.93/81.05/81.50/84.23%다. Diagnostic mean81.43%, population std1.89%p이며 official과의 차이는 반올림 전 값으로 약+1.18%p다. Best epoch를 official selection에 사용하거나 epoch18을 새 duration으로 바로 정하지 않는다.

[추측] 후반 test loss 증가와 best−final 차이는 possible late-stage overfitting 및 R4 duration sensitivity 검토를 뒷받침하는 패턴이다. 확정 진단은 아니며 특히 Fold3는 후반 train loss도 반등했다.

[확실] Combined confusion matrix는 `[[917,182],[200,775]]`이며 총2,074 videos, drowsy positive다. Pooled matrix metric과 fold mean은 다른 집계이므로 official 주값은 fold mean ± population std다.

[확실] **R2 4-FOLD FULL-RUN VALIDATED**는 workers4/pin=true/persistent=true/prefetch1, batch16/AMP=false에서 resume을 포함해 네 final 결과를 확보했다는 의미다. 무중단 완주나 optimal setting을 뜻하지 않는다. Workers2 hang, workers8 profile 성공 후1455 실패 이력은 유지했다.

[확실] Fold4는 **training result completed; MLflow terminal status stale**다. Final metrics/model/artifacts는 저장됐고 로컬과 일치하지만 종료 시 cp949 emoji output error로 parent/child RUNNING이 남았다. 결과 저장 확인 후 서버를 종료했으며 DB 상태를 임의 수정하지 않았다.

[확실] 의사결정 흐름은 **R0 frozen FC1 → R1 feature point sensitivity(official 개선 없음) → R2 representation adaptation(official F1 +2.30%p) → 후반 degradation 관찰 → R4 training-duration sensitivity 검토**다. R3 OPTIONAL / R4 RECOMMENDED / R5 DEFER로 분석을 마쳤지만 후속 조건 승인·사전 정의는 남아 있다. Epoch 값/config/추가 실험은 만들지 않았다.



# 11. Experiment Results Summary

[확실] **OFFICIAL final-epoch 4-fold mean.** Paper reference는 저장소의 문헌 기록이며 동일 조건 재현을 의미하지 않는다.

| Source / condition | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Role |
|---|---:|---:|---:|---:|---|
| Paper VGG19 Table III | 90.53 | 91.74 | 91.28 | 91.50 | Paper reference |
| Paper VGG16 Table III | 89.39 | 91.81 | 89.09 | 90.42 | Paper reference |
| R0 VGG19 fc1 | 79.94 | 80.88 | 75.28 | 77.94 | OFFICIAL |
| R0 VGG16 fc1 | 80.04 | 81.90 | 74.15 | 77.72 | OFFICIAL |
| R1 VGG19 fc2 | 79.51 | 81.45 | 73.24 | 76.91 | OFFICIAL |
| R2 4-fold | 81.58 | 81.04 | 79.49 | 80.24 | OFFICIAL / COMPLETED |

[확실] R2 std는 Accuracy2.19/Precision2.89/Recall2.06/F1 2.22%p다. R2−R0 F1 +2.30%p, R2−R1 F1 +3.33%p이며 single-fold과 diagnostic을 위 official mean에 섞지 않는다.

[확실] **POST-HOC DIAGNOSTIC ONLY:** best F1 mean은 R0 VGG19 80.70%, R0 VGG16 81.00%, R1 80.28%다. R2 Fold1 best epoch6 F1 78.93%는 별도 single-fold diagnostic이다.

# 12. Decisions Made So Far

[확실] 다음은 기존 config/docs/실행에서 확인된 결정이다. 새 실험 조건을 이 문서에서 확정하지 않는다.

| Decision | 이유 / 영향 |
|---|---|
| **Paper-informed reconstruction** | 미공개 조건과 구현 선택을 논문 사실에서 분리 |
| **20×4096 → single-bias LSTM512** | 공개 dimension/parameter count와 대응 |
| **Final epoch official** | Held-out test best-epoch selection 배제 |
| R0 frozen fc1 reference | 재학습 없이 완료 결과 보존 |
| R1 feature point만 변경 | 같은 dimension/평가로 sensitivity 비교 |
| R1 이후 fc1 reference 유지 | FC2 official F1 개선 없음 |
| R2 block5 fine-tuning | CNN training policy/domain adaptation 효과 검증 |
| CUDA required / no CPU fallback | R2 실행환경을 명시적으로 강제 |
| Epoch checkpoint/resume | 중단 시 완료 epoch 복구 가능하게 함 |
| Current workers4/prefetch1 | workers8 full failure 이후 공유 공급 부담 축소 |
| **Batch16 / AMP=false 유지** | 실행 최적화와 연구 조건 변경을 분리 |

# 13. Failed / Interrupted Experiments

[확실] 실패와 중단을 성공 결과에서 분리하고 기존 output/MLflow를 보존한다.

| Experiment | Status | Cause / evidence | Recoverable |
|---|---|---|---|
| Old R2 Fold1 | INTERRUPTED / NOT OFFICIAL | epoch25까지 metric; 당시 중간 checkpoint 없음 | No, epoch26 resume 불가 |
| First benchmark sandbox attempt | FAILED diagnostic | Worker spawn 제한; 원본 실패 기록 보존 | 새 진단 실행으로 재검증 |
| workers2 profile | INTERRUPTED / EXECUTION STABILITY ISSUE | 반복 hang/JSON 미생성; 정확 원인 미확인 | Training checkpoint 대상 아님 |
| 2026-10-04 R2 attempt | FAILED (MLflow) | DB에 별도 실패 기록; 이 문서의 근거로 정확 원인 미확인 | 미확인 |
| workers8 official, 2026-10-06 | FAILED | 1455 shared-storage 오류; 완료0/history·checkpoint 없음 | No |
| workers4 short profile, 2026-10-06 | INTERRUPTED | 사용자 요청으로 실행 프로세스 종료; 10-07 성공 파일과 구분 | Training checkpoint 대상 아님 |
| workers4 official Fold1 | COMPLETED / FOLD1 FULL-RUN VALIDATED | 완료30 epoch/history/checkpoint/final output; MLflow FINISHED | 완료 모델·checkpoint 보존; 실제 복원 미검증 |

[확실] Old run의 DB `RUNNING` 표시는 완료나 현재 실행 증거로 사용하지 않는다. Current run은 별도 시작 시각과 output/history로 식별했다.

# 14. Known Limitations

[확실] 원논문의 pretrained/freeze/FC 위치, sampling/preprocessing, optimizer/LR/weight decay/batch/epoch, initialization과 validation/checkpoint/fold aggregation 정책은 미공개로 기록되어 있다. 조건 불일치 가능성을 제거한 exact comparison은 아니다.

[확실] 현재 subject-wise split과 별도 validation은 없다. 매 epoch held-out test 관찰이 있어 best metric을 selection에 사용하지 않는 경계를 유지해야 한다. 2,074-video 결과만으로 실제 운전자 환경의 일반화를 검증했다고 볼 수 없다.

[확실] Windows DataLoader stability issue와 긴 R2 runtime이 관찰되었다. Workers4는 resume을 포함해 전체4-fold final을 확보했지만 무중단/반복 안정성은 보장하지 않는다. Error1455 당시 system commit/pagefile 수치와 workers2 hang 원인은 미확인이다.

[확실] Seed helper는 deterministic algorithms를 `warn_only=True`로 설정하며 adaptive pooling backward warning이 문서에 기록되어 있다. Bitwise CUDA determinism은 보장되지 않는다.

[확실] Profile의 allocated/reserved/peak는 allocator counter다. 일부 JSON은 physical10 GiB보다 큰 counter를 기록하지만 physical residency/spill 원인을 확정할 수 없다. Cold-start profile, 짧은 window, queue/cache/concurrent workload 때문에 진단 평균을 그대로 epoch 속도나 최적 설정으로 일반화하지 않는다.

# 15. Remaining Work

[확실] 다음 순서는 현재 문서 계획과 구현 상태를 바탕으로 정리한 남은 작업이며 이번 작성에서 실행하지 않았다.

1. [확실] Fold1~4 epoch30 official final 수집과 mean/population std 집계, R0/R1/R2 비교 해석은 COMPLETED다.
2. [확실] R4 고정15/30 diagnostic은 MIXED / INCONCLUSIVE로 마쳤다. F1 delta−0.22%p, epoch15 우위1/4folds여서 새 학습을 권장하지 않고 Stage2를 종료했다.
3. [확실] NEXT는 Stage3 S3-1 mean hidden-state pooling 설계 검토다. R2 reference와 fixed30을 유지하며 코드/config/training은 별도 작업이다. 이후 Behavior → Fusion → Final Comparison을 계획한다.

[확실] R3 OPTIONAL/NOT RUN, R4 DIAGNOSTIC COMPLETED/MIXED/NO NEW TRAINING RECOMMENDED, R5 DEFER/NOT RUN이다. 미검증 조건을 부정한 것이 아니라 Stage2의 범위를 닫고 Context 설계로 이동한 결정이다.

[확실] `train_r2_vgg19.py`는 `--fold` 없이 네 fold를 실행할 때 CV summary를 만든다. 이번 별도 실행 output은 최종 resume 경로를 골라 읽기 전용으로 집계하고 문서에 기록했다. 새 aggregate output 파일 생성이나 기존 결과 재학습/덮어쓰기는 수행하지 않았다.

# 16. Planned Context Improvement

[확실] 현재 Stage3는 **DESIGN ONLY / NOT RUN**이다. [Stage3 계획](context_model/stage3_context_improvement_plan.md)에서 BiLSTM/Attention/Pooling/Projection을 비교하고 S3-1 uniform mean pooling을 첫 단일 변경으로 추천했다. 기존 R2 reference와 fixed30을 유지하고 구현은 아직 없다.

[확실] 후보 BiLSTM/Attention/Temporal Pooling/Feature Bottleneck은 DOCUMENTED ONLY다. 첫 mean pooling 비교 조건과 판단 기준은 Stage3 계획에 설계했으며, 전용 code/config는 NOT STARTED, 실험 결과는 NOT AVAILABLE이다.

[확실] GRU 등 문서에서 확인되지 않은 후보를 이 계획에 추가하지 않았다. 현재 VGG spatial pooling과 계획된 temporal pooling은 구분한다.

# 17. Planned Behavior Detection

[확실] EAR/MAR/Head Pose/Blink/Yawn/temporal behavior rule은 **NOT STARTED**다. `behavior/__init__.py`는 설명만 있고 `configs/behavior/default.yaml`의 EAR/MAR/head_pose는 null이다. Threshold/duration/counter/reset/얼굴 미검출 처리의 입력·출력은 `NOT AVAILABLE`다.

[확실] 이 문서 작성 지시에서 제시된 향후 흐름은 다음과 같으며 아직 실행 가능한 pipeline이 아니다. 임의 threshold나 duration을 확정하지 않는다.

```text
face/landmark → EAR/MAR/pose → frame feature
→ temporal event → behavior decision
```

# 18. Planned Fusion Detection

[확실] 목적은 Context score와 Behavior event를 결합하는 것이다. 상태는 **PLANNED / NOT STARTED**다. `fusion/__init__.py`는 설명만 있고 config의 context/behavior_events/decision_rule은 null이다.

[확실] Late/score/rule/weighted fusion 중 구체 방식은 **NOT DECIDED**다. 가중치/threshold/temporal integration/평가 결과를 만들어 기록하지 않는다.

# 19. Final Evaluation Plan

[확실] 현재 확정된 비교는 R0 VGG19/VGG16, R1, R2 및 후속 refinement다. Accuracy/Precision/Recall/F1, 네 fold mean/population std, precision-recall balance와 fold variability를 보고한다. Paper Table III는 reference로 분리한다.

[확실] Context/Behavior/Fusion 이후 Final Comparison은 프로젝트 방향이다. Behavior-only 조건, Fusion variants, 구체 ablation matrix와 event-level 평가 protocol은 **NOT AVAILABLE**이며 향후 사전 정의해야 한다. 기존 video classification metric만으로 아직 없는 behavior event 검증을 완료했다고 표현하지 않는다.

# 20. Important Commands

[확실] 아래는 CLI에서 지원하는 명령의 문서화다. **이번 작업에서 실행하지 않았다.** Existing output을 보존하며 official Fold1 명령은 완료된 invocation의 기록이다. 재실행 용도가 아니다.

[확실] MLflow server의 기존 프로젝트 명령이다.

```powershell
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

[확실] R0/R1 extraction/training의 지원 명령이다. 완료된 cache/output이 연결된 기존 config로 재실행하거나 `--overwrite`를 사용하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/extract_paper_vgg_features.py --config configs/paper_reconstruction/vgg19_lstm.yaml
.\.venv\Scripts\python.exe scripts/train_paper_lstm_cv.py --config configs/paper_reconstruction/vgg19_lstm.yaml
.\.venv\Scripts\python.exe scripts/extract_paper_vgg_features.py --config configs/paper_reconstruction/vgg16_lstm.yaml
.\.venv\Scripts\python.exe scripts/train_paper_lstm_cv.py --config configs/paper_reconstruction/vgg16_lstm.yaml
.\.venv\Scripts\python.exe scripts/extract_paper_vgg_features.py --config configs/paper_reconstruction/vgg19_lstm_fc2.yaml
.\.venv\Scripts\python.exe scripts/train_paper_lstm_cv.py --config configs/paper_reconstruction/vgg19_lstm_fc2.yaml
```

[확실] Workers4/prefetch1의 bounded smoke/profile 명령 예시다. 출력은 미사용 이름이어야 하며 이번에 이 파일을 생성하지 않았다.

```powershell
.\.venv\Scripts\python.exe scripts/profile_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --batches 1 --test-batches 1 --num-workers 4 --pin-memory --persistent-workers --prefetch-factor 1 --output outputs/paper_reconstruction/refinement/R2_workers4_prefetch1_smoke_new.json
.\.venv\Scripts\python.exe scripts/profile_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --batches 5 --test-batches 1 --num-workers 4 --pin-memory --persistent-workers --prefetch-factor 1 --output outputs/paper_reconstruction/refinement/R2_workers4_prefetch1_profile5_new.json
```

[확실] 완료된 official Fold1 invocation은 다음과 같다. Existing directory라 재실행 시 train CLI가 중단한다.

```powershell
.\.venv\Scripts\python.exe scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --num-workers 4 --pin-memory --persistent-workers --prefetch-factor 1 --output-root outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers4_prefetch1_official
```

[확실] 아래 resume은 **학습 중단 후 신뢰하는 local checkpoint를 선택할 때 사용할 지원 명령**이다. 현재 run과 동시에 실행하지 않는다. Output은 새 경로이며 saved config와 같은 execution override를 유지한다.

```powershell
.\.venv\Scripts\python.exe scripts/train_r2_vgg19.py --config configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml --fold 1 --num-workers 4 --pin-memory --persistent-workers --prefetch-factor 1 --resume-from outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers4_prefetch1_official/fold_1/last_checkpoint.pt --output-root outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers4_prefetch1_resumed_new
```

# 21. Important Files

[확실] 핵심 source-of-truth와 역할은 다음과 같다. 모든 경로는 repository-relative다.

| Role | Path |
|---|---|
| Overview | `README.md` |
| Paper basis | `docs/paper_reconstruction/baseline_spec.md` |
| Data policies | `docs/paper_reconstruction/data_preparation.md` |
| Pipeline | `docs/paper_reconstruction/model_pipeline.md` |
| R0 results | `docs/paper_reconstruction/results.md` |
| Refinement/R1/troubleshooting | `docs/paper_reconstruction/reconstruction_refinement.md` |
| R0 configs | `configs/paper_reconstruction/vgg19_lstm.yaml`, `configs/paper_reconstruction/vgg16_lstm.yaml` |
| R1 config | `configs/paper_reconstruction/vgg19_lstm_fc2.yaml` |
| R2 config | `configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml` |
| Metadata/split/sequence creation | `scripts/build_sust_ddd_metadata.py`, `scripts/build_paper_cv_splits.py`, `scripts/build_paper_20frame_sequences.py` |
| Split/sequence summaries | `data/splits/paper_reconstruction/split_summary.json`, `data/sequences/paper_reconstruction_20frame_summary.json` |
| Feature extraction | `scripts/extract_paper_vgg_features.py`, `src/drowsiness_fusion/models/vgg_feature_extractor.py` |
| Baseline temporal model | `src/drowsiness_fusion/models/paper_lstm.py` |
| Baseline training | `scripts/train_paper_lstm_cv.py`, `scripts/train_paper_lstm_fold.py` |
| R2 model | `src/drowsiness_fusion/models/r2_vgg19_lstm.py` |
| R2 trainer | `src/drowsiness_fusion/training/r2_trainer.py` |
| R2 checkpoint/environment | `src/drowsiness_fusion/training/r2_execution.py` |
| R2 loader | `src/drowsiness_fusion/data/r2_dataloader.py` |
| R2 orchestration | `scripts/train_r2_vgg19.py` |
| Diagnostic scripts | `scripts/profile_r2_vgg19.py`, `scripts/benchmark_r2_dataloader.py`, `scripts/smoke_r2_vgg19.py` |
| R0 summaries/histories | `outputs/paper_reconstruction/vgg19_lstm/`, `outputs/paper_reconstruction/vgg16_lstm/` |
| R1 summary/history | `outputs/paper_reconstruction/refinement/R1_vgg19_fc2/` |
| Workers8 failed output | `outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers8_official/` |
| Current R2 history/environment/checkpoint | `outputs/paper_reconstruction/refinement/R2_cuda_fold1_workers4_prefetch1_official/fold_1/` |
| R2 Fold2 final | `outputs/paper_reconstruction/refinement/R2_cuda_fold2_workers4_prefetch1_official/fold_2/` |
| R2 Fold3 resumed final | `outputs/paper_reconstruction/refinement/R2_cuda_fold3_resume_next/fold_3/` |
| R2 Fold4 resumed final | `outputs/paper_reconstruction/refinement/R2_cuda_fold4_resumed_20261015/fold_4/` |
| Workers0/8 profiles | `outputs/r2_profile_fold1.json`, `outputs/r2_profile_fold1_workers8.json` |
| Initial/recheck benchmark | `outputs/r2_dataloader_benchmark_summary_20261002.json`, `outputs/r2_dataloader_benchmark_recheck.json` |
| Workers4 smoke/profile | `outputs/paper_reconstruction/refinement/R2_workers4_prefetch1_smoke_20261007.json`, `outputs/paper_reconstruction/refinement/R2_workers4_prefetch1_profile5_20261007.json` |
| Behavior/Fusion placeholders | `src/drowsiness_fusion/behavior/__init__.py`, `src/drowsiness_fusion/fusion/__init__.py`, `configs/behavior/default.yaml`, `configs/fusion/default.yaml` |
| MLflow local storage | `mlflow.db`, `mlartifacts/` |

# 22. Development Timeline

[확실] Git의 날짜는 **commit 기록 날짜**이며 최초 구현/실행 시각과 같다고 단정하지 않는다. 문서 날짜와 MLflow 시작 날짜는 해당 기록의 의미로 표시한다. 미확인 날짜를 새로 만들지 않는다.

| Date / source | Development event | Result / next question |
|---|---|---|
| 2026-09-26, git 기록 | 초기 폴더와 metadata/sequence 작업 기록 | Dataset preparation 기반 마련 |
| 2026-09-28, git 기록 | VGG16/VGG19 paper reconstruction baseline 완료 기록 | R0 official 결과와 diagnostic 분리 |
| 2026-09-28, git 기록 | Refinement roadmap, R1 FC2/MLflow 구현 기록 | 미공개 feature point sensitivity 검토 |
| 2026-09-29, MLflow/git | R1 4-fold 실행 및 R0 비교 문서화 | FC2 official F1 개선 없음; FC1 reference 유지 |
| 2026-09-29, MLflow 시작 | Old R2 Fold1 시작 | 이후 epoch25까지 metric; interrupted/no checkpoint |
| 날짜 미확인, 구현 흐름 | R2 설계 후 CUDA 정책·checkpoint/resume·timing 준비 | Old run 복원 불가를 새 run의 회복 기능으로 보완 |
| 2026-10-02, 문서 | workers0 CUDA profile/첫 worker benchmark | Loader 병목 관찰; 동시 영상 작업 때문에 재검증 필요 |
| 2026-10-04 문서·git에 반영 | 독립 benchmark 재검증, workers2 hang, workers8 short profile 선정 | 평균 개선과 안정성 한계 함께 기록 |
| 2026-10-04, MLflow | 별도 R2 FAILED attempt 기록 | 정확 cause/sequence는 미확인 |
| 2026-10-06, MLflow/사용자 traceback | workers8 official shared-memory1455 failure | 완료0/no checkpoint; 실행 설정 재검토 |
| 2026-10-06, 대화 기록 | workers4 short profile 사용자 요청 중단 | 이후 성공한 diagnostic과 구분 |
| 2026-10-07, diagnostic files | workers4/prefetch1 smoke1/profile5 완료 | Gradient policy 통과; 다음 full-run 검증 |
| 2026-10-07 20:48 KST, MLflow | workers4/prefetch1 official Fold1 시작 | 새 output/run, batch16/AMP=false 유지 |
| 2026-10-08 10:15:27 KST, snapshot | 당시 Fold1 완료10 epoch, checkpoint/history 존재 | 당시 RUNNING; official final은 NOT AVAILABLE |
| 2026-10-09 10:23:37 KST, MLflow 종료 | Workers4/prefetch1 Fold1 epoch30 완료 | FINISHED; official Fold1 F1 76.89%; 4-fold 결과는 NOT AVAILABLE |
| 2026-10-09 16:58:37 KST, MLflow 시작 | Fold2 child RUNNING, 별도 환경 JSON 생성 | 요청의 NOT RUN 전제와 차이; 문서 작업에서 실행하지 않음 |
| 2026-10-11 06:16:17 KST, MLflow 종료 | Fold2 epoch30 완료 | FINISHED; official F1 79.84%; 2/4 folds completed |
| 2026-10-11 18:00:39 KST, MLflow 시작 | Fold3 child RUNNING, 별도 환경 JSON | 요청의 NOT RUN 전제와 차이; 문서 작업에서 실행하지 않음 |
| 2026-10-12, output/실행 기록 | Fold3 epoch15 중단 → resume16 저장 → 사용자 요청 중단 → resume17~30 | Checkpoint 복원으로 완료 epoch 보존 |
| 2026-10-13 13:08:37 KST, final model/MLflow | Fold3 epoch30 완료 | FINISHED; final F1 81.36% |
| 2026-10-14 22:19:59 KST, checkpoint | Fold4 epoch21까지 저장 후 중단 | 다음 resume은 epoch22 |
| 2026-10-15 22:17:26 KST, final model | Fold4 resume22~30 완료 | Final F1 82.89%; 종료 시 cp949 오류로 MLflow 상태 잔존 |
| 2026-10-16, 집계/저장 검증 | 네 fold final 수집·4-fold 집계·해석; MLflow 저장 확인 후 서버 종료 | R2 official F1 80.24% ± 2.22%p; R4 권고 |
| 2026-10-18, 문서화 | README/refinement/development history 갱신 | R2 완료; 다음 R4 설계, epoch 값 미정 |
| 2026-10-18, R4 diagnostic | 공통15/30 비교: F1 delta−0.22%p, epoch15 우위1/4folds | MIXED / INCONCLUSIVE; 새 학습 권장 안 함 |
| 2026-10-18, 범위 결정·설계 | Stage2 종료; Stage3 R2 reference/mean pooling 설계 | Stage2 COMPLETED; Stage3 DESIGN ONLY / NOT RUN |

# 23. Status Matrix

[확실] **Snapshot: 2026-10-18 KST — Stage2 종료 후.** R2 결과와 기존 MLflow 저장 기록을 유지하고 R4 diagnostic 및 Stage3 설계 상태를 반영했다.

| Work | Status |
|---|---|
| Dataset preparation | COMPLETED |
| Paper basis/initial reconstruction | IMPLEMENTED; exact reproduction으로 주장하지 않음 |
| R0 VGG19/VGG16 | COMPLETED |
| R1 FC2 / 4-fold analysis | COMPLETED |
| R2 implementation | COMPLETED |
| CUDA/checkpoint/resume | IMPLEMENTED |
| workers0/8 short profiling | COMPLETED / DIAGNOSTIC ONLY |
| workers2 real profile | INTERRUPTED / EXECUTION STABILITY ISSUE |
| workers8 official | FAILED / completed0 / no checkpoint |
| workers4 smoke/profile5 | COMPLETED / DIAGNOSTIC ONLY |
| R2 Fold1 | **COMPLETED / completed30 of30 / official F1 76.89%** |
| R2 Fold2 | **COMPLETED / completed30 of30 / official F1 79.84%** |
| R2 completed folds / 4-fold training | **4 / 4 — COMPLETED** |
| R2 Fold3 | **COMPLETED / epoch30 / final F1 81.36%** |
| R2 Fold4 | **COMPLETED / epoch30 / final F1 82.89%; MLflow terminal status stale** |
| R2 official aggregate / interpretation | **COMPLETED / COMPLETED** |
| R2 official F1 | **80.24% ± 2.22%p (ddof=0)** |
| Stage2 | **COMPLETED — scope closed** |
| R3 | **OPTIONAL / NOT RUN** |
| R4 | **DIAGNOSTIC COMPLETED / MIXED / NO NEW TRAINING RECOMMENDED** |
| R5 | **DEFER / NOT RUN** |
| Stage3 | **DESIGN ONLY / NOT RUN — S3-1 mean hidden-state pooling** |
| Context candidates | DOCUMENTED ONLY |
| Context dedicated implementation | NOT STARTED |
| EAR/MAR/Head Pose/Blink/Yawn/temporal rule | NOT STARTED |
| Fusion implementation | NOT STARTED |
| Final ablation matrix | NOT AVAILABLE |

[확실] **NEXT:** Stage3 S3-1 설계 검토 후 별도 구현 범위 결정. **AFTER:** Context 실험 → Behavior Detection → Fusion Detection. 이번 작업은 설계까지이며 학습/config/구현을 수행하지 않았다.
