# S3-2 Learnable Temporal Attention Experiment Plan

[확실] 작성일: 2026-11-05 KST. **S3-2 ARCHITECTURE DESIGN COMPLETED / DESIGN ONLY / Implementation NOT STARTED / Training NOT RUN.** **Learnable Temporal Attention: LSTM의 각 시점 표현에 학습 가능한 중요도를 부여한 뒤 가중합하는 시간 정보 결합 방식**이다. 아래 선택 설계는 다음 구현 단계의 명세이며 구현된 모델이나 실험 결과가 아니다.

이 문서는 **무엇을 할 것인가**를 기록한다. 연구 질문·비교 기준·변경 범위·통제 조건·평가 원칙과 향후 순서를 정의한다. 실제 수행 작업·검증·오류·결정은 [진행 기록](s3_2_attention_progress.md)에 시간 순서로 남긴다. 이번 S3-2.1에서 확정한 설계와 이후 구현·실행에서 검증할 항목을 구분한다.

## 1. Background

[확실] R2 reference는 VGG19 CNN → 4096D frame features → LSTM512 → 마지막 timestep hidden state → classifier 구조다. S3-1은 같은 CNN/LSTM/classifier를 유지하고 시간 정보 결합 방식(temporal readout)을 마지막 hidden state에서20개 timestep hidden states의 동일 가중치 평균으로 바꿨다.

[확실] [S3-1 official 4-fold 결과](stage3_context_improvement_plan.md#s3-1-final-results--44-folds)는 R2 F1 **80.24% ± 2.22%p**, S3-1 F1 **77.26% ± 0.26%p**, raw 평균 차이 **−2.982288%p (약−2.98%p)**다. Mean은%, population std는%p이며 ddof=0이다. S3-1은 COMPLETED / NO IMPROVEMENT OVER R2 REFERENCE이고4/4 fold에서 final F1 improvement가 관찰되지 않았다.

[확실] 이 결과를 근거로 단순 평균에서 각 timestep의 기여도를 학습하는 방향으로 질문을 확장한다. S3-1의 결과가 Attention의 필요성이나 개선을 입증한 것은 아니다. Stage3는 IN PROGRESS이며 이번에는 실제 코드 확인과 Architecture Design만 수행한다.

## 2. Research Question

**Can learnable temporal attention improve the temporal representation compared with R2 final-hidden readout and S3-1 equal mean pooling?**

모든 timestep을 동일하게 평균하는 대신 모델이 각 timestep의 중요도를 학습하게 하면, R2의 마지막 hidden state 사용과 S3-1의 동일 가중치 평균보다 유용한 시간 표현을 얻을 수 있는가? 향후 동일 조건의 video-level 분류 지표로 비교할 계획이다. S3-2의 개선 여부는 **NOT AVAILABLE**이다.

## 3. Hypothesis

[추측] **HYPOTHESIS — 아직 검증되지 않음.** 모든 timestep이 졸음 판별에 동일하게 중요한 것은 아닐 수 있다. 단순 평균에서 중요한 temporal signal이 다른 시점 정보와 섞여 희석됐을 가능성이 있으므로, 중요한 시점의 기여도를 학습하는 방법이 더 유용할 수 있다는 가설을 검토한다.

정보 희석을 측정하거나 S3-1 성능 차이의 원인으로 증명한 실험은 없다. Attention을 추가하면 성능이 향상된다고 예측값을 만들지 않으며, 향후 Attention의 가중치만으로 원인이나 설명 가능성이 검증됐다고 주장하지 않는다.

## 4. Reference Models

| Reference | 시간 정보 결합 방식 | Official epoch30 F1 mean ± population std |
|---|---|---:|
| R2 / S3-0 | 마지막 timestep hidden state | 80.24% ± 2.22%p |
| S3-1 | 전체20개 timestep hidden states의 동일 가중치 평균 | 77.26% ± 0.26%p |

[확실] 두 reference는 완료된 기존 결과를 사용한다. Raw fold metric·집계·resume 이력은 [Stage3 최종 결과](stage3_context_improvement_plan.md#s3-1-final-results--44-folds)에 있다. S3-2 결과와 비교할 때 같은 fold끼리 raw metric 차이를 계산할 계획이며, 이번 문서 작업에서 reference를 재학습하거나 output을 변경하지 않는다.

## 5. Selected Architecture — Direct Scalar Temporal Attention

### 5.1 실제 코드 확인과 호환성

아래 파일을 읽고 구조·호출 방식·parameter grouping을 확인했다. 기존 코드와 YAML은 수정하지 않았다.

| 확인한 파일 | 확인 내용 |
|---|---|
| [R2 model](../../src/drowsiness_fusion/models/r2_vgg19_lstm.py) | `R2VGG19LSTM.forward()`가 `self.lstm(x)`의 마지막 hidden `[B,512]`를 head에 전달. `parameter_summary()`는 전체/학습/frozen parameter를 `numel()`로 계산 |
| [S3-1 model](../../src/drowsiness_fusion/models/stage3_temporal_mean.py) | `Stage3TemporalMean.forward()`의 FC1 뒤 reshape `[B,20,4096]` → `return_sequences=True` → `[B,20,512]` → `mean(dim=1)` → 같은 head |
| [공통 LSTM](../../src/drowsiness_fusion/models/paper_lstm.py) | `KerasCompatibleLSTM`: 단일-bias1-layer, 기본은 마지막 hidden; optional sequence는 `torch.stack(hidden_states, dim=1)`으로 반환 |
| [VGG 생성/feature 코드](../../src/drowsiness_fusion/models/vgg_feature_extractor.py) | ImageNet VGG19 생성 및 FC1 after ReLU4096D 정의 확인 |
| [S3-1 training script](../../scripts/train_stage3_s3_1.py), [R2 training script](../../scripts/train_r2_vgg19.py) | Block5와 `temporal_parameters()`를 명시적으로 수집한 두 Adam group; 누락·중복 검증; default forward의 logits를 loss에 전달 |
| [공통 trainer](../../src/drowsiness_fusion/training/r2_trainer.py) | train/evaluate 모두 `model(frames)`를 Tensor logits로 기대; 현재 AMP/scaler 경로 없음 |
| [S3-1 config](../../configs/context_model/stage3_s3_1_temporal_mean.yaml), [R2 config](../../configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml), [loader 설정](../../src/drowsiness_fusion/data/r2_dataloader.py) | 구조·LR·학습 조건과 loader 기본값/실행 override 차이 확인 |
| [기존 S3-1 검증 코드](../../tests/test_stage3_temporal_mean.py) | meta count, hidden/head hook, gradient, optimizer coverage 검증 방식과 기존 초기화 순서 확인. 이번에 실행·수정하지 않음 |

[확실] 현재 classifier는 별도 reusable `nn.Sequential`/head module이 아니다. `classifier_fc1 → classifier_relu → classifier_fc2 → classifier_sigmoid → classifier_fc3`라는 명시적 attribute와 호출 순서로 존재한다. 입력512D → Linear512 → ReLU → Linear64 → Sigmoid → Linear2 logits다. S3-2 전용 모델에서 이 구조·이름·초기화 순서를 유지할 계획이며, `self.classifier`가 이미 있다고 가정하지 않는다. 기존 R2/S3-1을 공통 factory/head로 리팩터링하지 않는다.

### 5.2 후보 비교와 선정

| 기준 | Candidate A — Direct Scalar | Candidate B — Projected / Additive |
|---|---|---|
| Score 개념 | `w^T h_t` (아래 bias 결정 참고) | 예: `v^T activation(W h_t + b) + c` |
| Single-change 비교 | 시간 정보 결합 방식만 교체; 원래 hidden 가중합 | 시간 결합 외 projection과 비선형 표현 용량이 추가됨 |
| Parameter / 설계 변수 | 선택안512개; hidden dimension·activation·dropout 추가 없음 | dimension d, activation, 각 layer 초기화/bias 등의 선택 증가 |
| Parameter 산술 예 | bias=False512; bias=True513 | 두 Linear 모두 bias=True인 예는 `512d + d + d + 1 = 514d + 1`; 채택한 구조가 아님 |
| Head / training 호환 | context512D; 기존 head와 두 optimizer group 유지 가능 | context를 원래 hidden의 가중합으로 정의하면 head 유지 가능하나 추가 scorer 설계 필요 |
| Weight 관찰 | sample별20개 상대 가중치 확인 가능 | 가능하나 가중치가 인과 설명이라는 보장은 동일하게 없음 |
| 결정 | **SELECTED — bias=False** | **NOT SELECTED for S3-2**; 성능이 나쁘다고 입증한 것은 아님 |

[설계 결정] 최소 parameter·hyperparameter로 **temporal weighting 자체**를 비교하기 위해 Candidate A를 선택한다. Transformer Multi-Head Self-Attention이나 query/key/value 구조가 아니다. CNN/LSTM/head, hidden 크기·layer 수와 training/evaluation 조건은 유지한다.

### 5.3 Score bias 결정과 초기화

[설계 결정] Score layer는 **`nn.Linear(512, 1, bias=False)`**로 정한다. 비교용 기본안의 bias=True513개에서 **softmax에 영향을 주지 않는 공통 scalar bias 한 개를 제외**한다.

```text
bias=True라면 s_t = w^T h_t + b
alpha_t = exp(w^T h_t + b) / sum_j exp(w^T h_j + b)
        = exp(w^T h_t) / sum_j exp(w^T h_j)
```

같은 scorer의 bias b는 모든 timestep에 공통이므로 정규화에서 상쇄된다. 이상적인 수학식에서 분류 loss의 bias 미분도0이다. 표현력을 추가하지 않는 parameter를 제거해 작은 구조와 명확한 gradient 검증을 유지한다. 기존 계획에서 bias는 NOT DECIDED였으므로 이는 예정된 설계 결정이며, 기존 구현의 bias를 변경한 것이 아니다.

[설계 결정] 초기화는 설치된 PyTorch `nn.Linear.reset_parameters()`의 **기본 weight 초기화**를 사용하도록 정한다. 로컬 source에서 `kaiming_uniform_(a=sqrt(5))`, 동등 범위 `[-1/sqrt(512), +1/sqrt(512)]`를 확인했다. 별도 zero/Xavier 초기화나 새 seed/hyperparameter는 추가하지 않는다. 학습 초기 Attention이 동일 가중치0.05라고 주장하지 않는다.

공통 VGG/LSTM/head를 기존과 같은 순서로 만든 **뒤에** scorer를 생성할 계획이다. 기존 공통 parameter 초기화에 scorer의 RNG 소비가 먼저 개입하지 않게 한다. 구현 단계에서 같은 seed와 같은 backbone 조건의 공통 초기 state를 직접 대조한다. 매 fold fresh ImageNet backbone과 seed42 초기화로 시작할 계획이며, 학습 완료된 R2/S3-1 checkpoint로 warm-start하지 않는다.

### 5.4 S3-1과 수학적 차이

```text
H = [h_1, ..., h_20]                    [B,20,512]
S3-1: z_b = (1/20) sum_t h_(b,t)        모든 가중치0.05
S3-2: s_(b,t) = w^T h_(b,t)             scalar per timestep
       alpha_(b,t) = softmax(s_b, dim=1)_t
       z_b = sum_t alpha_(b,t) * h_(b,t)
```

[설계 결정] Softmax는 **time axis dim=1**의20개 score에 적용한다. `[B,20,1]`의 마지막 singleton 축이나 batch 축에 적용하지 않는다. Attention score에는 추가 activation/projection/temperature/scaling을 넣지 않으며, 원래 LSTM hidden states를 그대로 가중합한다. 실제20 slots에 padding/mask/position encoding을 추가하지 않는다.

w는 sample/time에 공유하는 학습 parameter지만 **alpha는 입력 sequence의 hidden states에서 계산하는 sample별 값**이다. 고정된20개 전역 위치 가중치를 학습하는 방식이 아니다. 입력이 같거나 score가 같으면 가중치가 같을 수 있으므로 sample마다 반드시 다르다고 주장하지 않는다.

### 5.5 Tensor Shape Flow

기존 code에서 확인한 경로 뒤에 선택 Attention을 연결했을 때의 **설계 계약**이다. 아직 S3-2 실행으로 검증된 shape는 아니다.

| 단계 | Shape / 방향 |
|---|---|
| Frame sequence input | `[B,20,3,224,224]` |
| CNN 처리용 reshape | `[B*20,3,224,224]` |
| FC1 after ReLU / sequence 복원 | `[B,20,4096]` |
| LSTM `return_sequences=True` | `[B,20,512]` |
| Score Linear512→1 | `[B,20,1]` |
| Temporal softmax weight alpha | `[B,20,1]`, dim=1에서 합≈1 |
| Weighted hidden `alpha * H` | `[B,20,512]` |
| Context `sum(dim=1)` | `[B,512]` |
| Existing head | `[B,512] → [B,512] → [B,64] → [B,2]` |
| Default forward output | Tensor logits `[B,2]` |

B=1에서도 batch 축을 보존한다. 무차별 `squeeze()`로 batch/time 축을 제거하지 않는 방향으로 구현할 계획이다.

### 5.6 Expected Parameter Count

**EXPECTED / TO BE VERIFIED AT IMPLEMENTATION.** 이번에 기존 표준 PyTorch Linear만 meta device로 생성해 parameter shape/numel을 확인했다. S3-2 모델은 구현·생성하지 않았으며 아래 전체 count는 baseline에 추가 수를 더한 **예상값**이다.

| 구성 | Total | Trainable | Frozen | 검증 수준 |
|---|---:|---:|---:|---|
| 기존 R2/S3-1 baseline | 132,523,778 | 19,174,082 | 113,349,696 | 기존 code/metadata/검증 이력 |
| 선택 scorer: Linear512→1, bias=False | +512 | +512 | +0 | 표준 Linear `[1,512]` weight의 numel만 확인 |
| **Expected S3-2 selected design** | **132,524,290** | **19,174,594** | **113,349,696** | **EXPECTED / TO BE VERIFIED AT IMPLEMENTATION** |
| 비교용 bias=True안 (NOT SELECTED) | 132,524,291 | 19,174,595 | 113,349,696 | scorer513개일 때 산술; 선택 모델 actual count가 아님 |

구현 시 실제 S3-2 전체 모델에 기존 `parameter_summary()` 방식의 전체/trainable/frozen 집계를 적용하고 optimizer coverage와 함께 다시 검증한다. 이 함수는 annotation이 R2지만 실행 본문은 model parameters의 numel/requires_grad를 집계해 S3-1에서도 재사용하고 있다. 이번에는 함수를 변경하지 않았다.

### 5.7 Gradient Flow / Freeze Policy

```text
loss → classifier → context → weighted sum → original LSTM hidden states
                                         ↘ temporal softmax → score weight
hidden states → LSTM → FC1 operation → VGG19 Block5
```

이 그림의 화살표는 **backprop 경로**다. Hidden states에는 weighted-sum 직접 경로와 score/softmax 경로가 모두 연결될 예정이다.

| 부분 | Parameter 정책 / 예정 검증 |
|---|---|
| VGG19 Block1~4 | Frozen / parameter grad 없음; 기존 no_grad 구간 유지 |
| FC1 parameters | Frozen / parameter grad 없음 |
| FC1 operation / ReLU / avgpool | Block5로 gradient를 전달해야 함; 전체를 no_grad/detach하면 안 됨 |
| Block5 | Trainable / loss gradient 필요 |
| LSTM | Trainable / loss gradient 필요 |
| Attention score weight | Trainable / loss gradient 필요; 선택안에 bias parameter 없음 |
| Classifier | Trainable / loss gradient 필요 |

Gradient 검증은 구현 단계의 계획이다. 이번에 실제 gradient test를 실행하거나 통과했다고 쓰지 않는다. Nonzero gradient는 controlled non-degenerate input/label로 확인하며 모든 실제 batch에서 모든 parameter의 값이 nonzero여야 한다고 요구하지 않는다. S3-1 mean의 동일 timestep 직접 gradient 패턴을 Attention에 그대로 요구하지 않는다.

### 5.8 Optimizer Handling

[확실] 현재 `temporal_parameters()`는 **LSTM + classifier_fc1/fc2/fc3**만 명시적으로 yield한다. 새 module을 model attribute에 등록해도 이 iterator에 자동 포함되지 않는다. Training script는 이 목록과 Block5 목록을 만들어 중복/누락을 검사한 뒤 두 Adam group을 구성한다. Attention을 iterator에 넣지 않으면 trainable coverage 검사에서 오류가 난다.

[설계 결정] 향후 **S3-2 전용 model의 `temporal_parameters()`에 `attention_score.weight`를 명시적으로 포함**한다. 별도 세 번째 group이나 Attention 전용 LR을 만들지 않고 기존 temporal/head group **LR0.0001**을 사용한다. Block5 group은 LR0.00001, 공통 weight decay0.00001의 Adam을 유지한다. 두 group은 disjoint이며 trainable parameter 전체를 정확히 한 번 포함해야 한다.

Attention parameter를 추가하기 위한 **S3-2 model iterator 확장은 필요**하지만, Attention 전용 optimizer class나 trainer/loss 변경은 필요하지 않다. 향후 전용 S3-2 training 진입점이 두 group 패턴을 사용할 수 있도록 설계하며 기존 R2/S3-1 model/script/config는 수정하지 않는다. 지금은 어떤 진입점도 생성하지 않는다.

### 5.9 Attention Diagnostic Access

[설계 결정] Default **`forward(frames) → Tensor logits [B,2]`**를 유지한다. 기존 trainer/evaluator의 Tensor 계약을 바꾸지 않는다. 별도의 diagnostic method **`forward_with_attention(frames) → (logits, alpha)`**를 S3-2 전용 model에서 제공할 계획이다. Alpha shape는 `[B,20,1]`이다.

두 public 경로는 동일한 내부 forward 계산을 공유해 diagnostic 호출 한 번에 logits와 alpha를 얻도록 설계한다. 기본 training 호출에 tuple이나 flag를 요구하지 않는다. 최근 alpha를 model attribute에 graph째로 저장하는 cache, 자동 파일 저장, 매 batch 추가 forward는 넣지 않는다. Diagnostic은 필요할 때 caller가 `eval()`/`no_grad()`로 호출하고 저장이 필요하면 명시적으로 detach/cpu할 계획이다. 이번에는 method나 weight 저장 기능을 구현하지 않는다.

높은 alpha의 의미는 **“모델이 최종 분류 representation을 구성할 때 상대적으로 크게 반영한 timestep”**으로 제한한다. Hidden state는 그 시점까지의 과거 입력도 담으므로 특정 원본 frame의 단독 기여나 졸음의 원인으로 해석하지 않는다. Attention 관찰을 official checkpoint 선택에 사용하지 않는다.

### 5.10 Exclusions

이번 조건은 BiLSTM, Multi-Head Attention, Transformer/Self-Attention block, query/key/value, bottleneck/projection, positional encoding, Attention dropout/regularization, CNN/LSTM hidden/layer/head 변경을 제외한다. Scheduler/early stopping/augmentation 변경도 제외한다. 더 복잡한 구조의 성능이 나쁘다는 결론이 아니라, 하나의 주요 변수인 시간 정보 가중 결합을 분리하기 위한 범위 결정이다.

### 5.11 Implementation Validation Plan — NOT RUN

Test 파일이나 S3-2 모델을 이번에 생성하지 않는다. 다음 구현/검증 단계에서 수행할 최소 확인 항목이다.

| 예정 검증 | 확인할 내용 / 완료 기준 |
|---|---|
| Output shape | `[B,20,3,224,224] → [B,2]`; B=1/2에서 default return은 Tensor |
| Intermediate shape | feature `[B,20,4096]`, hidden `[B,20,512]`, score/alpha `[B,20,1]`, context `[B,512]` |
| Normalization / pooling | alpha finite/nonnegative, `sum(dim=1)≈1`; context가 original H의 가중합과 일치 |
| Input-dependent / axis | 비퇴화 synthetic sequence에서 입력에 따른 alpha 계산 확인; batch/마지막 singleton 축 softmax 오류 검출 |
| Uniform limit | 검증용으로 scorer weight를0으로 두면 alpha0.05이고 S3-1 mean/context와 일치; 선택 기본 초기화가0이라는 뜻은 아님 |
| Gradient / freeze | Block1~4/FC1 parameter grad 없음; Block5/LSTM/Attention/head에 유한한 grad 및 비퇴화 예제의 nonzero signal; FC1을 통한 Block5 경로 유지 |
| Expected count | 표준 layer512와 actual model132,524,290 / trainable19,174,594 / frozen113,349,696 비교 |
| Optimizer coverage | Attention을 temporal group에 포함; CNN/temporal disjoint; trainable 전체 포함; LR/weight decay 유지 |
| Forward / diagnostic parity | eval 상태에서 default와 diagnostic logits 일치; alpha shape/합 확인; graph cache나 기본 저장 부작용 없음 |
| Initialization | 같은 seed/backbone의 공통 state를 R2/S3-1과 비교; scorer는 기존 공통 module 생성 뒤에 생성 |
| R2 / S3-1 isolation | 기존 source/config/script/state 의미 불변; 기존 관련 회귀 검증을 후속 단계에서 수행 |
| Config / checkpoint identity | 향후 전용 condition/readout을 기록하고 R2/S3-1 checkpoint를 S3-2로 잘못 복원하지 않도록 검증 |

Full shape/parameter 검증은 meta 또는 기존의 bounded VGG-compatible mock 패턴부터 계획하고 pretrained 다운로드·원본 dataset 처리·full training은 구현 검증의 자동 부수 작업으로 실행하지 않는다. `parameter_summary`나 mock만의 count를 실제 S3-2 full-model count로 혼동하지 않는다.

## 6. Controlled Conditions

원칙적으로 R2/S3-1과 동일하게 유지할 **비교 기준**이다. 현재 S3-2 config를 생성하거나 실행 설정을 적용했다는 뜻은 아니다.

| 항목 | 유지할 기준 |
|---|---|
| Dataset / fold split | SUST-DDD2,074 videos; 같은 label-stratified4-fold / seed42; drowsy positive |
| Sequence length / preprocessing | 같은20-frame indices; RGB / bilinear224×224 / ImageNet normalization; no augmentation |
| VGG19 backbone / 학습 정책 | 같은 ImageNet VGG19; Block1~4 frozen; Block5 fine-tuning; FC1 frozen after ReLU4096D |
| LSTM / classifier | 기존 single-bias LSTM input4096 / hidden512 / layer1 및 기존 classifier 유지 |
| Optimizer policy | 기존 Adam 정책을 기준으로 비교 |
| Learning rate policy | Block5 0.00001 / LSTM-head-Attention 0.0001; Attention은 temporal group에 명시 포함 |
| Weight decay | 기존0.00001을 기준으로 비교 |
| Batch size / epochs / seed / AMP | 16 / fixed30 / 42 / false |
| DataLoader | 기존 workers4 / pin=true / persistent=true / prefetch1을 기준으로 비교 |
| Evaluation / official selection | Accuracy·Precision·Recall·F1·confusion matrix; fixed epoch30 final |
| Validation / scheduler / early stopping | 기존 none / none / none 정책 유지 예정 |

[확실] S3-1 YAML과 training code의 위 조건을 확인했다. R2 YAML의 기본 loader는 workers8/prefetch2지만 실제 완료된 R2 official은 CLI override의 workers4/prefetch1이었다([실행 환경 예](../../outputs/paper_reconstruction/refinement/R2_cuda_fold4_resumed_20261015/fold_4/experiment_environment.json)). S3-2는 S3-1 YAML과 기존 official 실행의 workers4/pin=true/persistent=true/prefetch1을 유지하도록 설계한다. YAML 기본값이 모두 같다고 주장하지 않는다.

Attention은 기존 temporal group에 명시 포함하며 group 수·기존 LR·weight decay는 바꾸지 않는다. S3-2 config는 아직 생성하지 않았다. 향후 통제 조건의 예외가 필요하면 변경 이유·영향을 실행 전에 계획과 progress에 기록한다.

## 7. Evaluation Policy

- **Official:** 각 fold의 fixed **epoch30 final result**를 사용할 계획이다.
- **Metrics:** Accuracy, Precision, Recall, F1, confusion matrix를 기록할 계획이다. Positive class는 drowsy다.
- **4-fold:** 각 fold 동일 가중치 mean과 **population std(ddof=0)**를 raw float에서 계산할 계획이다. Pooled confusion matrix metric이나 rounded 값에서 만든 평균과 구분한다.
- **Post-hoc best epoch:** **POST-HOC DIAGNOSTIC ONLY**다. Held-out fold의 최고 성능 epoch를 official model selection에 사용하지 않는다.
- **미완료 결과:** 실행 전에는 NOT RUN / NOT AVAILABLE, fold 실행 중간 수치는 INTERIM, 일부 fold 평균은 INTERIM / NOT OFFICIAL AGGREGATE로 표시한다. 네 fold의 epoch30과 metric 검증을 마친 뒤에만 OFFICIAL 4-FOLD RESULT로 기록한다.

중단 원칙도 기존 평가 정책과 구분한다. Test score를 보고 조기 종료하거나 official epoch를 바꾸지 않는다. 향후 설계 미확정, 통제 조건 불일치, 구조·gradient·parameter 검증 실패가 있으면 학습 단계로 넘어가지 않고 원인과 다음 조치를 기록할 계획이다. 실행 중 오류나 사용자 요청으로 중단되면 마지막 완료 epoch·원인·checkpoint 존재/복원 가능성을 확인해 기록하고, 검증된 checkpoint가 없으면 resume 가능으로 표시하지 않는다. 이는 향후 절차이며 이번에 중단/복원 기능을 구현하거나 실행하지 않는다.

Single seed, 동일 held-out test를 보며 후속 실험을 설계하는 exploratory model development, overlapping training folds, subject-wise 분할 정보 미확인과 CUDA/resume의 한계를 유지한다. Mean 차이만으로 통계적 유의성이나 실제 환경 일반화를 주장하지 않는다.

## 8. Success Criteria

최소 비교는 **S3-2 vs R2**와 **S3-2 vs S3-1**이다. Primary metric은 **official4-fold mean F1**이다. Fold별 F1 delta와 방향, population std, Precision/Recall, Accuracy와 confusion matrix를 함께 보고할 계획이다.

임의의 최소 개선폭이나 “몇% 이상이면 성공”이라는 threshold를 만들지 않는다. S3-1보다 높더라도 R2보다 낮다면 두 비교를 구분해 기록한다. Attention에 따른 parameter·비용 변화가 확인되면 성능과 함께 설명한다. 결과와 tradeoff를 그대로 제시하며 현재 성공/실패 판정이나 수치를 만들지 않는다.

## 9. Planned Execution Stages

S3-2.0 문서 setup과 이번 S3-2.1 설계 명세 작성을 수행했다. 구현·S3-2 test·학습은 아직 실행하지 않았다. 아래 후속 단계는 자동 실행하지 않는다.

| 단계 | 예정 작업 | 현재 상태 |
|---|---|---|
| S3-2.0 | Document setup — 계획/진행 기록 체계 마련 | COMPLETED |
| S3-2.1 | Architecture design — 후보 비교와 선택 명세 | COMPLETED / DESIGN ONLY |
| S3-2.2 | Implementation — 확정 설계의 model/config/trainer 연계 | NOT STARTED |
| S3-2.3 | Static / unit validation — 구조·gradient·parameter count 및 실행 전 검증 | NOT STARTED |
| S3-2.4 | Fold1 | NOT RUN |
| S3-2.5 | Fold2 | NOT RUN |
| S3-2.6 | Fold3 | NOT RUN |
| S3-2.7 | Fold4 | NOT RUN |
| S3-2.8 | 4-fold aggregation — raw final metric 집계·검증 | NOT STARTED |
| S3-2.9 | Final comparison and decision — R2/S3-1 비교 | NOT STARTED |

각 의미 있는 단계마다 **작업 수행 → 검증 → [상세 진행 기록](s3_2_attention_progress.md)과 [중앙 progress](../PROGRESS.md) 갱신 → 계획 상태/변경 반영 → git diff 확인 → 완료 보고** 순서를 따른다. 실패·중단·resume도 발생 순서대로 남긴다. 진행 기록을 나중에 한꺼번에 작성하지 않는다.

## 10. Current Status

| 항목 | 현재 상태 |
|---|---|
| Stage3 | IN PROGRESS |
| S3-1 / result | COMPLETED / NO IMPROVEMENT OVER R2 REFERENCE |
| S3-2 | **ARCHITECTURE DESIGN COMPLETED / DESIGN ONLY** |
| S3-2 plan / progress log | CREATED / CREATED |
| Architecture design | **COMPLETED** |
| Implementation / config / trainer integration / test code | NOT STARTED |
| Training | **NOT RUN** |
| Fold1 / Fold2 / Fold3 / Fold4 | NOT RUN / NOT RUN / NOT RUN / NOT RUN |
| Aggregate / actual S3-2 parameter count | NOT AVAILABLE / NOT VERIFIED — expected count는5.6절 |
| S3-2 MLflow run / checkpoint | NOT CREATED / NOT CREATED |
| Next | **S3-2 Implementation — NOT STARTED** |

