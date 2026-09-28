# SUST-DDD Paper Reconstruction Baseline Specification

이 문서는 baseline 조건을 `PAPER-REPORTED`, `STRUCTURALLY INFERRED`, `RECONSTRUCTION ASSUMPTION`으로 구분한다. 원논문 근거가 없는 구현 선택을 논문 조건처럼 기록하지 않는다.

## A. PAPER-REPORTED

### Dataset and evaluation

| Item | Reported value |
|---|---|
| Dataset | SUST-DDD |
| Total videos | 2,074 |
| Drowsy / not drowsy | 975 / 1,099 |
| Video unit | 10 sec |
| Input | 224×224×3 |
| Frames per video | 20 |
| Split | 75% training / 25% test |
| Cross-validation | k-fold, k=4 |

### Architecture statements

- CNN backbones: VGG16, VGG19, AlexNet, VGGFaceNet
- CNN에서 ReLU activation 사용
- VGG16/VGG19 pooling size는 2×2
- One LSTM layer
- Two fully connected layers와 final Softmax
- ReLU와 Sigmoid activation 사용

VGG19를 primary, VGG16을 comparison baseline으로 선택한 것은 현재 reconstruction의 범위이며 별도의 논문 조건이 아니다.

논문은 ReLU/Sigmoid의 정확한 layer 대응을 충분히 설명하지 않는다.

### Table III reference

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| VGG19 + LSTM | 90.53% | 91.74% | 91.28% | 91.50% |
| VGG16 + LSTM | 89.39% | 91.81% | 89.09% | 90.42% |

VGG19 F1은 본문 일부에서 91.46%, Table III에서 91.50%로 불일치한다. 이 프로젝트는 Table III의 91.50%를 reference 주값으로 사용한다.

## B. STRUCTURALLY INFERRED

Fig.1의 LSTM parameter count와 layer dimension으로부터 다음 구조를 역산했다.

```text
Keras-style single-bias LSTM parameters
= 4 × hidden_size × (input_size + hidden_size + 1)

9,439,232
= 4 × 512 × (input_size + 512 + 1)

input_size = 4,096
```

> The 4096-dimensional LSTM input was structurally inferred from the parameter count shown in Fig.1.

| Component | Dimension | Parameters |
|---|---|---:|
| LSTM | input 4,096 / hidden 512 / one layer | 9,439,232 |
| Dense 1 | 512 → 512 | 262,656 |
| Dense 2 | 512 → 64 | 32,832 |
| Output | 64 → 2 | 130 |
| Total | LSTM + dense classifier | 9,734,850 |

4096D는 실험 변수로 선택한 값이나 기존 512D GAP feature의 재사용이 아니다.

## C. RECONSTRUCTION ASSUMPTIONS

### Data and sampling

- Filename label: `d_*=drowsy(1)`, `n_*=not_drowsy(0)`
- Label-stratified 4-fold, seed 42
- Subject-wise split은 식별 정보가 없어 사용하지 않음
- 전체 유효 frame 범위에서 deterministic uniform 20-frame sampling
- Endpoint를 포함한 `np.linspace` 후 `np.rint` ties-to-even
- Short video는 중복 index를 허용해 20 slots를 유지하고 anomaly로 기록
- Invalid video는 가짜 frame index를 만들지 않고 anomaly로 기록
- 실제 dataset에서는 short/invalid/duplicate-index video가 모두 0

### Frame preprocessing

- OpenCV frame decode와 BGR→RGB 변환
- Bilinear resize to 224×224
- 0..1 tensor 변환 후 ImageNet normalization
- Data augmentation 없음
- Decode 실패 시 임의 frame 대체나 자동 제외 없이 오류

원논문은 sampling, resize interpolation, color conversion, normalization과 augmentation 세부 조건을 공개하지 않는다.

### VGG feature extraction

- torchvision ImageNet pretrained VGG16/VGG19
- Frozen backbone, offline inference
- Backbone별 별도 video-level `[20, 4096]` `.npy` cache
- 첫 번째 Linear와 그 뒤 ReLU까지 실행한 `fc1` 4096D representation 사용

> The paper does not specify the exact VGG feature extraction location. This reconstruction uses the 4096D first fully connected representation after ReLU.

### LSTM parameterization and initialization

일반 `torch.nn.LSTM`은 각 layer에 `bias_ih`와 `bias_hh` 두 bias vector를 두므로 input 4096, hidden 512에서 Fig.1보다 2,048개 많은 9,441,280 parameters가 된다. 현재 구현은 Fig.1의 9,439,232에 맞추기 위해 다음 parameter를 직접 정의한다.

- Input weight: `[4×512, 4096]`
- Recurrent weight: `[4×512, 512]`
- Single bias: `[4×512]`
- Batch-first input: `[B, 20, 4096]`

Initialization은 코드의 실제 동작에 따라 input weight, recurrent weight, bias 모두 `[-1/√512, +1/√512]` uniform distribution을 사용한다. Initial hidden/cell state는 0이다. 이는 Keras default kernel/recurrent/forget-gate initialization과 같다고 주장하지 않으며 reconstruction assumption이다.

### Sequence representation and classifier

```text
[B, 20, 4096]
→ final LSTM hidden state [B, 512]
→ Dense512 → ReLU
→ Dense64 → Sigmoid
→ Dense2 → logits
```

- 논문이 언급한 ReLU/Sigmoid의 위치를 각각 Dense512와 Dense64 뒤로 해석
- Attention, Dropout, BatchNorm 없음
- 추가 hidden layer 없음

### Softmax and loss

논문은 final Softmax를 명시한다. 현재 PyTorch training은 Dense2 raw logits를 `CrossEntropyLoss`에 전달한다. `CrossEntropyLoss`가 log-softmax와 negative log likelihood를 내부에서 처리하므로 training graph에 explicit Softmax를 넣지 않는다. Inference에서 probability가 필요할 때만 Softmax를 적용한다.

이는 Softmax binary classification과 기능적으로 대응하는 구현이지만, 원논문의 training graph와 완전히 동일하다고 주장하지 않는다.

### Training and evaluation

| Parameter | Implemented value | Classification |
|---|---|---|
| Optimizer | Adam | Reconstruction assumption |
| Learning rate | 0.0001 | Reconstruction assumption |
| Weight decay | 0.00001 | Reconstruction assumption |
| Batch size | 16 | Reconstruction assumption |
| Epochs | 30 fixed | Reconstruction assumption |
| Seed | 42 | Reconstruction assumption |
| Scheduler | None | Reconstruction assumption |
| Early stopping | None | Reconstruction assumption |
| Validation split | None | Reconstruction assumption |
| Best checkpoint selection | None | Reconstruction assumption |
| Loss | CrossEntropyLoss | Reconstruction assumption |
| Positive class | drowsy | Evaluation policy |
| Fold count | 4 | Paper-reported |
| Training/Test ratio | 75/25 | Paper-reported |
| Fold construction | Label-stratified | Reconstruction assumption |
| Aggregation | Mean + population std (`ddof=0`) | Reconstruction assumption |

각 fold의 25%를 training에서 제외된 held-out test fold로 해석했다. 별도 validation 없이 epoch 30 final metric을 official result로 사용했다. Held-out test history의 maximum F1 epoch는 post-hoc diagnostic일 뿐 official metric이나 checkpoint 선택에 사용하지 않는다.

## Known Unknowns

| Item | Paper | This reconstruction |
|---|---|---|
| Pretrained VGG | Not reported | ImageNet pretrained |
| Backbone freeze | Not reported | Frozen |
| CNN training mode | Not reported | Offline feature extraction; LSTM-only training |
| Feature extraction point | Not reported | First FC representation after ReLU (`fc1`) |
| Frame sampling | Not reported | Deterministic uniform 20 frames |
| Resize interpolation | Not reported | Bilinear |
| Color handling | Not reported | OpenCV BGR→RGB |
| Normalization | Not reported | ImageNet mean/std |
| Augmentation | Not reported | None |
| Activation placement | ReLU/Sigmoid named; placement unclear | Dense512→ReLU, Dense64→Sigmoid |
| Sequence representation | Not detailed | Final LSTM hidden state |
| LSTM parameterization | Fig.1 count reported | Custom single-bias implementation |
| LSTM initialization | Not reported | All parameters uniform `±1/√512`; zero initial states |
| Optimizer | Not reported | Adam |
| Learning rate | Not reported | 0.0001 |
| Weight decay | Not reported | 0.00001 |
| Batch size | Not reported | 16 |
| Epochs | Not reported | 30 fixed |
| Loss implementation | Softmax reported; details absent | Logits + CrossEntropyLoss |
| Validation | Not reported | Not used |
| Early stopping | Not reported | Not used |
| Scheduler | Not reported | None |
| Random seed | Not reported | 42 |
| Best checkpoint | Not reported | Not used; final epoch official |
| Test evaluation frequency | Not reported | Held-out test metrics recorded every epoch |
| Fold generation | Not detailed | Label-stratified 4-fold |
| Subject-wise split | Not reported | Unavailable |
| Fold aggregation | Not reported | Mean + population std (`ddof=0`) |

## 결과 및 상태

- Data preparation: `COMPLETED`
- VGG19/VGG16 feature extraction: `COMPLETED`
- VGG19/VGG16 4-fold training: `COMPLETED`
- Official final-epoch results: `RECORDED` ([results.md](results.md))
- Post-hoc best-epoch analysis: `DIAGNOSTIC ONLY`
- Paper Reconstruction: `COMPLETED`
- Validation: `NOT USED`
- Early stopping: `NOT USED`
- Best checkpoint selection: `NOT USED`
- Official result: `FIXED EPOCH / FINAL EPOCH`
- Context Model Improvement: `NOT STARTED`

Context Model Improvement에서도 Paper Reconstruction과 동일한 4-fold 75% training / 25% held-out test 프로토콜을 유지한다. 별도 validation split, early stopping, best checkpoint selection은 도입하지 않으며, 사전에 정의한 fixed epoch의 final result를 공식 결과로 사용한다. Held-out test 결과를 기준으로 best epoch를 선택하지 않는다.

기존 저장소의 32-frame sequence, 512D GAP feature, LSTM128, 기존 split과 기존 성능은 이 baseline에 사용하지 않았다.
