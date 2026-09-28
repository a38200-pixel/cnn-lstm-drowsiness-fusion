# Paper Reconstruction Model Pipeline

## 상태

- Model pipeline: `IMPLEMENTED`
- VGG19 feature extraction: `COMPLETED`
- VGG19 4-fold training: `COMPLETED`
- VGG19 official result: `RECORDED`
- VGG19 early stopping / best-epoch selection: `NOT USED`
- VGG16 feature extraction: `COMPLETED`
- VGG16 4-fold training: `COMPLETED`
- VGG16 official result: `RECORDED`
- Initial paper-informed reconstruction baseline: `COMPLETED`
- Reconstruction refinement: `PLANNED / NEXT`
- Post-hoc best-epoch analysis: `DIAGNOSTIC ONLY`

## Pipeline

```text
20-frame sequence metadata
→ source_frame_index 기반 RGB decode
→ deterministic 224×224 preprocessing
→ VGG19 또는 VGG16 fc1 4096D feature
→ video별 [20, 4096] .npy cache
→ single-bias LSTM512
→ Dense512 + ReLU
→ Dense64 + Sigmoid
→ Dense2 logits
→ CrossEntropyLoss / held-out test fold evaluation
```

Frame decode 실패는 임의 frame 대체나 자동 제외 없이 오류로 처리한다. Feature cache는 shape가 정확히 20×4096이고 NaN/Inf가 없어야 한다.

## Paper-reported

- 20 frames/video
- 224×224×3 input
- VGG19 primary, VGG16 secondary
- LSTM
- 4-fold, 75% training / 25% test
- One LSTM layer, two fully connected layers, final Softmax
- ReLU와 Sigmoid activation 사용

## Structurally inferred

- Fig.1 parameter count로 결정되는 LSTM input 4,096
- LSTM hidden 512
- Dense 512, Dense 64, Dense 2

LSTM parameter 공식은 `4 × ((4096 + 512) × 512 + 512) = 9,439,232`다.

## Paper-unknown

- ImageNet pretrained 여부
- CNN freeze/partial/full fine-tuning 여부
- 정확한 VGG FC extraction 위치
- frame sampling 방식과 resize interpolation
- normalization
- optimizer, learning rate, weight decay, batch size, epochs
- scheduler, early stopping, random seed
- augmentation
- 정확한 fold 생성 및 metric aggregation 방식
- activation의 정확한 layer 대응과 LSTM initialization

## Reconstruction assumptions

- torchvision ImageNet pretrained VGG 사용
- backbone freeze와 offline feature cache
- 첫 번째 Linear 뒤 ReLU(`fc1`)의 4096D representation 사용
- OpenCV BGR→RGB, bilinear 224×224 resize, ImageNet normalization
- augmentation 없음
- Fig.1 count를 맞추기 위한 Keras-compatible single-bias LSTM
- Input/recurrent weight와 single bias를 모두 uniform `±1/√512`로 초기화
- Initial hidden/cell state는 0; Keras default initialization과 동일하다고 가정하지 않음
- batch-first `[B, 20, 4096]`, 마지막 hidden state 사용
- Dense512→ReLU→Dense64→Sigmoid→Dense2 logits
- Dense2 raw logits를 CrossEntropyLoss에 전달하고 Softmax는 probability inference에서만 사용
- Dropout, BatchNorm, scheduler, early stopping 없음
- Adam, epochs 30, batch size 16, learning rate 0.0001, weight decay 0.00001
- seed 42, held-out test fold evaluation, drowsy positive class
- CV mean과 population std(`ddof=0`)

## Feature cache

각 backbone은 별도 cache와 manifest를 사용한다.

```text
data/features/paper_reconstruction/vgg19/<video_id>.npy
data/features/paper_reconstruction/vgg19_manifest.csv
data/features/paper_reconstruction/vgg16/<video_id>.npy
data/features/paper_reconstruction/vgg16_manifest.csv
```

Manifest 컬럼은 `video_id`, `label`, `feature_path`, `sequence_length`, `feature_dim`, `backbone`이다.

## Evaluation policy

- Stratified 4-fold reconstruction
- Fold마다 fixed 30 epochs
- Early stopping과 scheduler 없음
- Best checkpoint selection 없음
- Epoch 30 final metric을 공식 fold metric으로 사용
- 4개 final metric의 mean과 population std(`ddof=0`) 사용
- Holdout history의 best epoch는 training behavior 진단에만 사용

현재 reconstruction에서는 held-out test fold를 매 epoch 기록했다. 이 history에서 best epoch를 사후 선택하면 test fold가 validation 역할까지 하게 되므로, 공식 결과는 사전 정의한 final-epoch 정책을 유지한다. 원논문의 validation 및 epoch/checkpoint selection 정책은 공개되지 않았다.

## 결과 보존

VGG19와 VGG16 feature cache, history, metrics, checkpoint 및 `cv_summary.json`은 완료된 공식 baseline 산출물로 보존한다. 결과 정리를 위해 재추출·재학습하거나 기존 파일을 덮어쓰지 않는다.

## 생성 결과

Feature extraction은 `.npy` cache와 manifest를 만들었다. 단일 fold training은 `history.json`, `metrics.json`, `final_model.pt`를 만들었으며 CV 실행은 `cv_summary.json`을 만들었다. 두 backbone의 공식 결과와 post-hoc 진단은 [results.md](results.md)에 분리해 기록했다.

## 다음 단계

Proposed Context Model Improvement 전에 [reconstruction_refinement.md](reconstruction_refinement.md)의 계획에 따라 feature extraction 위치와 기타 미공개 조건의 민감도를 검토한다. 핵심 4096D→LSTM512 구조와 동일한 4-fold/fixed-epoch 평가 정책은 유지한다.
