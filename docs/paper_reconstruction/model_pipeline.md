# Paper Reconstruction Model Pipeline

## 상태

- Model pipeline: `IMPLEMENTED`
- Feature extraction: `USER EXECUTION REQUIRED`
- VGG19 training: `NOT RUN`
- VGG16 training: `NOT RUN`
- 4-fold evaluation: `NOT RUN`

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
→ CrossEntropyLoss / holdout evaluation
```

Frame decode 실패는 임의 frame 대체나 자동 제외 없이 오류로 처리한다. Feature cache는 shape가 정확히 20×4096이고 NaN/Inf가 없어야 한다.

## Paper-confirmed

- 20 frames/video
- 224×224×3 input
- VGG19 primary, VGG16 secondary
- LSTM
- 4-fold, 75% train / 25% holdout
- Fig.1 parameter count로 결정되는 LSTM input 4,096
- LSTM hidden 512, one layer
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
- 첫 번째 FC 뒤 ReLU(`fc1`)의 4096D representation 사용
- OpenCV BGR→RGB, bilinear 224×224 resize, ImageNet normalization
- augmentation 없음
- Fig.1 count를 맞추기 위한 Keras-compatible single-bias LSTM
- batch-first `[B, 20, 4096]`, 마지막 hidden state 사용
- Dense512→ReLU→Dense64→Sigmoid→Dense2 logits
- Softmax는 probability가 필요한 inference에서만 사용
- Dropout, BatchNorm, scheduler, early stopping 없음
- Adam, epochs 30, batch size 16, learning rate 0.0001, weight decay 0.00001
- seed 42, holdout fold evaluation, drowsy positive class
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

## 사용자 실행

VGG19:

```powershell
.\.venv\Scripts\python.exe scripts\extract_paper_vgg_features.py --config configs\paper_reconstruction\vgg19_lstm.yaml
.\.venv\Scripts\python.exe scripts\train_paper_lstm_fold.py --config configs\paper_reconstruction\vgg19_lstm.yaml --fold 1
.\.venv\Scripts\python.exe scripts\train_paper_lstm_cv.py --config configs\paper_reconstruction\vgg19_lstm.yaml
```

VGG16은 config 파일을 `vgg16_lstm.yaml`로 변경한다. Pretrained weight가 로컬 torchvision cache에 없으면 최초 사용자 실행에서 다운로드가 필요할 수 있다.

## 생성 결과

Feature extraction은 `.npy` cache와 manifest를 만든다. 단일 fold training은 `history.json`, `metrics.json`, `final_model.pt`를 만들며 CV 실행은 추가로 `cv_summary.json`을 만든다. 현재 이 작업들은 실행되지 않았고 성능 결과도 없다.

## 다음 단계

VGG19 cache 생성 후 manifest validation을 확인하고 fold 1 smoke training을 수행한다. 메모리와 실행 시간을 확인한 뒤 VGG19 4-fold, VGG16 cache 및 4-fold 순서로 진행한다.
