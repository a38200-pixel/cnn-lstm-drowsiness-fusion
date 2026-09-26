# CNN-LSTM Drowsiness Fusion

SUST-DDD 원논문의 VGG19+LSTM 및 VGG16+LSTM baseline을 재구성하고, 이후 행동 규칙과 fusion detection으로 확장하기 위한 프로젝트입니다.

## 현재 상태

- `STEP PAPER-BASELINE PIPELINE IMPLEMENTED`
- `DATA PREPARATION: VERIFIED`
- `FEATURE EXTRACTION: USER EXECUTION REQUIRED`
- `VGG19 TRAINING: NOT RUN`
- `VGG16 TRAINING: NOT RUN`
- `4-FOLD EVALUATION: NOT RUN`
- `TRAINING: NOT RUN`
- `CNN FEATURE EXTRACTION: NOT RUN`

Metadata, 4-fold split, 20-frame sequence metadata는 사용자 실행으로 생성 및 검증되었습니다. 현재 확인된 결과는 2,074 videos, 41,480 sequence rows, 20 rows/video이며 short/invalid/duplicate frame index는 모두 0입니다. VGG feature extraction과 LSTM training은 실행하지 않았습니다.

## Paper-confirmed

- Dataset: SUST-DDD
- Total videos: 2,074
- Drowsy: 975
- Not drowsy: 1,099
- Video length: 10 sec
- Input: 20 frames per video, 224×224×3
- Backbone: VGG19 primary, VGG16 secondary
- LSTM input: 4,096D (Fig.1 parameter count로 구조적으로 결정)
- LSTM hidden: 512, one layer
- Classifier dimensions: 512 → 512 → 64 → 2
- 4-fold cross validation
- 각 fold: 75% train / 25% holdout

전체 모델 명세와 아직 확인되지 않은 조건은 [baseline_spec.md](docs/paper_reconstruction/baseline_spec.md)에 정리되어 있습니다.

## Reconstruction assumptions

원논문에 정확한 생성 방법이 없는 항목은 논문 조건이 아니라 다음 재구성 가정으로 관리합니다.

- label별 stratified 4-fold 생성
- random seed `42`
- subject-wise split `false`: subject 정보 확인 불가
- 영상 전체 구간의 deterministic uniform 20-frame sampling
- sampling 실수 위치를 `np.rint`로 가장 가까운 정수로 변환하며 정확한 `.5`는 ties-to-even
- 짧은 영상의 중복 index를 제거하지 않고 20 slot을 유지하며 summary에 anomaly로 기록
- 무효한 `frame_count` 또는 `fps`는 가짜 index를 만들지 않고 summary에 기록
- torchvision VGG의 ImageNet pretrained weight 사용 및 backbone freeze
- 첫 번째 FC의 ReLU 출력(`fc1`)을 4096D frame feature로 사용
- bilinear 224×224 resize, RGB, ImageNet normalization, augmentation 없음
- classifier activation을 Dense512→ReLU, Dense64→Sigmoid로 해석
- CrossEntropyLoss를 사용하고 모델은 Softmax가 아닌 logits 반환
- Adam, epochs 30, batch size 16, learning rate 0.0001, weight decay 0.00001

상세 데이터 정책은 [data_preparation.md](docs/paper_reconstruction/data_preparation.md), 모델과 학습 정책은 [model_pipeline.md](docs/paper_reconstruction/model_pipeline.md)를 참고합니다.

## Prepared outputs

다음 데이터 준비 파일이 생성되어 있습니다.

```text
data/metadata/sust_ddd_metadata.csv
data/splits/paper_reconstruction/
├─ fold_1_train.csv
├─ fold_1_holdout.csv
├─ fold_2_train.csv
├─ fold_2_holdout.csv
├─ fold_3_train.csv
├─ fold_3_holdout.csv
├─ fold_4_train.csv
├─ fold_4_holdout.csv
└─ split_summary.json
data/sequences/paper_reconstruction_20frame.csv
data/sequences/paper_reconstruction_20frame_summary.json
```

- Metadata CSV: 원본 경로, label, label ID, container metadata
- Fold CSV: fold별 train 또는 holdout video 목록
- Split summary: label 분포, seed, subject-wise 상태, holdout coverage 검증
- Sequence CSV: 영상별 20개 target/source frame index와 timestamp
- Sequence summary: row 수, 짧은/무효 영상, 중복 index, sampling 정책

생성 파일은 재생성 가능한 로컬 산출물이므로 Git에서 제외됩니다.

Feature extraction 실행 후에는 다음 cache가 생성됩니다.

```text
data/features/paper_reconstruction/vgg19/<video_id>.npy
data/features/paper_reconstruction/vgg19_manifest.csv
data/features/paper_reconstruction/vgg16/<video_id>.npy
data/features/paper_reconstruction/vgg16_manifest.csv
```

## 설정과 사용자 실행

[data.yaml](configs/paper_reconstruction/data.yaml)에 raw dataset 경로, 출력 경로, seed와 sampling 설정이 있습니다. 기본 raw root는 다음과 같습니다.

```text
C:/Users/AISW_203_113/Documents/Datasets/SUST Driver Drowsiness Dataset
```

원본 영상은 항상 read-only로 사용하며 복사, 변환, 재인코딩하지 않습니다. 설치 후 전체 데이터 작업은 사용자가 다음 순서로 실행합니다.

```powershell
python -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\build_sust_ddd_metadata.py
.\.venv\Scripts\python.exe scripts\build_paper_cv_splits.py
.\.venv\Scripts\python.exe scripts\build_paper_20frame_sequences.py
```

두 번째와 세 번째 스크립트는 앞 단계 metadata CSV가 없으면 오류를 출력합니다. Metadata 영상 수 또는 label 수가 논문 기대값과 다르면 자동 수정하거나 제외하지 않고 오류를 출력합니다.

VGG19 feature extraction과 단일/전체 fold 학습 예시:

```powershell
.\.venv\Scripts\python.exe scripts\extract_paper_vgg_features.py --config configs\paper_reconstruction\vgg19_lstm.yaml
.\.venv\Scripts\python.exe scripts\train_paper_lstm_fold.py --config configs\paper_reconstruction\vgg19_lstm.yaml --fold 1
.\.venv\Scripts\python.exe scripts\train_paper_lstm_cv.py --config configs\paper_reconstruction\vgg19_lstm.yaml
```

VGG16은 config 경로를 `vgg16_lstm.yaml`로 바꿔 같은 명령을 사용합니다.

## 이번 단계에서 하지 않은 작업

- 전체 2,074개 영상의 frame decode
- 전체 41,480 frames VGG inference 및 feature cache 생성
- VGG16/VGG19 LSTM training과 checkpoint 생성
- 실제 4-fold evaluation 및 성능 집계
- augmentation, face crop, behavior/fusion 구현

## 프로젝트 단계

1. SUST-DDD Paper Reconstruction: VGG19+LSTM, VGG16+LSTM, 4-fold CV
2. Context Model Improvement
3. Behavior Rule Detection: EAR, MAR, Head Pose
4. Fusion Detection: context prediction, behavior events, final decision

## 다음 단계

VGG19 feature cache를 먼저 생성하고 manifest의 20×4096 shape와 finite-value 검증을 확인합니다. 이후 fold 1 smoke training을 사용자 환경에서 실행한 뒤 전체 4-fold로 확장하고, 같은 절차를 VGG16에 적용합니다.
