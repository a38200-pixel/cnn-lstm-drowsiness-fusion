# SUST-DDD Paper Reconstruction Baseline Specification

이 문서는 원논문에서 확정된 조건, 아직 알 수 없는 조건, 실행을 위해 도입한 reconstruction assumption을 분리한다. 원논문 근거가 없는 값을 논문 조건처럼 확정하지 않는다.

## Paper-confirmed

| 항목 | 값 | 비고 |
|---|---:|---|
| Dataset | SUST-DDD | |
| Total videos | 2,074 | |
| Drowsy | 975 | `label_id=1`은 프로젝트 표현 정책 |
| Not drowsy | 1,099 | `label_id=0`은 프로젝트 표현 정책 |
| Video length | 10 sec | |
| Input size | 224×224×3 | |
| Frames per video | 20 | 정확한 sampling은 UNKNOWN |
| Primary backbone | VGG19 | |
| Secondary backbone | VGG16 | |
| LSTM input dimension | 4,096 | Fig. 1 parameter count로 역산된 구조적 확정값 |
| LSTM hidden size | 512 | |
| LSTM layers | 1 | |
| Cross validation | 4-fold | |
| Fold 구성 | 75% train / 25% holdout | |

Classifier 구조:

```text
LSTM512
→ Dense512
→ activation
→ Dense64
→ activation
→ Dense2
→ Softmax
```

논문 문구에 ReLU와 Sigmoid가 언급되지만 정확한 layer 대응은 완전히 명확하지 않다. 초기 구현은 Dense512 뒤 ReLU, Dense64 뒤 Sigmoid로 해석한다.

Fig.1의 LSTM parameter count는 Keras 공식
`4 × ((input_size + hidden_size) × hidden_size + hidden_size)`를 따른다.
hidden size 512와 parameter 9,439,232를 대입하면 input size는 4,096으로 결정된다.

## Reconstruction assumptions for data preparation

- label별 stratified 4-fold
- seed `42`
- subject-wise split `false / unavailable`
- deterministic uniform 20-frame sampling
- 첫 유효 frame과 마지막 유효 frame을 포함하는 `np.linspace`
- `np.rint` 정수 변환과 ties-to-even 정책
- 짧은 영상은 중복 index를 유지하고 anomaly로 보고
- invalid metadata는 가짜 sequence를 만들지 않고 anomaly로 보고

## Reconstruction assumptions for model pipeline

- torchvision VGG19/VGG16과 ImageNet pretrained weight
- CNN backbone freeze 및 offline feature cache
- 첫 번째 FC와 ReLU 출력(`fc1`)을 4096D feature로 사용
- bilinear 224×224 resize, RGB 변환, ImageNet normalization
- augmentation 없음
- Keras parameter count와 맞는 single-bias, batch-first LSTM 구현
- 마지막 timestep hidden state 사용
- Dense512→ReLU→Dense64→Sigmoid→Dense2 logits
- Dropout과 BatchNorm 없음
- CrossEntropyLoss, Adam, epochs 30, batch size 16
- learning rate 0.0001, weight decay 0.00001, seed 42
- holdout fold를 evaluation 대상으로 사용하고 별도 test/validation split 없음
- 4-fold 표준편차는 population std(`ddof=0`)

정확한 fold 알고리즘과 sampling 방식은 공개되지 않았으므로 결과 문서에는 `paper-informed stratified reconstruction`과 `reconstruction assumption`을 표시한다.

## UNKNOWN

- 원논문의 정확한 20-frame sampling 방식
- pretrained 여부
- CNN freeze/fine-tuning 여부
- 정확한 VGG FC extraction 위치
- optimizer
- learning rate
- batch size
- epochs
- weight decay
- augmentation
- normalization
- loss 세부 구현
- 원논문의 random seed
- subject-wise split 여부
- fold 결과 aggregation 방식
- activation의 정확한 layer 대응
- LSTM initialization 세부 조건
- scheduler 및 early stopping

## 현재 상태

- 데이터 준비 코드: `IMPLEMENTED`
- 실제 metadata/split/sequence: `VERIFIED` (2,074 videos / 41,480 rows)
- 모델 pipeline: `IMPLEMENTED`
- Feature extraction: `USER EXECUTION REQUIRED`
- CNN feature extraction: `NOT RUN`
- Training: `NOT RUN`

기존 저장소의 32-frame sequence, 512D feature, 기존 split 및 성능은 이 baseline 결과로 사용하지 않는다.
