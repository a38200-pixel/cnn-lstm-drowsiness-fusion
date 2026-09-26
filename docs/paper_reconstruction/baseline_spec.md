# SUST-DDD Paper Reconstruction Baseline Specification

이 문서는 paper reconstruction baseline의 확정 조건과 미확정 조건을 분리한다. `UNKNOWN` 항목은 근거를 확보하고 decision 문서를 남기기 전까지 임의의 기본값으로 확정하지 않는다.

## 확정 조건

| 항목 | 값 | 비고 |
|---|---:|---|
| Dataset | SUST-DDD | |
| Total videos | 2,074 | |
| Drowsy | 975 | |
| Not drowsy | 1,099 | |
| Video length | 10 sec | |
| Input size | 224 × 224 × 3 | |
| Frames per video | 20 | sampling 방식은 UNKNOWN |
| Primary backbone | VGG19 | |
| Secondary backbone | VGG16 | |
| LSTM input dimension | 4,096 | Fig. 1 parameter count로 역산된 구조적 확정값 |
| LSTM hidden size | 512 | |
| LSTM layers | 1 | |
| Cross validation | 4-fold | |
| Fold 구성 | 75% train / 25% holdout | |

## Classifier

```text
LSTM512
→ Dense512
→ activation
→ Dense64
→ activation
→ Dense2
→ Softmax
```

두 hidden activation의 종류는 아직 `UNKNOWN`이다.

## UNKNOWN

- 20-frame sampling 방식
- pretrained 여부
- CNN freeze/fine-tuning 여부
- optimizer
- learning rate
- batch size
- epochs
- weight decay
- augmentation
- normalization
- loss 세부 구현
- random seed
- subject-wise split 여부
- fold 결과 aggregation 방식

## 재구성 원칙

- UNKNOWN 항목은 실험 편의를 이유로 논문 조건처럼 기록하지 않는다.
- 실행을 위해 가정이 필요하면 별도 config와 `docs/decisions/`의 결정 기록에 명시한다.
- 기존 저장소의 32-frame/512D feature 실험, split, 모델 성능은 이 baseline의 결과로 간주하지 않는다.
- raw dataset은 외부 data root로 참조하며 저장소에 복제하지 않는다.
