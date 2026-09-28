# Paper Reconstruction Results

## 1. Scope

이 문서는 SUST-DDD 원논문에서 공개한 구조적 정보를 바탕으로 구현한 VGG19+LSTM 및 VGG16+LSTM paper-informed reconstruction 결과를 기록한다. 두 backbone의 feature extraction과 4-fold training은 완료되었으며, 기존 history, metrics, checkpoint와 `cv_summary.json`은 수정하지 않았다.

4096D LSTM 입력은 Fig.1의 LSTM parameter count `9,439,232`와 hidden size 512로부터 구조적으로 역산한 조건이다. 반면 pretrained/freeze 정책과 정확한 FC feature 위치 등은 공개되지 않아 reconstruction assumption으로 분리한다.

## 2. Evaluation Protocol

- Dataset: SUST-DDD, 2,074 videos (drowsy 975 / not drowsy 1,099)
- Input: video당 deterministic uniform 20 frames, 224×224 RGB
- Paper statement: 75% training / 25% test, k-fold cross-validation, k=4
- Reconstruction interpretation: 각 fold의 75%는 train, 나머지 25%는 training에서 제외된 held-out test fold
- Separate validation set: 현재 reconstruction에서는 사용하지 않음
- Training: fixed 30 epochs, Adam, batch size 16, seed 42
- Scheduler / early stopping / dropout: 사용하지 않음
- Official fold metric: epoch 30 final metric
- Aggregation: 4-fold mean과 population std(`ddof=0`)
- Positive class: drowsy

기존 파일명에는 `holdout`이 사용되지만 결과 문서에서는 원논문의 25% test 표현에 맞춰 `held-out test fold`라고 부른다.

원논문은 75% training / 25% test 및 k=4 cross-validation을 명시하지만 validation, epoch selection, early stopping 및 checkpoint selection 정책은 공개하지 않았다. 따라서 fixed 30 epochs와 final-epoch 평가는 원논문의 확정 조건이 아니라 현재 reconstruction의 사전 정의 정책이다.

## 3. VGG19 + LSTM

### Official final-epoch fold results

| Fold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.7726396917148363 | 0.7647058823529411 | 0.7459016393442623 | 0.7551867219917013 |
| 2 | 0.8092485549132948 | 0.8436018957345972 | 0.7295081967213115 | 0.7824175824175824 |
| 3 | 0.7957610789980732 | 0.7948717948717948 | 0.7622950819672131 | 0.7782426778242677 |
| 4 | 0.8201160541586073 | 0.8318584070796460 | 0.7736625514403292 | 0.8017057569296375 |

### Official 4-fold aggregation

| Metric | Mean | Population std | Mean (%) |
|---|---:|---:|---:|
| Accuracy | 0.7994413449462029 | 0.017716503194334772 | 79.94% |
| Precision | 0.8087594950097448 | 0.031149404610367477 | 80.88% |
| Recall | 0.7528418673682791 | 0.016699484291199877 | 75.28% |
| F1 | 0.7793881847907973 | 0.016539908282544240 | 77.94% |

## 4. VGG16 + LSTM

### Official final-epoch fold results

| Fold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.7803468208092486 | 0.7663934426229508 | 0.7663934426229508 | 0.7663934426229508 |
| 2 | 0.8188824662813102 | 0.8289473684210527 | 0.7745901639344263 | 0.8008474576271186 |
| 3 | 0.8111753371868978 | 0.8379629629629629 | 0.7418032786885246 | 0.7869565217391304 |
| 4 | 0.7911025145067698 | 0.8426395939086294 | 0.6831275720164609 | 0.7545454545454546 |

### Official 4-fold aggregation

| Metric | Mean | Population std | Mean (%) |
|---|---:|---:|---:|
| Accuracy | 0.8003767846960566 | 0.015380836515636724 | 80.04% |
| Precision | 0.8189858419788989 | 0.030760459929389413 | 81.90% |
| Recall | 0.7414786143155907 | 0.035784329433760980 | 74.15% |
| F1 | 0.7771857191336636 | 0.017919272201985772 | 77.72% |

## 5. VGG19 vs VGG16 Official Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| VGG19 + LSTM | 79.94% | 80.88% | 75.28% | 77.94% |
| VGG16 + LSTM | 80.04% | 81.90% | 74.15% | 77.72% |

동일 reconstruction 조건에서 VGG16은 Accuracy와 Precision이 소폭 높고, VGG19는 Recall과 F1이 소폭 높았다. 차이는 매우 작으며 현재 결과만으로 특정 backbone의 명확한 성능 우위를 주장하지 않는다.

## 6. Post-hoc Best-Epoch Diagnostic

> **POST-HOC DIAGNOSTIC ONLY — 공식 reconstruction 결과가 아니다.**

### VGG19 diagnostic best-F1 epochs

| Fold | Epoch | Train loss | Held-out test loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.40373916538773624 | 0.4731928744068036 | 0.8092485549132948 | 0.8008298755186722 | 0.7909836065573771 | 0.7958762886597939 |
| 2 | 9 | 0.32952507398423660 | 0.43853935157173174 | 0.8227360308285164 | 0.8362831858407079 | 0.7745901639344263 | 0.8042553191489362 |
| 3 | 25 | 0.07517169464223830 | 0.5568940764677088 | 0.8131021194605009 | 0.8075313807531381 | 0.7909836065573771 | 0.7991718426501035 |
| 4 | 26 | 0.08553850663348544 | 0.5163501223235804 | 0.8336557059961315 | 0.8030888030888030 | 0.8559670781893004 | 0.8286852589641435 |

| Metric | Diagnostic mean | Population std |
|---|---:|---:|
| Accuracy | 0.8196856027996109 | 0.009443822265621057 |
| Precision | 0.8119333113003303 | 0.014263633779086696 |
| Recall | 0.8031311138096202 | 0.031230389112992835 |
| F1 | 0.8069971773557443 | 0.012872457991015612 |

### VGG16 diagnostic best-F1 epochs

| Fold | Epoch | Train loss | Held-out test loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28 | 0.041259899094604986 | 0.7695433464913355 | 0.7957610789980732 | 0.7555555555555555 | 0.8360655737704918 | 0.7937743190661478 |
| 2 | 22 | 0.086189398548035760 | 0.5958575900923999 | 0.8227360308285164 | 0.8040000000000000 | 0.8237704918032787 | 0.8137651821862347 |
| 3 | 5 | 0.385219046195603650 | 0.42724807287686595 | 0.8111753371868978 | 0.7874015748031497 | 0.8196721311475410 | 0.8032128514056226 |
| 4 | 7 | 0.363261498391207200 | 0.4126242961828206 | 0.8317214700193424 | 0.7932330827067670 | 0.8683127572016461 | 0.8290766208251474 |

| Metric | Diagnostic mean | Population std |
|---|---:|---:|
| Accuracy | 0.8153484792582075 | 0.013451134788245788 |
| Precision | 0.7850475532663681 | 0.018038275110033370 |
| Recall | 0.8369552384807394 | 0.019082903524282830 |
| F1 | 0.8099572433707881 | 0.013109394096352770 |

Best-epoch values were obtained retrospectively from the per-epoch held-out test metrics. Because the held-out fold was not an independent validation set, these values are reported only as diagnostic observations and are not used as the official reconstruction result.

위 값은 매 epoch 기록된 held-out test metric을 실험 종료 후 되돌아보며 선택한 값이다. 별도 validation set이 아니므로 이를 기준으로 epoch를 선택하면 test fold가 모델 선택에 사용된다. 따라서 best-epoch 값은 과적합 경향을 살펴보는 진단 정보일 뿐 공식 metric이나 checkpoint 선택에 사용하지 않는다.

## 7. Training Behavior

- 두 backbone 모두 train loss는 전반적으로 감소했다.
- 일부 fold에서는 후반부 train loss가 감소하는 동안 held-out test loss가 증가했다.
- Best-F1 epoch는 VGG19에서 4/9/25/26, VGG16에서 28/22/5/7로 fold마다 크게 달랐다.
- 하나의 고정된 최적 epoch가 모든 fold에서 공통으로 나타나지 않았다.
- 과적합이 나타나는 시점도 fold마다 달라 fixed epoch 하나만으로 일반화하기 어렵다.

이 분석은 학습 거동과 과적합 경향을 확인하기 위한 post-hoc diagnostic only이다. 개선 실험에서도 held-out test 결과를 이용해 epoch를 선택하지 않으며, 공식 결과는 사전에 정의한 fixed epoch의 final result로 평가한다.

## 8. Original Paper Reference

원논문 Table III:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| VGG19 + LSTM | 90.53% | 91.74% | 91.28% | 91.50% |
| VGG16 + LSTM | 89.39% | 91.81% | 89.09% | 90.42% |

원논문 본문의 VGG19 F1은 91.46%로 기재된 부분이 있으나 Table III는 91.50%다. 이 문서에서는 Table III의 91.50%를 주값으로 사용한다.

현재 reconstruction official mean:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| VGG19 + LSTM | 79.94% | 80.88% | 75.28% | 77.94% |
| VGG16 + LSTM | 80.04% | 81.90% | 74.15% | 77.72% |

공개된 구조적 정보를 기반으로 한 paper-informed reconstruction이며, 공개되지 않은 학습 및 평가 조건으로 인해 원논문의 수치와 직접 동일 조건 비교할 수 없다. 따라서 수치 차이만으로 재현 실패라고 단정하지 않는다.

## 9. Reconstruction Limitations

원논문에서 다음 조건을 확인할 수 없다.

- ImageNet pretrained 여부
- CNN freeze / partial fine-tuning / full fine-tuning 정책
- 정확한 VGG FC feature extraction 위치
- 정확한 frame sampling과 resize/normalization
- augmentation
- optimizer, learning rate, weight decay, batch size, epoch 수
- scheduler와 early stopping
- validation set 존재 여부와 비율
- epoch 및 checkpoint selection 정책
- test를 매 epoch 평가했는지 여부
- LSTM initialization
- 정확한 fold 생성 방식과 subject-wise split 여부
- fold metric aggregation 방식

## 10. Conclusion

VGG19와 VGG16 initial paper-informed reconstruction baseline의 feature extraction 및 fixed 30-epoch 4-fold training을 완료했다. 공식 final-epoch F1 mean은 VGG19 77.94%, VGG16 77.72%였으며 두 backbone의 평균 성능은 매우 유사했다. Post-hoc best-F1 분석은 과적합과 fold별 최적 시점 변동을 보여주지만 공식 결과로 사용하지 않는다. 원논문과의 성능 차이 때문에 Paper Reconstruction 전체가 완료되었다고 표현하지 않으며, 미공개 조건에 대한 refinement가 남아 있다.

## 11. Next Step

Proposed Context Model Improvement로 바로 넘어가지 않고 [reconstruction_refinement.md](reconstruction_refinement.md)의 R0–R5 계획에 따라 미공개 reconstruction condition의 sensitivity를 검토한다. 모든 refinement 실험은 기존 4-fold 75% training / 25% held-out test와 fixed-epoch final-result 정책을 유지하며, validation, early stopping, best checkpoint selection 또는 test 기반 best-epoch selection을 사용하지 않는다.
