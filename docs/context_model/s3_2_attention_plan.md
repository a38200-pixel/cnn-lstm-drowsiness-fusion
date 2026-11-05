# S3-2 Learnable Temporal Attention Experiment Plan

[확실] 작성일: 2026-11-05 KST. **S3-2 PLANNING / Architecture design NOT STARTED / Implementation NOT STARTED / Training NOT RUN.** Provisional name은 **S3-2 Learnable Temporal Attention**이다. 여러 시점 중 어떤 시점이 중요한지를 학습하는 Learnable Temporal Attention을 검토하기 위한 문서이며 상세 구조는 미확정이다.

이 문서는 **무엇을 할 것인가**를 기록한다. 연구 질문·비교 기준·변경 범위·통제 조건·평가 원칙과 향후 순서를 정의한다. 실제 수행 작업·검증·오류·결정은 [진행 기록](s3_2_attention_progress.md)에 시간 순서로 남긴다. 다음 설계 단계에서 정할 항목을 이미 결정된 것으로 해석하지 않는다.

## 1. Background

[확실] R2 reference는 VGG19 CNN → 4096D frame features → LSTM512 → 마지막 timestep hidden state → classifier 구조다. S3-1은 같은 CNN/LSTM/classifier를 유지하고 시간 정보 결합 방식(temporal readout)을 마지막 hidden state에서20개 timestep hidden states의 동일 가중치 평균으로 바꿨다.

[확실] [S3-1 official 4-fold 결과](stage3_context_improvement_plan.md#s3-1-final-results--44-folds)는 R2 F1 **80.24% ± 2.22%p**, S3-1 F1 **77.26% ± 0.26%p**, raw 평균 차이 **−2.982288%p (약−2.98%p)**다. Mean은%, population std는%p이며 ddof=0이다. S3-1은 COMPLETED / NO IMPROVEMENT OVER R2 REFERENCE이고4/4 fold에서 final F1 improvement가 관찰되지 않았다.

[확실] 이 결과를 근거로 단순 평균에서 각 timestep의 기여도를 학습하는 방향으로 질문을 확장한다. S3-1의 결과가 Attention의 필요성이나 개선을 입증한 것은 아니다. Stage3는 IN PROGRESS이며 이번에는 문서 체계만 마련한다.

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

## 5. Intended Change

현재 정하는 **변경 범위**는 다음과 같다. 구현된 S3-2 구조를 뜻하지 않는다.

- 기존 VGG19 backbone을 유지할 계획이다.
- 기존 LSTM과 classifier 구조를 유지할 계획이다.
- LSTM 출력과 classifier 사이의 시간 정보 결합 부분만 변경할 계획이다.
- 여러 timestep의 중요도를 학습할 수 있도록 Attention layer를 추가하는 방향을 검토한다.

아래 세부 사항은 **다음 Architecture Design 단계에서 검토할 미결정 항목**이다.

| 미결정 항목 | 현재 상태 |
|---|---|
| Attention scoring formula / 구현 계열 | NOT DECIDED |
| Hidden projection 여부 | NOT DECIDED |
| Nonlinear activation / bias 여부 | NOT DECIDED |
| Parameter initialization | NOT DECIDED |
| Attention dimension | NOT DECIDED |
| Dropout 추가 여부 | NOT DECIDED |
| Attention regularization / temperature | NOT DECIDED |
| Exact parameter count | NOT AVAILABLE — 설계·구현 후 검증 예정 |
| Attention parameter의 optimizer group / LR 처리 | NOT DECIDED |

특정 Linear layer, additive/dot-product 방식, query/key/value 또는 multi-head 구조를 현재 채택하지 않는다. R2/S3-1의 parameter count를 S3-2의 값으로 복사하지 않는다. 향후 설계 결정과 근거를 진행 기록에 남긴 후 구현 단계로 넘어갈 계획이다.

## 6. Controlled Conditions

원칙적으로 R2/S3-1과 동일하게 유지할 **비교 기준**이다. 현재 S3-2 config를 생성하거나 실행 설정을 적용했다는 뜻은 아니다.

| 항목 | 유지할 기준 |
|---|---|
| Dataset / fold split | SUST-DDD2,074 videos; 같은 label-stratified4-fold / seed42; drowsy positive |
| Sequence length / preprocessing | 같은20-frame indices; RGB / bilinear224×224 / ImageNet normalization; no augmentation |
| VGG19 backbone / 학습 정책 | 같은 ImageNet VGG19; Block1~4 frozen; Block5 fine-tuning; FC1 frozen after ReLU4096D |
| LSTM / classifier | 기존 single-bias LSTM hidden512 및 기존 classifier 유지 예정 |
| Optimizer policy | 기존 Adam 정책을 기준으로 비교 |
| Learning rate policy | 기존 Block5 0.00001 / temporal-head 0.0001을 기준으로 비교; Attention parameter 처리 미정 |
| Weight decay | 기존0.00001을 기준으로 비교 |
| Batch size / epochs / seed / AMP | 16 / fixed30 / 42 / false |
| DataLoader | 기존 workers4 / pin=true / persistent=true / prefetch1을 기준으로 비교 |
| Evaluation / official selection | Accuracy·Precision·Recall·F1·confusion matrix; fixed epoch30 final |
| Validation / scheduler / early stopping | 기존 none / none / none 정책 유지 예정 |

Attention parameter 때문에 optimizer parameter group 등의 조정이 필요하면, **필요한 이유와 비교에 미치는 영향을 먼저 문서화한 뒤** 설계에서 결정한다. 지금은 새 group·Attention LR·초기화·parameter 수를 임의 확정하지 않는다. 기존 통제 조건의 예외가 필요하면 변경 전후와 이유를 계획 및 진행 문서에 남기고, 다른 조건의 결과를 동일 조건 비교로 취급하지 않는다.

## 7. Evaluation Policy

- **Official:** 각 fold의 fixed **epoch30 final result**를 사용할 계획이다.
- **Metrics:** Accuracy, Precision, Recall, F1, confusion matrix를 기록할 계획이다. Positive class는 drowsy다.
- **4-fold:** 각 fold 동일 가중치 mean과 **population std(ddof=0)**를 raw float에서 계산할 계획이다. Pooled confusion matrix metric이나 rounded 값에서 만든 평균과 구분한다.
- **Post-hoc best epoch:** **DIAGNOSTIC ONLY**다. Held-out fold의 최고 성능 epoch를 official model selection에 사용하지 않는다.
- **미완료 결과:** 실행 전에는 NOT RUN / NOT AVAILABLE, fold 실행 중간 수치는 INTERIM, 일부 fold 평균은 INTERIM / NOT OFFICIAL AGGREGATE로 표시한다. 네 fold의 epoch30과 metric 검증을 마친 뒤에만 OFFICIAL 4-FOLD RESULT로 기록한다.

중단 원칙도 기존 평가 정책과 구분한다. Test score를 보고 조기 종료하거나 official epoch를 바꾸지 않는다. 향후 설계 미확정, 통제 조건 불일치, 구조·gradient·parameter 검증 실패가 있으면 학습 단계로 넘어가지 않고 원인과 다음 조치를 기록할 계획이다. 실행 중 오류나 사용자 요청으로 중단되면 마지막 완료 epoch·원인·checkpoint 존재/복원 가능성을 확인해 기록하고, 검증된 checkpoint가 없으면 resume 가능으로 표시하지 않는다. 이는 향후 절차이며 이번에 중단/복원 기능을 구현하거나 실행하지 않는다.

Single seed, 동일 held-out test를 보며 후속 실험을 설계하는 exploratory model development, overlapping training folds, subject-wise 분할 정보 미확인과 CUDA/resume의 한계를 유지한다. Mean 차이만으로 통계적 유의성이나 실제 환경 일반화를 주장하지 않는다.

## 8. Success Criteria

최소 비교는 **S3-2 vs R2**와 **S3-2 vs S3-1**이다. Primary metric은 **official4-fold mean F1**이다. Fold별 F1 delta와 방향, population std, Precision/Recall, Accuracy와 confusion matrix를 함께 보고할 계획이다.

임의의 최소 개선폭이나 “몇% 이상이면 성공”이라는 threshold를 만들지 않는다. S3-1보다 높더라도 R2보다 낮다면 두 비교를 구분해 기록한다. Attention에 따른 parameter·비용 변화가 확인되면 성능과 함께 설명한다. 결과와 tradeoff를 그대로 제시하며 현재 성공/실패 판정이나 수치를 만들지 않는다.

## 9. Planned Execution Stages

아래는 앞으로의 순서다. **S3-2.0 문서 생성만 수행했으며 나머지는 실행하지 않았다.** 다음 단계 시작을 이번 문서 생성 작업의 일부로 해석하지 않는다.

| 단계 | 예정 작업 | 현재 상태 |
|---|---|---|
| S3-2.0 | Document setup — 계획/진행 기록 체계 마련 | COMPLETED |
| S3-2.1 | Architecture design — 미결정 항목과 비교 조건 검토 | NOT STARTED |
| S3-2.2 | Implementation — 확정 설계의 model/config/trainer 연계 | NOT STARTED |
| S3-2.3 | Static / unit validation — 구조·gradient·parameter count 및 실행 전 검증 | NOT STARTED |
| S3-2.4 | Fold1 | NOT RUN |
| S3-2.5 | Fold2 | NOT RUN |
| S3-2.6 | Fold3 | NOT RUN |
| S3-2.7 | Fold4 | NOT RUN |
| S3-2.8 | 4-fold aggregation — raw final metric 집계·검증 | NOT STARTED |
| S3-2.9 | Final comparison and decision — R2/S3-1 비교 | NOT STARTED |

각 의미 있는 단계마다 **작업 수행 → 검증 → [진행 기록](s3_2_attention_progress.md) 갱신 → git diff 확인 → 완료 보고** 순서를 따른다. 실패·중단·resume도 발생 순서대로 남긴다. 진행 기록을 나중에 한꺼번에 작성하지 않는다.

## 10. Current Status

| 항목 | 현재 상태 |
|---|---|
| Stage3 | IN PROGRESS |
| S3-1 / result | COMPLETED / NO IMPROVEMENT OVER R2 REFERENCE |
| S3-2 | **PLANNING** |
| S3-2 plan / progress log | CREATED / CREATED |
| Architecture design | **NOT STARTED** |
| Implementation / config / trainer integration / test code | NOT STARTED |
| Training | **NOT RUN** |
| Fold1 / Fold2 / Fold3 / Fold4 | NOT RUN / NOT RUN / NOT RUN / NOT RUN |
| Aggregate / S3-2 parameter count | NOT AVAILABLE / NOT AVAILABLE |
| S3-2 MLflow run / checkpoint | NOT CREATED / NOT CREATED |
| Next | **S3-2 Architecture Design** |

