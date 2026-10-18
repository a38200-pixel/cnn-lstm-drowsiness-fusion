# R4 Training Duration Refinement

[확실] 작성일: 2026-10-18 KST. 최신 상태: **R4 DIAGNOSTIC COMPLETED / R4 NEW TRAINING NOT RUN**. 아래 설계는 분석 전에 고정했으며 같은 날짜의 Diagnostic Result에 결과를 추가했다. 기존 R2 held-out test history를 이미 관찰했으므로 **test 노출 이전의 독립적 preregistration으로 주장하지 않는다.** 작성 후 변경은 이유와 날짜를 남기고, 변경 전후 결과를 선택적으로 보고하지 않는다.

[확실] 이 계획서만 생성한다. 학습/profile/benchmark/dataset scan, config 생성, 코드 변경, R3/R5 실행, Stage3 구현을 수행하지 않는다. 기존 결과를 수정하지 않는다.

## Motivation

[확실] R2 block5 representation adaptation은 R0 frozen FC1 대비 official F1 mean을 2.30%p 개선했다. 그러나 final과 best 사이 차이 및 후반 test loss 증가가 관찰되어 training duration sensitivity를 검토할 근거가 있다. R4는 새로운 architecture가 아니라 **R2 architecture/training policy에서 duration만 비교하는 refinement**다.

[확실] 기존 30-epoch 실행과 동일한 짧은 prefix를 다시 계산해도 독립적인 test 검증 자료가 새로 생기는 것은 아니다. 장시간 재학습에 앞서 기존 trajectory로 답할 수 있는 질문과 새 실행이 필요한 질문을 분리한다.

## R2 Evidence

[확실] 출처는 `reconstruction_refinement.md`의 R2 4-fold 분석과 아래 최종 output의 history/metrics다. 모든 official 값은 epoch30 final, mean ± population std(ddof=0)다.

| Evidence | Value |
|---|---|
| R2 official F1 | **80.24% ± 2.22%p** |
| Post-hoc best F1 | **81.43% ± 1.89%p — DIAGNOSTIC ONLY** |
| Best−final mean difference | 약 **+1.18%p**, 반올림 전 float로 계산 |
| Fold별 best F1 epoch | 6 / 25 / 18 / 18 |

| Fold | Early10 mean test loss | Last10 mean test loss |
|---|---:|---:|
| 1 | 0.5113 | 0.8366 |
| 2 | 0.4853 | 0.7256 |
| 3 | 0.4723 | 0.7206 |
| 4 | 0.4234 | 0.5685 |

[확실] 모든 fold에서 마지막10 평균 test loss가 초반10보다 높았다. Fold3의 후반 train loss 반등도 있어 모든 fold의 train loss가 단조 감소했다고 설명할 수 없다.

[추측] 이는 possible late-stage overfitting과 duration sensitivity의 분석 근거이며, overfitting 확정이나 특정 epoch의 최적성을 입증하지 않는다.

[확실] 원본 경로는 `outputs/paper_reconstruction/refinement/` 기준 다음과 같다. Fold3/4는 최종 resume history를 사용하고 이전 부분 실행을 중복 fold로 집계하지 않는다.

| Fold | Final source |
|---|---|
| 1 | `R2_cuda_fold1_workers4_prefetch1_official/fold_1/` |
| 2 | `R2_cuda_fold2_workers4_prefetch1_official/fold_2/` |
| 3 | `R2_cuda_fold3_resume_next/fold_3/` |
| 4 | `R2_cuda_fold4_resumed_20261015/fold_4/` |

## Research Question

[확실] 연구 질문은 **R2에서 관찰된 late-stage test degradation이 fixed30 duration과 관련되어 있는가?**다. 권장 분석의 구체적인 대비는 같은 recorded trajectory의 epoch15 대 epoch30이다.

[확실] 이 대비는 해당 seed·fold·실행 경로에서 training을 더 진행했을 때 metric/loss가 어떻게 달라졌는지 보여준다. 다른 seed, 새 subject, 독립 test나 모든 가능한 duration의 최적성을 검증하지 않는다.

## Leakage Risk

[확실] R2의 전체 epoch별 test metric과 best epochs6/25/18/18을 이미 분석했다. 따라서 epoch18을 최적값으로 채택하거나 fold마다 다른 best epoch를 선택하는 것은 금지한다. 새로 같은 test fold에 재학습해도 이미 노출된 test 정보가 사라지지 않는다.

[확실] **15 선택 근거는 기존30의 절반이라는 단순한 상대 duration**이다. 가장 적은 후보 한 개로 충분한 training-length 대비를 만들고, best epoch를 그대로 복사하지 않는다. 다만 이 근거가 사후 test 노출 사실을 없애거나 선택의 완전한 독립성을 입증하지는 않는다.

[확실] 이후 수행할 분석에는 공통15/30만 사용한다. 결과를 본 뒤20/18/다른 epoch로 후보를 바꾸거나 추가하지 않는다. 모든 fold를 보고하며 favorable fold만 선택하지 않는다. 변경이 필요하면 별도 후속 계획과 노출 이력을 공개한다.

[확실] 기존 test에 의존하는 analysis는 **retrospective duration sensitivity / POST-HOC DIAGNOSTIC ONLY**다. 미래의 독립적인 confirmatory 평가가 필요하면 별도 평가 설계를 먼저 정해야 하며 이번 범위에서 validation이나 새 split을 추가하지 않는다.

## Fixed Conditions

[확실] 비교의 기반은 R2이며 아래 값은 duration 외 변경하지 않는다. 향후 새 실행을 따로 선택할 때도 같은 계약을 적용한다.

| Item | Fixed value |
|---|---|
| Backbone | ImageNet-pretrained VGG19 |
| Block1~4 / Block5 | frozen / trainable |
| FC1 / feature | frozen / fc1 after ReLU4096D |
| Temporal model | single-bias LSTM input4096, hidden512, one layer |
| Head | Dense512→ReLU→Dense64→Sigmoid→Dense2 logits |
| Optimizer | Adam |
| Block5 / LSTM-head LR | 0.00001 / 0.0001 |
| Weight decay | 0.00001 |
| Batch / seed / AMP | **16 / 42 / false** |
| Fold | 기존 label-stratified4-fold, same assignments |
| Sampling | same deterministic uniform20 frames |
| Preprocessing | same RGB/bilinear224×224/ImageNet normalization |
| Augmentation | none |
| Validation / scheduler / early stopping | none / none / none |
| DataLoader | workers4, pin_memory=true, persistent_workers=true, prefetch_factor=1 |
| Positive class | drowsy |

[확실] Duration 이외 model/training/data/evaluation 설정이 다르면 단일변수 비교로 취급하지 않는다. 기존 실행의 버전·환경·resume 이력도 보존한다.

## Candidate Designs

[확실] Duration 후보 비교는 아래와 같다. 30은 조건이 맞는 기존 R2 official reference를 재사용한다. 비용은 새로 처음부터 학습하는 경우의 추가 epoch 수다.

| Duration design | Methodological rationale | Additional epochs / fold runs |
|---|---|---|
| Option A: **15 vs30** | 절반 duration의 큰 대비, 후보1개로 제한 | 새 학습 시60 epochs / 4folds; history 분석이면0 |
| Option B: 20 vs30 | 30의2/3, 더 작은 duration 차이 | 새 학습 시80 epochs / 4folds |
| Option C: 15 vs20 vs30 | 서로 다른 짧은 duration을 비교 가능하나 선택 자유도와 비용 증가 | 새 학습 시140 epochs / 8fold runs |

[확실] 여기서20이15보다 타당하다는 별도 이론·사전 검증 근거는 없다. Best18에 가까운 값을 선택하는 이유로20을 채택하지 않는다. 세 후보를 모두 실행해야 연구 질문에 답할 수 있다는 근거도 현재 없다.

[확실] 실행 전략 A/B/C와 위 duration Option A/B/C는 구분한다.

| Execution strategy | Advantage | Limitation / cleanliness | Additional compute |
|---|---|---|---|
| **A: existing history diagnostic** | 동일 recorded trajectory의 직접 prefix 대비; 재학습 없음 | test 노출 후의 retrospective 분석; 새 official15 조건 아님 | **0fold runs** |
| B: fixed15 새4-fold | 사전에 정한 중단 길이를 실제 실행하고15 모델을 확보 | 동일 seed/test 재사용으로 독립 검증은 아님; CUDA/resume 차이 개입 가능 | 60epochs / 약70~75시간 |
| C: fixed15와20 새각4-fold | duration response를 추가 관찰 | 비용·다중 비교 증가; 동일 test 노출 한계 유지 | 140epochs / 약163~175시간 |
| D: 다른 seed/새 독립 평가 | 별도 재현성/일반화 질문 검토 가능 | duration 외 변수 또는 평가 protocol 변경; 이번 단일변수 R4 범위 밖 | 미정, 이번에는 제안·실행하지 않음 |

[확실] 기존 epoch15 checkpoint가 같은 fixed15 학습인가를 판단하려면 **명목 seed만이 아니라 optimizer state·초기 weight·배치 순서·RNG·연산·실행 이력까지** 대응해야 한다. 재학습은 별도 실행이며 같은 recorded prefix를 조회하는 것과 동일한 절차는 아니다.

[확실] 코드상 `training.epochs`는 검증과 epoch loop 종료 등에 쓰이고, Adam update에 총 학습 길이를 전달하거나 duration-dependent scheduler를 사용하지 않는다. 따라서 초기화와 각 update 입력/연산이 같다는 조건에서는 fixed15의15번째 update-epoch 상태가 fixed30의15번째 epoch 상태와 대응한다. Adam의 moment/step도 같은 prefix를 따르면 같고, 단순히30을 목표로 했다는 이유로 앞15 epoch가 달라지지는 않는다.

[확실] 이 대응은 조건부다. CUDA deterministic 설정은 warn_only=True이고 Fold3/4에는 resume 이력이 있다. RNG/optimizer/shuffle generator를 복원해도 multiprocess 재시작과 미지원 deterministic 연산 때문에 새 실행과의 bitwise 동일성을 보장하지 않는다. Epoch15 metric은 해당 recorded trajectory의 관찰값으로는 재사용 가능하지만 새 fixed15 실행과 정확히 같다고 단정하지 않는다.

[확실] `last_checkpoint.pt`는 매 epoch 덮어쓰므로 현재 최종30 checkpoint를 거꾸로 계산해15 weight/optimizer를 복원할 수 없다. History는 metric 기록이지 모델 state가 아니다. Epoch15 checkpoint의 별도 보존은 미확인이다. 새15 모델 자체가 필요할 때는 별도 보존 checkpoint 확인 또는 새 학습이 필요하지만 이번 권장 분석에는 필요하지 않다.

## Recommended Design

[확실] **최종 추천: execution strategy A, common fixed contrast15 vs30.** 기존 R2 history의 epoch15와 epoch30만 네 fold에서 한 번 집계해 diagnostic으로 보고한다. 정확한 duration 후보는15이고 reference는 기존30이다. 추가 학습 수는 **0fold runs**, R2 epoch30 재학습은 하지 않는다.

[확실] 이유는 동일 trajectory의 metric이 이미 기록되어 있고, 새 동일-seed fixed15 학습이 test leakage를 제거하거나 독립적인 검증 정보를 주지는 않기 때문이다. 현재 질문을 먼저 최소 비용의 retrospective 관찰로 답한다. 새 모델 artifact 확보나 반복 실행 검증을 요구할 때만 strategy B를 별도 검토한다.

[확실] 30-reference 재사용은 동일 architecture/optimizer/LR/data/fold/seed/실행 설정을 비교한다는 조건에서 가능하다. 기존 결과를 새 실행이라고 부르지 않고, Fold3/4의 resume 및 Fold4의 terminal status stale도 명시한다. Final metric/model/artifact 저장은 확인되었고 DB 상태를 임의 수정하지 않는다.

[확실] **Official comparison boundary:** 기존 R2 epoch30은 official4-fold reference로 유지하고, history15는 diagnostic duration observation으로 병기한다. 권장안에는 새 R4 official aggregate가 생기지 않는다. Strategy B를 향후 승인해 실제로 실행한다면 project policy상 fixed15 final만 해당 조건의 official metric으로 보고할 수 있지만, 동일한 사전 노출 test를 사용한 sensitivity 실험이라는 한계를 함께 명시해야 한다.

[확실] 현재 `scripts/train_r2_vgg19.py`의 config 검증은 fixed30을 강제한다. Strategy B는 현 CLI에15 config만 주어 바로 실행할 수 없고 별도 승인된 구현 검토가 필요하다. 이번 작업에서 config/코드를 바꾸거나 guard를 우회하지 않는다.

## Evaluation Policy

[확실] 향후 승인될 분석은 동일 fold의 epoch15와30 값을 paired comparison한다. Accuracy/Precision/Recall/F1 각각의4-fold mean/population std(ddof=0), fold별 delta, Precision/Recall tradeoff, train/test loss를 보고한다. Primary comparison은 F1이며 다른 metric을 보고 유리한 primary로 바꾸지 않는다.

[확실] Fold별 best epoch, early stopping, validation 추가를 사용하지 않는다. 새로 실행하는 duration condition이 있다면 사전에 고정한 final epoch만 official로 사용하고 best test epoch는 POST-HOC DIAGNOSTIC ONLY를 유지한다. 이 원칙이 기존 history15를 소급해 official로 승격한다는 의미는 아니다.

[확실] Pooled confusion matrix와 fold mean을 혼동하지 않고, 동일 fold를 독립적인 네 번의 모집단 실험처럼 취급해 근거 없는 유의성을 주장하지 않는다. 기존 test를 다시 관찰하는 분석의 모든 결과와 null/negative 방향도 보고한다.

## Expected Compute

[확실] **권장안은 추가 학습0, GPU 학습 점유시간0**이다. 작은 JSON/history의15/30 집계만 필요하며 전체 video scan이나 feature extraction은 필요하지 않다. 계획 작성 시에는 비교 metric을 집계하지 않았고, 후속 요청으로 수행한 결과는 아래 Diagnostic Result에 기록했다.

[확실] 비교용 비용 추정의 기준은 기존 history에서 timing이 있는 epoch의 fold별 평균 약69.85~75.17분이다. Fold3는28개, Fold4는29개 timing만 최종 history에 남아 있어 이를 완전한 총 wall-time으로 사용하지 않았다. Epoch30 한 개의67~73분을 전체30-epoch 시간으로 혼동하지 않는다.

| Optional new training | Extra epochs | Approximate single-machine training occupancy |
|---|---:|---:|
| fixed15 ×4folds | 60 | 약70~75시간 |
| fixed20 ×4folds | 80 | 약93~100시간 |
| fixed15+20 ×각4folds | 140 | 약163~175시간 |
| reference30 ×4folds 재학습 | 120 | 약140~150시간; 권장하지 않음 |

[추측] 위 시간은 관측 평균을 약70~75분/epoch로 반올림해 곱한 참고 추정이다. 시작/저장·시스템 부하·Windows stall·중단/재실행에 따라 달라지며 예약 시간이나 보장된 종료 시각이 아니다. GPU kernel의 실제 active time과 장비 점유시간도 다르다.

## Decision Rule

[확실] 분석 실행 전에 아래 보고·판단 순서를 고정한다. 새로운 최소 개선폭이나 임의의 통계 유의성 threshold는 만들지 않는다.

1. [확실] 네 fold의 epoch15와30이 모두 존재하고 조건/출처가 대응하는지 확인한다. 누락·불일치는 평균으로 대체하지 말고 비교 불가로 보고한다.
2. [확실] Primary는 mean F1(15)−mean F1(30)의 방향이다. 양수면 해당 recorded trajectories에서 shorter duration이 나은 관찰,0이면 동일, 음수면 fixed15 이점이 관찰되지 않은 것으로 기술한다. 확정적인 일반화 성능 개선으로 표현하지 않는다.
3. [확실] F1 std와 fold별 delta, Precision/Recall tradeoff를 함께 보고한다. Mean F1이 올라도 std 증가나 Precision/Recall 저하가 있으면 상충 관계를 명시하고 무조건 우월로 판정하지 않는다.
4. [확실] Train/test loss 차이와 후반 변화를 보조 근거로 사용한다. Loss가 낮다는 이유만으로 F1 기준 결과를 뒤집거나 overfitting을 확정하지 않는다.
5. [확실] Diagnostic 결과만으로 official reference를 교체하지 않는다. 개선 관찰은 후속 reference-policy 검토 근거이고, 정책 반영은 별도 결정·노출 한계 공개가 필요하다. 실제 신규 fixed-duration 실험을 승인·실행했다면 그 사전 고정 final 결과로 같은 기준을 적용한다.
6. [확실] 이점이 없거나 tradeoff로 실질적인 이득을 판단하기 어려우면 후보를 즉석 확장하지 않고 Stage2 종료 여부를 검토한다. R3는 OPTIONAL, R5는 DEFER를 유지하고 이후 Stage3 이동 여부를 결정한다.

## What R4 Will Not Claim

[확실] 이 계획은 test leakage가 해소되었다거나 독립적인 preregistered confirmation이라고 주장하지 않는다.15/18/20이 전역 optimal epoch, 모든 seed에서 최선, 원논문 duration 복원, paper gap 해결, overfitting confirmed라는 결론을 내리지 않는다.

[확실] 동일 seed라는 이유만으로 새 실행의 trajectory가 bitwise 같다고 주장하지 않고, history metric을 checkpoint weight로 취급하지 않는다. 새 동일-test 재학습만으로 평가 독립성이 생긴다고 주장하지 않는다.

## Execution Status

| Work | Status |
|---|---|
| R4 plan | DESIGN ONLY로 고정한 분석 범위 유지; 아래 Diagnostic Result 추가 |
| Registered analysis contrast | **15 vs30; common across all4folds** |
| Recommended strategy | **A — existing history diagnostic; no additional training** |
| Plan-specific15/30 aggregate | **R4 DIAGNOSTIC COMPLETED** |
| New R4 fixed-duration training / official aggregate | **NOT RUN / NOT AVAILABLE** |
| R4 config / code change | **NOT CREATED / NOT CHANGED** |
| Existing R2 epoch30 reference | COMPLETED; reusable subject to stated conditions |
| R3 / R5 | **OPTIONAL / DEFER** |
| Stage3 Context implementation | **NOT STARTED** |

[확실] 고정15/30 집계를 아래에 기록했다. **NEXT 권고:** R4를 diagnostic으로 종료하고 Stage2 종료 여부를 결정한 뒤 Stage3 Context Improvement 범위를 설계한다. Stage2 종료나 Stage3 구현은 이번 작업에서 확정·실행하지 않는다. 새 학습과 official reference 변경은 없다.

## Diagnostic Result

[확실] 분석일: **2026-10-18 KST**. **R4 DIAGNOSTIC: COMPLETED / R4 NEW TRAINING: NOT RUN**. 사전 고정한 공통 epoch15와30만 사용했고 다른 duration을 탐색하거나 후보를 변경하지 않았다. 이 절은 **R4 DIAGNOSTIC ONLY**인 retrospective sensitivity analysis다. Epoch30은 기존 R2 official reference로 유지한다.

### Source and validation

[확실] R2 Evidence에 명시한 네 최종 output의 `history.json`만 source of truth로 사용했다. Fold3는 `R2_cuda_fold3_resume_next/fold_3/history.json`, Fold4는 `R2_cuda_fold4_resumed_20261015/fold_4/history.json`이며 앞선 부분 실행을 중복 집계하지 않았다. 네 history 모두 epoch1~30 연속성과 epoch15/30 행의 존재를 확인했다. 추정·보간·checkpoint 역직렬화는 수행하지 않았다.

[확실] 계산에는 history의 원래 float precision을 사용했다. Metric은 %로 소수 둘째 자리, loss는 소수 여섯째 자리로 표시했다. Std는 population std(ddof=0)이며 std 단위는 %p다. 표의 표시값을 다시 빼서 delta를 계산하지 않았다.

### Fold-level epoch15 versus epoch30

| Fold | Epoch | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Train Loss | Test Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 diagnostic | 78.61 | 77.37 | 77.05 | 77.21 | 0.194189 | 0.619931 |
| 1 | 30 official | 78.23 | 76.73 | 77.05 | 76.89 | 0.012044 | 0.900131 |
| 2 | 15 diagnostic | 80.73 | 80.00 | 78.69 | 79.34 | 0.109118 | 0.591543 |
| 2 | 30 official | 81.12 | 80.17 | 79.51 | 79.84 | 0.012602 | 0.764611 |
| 3 | 15 diagnostic | 83.24 | 85.84 | 77.05 | 81.21 | 0.091132 | 0.540609 |
| 3 | 30 official | 83.04 | 84.21 | 78.69 | 81.36 | 0.123243 | 0.648642 |
| 4 | 15 diagnostic | 83.17 | 81.20 | 83.54 | 82.35 | 0.075964 | 0.508126 |
| 4 | 30 official | 83.95 | 83.06 | 82.72 | 82.89 | 0.025707 | 0.601602 |

### Paired changes: epoch15 minus epoch30

[확실] Metric 변화는 %p, test loss 변화는 원래 loss 단위다. 음수 test loss delta는 epoch15의 loss가 낮다는 뜻이다.

| Fold | Accuracy delta (%p) | Precision delta (%p) | Recall delta (%p) | F1 delta (%p) | Test Loss delta |
|---|---:|---:|---:|---:|---:|
| 1 | +0.39 | +0.63 | 0.00 | **+0.32** | −0.280200 |
| 2 | −0.39 | −0.17 | −0.82 | **−0.50** | −0.173068 |
| 3 | +0.19 | +1.63 | −1.64 | **−0.15** | −0.108033 |
| 4 | −0.77 | −1.86 | +0.82 | **−0.53** | −0.093476 |
| **Mean delta** | **−0.15** | **+0.06** | **−0.41** | **−0.22** | — |

### Four-fold aggregate and primary comparison

| Duration / role | Accuracy (%) ± std (%p) | Precision (%) ± std (%p) | Recall (%) ± std (%p) | F1 (%) ± std (%p) |
|---|---:|---:|---:|---:|
| **Epoch15 — R4 DIAGNOSTIC ONLY** | **81.44 ± 1.92** | **81.10 ± 3.07** | **79.08 ± 2.66** | **80.03 ± 1.95** |
| **Epoch30 — R2 OFFICIAL** | **81.58 ± 2.19** | **81.04 ± 2.89** | **79.49 ± 2.06** | **80.24 ± 2.22** |

[확실] Primary result의 원래 float 값은 mean F1(15)=0.8002716989713582, mean F1(30)=0.8024238415748962, 차이=(15−30)×100=**−0.2152142603538021%p**, 표시값 **−0.22%p**다. Epoch15가 더 높은 fold는 **1개**, epoch30이 더 높은 fold는 **3개**, 동률은 **0개**다. Shorter duration의 일관된 F1 개선은 관찰되지 않았다.

[확실] Epoch15의 F1 std는1.95%p로 epoch30의2.22%p보다 작다. 그러나 Recall std는2.66%p 대2.06%p, Precision std는3.07%p 대2.89%p로 더 크다. 따라서 variability가 모든 metric에서 개선되었다고 해석하지 않는다.

### Precision/Recall tradeoff

[확실] Mean Precision은 epoch15에서 **+0.06%p**, mean Recall은 **−0.41%p**다. Aggregate 수준에서 미세한 Precision 상승과 Recall 감소가 동반되었고 F1 mean도 낮아졌다. Fold3는 Precision 상승/Recall 감소, Fold4는 반대 tradeoff를 보여 모든 fold가 같은 방향으로 변하지 않았다.

### Loss behavior

[확실] Epoch15 test loss는 네 fold 모두 epoch30보다 낮다. 이는 기존 R2 분석에서 마지막10 epoch 평균 test loss가 초반10보다 높은 관찰과 방향이 일치한다.

| Fold | Early10 mean Test Loss | Last10 mean Test Loss |
|---|---:|---:|
| 1 | 0.5113 | 0.8366 |
| 2 | 0.4853 | 0.7256 |
| 3 | 0.4723 | 0.7206 |
| 4 | 0.4234 | 0.5685 |

[확실] Fold1/2/4의 train loss는 epoch30에서 더 낮지만 Fold3는 반대로 epoch15보다 epoch30에서 높다. 모든 fold에 대해 train loss 단조 감소를 주장하지 않는다. Test loss의 일관된 하락은 F1의 일관된 개선과 동일하지 않다.

[추측] 이 loss 패턴은 possible late-stage overfitting/degradation의 후속 해석 근거다. 다만 calibration이나 오류별 confidence를 별도로 분석하지 않았으므로 loss/F1 차이의 원인을 확정하지 않는다.

### Methodological limitation and decision

[확실] **Classification: B — MIXED / INCONCLUSIVE.** Primary F1 mean은 낮아지고3/4 fold에서 epoch30이 더 높으므로 shorter duration의 일관된 diagnostic benefit은 없다. 반면 모든 fold의 test loss가 낮고 F1 std도 작아 secondary diagnostic benefit은 존재한다. 이 상충 때문에 A의 일관 개선이나 모든 지표에서 이점이 전혀 없다는 결론으로 일반화하지 않는다. 임의 threshold나 유의성 기준을 추가하지 않았다.

[확실] 이미 held-out test history를 본 뒤 수행한 분석이며 independent validation이 아니다. Fold3/4 resume과 CUDA warn_only의 한계도 유지한다. Epoch15는 새로운 official result/model, 자동 교체할 R2 reference 또는 optimal duration이 아니다. Epoch30 official 결과는 그대로 유지한다.

[확실] **R4를 diagnostic으로 종료할 것을 권고한다.** Same-seed fixed15의 새4-fold 학습은 약70~75시간의 장비 점유가 예상되지만, test 노출 문제를 제거하지 못하고 현재 primary 비교에서도 이점을 보이지 않았다. 새 모델 확보나 별도 재현성 질문이 없는 현재 목적에서는 추가 학습을 권장하지 않는다. 이 시간은 기존 Expected Compute의 추정이며 새 측정 결과가 아니다.

[확실] **Stage2 종료를 검토하고 Stage3 Context Improvement 설계로 이동할 것을 권고한다.** 이는 종료 결정의 지원 자료이며 이번 작업에서 Stage2 종료를 확정하거나 다른 문서 상태를 바꾸지 않는다. R3는 OPTIONAL, R5는 DEFER를 유지하고 후보를 즉석 확장하지 않는다. Stage3 구현은 NOT STARTED다.

[확실] 변경 범위는 이 문서 하나다. Training/config/model/trainer/기존 output과 README/refinement/development history는 수정하지 않았다.
