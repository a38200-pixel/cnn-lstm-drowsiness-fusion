# Project Execution Plan

[확실] 도입일: 2026-11-05 KST. **Central Execution Plan ACTIVE.** 이 문서는 **현재 프로젝트 실행 계획의 source of truth**다. 공통 작업 방식은 [PROJECT_WORKFLOW](PROJECT_WORKFLOW.md), 실제 수행 기록은 [PROGRESS](PROGRESS.md)를 따른다. 기존 상세 실험 계획·결과를 대체하지 않고 현재 단계와 다음 작업을 연결한다.

## 1. Current Project State

| 범위 | 실제 현재 상태 | 근거 / 다음 방향 |
|---|---|---|
| Project Workflow / Central Execution Plan / Central Progress Log | ACTIVE / ACTIVE / ACTIVE | WF-1 COMPLETED; 중앙 Progress Step 1에 검증 기록 |
| Stage1 — Paper-informed Reconstruction / R0 | COMPLETED | VGG19/VGG16 frozen feature baseline 완료 |
| Stage2 — Reconstruction Refinement | COMPLETED | R1/R2 및 R4 diagnostic 완료; R3/R5 미실행 상태 보존 |
| Stage3 — Context Model Improvement | IN PROGRESS | S3-1 완료; S3-2 Architecture Design COMPLETED |
| S3-0 | R2 REFERENCE | 기존 R2 official 결과 재사용 |
| S3-1 | COMPLETED | NO IMPROVEMENT OVER R2 REFERENCE |
| S3-2 | ARCHITECTURE DESIGN COMPLETED | Implementation NOT STARTED; Training NOT RUN |
| Stage4 — Behavior Detection | PLANNED / NOT STARTED | EAR / MAR / Head Pose 기반 탐지 예정 |
| Stage5 — Context + Behavior Fusion | PLANNED / NOT STARTED | Context prediction과 Behavior events 결합 예정 |

WF-1과 이번 **S3-2.1 Architecture Design은 COMPLETED**다. **Next = S3-2.2 Implementation — NOT STARTED**이며 이번 작업에서는 시작하지 않는다.

## 2. WF-1 — Project Workflow System Introduction

- **Purpose:** Codex와 Claude Code가 같은 계획·진행 관리 규칙을 사용하도록 공통 운영 계층을 도입한다.
- **Status:** COMPLETED — 중앙 Progress Step 1에 검증 결과 기록.
- **Scope:** 공통 entry point, 작업 규칙, 중앙 실행 계획, 중앙 진행 index를 작성한다. 기존 연구 상태를 연결한다.
- **Change Targets:** `AGENTS.md`, `CLAUDE.md`, `docs/PROJECT_WORKFLOW.md`, `docs/EXECUTION_PLAN.md`, `docs/PROGRESS.md`; README Documentation Index에 최소 링크 추가.
- **Out of Scope:** 모델·trainer·config·training script·unit test, S3-2 architecture 결정, 학습·MLflow run·checkpoint 생성, 기존 output·상세 문서·Git history 변경, commit/push.
- **Execution Order:** 기존 git 상태와 상세 문서 확인 → 중앙 계획 도입 → workflow/entry point/progress 작성 → 문서·상태·범위 검증 → 중앙 progress 기록 → 계획 상태 반영 → git diff 확인.
- **Validation:** 다섯 파일 존재, 링크, 공통 규칙 일치, 실제 연구 상태, 상세 문서 보존, 변경 범위, `git diff --check`.
- **Completion Criteria:** 공통 운영 문서를 읽고 현재 단계만 수행·검증·기록할 경로가 연결되고, 검증 결과가 중앙 progress에 남아 있으며 운영 문서가 ACTIVE다.
- **Expected Artifacts:** 다섯 운영 파일과 README 링크, 중앙 Progress Step 1, 초기 Plan Change Log entry.
- **Next:** S3-2.1 Architecture Design — NOT STARTED.

## 3. Stage1 — Paper-informed Reconstruction

[확실] **COMPLETED.** SUST-DDD 논문의 공개 정보로 CNN-LSTM 구조를 재구성했다. R0 VGG19/VGG16 frozen feature baseline은 완료 상태다. 원논문의 미공개 조건 때문에 exact reproduction으로 주장하지 않는다.

완료된 과거 작업을 새 PLANNED 단계로 만들지 않는다. 당시 계획·검증·결과는 [논문 근거](paper_reconstruction/baseline_spec.md), [결과](paper_reconstruction/results.md), [개발 이력](PROJECT_DEVELOPMENT_HISTORY.md)에 보존한다. 이번 도입에서 과거 작업 순서나 검증을 새로 만들어 기록하지 않는다.

## 4. Stage2 — Reconstruction Refinement

[확실] **COMPLETED — scope closed.** R1 FC2 feature experiment와 R2 VGG19 Block5 fine-tuning이 완료됐다. R4 training-duration diagnostic은 **COMPLETED / MIXED / INCONCLUSIVE / NO NEW TRAINING RECOMMENDED**, R3는 **OPTIONAL / NOT RUN**, R5는 **DEFER / NOT RUN**이다.

| R2 official4-fold reference | Mean (%) ± population std (%p), ddof=0 |
|---|---:|
| Accuracy | 81.58% ± 2.19%p |
| Precision | 81.04% ± 2.89%p |
| Recall | 79.49% ± 2.06%p |
| F1 | 80.24% ± 2.22%p |

근거는 [Refinement](paper_reconstruction/reconstruction_refinement.md)와 [R4 diagnostic](paper_reconstruction/r4_training_duration_plan.md)이다. 기존 결과를 이번 작업에서 다시 계산하거나 변경하지 않는다.

## 5. Stage3 — Context Model Improvement

### S3-0 — R2 Reference

[확실] **REFERENCE.** 마지막 timestep의 LSTM hidden state를 classifier에 입력하는 기존 R2 결과를 사용한다. Official F1은 **80.24% ± 2.22%p**다. 재학습한 새 결과로 취급하지 않는다.

### S3-1 — Temporal Mean Pooling

[확실] **COMPLETED.** 주요 변경은 LSTM final hidden → 전체 timestep hidden states의 동일 가중치 평균이다. 네 fold 모두 **COMPLETED VIA RESUME**이며 official은 각 fold epoch30 final이다.

| S3-1 official4-fold result | Mean (%) ± population std (%p), ddof=0 |
|---|---:|
| Accuracy | 79.02% ± 1.01%p |
| Precision | 79.14% ± 3.70%p |
| Recall | 75.80% ± 3.67%p |
| F1 | 77.26% ± 0.26%p |

[확실] R2 대비 raw mean F1 차이는 약 **−2.98%p**다. 판단은 **NO IMPROVEMENT OVER R2 REFERENCE**이며 R2 reference를 유지한다. 상세 raw metric·집계·resume·진단·한계는 [S3-1 최종 결과](context_model/stage3_context_improvement_plan.md#s3-1-final-results--44-folds)에 있다.

### S3-2 — Learnable Temporal Attention

[확실] **ARCHITECTURE DESIGN COMPLETED / DESIGN ONLY.** [실험 계획](context_model/s3_2_attention_plan.md)과 [상세 진행 기록](context_model/s3_2_attention_progress.md)은 CREATED다. Implementation은 **NOT STARTED**, Training 및 Fold1~4는 **NOT RUN**, Aggregate는 **NOT AVAILABLE**이다.

연구 질문은 “모든 timestep을 동일하게 평균하는 대신 모델이 중요한 timestep의 기여도를 학습하면 R2/S3-1보다 시간 표현을 개선할 수 있는가?”다. 선택 설계는 **Direct Scalar Temporal Attention / Linear(512,1,bias=False) → temporal softmax(dim=1) → original hidden weighted sum**이다. 추가 scorer512개와 전체 parameter count는 구현 전 예상값으로 구분한다. 세부 근거·초기화·optimizer·gradient·diagnostic·검증 계획은 [S3-2 상세 설계](context_model/s3_2_attention_plan.md#5-selected-architecture--direct-scalar-temporal-attention)에 기록한다.

### S3-2.1 — Architecture Design

- **Purpose:** 기존 VGG19/LSTM/classifier 유지 원칙 아래 시간 정보 결합 방식(temporal readout)의 Attention 설계와 비교 조건을 검토한다.
- **Status:** COMPLETED / DESIGN ONLY — 상세 Progress Step 1 및 중앙 Progress Step 2에 기록.
- **Scope:** 후보 비교, 선택 설계·근거·parameter/optimizer 영향, shape/gradient/diagnostic과 후속 검증 계획을 문서화한다.
- **Change Targets:** `s3_2_attention_plan.md`, `s3_2_attention_progress.md`, 이 중앙 계획과 중앙 progress를 갱신한다. `README.md`와 `stage3_context_improvement_plan.md`는 설계 완료 상태/Next의 최소 동기화만 허용한다.
- **Out of Scope:** model/config/trainer/training script/test 구현, 학습·MLflow run·checkpoint 생성, 기존 R2/S3-1 결과 변경.
- **Execution Order:** 공통 규칙과 현재 계획·진행 기록 확인 → 기존 비교 조건 확인 → 설계 대안 검토 → 결정과 미결정 사항 문서화 → 검증 → 상세/중앙 progress 갱신 → 다음 계획 반영.
- **Validation:** 상세 설계와 연구 질문·통제 조건의 일관성, parameter/학습 정책 영향의 근거, 계획되지 않은 구현·실행 부재, 문서 링크·상태·git diff 확인.
- **Completion Criteria:** 선택 구조·근거·tensor/gradient flow·expected count·optimizer/diagnostic·제외 범위·구현 검증 계획이 기존 code와 호환되고 상세/중앙 문서가 일치하면 완료로 기록한다.
- **Expected Artifacts:** 상세 Attention 설계 명세와 두 progress의 결정/검증 이력. S3-2 코드나 실험 결과는 이 단계 산출물이 아니다.
- **Next:** S3-2.2 Implementation — NOT STARTED; 확정된 설계와 범위를 계획에 반영한 뒤 진행할 단계다.

### S3-2.2 — Implementation (Next)

- **Purpose:** 확정된 Direct Scalar Temporal Attention 설계를 기존 비교 조건을 유지하며 구현할 다음 단계다.
- **Status:** NOT STARTED. 이번 S3-2.1 작업에서는 실행하지 않는다.
- **Scope:** 향후 S3-2 전용 model/config/training 진입점과 선택 Attention을 연결하고 기본 Tensor logits 계약·두 optimizer group을 유지할 계획이다.
- **Change Targets:** 다음 작업 시작 전 S3-2 전용 파일의 경로·범위를 구체화한다. 기존 R2/S3-1 model/config/script와 공통 LSTM/trainer는 변경하지 않는 방향이다. 상세/중앙 plan/progress는 갱신한다.
- **Out of Scope:** Architecture 임의 변경, 추가 readout/augmentation/scheduler/early stopping, pretrained weight 다운로드·원본 dataset 처리·full training·MLflow run·checkpoint 생성의 자동 실행.
- **Execution Order:** 공통 규칙/선택 명세 확인 → 파일 범위 구체화 → 전용 구현 → 필요한 bounded 검증 → 상세/중앙 progress → 계획 상태 반영 → git diff. 후속 S3-2.3/학습으로 임의 확대하지 않는다.
- **Validation:** 상세 plan5.11절의 shape·normalization·freeze/gradient·parameter/optimizer coverage·diagnostic parity·initialization·reference isolation 기준으로 후속 검증을 설계했다. 실제 구현/검증 결과는 아직 NOT RUN이다.
- **Completion Criteria:** 선택 설계와 구현이 일치하고 실시한 검증/미실시 검증 및 다음 S3-2.3 범위가 문서에 구분돼야 한다. 구현 전 예상 count를 actual로 쓰지 않는다.
- **Expected Artifacts:** 향후 전용 S3-2 코드와 구현/검증 기록. 현재 S3-2 model/config/script/test는 생성하지 않았다.
- **Next:** S3-2.3 Static / unit validation — NOT STARTED.

### Later S3-2 Stages

| 단계 | 현재 상태 | 상세 기준 |
|---|---|---|
| S3-2.0 Document setup | COMPLETED | 기존 상세 Progress Step 0 |
| S3-2.1 Architecture Design | COMPLETED | 선택 명세 완료; DESIGN ONLY |
| S3-2.2 Implementation | NOT STARTED | 설계 완료 후 범위와 검증 기준 구체화 |
| S3-2.3 Static / unit validation | NOT STARTED | 구현 후 구조·gradient·parameter count 등 검증 예정 |
| S3-2.4~S3-2.7 Fold1~Fold4 | NOT RUN | 기존 상세 계획의 실행 순서; 지금 실행하지 않음 |
| S3-2.8 4-fold aggregation | NOT STARTED | Aggregate NOT AVAILABLE |
| S3-2.9 Final comparison and decision | NOT STARTED | R2/S3-1 대비 결과 NOT AVAILABLE |

뒤 단계는 현재 실행 가능한 작업 지시가 아니다. 각 단계에 들어가기 전에 아래 공통 Step Fields를 채우고 상세 계획과 대조한다. 단계를 임의로 묶어서 구현·학습까지 이어가지 않는다.

## 6. Stage4 — Behavior Detection

- **Purpose:** EAR / MAR / Head Pose 기반 행동 이상 신호 탐지를 향후 검토한다.
- **Status:** PLANNED / NOT STARTED.
- **Scope:** 현재는 향후 방향만 존재하며 event 정의·threshold·평가 protocol은 미정이다.
- **Change Targets:** 실제 단계 계획 시 필요한 문서·파일을 특정할 예정이다. 이번 변경 대상은 없다.
- **Out of Scope:** 현재 단계에서의 Behavior 설계 확정·구현·데이터 처리·실행.
- **Execution Order:** 선행 Context 검토 → Behavior 상세 계획 → 해당 단계 작업·검증·기록. 현재 실행하지 않는다.
- **Validation:** 구체 검증 방법은 실제 시작 전 상세 계획에 정의한다. 현재 NOT RUN.
- **Completion Criteria:** 상세 범위와 평가 기준이 미정이므로 현재 완료 판정하지 않는다.
- **Expected Artifacts:** 향후 Behavior 계획·진행·결과 기록. 현재 구현/결과 NOT AVAILABLE.
- **Next:** Stage5와의 연결은 선행 결과와 정의된 interface를 근거로 별도 계획한다.

## 7. Stage5 — Context + Behavior Fusion

- **Purpose:** Context prediction과 Behavior events를 결합하는 Fusion Detection을 향후 검토한다.
- **Status:** PLANNED / NOT STARTED.
- **Scope:** 현재는 결합 방향만 존재하며 구체 fusion rule·가중치·평가 protocol은 미정이다.
- **Change Targets:** 실제 단계 계획 시 특정한다. 이번 변경 대상은 없다.
- **Out of Scope:** 현재 단계에서의 Fusion 구조 확정·구현·실행.
- **Execution Order:** Context/Behavior 결과와 interface 확인 → 상세 결합·평가 계획 → 해당 단계 작업·검증·기록. 현재 실행하지 않는다.
- **Validation:** timestamp 정렬·event aggregation·기여도 비교 등 구체 방법은 시작 전 상세 계획에 정의한다. 현재 NOT RUN.
- **Completion Criteria:** 아직 확정하지 않았으며 현재 완료 판정하지 않는다.
- **Expected Artifacts:** 향후 Fusion 계획·진행·비교 결과. 현재 결과 NOT AVAILABLE.
- **Next:** 최종 비교의 범위와 조건은 선행 결과에 따라 별도 정의한다.

## 8. Step Fields and Plan Maintenance

모든 주요 작업은 실행 전에 최소 **Purpose / Status / Scope / Change Targets / Out of Scope / Execution Order / Validation / Completion Criteria / Expected Artifacts / Next**를 갖춘다. 완료된 Stage1/2와 S3-1은 기존 상세 기록을 근거로 요약했으며, 당시 계획을 새로 꾸며내지 않는다.

새 작업이나 변경된 접근법은 이 계획에 먼저 기록한 뒤 수행한다. 변경 이유·영향은 마지막 Plan Change Log에 남기며, 실제 진행은 [PROGRESS](PROGRESS.md)에 남긴다. 정상적인 단계 상태 갱신과 연구 조건/접근법 변경을 구분한다. 후자라면 변경 작업을 시작하기 전에 계획을 먼저 갱신한다.

## Plan Change Log

### 2026-11-05 — Central Execution Plan Introduced

- **Previous Plan:** 기존 프로젝트 계획과 진행 기록이 개발 이력 및 실험별 상세 문서에 분산돼 있었다. 과거 변경 연혁을 이번에 소급 재구성하지 않는다.
- **Reason:** 공통 entry point와 중앙 계획/진행 index로 현재 단계·검증·계획 변경을 일관되게 관리하려는 사용자 요청.
- **Updated Plan:** 현재 실제 연구 상태를 바탕으로 중앙 실행 계획을 도입하고, 이번 WF-1과 Next S3-2.1을 구분한다.
- **Affected Steps:** 프로젝트 공통 작업 절차와 이후 모든 주요 단계. 완료된 Stage1/2/S3-1 및 기존 결과는 유지한다.
- **Impact:** 운영 문서와 작업 기록 경로만 추가한다. S3-2 세부 설계·코드·config·학습·실험 결과 변경은 없다.

### 2026-11-05 — S3-2 Design Status Synchronization Scope

- **Previous Plan:** S3-2.1의 변경 대상은 상세 plan/progress와 중앙 plan/progress 네 문서였다.
- **Reason:** 설계 완료 후 README와 Stage3 개요에 PLANNING / Architecture design NOT STARTED가 남으면 현재 상태가 불일치한다.
- **Updated Plan:** 필수 네 문서 외에 README와 Stage3 개요의 현재 상태/Next만 최소 동기화한다. 기존 결과와 과거 Timeline은 보존한다.
- **Affected Steps:** S3-2.1 문서 갱신 범위. 구현·학습 단계의 실행 권한은 확대하지 않는다.
- **Impact:** 문서 두 개의 상태/Next 문구만 추가 변경한다. Architecture 선택은 원래 예정된 설계 결정이며 기존 연구 조건·코드·config·결과는 변경하지 않는다.
