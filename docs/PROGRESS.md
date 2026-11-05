# Project Progress Log

[확실] 중앙 기록 도입일: 2026-11-05 KST. **Central Progress Log ACTIVE.** 이 문서는 **전체 프로젝트 진행 기록의 중앙 index**다. 공통 규칙은 [PROJECT_WORKFLOW](PROJECT_WORKFLOW.md), 현재 실행 계획은 [EXECUTION_PLAN](EXECUTION_PLAN.md)에 있다. 과거 상세 기록을 복사하거나 대체하지 않고 요약과 링크를 제공한다.

## Existing Detailed History

- [Project Development History](PROJECT_DEVELOPMENT_HISTORY.md): 기존 연구·실행·실패·중단·resume의 누적 이력.
- [Paper reconstruction specification](paper_reconstruction/baseline_spec.md), [data preparation](paper_reconstruction/data_preparation.md), [model pipeline](paper_reconstruction/model_pipeline.md), [results](paper_reconstruction/results.md): Stage1/R0 상세 근거와 완료 결과.
- [Reconstruction refinement](paper_reconstruction/reconstruction_refinement.md), [R4 duration diagnostic](paper_reconstruction/r4_training_duration_plan.md): Stage2 상세 실험·진단·결정.
- [Stage3 plan and S3-1 final results](context_model/stage3_context_improvement_plan.md): R2 reference와 S3-1 official4-fold 비교.
- [S3-2 experiment plan](context_model/s3_2_attention_plan.md), [S3-2 progress log](context_model/s3_2_attention_progress.md): S3-2 계획과 세부 진행 기록. 문서 setup Step 0은 완료됐고 Architecture Design Step 1은 이번 중앙 Step 2와 연결한다. 구현·학습은 시작하지 않았다.

S3-2 관련 주요 작업을 수행하면 **이 중앙 로그에는 요약과 링크**, **S3-2 progress에는 세부 내용**을 남긴다. 두 문서를 필요한 범위에서 갱신하며 같은 내용을 장문으로 중복 작성하지 않는다.

## Historical Summary

아래는 기존 문서에서 확인한 상태 요약이다. 중앙 도입 이전의 작업·실행 시각을 소급해 만들어 기록하지 않는다.

| 범위 | 현재 확인된 상태 |
|---|---|
| Stage1 / R0 VGG19·VGG16 | COMPLETED |
| Stage2 / R1·R2 | COMPLETED |
| R4 training-duration diagnostic | COMPLETED / MIXED / INCONCLUSIVE |
| R3 / R5 | OPTIONAL / NOT RUN; DEFER / NOT RUN |
| Stage3 | IN PROGRESS |
| S3-0 | R2 REFERENCE; F1 80.24% ± 2.22%p |
| S3-1 | COMPLETED; F1 77.26% ± 0.26%p; NO IMPROVEMENT OVER R2 REFERENCE |
| S3-2 | ARCHITECTURE DESIGN COMPLETED; plan/progress UPDATED |
| S3-2 Architecture Design / Implementation | COMPLETED / NOT STARTED |
| S3-2 Training / Aggregate | NOT RUN / NOT AVAILABLE |
| Stage4 / Stage5 | PLANNED / NOT STARTED |

## Central Progress Timeline

새 중앙 Step 번호는 이 운영 체계 도입부터 시작한다. 기존 S3-2 progress의 Step 번호와는 별개다.

### Step 1 — Project Workflow System Introduced

- **Date:** 2026-11-05 KST.
- **Status:** COMPLETED.
- **Purpose:** Codex와 Claude Code가 동일한 계획·실행·검증·진행 기록·계획 변경 반영 규칙을 사용하도록 운영 계층을 도입한다.
- **Plan Reference:** [EXECUTION_PLAN — WF-1](EXECUTION_PLAN.md#2-wf-1--project-workflow-system-introduction).
- **Files Changed:** `AGENTS.md`, `CLAUDE.md`, `docs/PROJECT_WORKFLOW.md`, `docs/EXECUTION_PLAN.md`, `docs/PROGRESS.md` 생성; `README.md` Documentation Index에 링크 세 개 추가.
- **Validation:** 작업 전 git 상태와 diff를 확인했다. 다섯 파일의 존재, 짧고 동일한 entry point, 링크/anchor, 계획·progress 필수 필드, 실제 연구 상태, 실험 평가 규칙, 중앙/상세 로그 관계, Markdown·UTF-8·공백을 검증했다. README diff는 링크 세 줄뿐이다. 기존 source/scripts/configs/tests/상세 docs와 MLflow DB의 content hash, 기존 output의 파일 크기/수정 시각 목록이 작업 전과 같음을 확인했다. 전체 `git diff --check`는 권한 제한 밖의 읽기 전용 검사에서 통과했다.
- **Result:** Project Workflow / Central Execution Plan / Central Progress Log **ACTIVE**. 중앙 계획·진행 기록 운영 체계를 마련했다. 모델·trainer·config·test·기존 실험 결과·상세 문서는 보존했고 S3-2 설계·구현·학습·MLflow run·checkpoint·commit/push를 수행하지 않았다.
- **Issues:** 기존 `.tmp_resume_tests_verified/` 아래 추적 파일의 삭제 표시와 권한 제한이 작업 전부터 있었다. 이번 작업과 분리해 보존했다. 최초 sandbox 내 diff 검사는 해당 파일의 접근 오류가 있었고, 권한 제한 밖의 읽기 전용 재검사는 통과했다. Entry point 동일성 검사의 첫 시도는 파일 끝 빈 줄 차이로 실패했고, 끝 개행을 통일한 뒤 재검증을 통과했다.
- **Decision:** 계획 변경 발생: 중앙 실행 계획 도입을 [초기 Plan Change Log](EXECUTION_PLAN.md#plan-change-log)에 기록했다. 기존 연구 계획/결과는 변경하지 않았다. 앞으로 현재 실행 계획 단계를 확인해 수행하고 각 주요 단계 후 중앙 progress를 갱신하며, S3-2 세부 작업은 기존 상세 progress에도 기록한다.
- **Next:** S3-2.1 Architecture Design — NOT STARTED. S3-2는 PLANNING, Implementation NOT STARTED, Training NOT RUN이다.


### Step 2 — S3-2 Architecture Design

- **Date:** 2026-11-05 KST.
- **Status:** COMPLETED / DESIGN ONLY.
- **Purpose:** R2/S3-1과 비교할 최소 learnable temporal weighting 설계를 확정한다.
- **Plan Reference:** [EXECUTION_PLAN — S3-2.1](EXECUTION_PLAN.md#s3-21--architecture-design).
- **Files Changed:** `docs/context_model/s3_2_attention_plan.md`, `docs/context_model/s3_2_attention_progress.md`, `docs/EXECUTION_PLAN.md`, `docs/PROGRESS.md`; README/Stage3 개요는 사전 기록한 문서 동기화 범위에서 상태/Next만 갱신했다.
- **Validation:** 실제 R2/S3-1/common LSTM/optimizer/trainer/config source를 읽고 shape·head·optimizer coverage를 확인했다. 표준 PyTorch Linear의 meta count와 기본 초기화 source만 확인했다. 문서 링크/anchor·shape/expected count·현재 상태·기록 보존·표·공백과 전체 `git diff --check`를 검증했다. Protected source/config/script/test·기존 상세 문서·MLflow DB의 content hash와 기존 output의 파일 크기/수정 시각 목록이 작업 전과 일치했다. S3-1 final metric 구간과 상세 Step 0도 HEAD와 일치한다. S3-2 구현 테스트는 NOT RUN이다.
- **Result:** **S3-2 ARCHITECTURE DESIGN COMPLETED.** 선택안은 **Direct Scalar Temporal Attention / Linear512→1, bias=False / temporal softmax / weighted sum**이다. 공통 scalar bias의 상쇄를 근거로 불필요한 parameter 한 개를 제외했다. 상세 shape/count/gradient/optimizer/diagnostic은 [S3-2 Progress Step 1](context_model/s3_2_attention_progress.md#step-1--s3-2-architecture-design)에 기록한다.
- **Issues:** 기존 `temporal_parameters()`에는 Attention이 자동 포함되지 않으므로 전용 iterator에 명시 포함하도록 설계했다. 기존 임시 테스트 파일의 접근 제한에 따른 diff 재검사도 상세 기록에 남겼다.
- **Decision:** Architecture 후보 중 최소 설계 변수를 가진 A를 선택한다. Bias와 초기화 등은 원래 미정 항목의 설계 결정이며 계획 이탈이 아니다. **계획 변경 발생:** README/Stage3의 상태 동기화 범위 추가만 작업 전에 [Plan Change Log](EXECUTION_PLAN.md#plan-change-log)에 기록했다. 구현·학습 범위는 확대하지 않았다.
- **Next:** **S3-2 Implementation — NOT STARTED**. Training NOT RUN.
