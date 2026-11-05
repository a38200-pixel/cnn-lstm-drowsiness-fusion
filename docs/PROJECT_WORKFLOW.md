# Project Working Rules

[확실] 도입일: 2026-11-05 KST. **Project Workflow ACTIVE.** 이 문서는 **프로젝트 전체 작업 방식의 source of truth**다. 현재 실행 계획은 [EXECUTION_PLAN](EXECUTION_PLAN.md), 실제 수행 기록의 중앙 index는 [PROGRESS](PROGRESS.md)다. 저장소 entry point는 [AGENTS.md](../AGENTS.md)와 [CLAUDE.md](../CLAUDE.md)이며 같은 공통 규칙을 사용한다.

## 1. 작업 시작 전

1. 사용하는 agent의 entry point인 AGENTS.md 또는 CLAUDE.md를 확인한다.
2. 이 PROJECT_WORKFLOW.md를 읽는다.
3. EXECUTION_PLAN.md를 읽는다.
4. PROGRESS.md의 최신 관련 기록과 연결된 상세 계획·진행 문서를 읽는다.
5. 현재 요청이 어느 실행 계획 단계에 해당하는지 확인한다.
6. `git status`와 `git diff --check`로 기존 변경 상태를 확인한다. 현재 작업과 무관한 uncommitted change를 임의 수정·삭제·복원하지 않는다.

계획에 없는 새 작업이면 바로 구현하지 않는다. 사용자 요청의 범위에 맞춰 중앙 실행 계획에 새 단계 또는 계획 변경을 먼저 기록한 뒤 수행한다. 완료된 과거 작업을 다시 PLANNED로 만들지 않는다.

## 2. Execution Plan 우선

EXECUTION_PLAN.md는 현재 프로젝트 실행 계획의 기준이다. 실험별 상세 계획은 연구 질문·통제 조건·평가 기준을 보완한다. 서로 다른 상태나 조건을 발견하면 근거를 확인해 계획을 먼저 맞춘다. 계획을 조용히 바꾸거나 상세 실험 조건을 암묵적으로 덮어쓰지 않는다.

모든 주요 실행 단계는 최소 다음을 포함한다.

- 목적(Purpose), 현재 상태(Status), 작업 범위(Scope).
- 변경 대상(Change Targets), 변경하지 않을 범위(Out of Scope).
- 실행 순서(Execution Order), 검증 방법(Validation).
- 완료 기준(Completion Criteria), 예상 산출물(Expected Artifacts), Next.

아직 구체화되지 않은 미래 단계는 PLANNED / NOT STARTED로 두고, 실제 실행 전 위 항목을 채운다. 예상 산출물을 실제 생성된 결과처럼 쓰지 않는다.

## 3. 한 단계씩 진행

각 주요 단계는 **계획 재확인 → 실행 → 검증 → 진행 기록 → 계획 변경/상태 반영 → git diff 확인 → Next 확인** 순서로 수행한다. 현재 단계만 수행하며, 요청 범위 밖의 여러 주요 단계를 임의로 이어서 실행하지 않는다. Next를 적었다는 이유로 다음 구현·학습까지 자동 실행하지 않는다.

단계가 끝나면 검증 결과를 기록하고 완료 기준을 충족했을 때만 COMPLETED로 표시한다. 단순 실행 명령 발행이나 프로세스 시작을 완료로 취급하지 않는다.

## 4. 중앙 Progress와 상세 기록

모든 의미 있는 작업 단계 후에는 PROGRESS.md를 반드시 갱신한다. 중앙 로그에는 요약·상태·근거·상세 기록 링크를 남기고, 실험별 상세 로그에는 실제 변경·검증·오류·metric/output 경로를 남긴다. 기존 개발 이력과 상세 문서를 삭제하거나 대체하지 않는다.

S3-2 작업에서는 다음 두 문서를 모두 필요한 범위에서 갱신한다.

- [중앙 PROGRESS](PROGRESS.md): 작업 요약 + 상세 entry 링크.
- [S3-2 progress](context_model/s3_2_attention_progress.md): 구조·실험의 세부 이력.

같은 내용을 두 문서에 장문으로 복사하지 않는다. S3-2 외 작업도 관련 상세 로그가 있으면 같은 원칙으로 연결한다. 기록을 나중에 한꺼번에 몰아서 작성하지 않는다.

```text
### Step N — <단계명>
Date: 실제 날짜, 가능하면 KST 시각과 근거
Status: PLANNED / IN PROGRESS / COMPLETED / BLOCKED / INTERRUPTED / RESUMED 등
Purpose: 왜 수행했는가
Plan Reference: EXECUTION_PLAN 단계 및 관련 상세 계획 링크
Files Changed: 실제 변경 파일; 없으면 NONE
Validation: 실제 수행한 검증과 결과; 미수행이면 NOT RUN
Result: 확인된 결과; 아직 없으면 NOT AVAILABLE
Issues: 발생한 문제; 없으면 NONE, 원인 미확인이면 UNKNOWN
Decision: 결과에 근거한 결정과 미결정 사항
Next: 다음 단계와 아직 시작하지 않은 작업
```

FAILED 등 실제 상황을 더 잘 표현하는 상태도 사용할 수 있다. 진행 상태와 실험 결과 유무를 구분한다. BLOCKED/INTERRUPTED/RESUMED는 발생 근거와 범위를 함께 기록하며 성공 후 이전 기록을 지우지 않는다.

## 5. 실패·중단·Resume 기록

Training interruption, CUDA/DataLoader/MLflow 오류, checkpoint resume, config correction을 실제 발생 순서대로 남긴다. 실패한 시도가 나중에 성공했더라도 삭제하지 않는다.

학습 중단·재개에는 최소 다음을 확인해 기록한다.

- 어느 fold의 마지막 완료 epoch인지, 진행 중이던 epoch와 미저장 작업은 무엇인지.
- 확인된 원인 또는 UNKNOWN, checkpoint 존재 여부와 actual 경로.
- 복원 가능 여부와 검증 근거, 실제 resume start_epoch와 새 output 경로.
- MLflow 상태와 실제 학습 완료·프로세스 생존 여부의 차이.

아직 실행하지 않은 resume을 RESUMED로 기록하지 않는다. 완료된 최종 resume output만 집계하고 이전 부분 실행을 중복 fold로 세지 않는다. Resume 완료를 무중단 완주나 bitwise 동일성으로 표현하지 않는다.

## 6. 계획 변경 규칙

기존 계획을 바꿔야 한다면 **변경 이유 분석 → EXECUTION_PLAN 수정 및 Plan Change Log 기록 → 변경된 접근법 수행 → 검증 → PROGRESS에 계획 변경 발생 기록** 순서를 따른다. 변경 작업을 수행한 뒤 계획을 소급해서 맞추지 않는다.

계획 변경 기록은 최소 다음을 포함한다.

```text
### YYYY-MM-DD — <변경 제목>
Previous Plan: 이전 계획
Reason for Change / Reason: 변경 이유와 근거
Updated Plan: 변경 후 계획
Affected Steps: 영향을 받는 단계
Expected Impact / Impact: 비교 조건·범위·비용·산출물에 대한 예상 영향
```

실험별 계획도 변경이 필요하면 중앙 계획과 함께 일치시킨다. 단순한 실행 상태 갱신과 조건·접근법 변경을 구분하며, 완료된 과거 기록을 삭제하거나 새 계획으로 덮어쓰지 않는다.

## 7. 실험 결과와 평가 원칙

실행하지 않은 결과·parameter count·학습 성능을 생성하지 않는다. 미시작은 NOT STARTED, 학습 미실행은 NOT RUN, 결과 없음은 NOT AVAILABLE로 표시한다. 실제 상황에 따라 IN PROGRESS / COMPLETED / BLOCKED 등을 사용한다.

| 구분 | 표시 / 기본 원칙 |
|---|---|
| 실행 중간 수치 | INTERIM / NOT OFFICIAL |
| 네 fold 미완료 평균 | INTERIM / NOT OFFICIAL AGGREGATE |
| 네 fold final 결과와 검증 완료 | OFFICIAL 4-FOLD RESULT |
| Official epoch | 사전에 정한 fixed final epoch; 현재 R2/S3-1과 S3-2 계획은 epoch30 |
| Held-out test best epoch | POST-HOC DIAGNOSTIC ONLY; official selection에 사용하지 않음 |
| 4-fold aggregate | Raw metric의 equal fold mean |
| Std | Population std, ddof=0 |

특정 실험 계획에서 다른 평가 방식을 사전에 정의했다면 그 계획을 따르고 차이를 명시한다. 결과를 본 뒤 조용히 정책을 바꾸거나 best epoch를 official로 승격하지 않는다. Mean/std의 단위, raw 값과 표시 반올림 값, pooled metric과 fold mean을 구분한다.

기존 검증된 결과를 문서 정리만으로 수정하지 않는다. 수치 불일치가 있으면 원본 metric/history를 재확인·재계산하고 근거와 수정 이유를 기록한다.

## 8. 실험 통제

새 실험은 가능한 한 한 번에 하나의 주요 조건만 변경한다. 비교 시 dataset, fold split, preprocessing, sequence length, seed, evaluation policy, training policy를 유지하고 변경 항목은 문서화한다.

Architecture 변경 때문에 parameter count·optimizer group·LR·비용이 달라지면 이유와 비교 영향을 먼저 계획에 남긴다. 결과를 근거로 성능이나 일반화를 과장하지 않으며 동일 test 재사용·single seed·resume 등의 한계를 해당 실험 문서에 유지한다.

## 9. 구현 스타일

사람이 읽기 쉬운 구조를 우선한다. 불필요한 factory·abstraction·orchestration·숨겨진 동작을 만들지 않는다. 학습 script는 가능한 한 다음 흐름이 눈에 보이게 작성한다.

```text
config → device → dataset/dataloader → model → optimizer → MLflow
→ epoch loop → evaluation → saving
```

재사용성이 명확한 코드만 src로 분리한다. 이 규칙 도입을 이유로 기존 코드를 범위 밖에서 일괄 재구성하지 않는다.

## 10. 코드 주석

향후 새 Python 파일에는 한국어 docstring/comment로 역할·입력·출력·하지 않는 일을 명시한다. 주요 함수/클래스는 무엇을 하는지 설명하고, 구현 선택의 이유가 필요한 경우 왜 그렇게 하는지도 적는다. 모델명·tensor shape·metric 이름 등 필요한 기술 용어는 영어를 유지할 수 있다.

## 11. 코드·문서 일치

코드 또는 실험 상태가 바뀌면 관련 progress와 계획의 현재 상태를 함께 맞춘다. 실행 전 계획과 실제 결과를 구분하고 기존 상세 이력은 보존한다.

README는 프로젝트 개요·현재 상태·최종/중요 결과·상세 문서 링크 역할을 유지한다. 운영 규칙을 README에 장문으로 복제하지 않고 중앙 문서 링크로 연결한다.

## 12. 작업 종료

종료 전 현재 계획 단계, 범위 초과 수정, 수행한 테스트/검증, PROGRESS 갱신, 계획 변경 시 EXECUTION_PLAN 반영, `git diff --check` 결과를 확인한다. 검증 실패나 접근 권한 제한이 있으면 확인 가능한 범위와 남은 제약을 구분하며 통과한 것으로 쓰지 않는다. 기존 uncommitted change는 현재 작업 변경과 구분한다.

완료 보고에는 **수행 단계 / 수정 파일 / 검증 / progress 업데이트 / execution plan 변경 여부 / 현재 상태 / Next**를 포함한다. 명시된 작업 범위를 넘어 commit/push·구현·학습을 자동 실행하지 않는다.
