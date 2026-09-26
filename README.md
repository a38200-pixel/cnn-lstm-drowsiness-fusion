# CNN-LSTM Drowsiness Fusion

SUST-DDD 원논문의 VGG19+LSTM 및 VGG16+LSTM baseline을 재구성하고, 이후 행동 규칙과 fusion detection으로 확장하기 위한 프로젝트입니다.

현재 저장소에는 프로젝트 골격과 범용 데이터/재현성/평가 유틸리티만 있습니다. 기존 프로젝트의 split, 전처리 결과, feature, checkpoint, audit 산출물 및 성능 수치는 이 프로젝트의 baseline으로 사용하지 않습니다.

## 데이터 경로

Raw SUST-DDD 데이터는 저장소 밖에 두고 `SUST_DDD_ROOT` 환경 변수 또는 향후 스크립트의 명시적 인자로 지정합니다.

PowerShell 예시:

```powershell
$env:SUST_DDD_ROOT = "D:\datasets\SUST-DDD"
```

`data/metadata/`에는 재생성 가능한 메타데이터만, `data/splits/`에는 이 프로젝트에서 새로 생성한 split 정의만 둡니다. 두 디렉터리의 생성 산출물은 기본적으로 Git에서 제외됩니다.

## 프로젝트 단계

### Phase 1 — SUST-DDD Paper Reconstruction

- VGG19+LSTM
- VGG16+LSTM
- 4-fold cross validation

논문에서 확인되지 않은 조건은 [baseline_spec.md](docs/paper_reconstruction/baseline_spec.md)에 `UNKNOWN`으로 유지합니다.

### Phase 2 — Context Model Improvement

Paper reconstruction 결과를 기준선으로 삼아 context model을 개선합니다.

### Phase 3 — Behavior Rule Detection

- EAR
- MAR
- Head Pose

### Phase 4 — Fusion Detection

- Context prediction
- Behavior rule events
- Final fusion decision

## 구조

```text
configs/                     실험군별 설정
data/metadata/               재생성 가능한 비디오 메타데이터
data/splits/                 이 프로젝트에서 생성할 CV split
docs/paper_reconstruction/   논문 재구성 기준
docs/decisions/              설계 결정 기록
src/drowsiness_fusion/       Python 패키지
scripts/                     실행 진입점
tests/                       단위 테스트
outputs/                     로컬 실행 결과(Git 제외)
```

## 개발 설치

```powershell
python -m pip install -e ".[dev]"
```

현재 단계에서는 데이터 전처리, frame sequence 생성, CNN feature extraction 및 training을 실행하지 않습니다.
