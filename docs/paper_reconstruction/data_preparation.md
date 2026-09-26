# Paper Reconstruction Data Preparation

## 이번 단계에서 구현한 내용

- SUST-DDD `d_*.mp4` / `n_*.mp4` label parsing과 metadata 생성
- 논문 기대 universe count 검증
- seed가 고정된 label-stratified 4-fold 생성과 coverage 검증
- deterministic uniform 20-frame sequence metadata 생성
- short/invalid video와 duplicate frame index summary
- synthetic/mock 기반 targeted test

모든 raw 영상 접근은 read-only다. 영상 복사, 변환, 재인코딩, frame 추출은 하지 않는다.

## 생성 파일

| 파일 | 역할 |
|---|---|
| `data/metadata/sust_ddd_metadata.csv` | 영상별 label, 경로, frame count, fps, 크기, duration |
| `data/splits/paper_reconstruction/fold_*_train.csv` | 해당 fold의 75% train 목록 |
| `data/splits/paper_reconstruction/fold_*_holdout.csv` | 해당 fold의 25% holdout 목록 |
| `data/splits/paper_reconstruction/split_summary.json` | 분포, seed, subject-wise 상태, coverage 검증 |
| `data/sequences/paper_reconstruction_20frame.csv` | 영상별 20개 frame index/timestamp |
| `data/sequences/paper_reconstruction_20frame_summary.json` | row 수와 sampling anomaly |

## Metadata 정책

- `d_*.mp4`는 `drowsy`, `label_id=1`
- `n_*.mp4`는 `not_drowsy`, `label_id=0`
- 기대값은 total 2,074, drowsy 975, not drowsy 1,099
- 기대값 불일치, 알 수 없는 prefix, 중복 video ID를 자동 수정하거나 제외하지 않는다.
- 읽히지 않는 영상의 container 수치는 0으로 유지해 후속 anomaly 검증에서 드러나게 한다.

## Fold 정책

원논문 확정값은 4-fold와 각 fold의 75% train / 25% holdout이다. 정확한 생성 알고리즘과 subject 정보는 공개되지 않았다.

재구성 구현은 video ID를 label별로 정렬하고 seed `42`로 shuffle한 뒤 round-robin으로 네 holdout에 배정한다. 따라서 각 video는 정확히 한 번 holdout에 포함되고 나머지 세 fold에서는 train에 포함된다. 이 방식은 `paper-informed stratified reconstruction`이다.

## Sequence sampling 정책

원논문은 영상당 20 frames와 224×224×3 입력만 확정한다. 정확한 sampling 방식은 `UNKNOWN`이다.

최초 reconstruction assumption은 다음과 같다.

1. frame index `0`부터 `frame_count - 1`까지 `np.linspace`로 20개 위치를 만든다.
2. `np.rint`로 가장 가까운 정수로 변환한다. 정확한 `.5`는 NumPy의 ties-to-even 정책을 따른다.
3. random sampling은 사용하지 않는다.
4. `frame_count < 20`이면 중복 index를 제거하지 않고 20 slot을 유지한다.
5. `frame_count <= 0` 또는 `fps <= 0`이면 가짜 sequence 행을 만들지 않고 invalid video로 summary에 기록한다.
6. 이 단계에서는 target/source index를 동일하게 기록하며 실제 frame decode는 하지 않는다.

## 사용자 실행

```powershell
.\.venv\Scripts\python.exe scripts\build_sust_ddd_metadata.py
.\.venv\Scripts\python.exe scripts\build_paper_cv_splits.py
.\.venv\Scripts\python.exe scripts\build_paper_20frame_sequences.py
```

각 단계는 앞 단계 파일이 없으면 오류를 출력한다. 다른 config를 쓰려면 각 명령에 `--config <path>`를 전달할 수 있다.

## 실제 생성 결과

- Videos: 2,074
- Drowsy: 975
- Not drowsy: 1,099
- Sequence rows: 41,480
- Rows per video: 20
- Short / invalid / duplicate index videos: 0 / 0 / 0
- 4-fold holdout coverage: valid, 각 video 정확히 1회 holdout

이 결과 파일은 사용자가 실행해 생성했으며 이번 모델 pipeline 작업에서는 수정하지 않았다.

## Codex가 실행하지 않은 장시간 작업

- frame/image 추출과 CNN 관련 작업
- 학습과 checkpoint 생성

## 상태와 다음 단계

- 상태: `DATA PREPARATION VERIFIED`
- 다음 단계: VGG19/VGG16 offline feature cache를 사용자 환경에서 생성한다.
