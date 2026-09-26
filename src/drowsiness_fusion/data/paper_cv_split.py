"""Paper-informed stratified 4-fold split을 생성하는 모듈이다.

주요 입력은 SUST-DDD metadata 행, fold 수, seed이며, 주요 출력은 fold별 train/holdout
CSV와 검증 summary다. subject 정보를 추정하거나 별도 test split을 만들지 않는다.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from .sust_ddd_metadata import LABEL_IDS, METADATA_COLUMNS

SPLIT_COLUMNS = ("video_id", "label", "label_id", "video_path")


def read_metadata_csv(path: str | Path) -> list[dict[str, str]]:
    """metadata CSV를 읽고 split에 필요한 컬럼과 video_id 유일성을 검사한다.

    입력은 metadata CSV 경로이고 반환값은 행 목록이다. 앞 단계 파일이 없거나 schema가
    다르면 후속 산출물을 만들지 않고 명확한 오류를 낸다.
    """

    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"metadata CSV가 없습니다. 먼저 metadata script를 실행하세요: {metadata_path}"
        )
    with metadata_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        required = set(METADATA_COLUMNS)
        if not required.issubset(columns):
            missing = sorted(required - columns)
            raise ValueError(f"metadata CSV 필수 컬럼이 없습니다: {missing}")
        rows = list(reader)

    video_ids = [row["video_id"] for row in rows]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("metadata CSV에 중복 video_id가 있습니다.")
    if not rows:
        raise ValueError("metadata CSV가 비어 있습니다.")
    return rows


def make_stratified_folds(
    rows: Sequence[Mapping[str, str]], folds: int = 4, seed: int = 42
) -> list[dict[str, list[dict[str, str]]]]:
    """label 비율을 유지하는 재현 가능한 K-fold train/holdout을 만든다.

    label별로 video_id 정렬 후 seed 기반 shuffle과 round-robin 배정을 사용한다. 논문에
    정확한 알고리즘이 없어 이 정책을 reconstruction assumption으로 명시한다.
    """

    if folds < 2:
        raise ValueError("folds는 2 이상이어야 합니다.")
    if len(rows) < folds:
        raise ValueError("sample 수가 folds보다 작습니다.")

    rows_by_label: dict[str, list[dict[str, str]]] = {}
    seen_ids: set[str] = set()
    for source_row in rows:
        row = dict(source_row)
        video_id = row.get("video_id", "")
        label = row.get("label", "")
        if not video_id or video_id in seen_ids:
            raise ValueError(f"비어 있거나 중복된 video_id입니다: {video_id!r}")
        if label not in LABEL_IDS:
            raise ValueError(f"지원하지 않는 label입니다: {label!r}")
        seen_ids.add(video_id)
        rows_by_label.setdefault(label, []).append(row)

    holdout_ids_by_fold = [set() for _ in range(folds)]
    rng = random.Random(seed)
    for label in sorted(rows_by_label):
        label_rows = sorted(rows_by_label[label], key=lambda row: row["video_id"])
        rng.shuffle(label_rows)
        for index, row in enumerate(label_rows):
            holdout_ids_by_fold[index % folds].add(row["video_id"])

    ordered_rows = sorted((dict(row) for row in rows), key=lambda row: row["video_id"])
    result = []
    for holdout_ids in holdout_ids_by_fold:
        holdout = [row for row in ordered_rows if row["video_id"] in holdout_ids]
        train = [row for row in ordered_rows if row["video_id"] not in holdout_ids]
        result.append({"train": train, "holdout": holdout})
    return result


def validate_fold_coverage(
    source_rows: Sequence[Mapping[str, str]],
    fold_rows: Sequence[Mapping[str, Sequence[Mapping[str, str]]]],
) -> dict[str, object]:
    """fold 간 분리와 전체 holdout coverage를 검증한다.

    반환 summary에는 각 fold의 overlap과 각 video가 정확히 한 번 holdout인지 포함한다.
    실패를 자동 보정하지 않으며 ``valid``를 false로 기록한다.
    """

    universe = {row["video_id"] for row in source_rows}
    holdout_counts: Counter[str] = Counter()
    fold_checks = []
    for fold_number, fold in enumerate(fold_rows, start=1):
        train_ids = {row["video_id"] for row in fold["train"]}
        holdout_ids = {row["video_id"] for row in fold["holdout"]}
        holdout_counts.update(holdout_ids)
        fold_checks.append(
            {
                "fold": fold_number,
                "train_holdout_disjoint": train_ids.isdisjoint(holdout_ids),
                "covers_universe": train_ids | holdout_ids == universe,
                "train_count": len(train_ids),
                "holdout_count": len(holdout_ids),
            }
        )

    coverage_exact = set(holdout_counts) == universe and all(
        holdout_counts[video_id] == 1 for video_id in universe
    )
    valid = coverage_exact and all(
        check["train_holdout_disjoint"] and check["covers_universe"] for check in fold_checks
    )
    return {
        "valid": valid,
        "holdout_coverage_exactly_once": coverage_exact,
        "fold_checks": fold_checks,
    }


def _label_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts = Counter(row["label"] for row in rows)
    return {label: counts.get(label, 0) for label in LABEL_IDS}


def build_split_summary(
    source_rows: Sequence[Mapping[str, str]],
    fold_rows: Sequence[Mapping[str, Sequence[Mapping[str, str]]]],
    seed: int,
) -> dict[str, object]:
    """원본 행, fold 행, seed를 받아 분포와 coverage가 포함된 summary를 반환한다.

    summary는 JSON 직렬화 가능한 값만 포함하며 subject-wise를 사용할 수 없다는 상태와
    이 구현이 paper-informed reconstruction이라는 점을 명시한다.
    """

    validation = validate_fold_coverage(source_rows, fold_rows)
    folds = []
    for fold_number, fold in enumerate(fold_rows, start=1):
        folds.append(
            {
                "fold": fold_number,
                "train_count": len(fold["train"]),
                "holdout_count": len(fold["holdout"]),
                "train_label_counts": _label_counts(fold["train"]),
                "holdout_label_counts": _label_counts(fold["holdout"]),
            }
        )
    return {
        "method": "paper-informed stratified reconstruction",
        "total_count": len(source_rows),
        "label_counts": _label_counts(source_rows),
        "fold_count": len(fold_rows),
        "seed": seed,
        "stratified": True,
        "subject_wise": False,
        "subject_wise_status": "unavailable",
        "folds": folds,
        "coverage_validation": validation,
    }


def write_fold_outputs(
    fold_rows: Sequence[Mapping[str, Sequence[Mapping[str, str]]]],
    summary: Mapping[str, object],
    output_dir: str | Path,
) -> None:
    """fold 행과 summary를 받아 8개 CSV와 ``split_summary.json``을 저장한다.

    반환값은 없으며, label이나 fold membership을 저장 단계에서 변경하지 않는다.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for fold_number, fold in enumerate(fold_rows, start=1):
        for split_name in ("train", "holdout"):
            path = destination / f"fold_{fold_number}_{split_name}.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=SPLIT_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(fold[split_name])
    (destination / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
