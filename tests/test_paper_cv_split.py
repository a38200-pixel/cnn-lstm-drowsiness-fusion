"""Paper-informed stratified 4-fold의 분리와 coverage를 synthetic 행으로 검증한다.

실제 2,074개 metadata나 full dataset 산출물은 사용하거나 생성하지 않는다.
"""

from collections import Counter

from drowsiness_fusion.data.paper_cv_split import (
    make_stratified_folds,
    validate_fold_coverage,
)


def _synthetic_rows() -> list[dict[str, str]]:
    rows = []
    for label, prefix in (("drowsy", "d"), ("not_drowsy", "n")):
        for index in range(8):
            rows.append(
                {
                    "video_id": f"{prefix}_{index}",
                    "label": label,
                    "label_id": "1" if label == "drowsy" else "0",
                    "video_path": f"C:\\dataset\\{prefix}_{index}.mp4",
                }
            )
    return rows


def test_stratified_four_fold_is_reproducible_and_balanced() -> None:
    rows = _synthetic_rows()
    first = make_stratified_folds(rows, folds=4, seed=42)
    second = make_stratified_folds(rows, folds=4, seed=42)

    assert first == second
    for fold in first:
        assert Counter(row["label"] for row in fold["holdout"]) == {
            "drowsy": 2,
            "not_drowsy": 2,
        }


def test_train_holdout_are_disjoint_and_each_sample_is_holdout_once() -> None:
    rows = _synthetic_rows()
    folds = make_stratified_folds(rows, folds=4, seed=42)
    validation = validate_fold_coverage(rows, folds)

    assert validation["valid"] is True
    assert validation["holdout_coverage_exactly_once"] is True
    for fold in folds:
        train_ids = {row["video_id"] for row in fold["train"]}
        holdout_ids = {row["video_id"] for row in fold["holdout"]}
        assert train_ids.isdisjoint(holdout_ids)
