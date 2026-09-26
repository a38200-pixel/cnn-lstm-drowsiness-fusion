"""Drowsy-positive fold metric과 4-fold mean/std 계산을 synthetic label로 검증한다.

실제 model inference, fold training, 성능 결과 생성은 하지 않는다.
"""

from drowsiness_fusion.evaluation.classification_metrics import (
    calculate_classification_metrics,
    summarize_fold_metrics,
)


def test_drowsy_is_positive_class() -> None:
    metrics = calculate_classification_metrics(
        targets=[0, 0, 1, 1], predictions=[0, 1, 1, 1]
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.8


def test_fold_summary_uses_population_standard_deviation() -> None:
    folds = [
        {"accuracy": value, "precision": value, "recall": value, "f1": value}
        for value in (0.6, 0.7, 0.8, 0.9)
    ]
    summary = summarize_fold_metrics(folds)

    assert summary["accuracy"]["mean"] == 0.75
    assert round(summary["accuracy"]["std"], 6) == 0.111803
