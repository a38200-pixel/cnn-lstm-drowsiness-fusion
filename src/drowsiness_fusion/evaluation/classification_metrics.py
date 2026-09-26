"""Drowsy를 positive class로 사용하는 fold 및 4-fold classification metric 모듈이다.

주요 입력은 binary target/prediction과 fold metric 목록이고 주요 출력은 accuracy,
precision, recall, F1 및 mean/std다. 모델 inference나 checkpoint 저장은 하지 않는다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean, pstdev

from .metrics import binary_classification_metrics

METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def calculate_classification_metrics(
    targets: Sequence[int], predictions: Sequence[int]
) -> dict[str, float | list[list[int]]]:
    """Drowsy(``1``)를 positive로 accuracy, precision, recall, F1을 반환한다."""

    metrics = binary_classification_metrics(targets, predictions, positive_label=1)
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["positive_precision"]),
        "recall": float(metrics["positive_recall"]),
        "f1": float(metrics["positive_f1"]),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def summarize_fold_metrics(
    fold_metrics: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Fold metric 목록에서 각 metric의 mean과 population std를 계산한다.

    원논문 aggregation 방식이 공개되지 않아 초기 구현은 네 fold 전체를 모집단으로 보는
    ``ddof=0`` 표준편차를 사용한다. 비어 있거나 NaN/Inf인 입력은 오류로 처리한다.
    """

    if not fold_metrics:
        raise ValueError("요약할 fold metric이 없습니다.")
    summary = {}
    for metric_name in METRIC_NAMES:
        values = [float(metrics[metric_name]) for metrics in fold_metrics]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{metric_name}에 NaN 또는 Inf가 있습니다.")
        summary[metric_name] = {"mean": mean(values), "std": pstdev(values)}
    return summary
