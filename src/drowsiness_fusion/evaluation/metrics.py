"""Model-independent classification metrics."""

from __future__ import annotations

from collections.abc import Sequence


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_classification_metrics(
    targets: Sequence[int], predictions: Sequence[int], *, positive_label: int = 1
) -> dict[str, float | list[list[int]]]:
    """Calculate binary accuracy, macro F1, and positive-class metrics."""

    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must have the same non-zero length")
    if positive_label not in (0, 1):
        raise ValueError("positive_label must be 0 or 1")

    confusion = [[0, 0], [0, 0]]
    for target, prediction in zip(targets, predictions):
        if target not in (0, 1) or prediction not in (0, 1):
            raise ValueError("binary metrics accept only labels 0 and 1")
        confusion[target][prediction] += 1

    class_metrics: dict[int, tuple[float, float, float]] = {}
    for label in (0, 1):
        true_positive = confusion[label][label]
        false_positive = confusion[1 - label][label]
        false_negative = confusion[label][1 - label]
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        class_metrics[label] = (precision, recall, f1)

    correct = confusion[0][0] + confusion[1][1]
    positive_precision, positive_recall, positive_f1 = class_metrics[positive_label]
    return {
        "accuracy": correct / len(targets),
        "macro_f1": (class_metrics[0][2] + class_metrics[1][2]) / 2,
        "positive_precision": positive_precision,
        "positive_recall": positive_recall,
        "positive_f1": positive_f1,
        "confusion_matrix": confusion,
    }
