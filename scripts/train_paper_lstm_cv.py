"""Cached VGG feature로 네 fold를 순차 학습하고 metric을 집계하는 스크립트다.

주요 입력은 baseline YAML이며 주요 출력은 fold별 산출물과 CV mean/std JSON이다.
CNN inference, 별도 test split, 여러 실험 조합 탐색은 하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_fusion.evaluation.classification_metrics import (  # noqa: E402
    summarize_fold_metrics,
)
from drowsiness_fusion.training.paper_trainer import run_fold_training  # noqa: E402


def load_config(path: Path) -> dict:
    """Baseline YAML을 읽어 mapping으로 반환하며 파일/형식 오류를 즉시 알린다."""

    if not path.is_file():
        raise FileNotFoundError(f"config 파일이 없습니다: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config 최상위 값은 mapping이어야 합니다: {path}")
    return config


def main() -> None:
    """Fold 1부터 4까지 학습하고 accuracy/precision/recall/F1 mean/std를 저장한다."""

    parser = argparse.ArgumentParser(description="Paper reconstruction 4-fold 학습")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    fold_metrics = []
    for fold in range(1, int(config["cross_validation"]["folds"]) + 1):
        fold_metrics.append(run_fold_training(config, fold, PROJECT_ROOT))
    summary = {
        "fold_metrics": fold_metrics,
        "aggregation": "mean and population std (ddof=0)",
        "summary": summarize_fold_metrics(fold_metrics),
    }
    output_root = PROJECT_ROOT / config["output"]["root"]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "cv_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"4-fold summary 생성: {summary_path}")


if __name__ == "__main__":
    main()
