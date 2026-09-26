"""Cached VGG feature로 paper LSTM의 단일 fold를 학습하는 실행 스크립트다.

주요 입력은 baseline YAML과 fold 번호이며 주요 출력은 history, metric, checkpoint다.
CNN feature extraction, 별도 test split, 4-fold aggregation은 하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
    """요청한 fold 하나를 config 그대로 학습하고 holdout에서 평가한다."""

    parser = argparse.ArgumentParser(description="Paper reconstruction 단일 fold 학습")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    metrics = run_fold_training(load_config(args.config), args.fold, PROJECT_ROOT)
    print(f"fold {args.fold} 완료: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
