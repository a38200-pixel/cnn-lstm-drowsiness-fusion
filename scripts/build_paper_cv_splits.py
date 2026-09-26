"""Metadata CSV에서 paper-informed stratified 4-fold CSV를 만드는 스크립트다.

입력은 metadata CSV와 data config이며 출력은 8개 fold CSV 및 summary JSON이다.
subject를 추정하거나 별도 test split을 생성하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_fusion.data.paper_cv_split import (  # noqa: E402
    build_split_summary,
    make_stratified_folds,
    read_metadata_csv,
    write_fold_outputs,
)


def load_config(path: Path) -> dict:
    """YAML config를 읽어 dict로 반환하고 부재 또는 형식 오류를 명확히 알린다."""

    if not path.is_file():
        raise FileNotFoundError(f"config 파일이 없습니다: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config 최상위 값은 mapping이어야 합니다: {path}")
    return config


def main() -> None:
    """metadata를 4-fold로 나누고 coverage 검증을 통과한 경우에만 저장한다."""

    parser = argparse.ArgumentParser(description="SUST-DDD paper reconstruction 4-fold 생성")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/paper_reconstruction/data.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = PROJECT_ROOT / config["metadata"]["output"]
    cv_config = config["cross_validation"]
    if not cv_config.get("stratified", False):
        raise ValueError("paper reconstruction split은 stratified=true여야 합니다.")
    if cv_config.get("subject_wise") is not False:
        raise ValueError("subject 정보가 없어 subject_wise는 false여야 합니다.")

    rows = read_metadata_csv(metadata_path)
    fold_rows = make_stratified_folds(
        rows, folds=int(cv_config["folds"]), seed=int(cv_config["seed"])
    )
    summary = build_split_summary(rows, fold_rows, seed=int(cv_config["seed"]))
    if not summary["coverage_validation"]["valid"]:
        raise RuntimeError("fold coverage 검증에 실패하여 산출물을 저장하지 않습니다.")

    output_dir = PROJECT_ROOT / cv_config["output_dir"]
    write_fold_outputs(fold_rows, summary, output_dir)
    print(f"4-fold split 생성: {output_dir}")


if __name__ == "__main__":
    main()
