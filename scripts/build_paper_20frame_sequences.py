"""Metadata CSV에서 deterministic 20-frame sequence metadata를 만드는 스크립트다.

입력은 metadata CSV와 data config이며 출력은 sequence CSV와 anomaly summary JSON이다.
실제 frame/image 추출, resize, normalization, CNN feature extraction은 하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_fusion.data.paper_cv_split import read_metadata_csv  # noqa: E402
from drowsiness_fusion.data.paper_sequence import (  # noqa: E402
    build_sequence_metadata,
    write_sequence_outputs,
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
    """metadata에 sampling index를 붙이고 CSV 및 summary를 저장한다."""

    parser = argparse.ArgumentParser(description="SUST-DDD 20-frame sequence metadata 생성")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/paper_reconstruction/data.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_path = PROJECT_ROOT / config["metadata"]["output"]
    sequence_config = config["sequence"]
    if sequence_config.get("sampling") != "uniform":
        raise ValueError("현재 reconstruction assumption은 sampling=uniform입니다.")
    if sequence_config.get("deterministic") is not True:
        raise ValueError("sequence sampling은 deterministic=true여야 합니다.")

    metadata_rows = read_metadata_csv(metadata_path)
    rows, summary = build_sequence_metadata(
        metadata_rows, frames_per_video=int(sequence_config["frames_per_video"])
    )
    output_path = PROJECT_ROOT / sequence_config["output"]
    summary_path = PROJECT_ROOT / sequence_config["summary"]
    write_sequence_outputs(rows, summary, output_path, summary_path)
    print(f"20-frame sequence metadata 생성: {output_path}")
    print(f"sequence summary 생성: {summary_path}")


if __name__ == "__main__":
    main()
