"""SUST-DDD raw MP4를 읽어 metadata CSV를 만드는 실행 스크립트다.

입력은 ``configs/paper_reconstruction/data.yaml``의 raw root이고 출력은 metadata CSV다.
영상을 복사·변환·재인코딩하거나 frame을 추출하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_fusion.data.sust_ddd_metadata import (  # noqa: E402
    build_metadata_rows,
    find_mp4_files,
    validate_expected_universe,
    write_metadata_csv,
)


def load_config(path: Path) -> dict:
    """YAML config를 읽어 dict로 반환하며 파일 부재와 잘못된 형식을 즉시 알린다."""

    if not path.is_file():
        raise FileNotFoundError(f"config 파일이 없습니다: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config 최상위 값은 mapping이어야 합니다: {path}")
    return config


def main() -> None:
    """전체 MP4 metadata를 만들고 논문 기준 universe 검증 후 CSV를 저장한다."""

    parser = argparse.ArgumentParser(description="SUST-DDD metadata CSV 생성")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/paper_reconstruction/data.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_root = Path(config["dataset"]["raw_root"])
    output_path = PROJECT_ROOT / config["metadata"]["output"]

    video_paths = find_mp4_files(raw_root)
    rows = build_metadata_rows(video_paths)
    validation = validate_expected_universe(rows)
    if not validation["valid"]:
        details = "\n".join(f"- {error}" for error in validation["errors"])
        raise RuntimeError(
            "SUST-DDD universe가 논문 기준 기대값과 다릅니다. "
            "자동 수정하거나 제외하지 않았습니다.\n" + details
        )

    write_metadata_csv(rows, output_path)
    print(f"metadata 생성: {output_path}")


if __name__ == "__main__":
    main()
