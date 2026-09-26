"""SUST-DDD 원본 영상의 메타데이터 CSV를 만드는 모듈이다.

주요 입력은 읽기 전용 SUST-DDD raw root이며, 주요 출력은 영상별 메타데이터 행과
``sust_ddd_metadata.csv``이다. 영상 복사, 변환, 재인코딩, frame 추출은 하지 않는다.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .sust_ddd import LABEL_DROWSY, LABEL_NOT_DROWSY, parse_label

METADATA_COLUMNS = (
    "video_id",
    "label",
    "label_id",
    "video_path",
    "filename",
    "frame_count",
    "fps",
    "width",
    "height",
    "duration_sec",
)
EXPECTED_COUNTS = {
    "total": 2074,
    LABEL_DROWSY: 975,
    LABEL_NOT_DROWSY: 1099,
}
LABEL_IDS = {LABEL_NOT_DROWSY: 0, LABEL_DROWSY: 1}


def find_mp4_files(raw_root: str | Path) -> list[Path]:
    """raw root 아래의 MP4 파일을 결정적인 순서로 찾는다.

    ``raw_root``는 외부 데이터셋 경로다. 반환값은 대소문자를 구분하지 않고 찾은
    ``.mp4`` 경로 목록이다. 다른 확장자를 임의로 데이터셋에 포함하지 않는다.
    """

    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SUST-DDD raw root를 찾을 수 없습니다: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def read_video_metadata(video_path: str | Path) -> dict[str, str | int | float]:
    """한 MP4의 container 메타데이터를 읽는다.

    입력 영상은 읽기 전용으로 열며, 반환값은 CSV 한 행이다. 열리지 않는 영상도
    삭제하지 않고 수치 필드를 0으로 남겨 후속 sequence summary에서 드러나게 한다.
    """

    import cv2

    path = Path(video_path).expanduser().resolve()
    label = parse_label(path.name)
    capture = cv2.VideoCapture(str(path))
    try:
        if capture.isOpened():
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            frame_count = 0
            fps = 0.0
            width = 0
            height = 0
    finally:
        capture.release()

    duration_sec = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    return {
        "video_id": path.stem,
        "label": label,
        "label_id": LABEL_IDS[label],
        "video_path": str(path),
        "filename": path.name,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def validate_expected_universe(
    rows: Sequence[Mapping[str, object]],
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> dict[str, object]:
    """metadata universe가 논문 기준 기대 개수와 일치하는지 검증한다.

    입력은 metadata 행과 기대 count다. 실제 count와 오류 목록을 반환한다. 불일치한
    영상을 자동 제외하거나 label을 고치지 않는 것이 중요한 정책이다.
    """

    video_ids = [str(row["video_id"]) for row in rows]
    label_counts = Counter(str(row["label"]) for row in rows)
    errors: list[str] = []

    duplicate_count = len(video_ids) - len(set(video_ids))
    if duplicate_count:
        errors.append(f"중복 video_id가 {duplicate_count}개 있습니다.")
    if len(rows) != expected_counts["total"]:
        errors.append(
            f"전체 영상 수가 기대값과 다릅니다: actual={len(rows)}, "
            f"expected={expected_counts['total']}"
        )
    for label in (LABEL_DROWSY, LABEL_NOT_DROWSY):
        actual = label_counts[label]
        expected = expected_counts[label]
        if actual != expected:
            errors.append(f"{label} 수가 기대값과 다릅니다: actual={actual}, expected={expected}")

    return {
        "valid": not errors,
        "total": len(rows),
        "label_counts": dict(label_counts),
        "duplicate_video_id_count": duplicate_count,
        "errors": errors,
    }


def build_metadata_rows(video_paths: Iterable[Path]) -> list[dict[str, str | int | float]]:
    """발견된 영상 경로를 metadata 행으로 변환한다.

    경로 순서를 유지해 행을 만들고 video_id 중복 시 오류를 낸다. label 해석 실패도
    조용히 건너뛰지 않고 ``ValueError``로 전달한다.
    """

    rows = [read_video_metadata(path) for path in video_paths]
    video_ids = [str(row["video_id"]) for row in rows]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("중복 video_id가 있어 metadata를 만들 수 없습니다.")
    return rows


def write_metadata_csv(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> None:
    """검증된 metadata 행을 UTF-8 CSV로 저장한다.

    입력은 metadata 행과 출력 경로이며 반환값은 없다. 컬럼 순서를 고정하지만 데이터의
    label이나 수치를 수정하지 않는다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
