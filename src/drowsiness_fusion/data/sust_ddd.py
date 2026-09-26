"""SUST-DDD 경로 검색과 파일명 label 해석을 제공하는 공통 모듈이다.

주요 입력은 외부 raw data root와 영상 파일명이고, 주요 출력은 정렬된 영상 경로와
``drowsy``/``not_drowsy`` label이다. 데이터 복사, 영상 변환, frame 추출은 하지 않는다.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

VIDEO_EXTENSIONS = frozenset({".avi", ".mkv", ".mov", ".mp4"})
LABEL_DROWSY = "drowsy"
LABEL_NOT_DROWSY = "not_drowsy"
_VIDEO_ID_PATTERN = re.compile(r"^(?P<prefix>[dn])_(?P<identifier>.+)$", re.IGNORECASE)


def resolve_data_root(value: str | Path | None = None) -> Path:
    """명시한 경로 또는 ``SUST_DDD_ROOT``를 검증해 절대 경로로 반환한다."""

    raw_value = value if value is not None else os.getenv("SUST_DDD_ROOT")
    if raw_value is None or not str(raw_value).strip():
        raise ValueError("Set SUST_DDD_ROOT or pass an explicit SUST-DDD data root")
    root = Path(raw_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SUST-DDD data root does not exist: {root}")
    return root


def parse_label(filename_or_id: str | Path) -> str:
    """파일명/ID의 ``d_``/``n_`` prefix를 label로 바꾸며 그 외 값은 거부한다."""

    # 역슬래시를 먼저 정규화해 현재 OS와 무관하게 Windows 경로를 동일하게 해석한다.
    video_id = Path(str(filename_or_id).replace("\\", "/")).stem
    match = _VIDEO_ID_PATTERN.fullmatch(video_id)
    if match is None:
        raise ValueError(f"Unsupported SUST-DDD video name: {filename_or_id}")
    return LABEL_DROWSY if match.group("prefix").casefold() == "d" else LABEL_NOT_DROWSY


def discover_videos(data_root: str | Path) -> list[Path]:
    """data root 아래의 지원 영상 파일을 재귀 검색해 결정적 순서로 반환한다."""

    root = resolve_data_root(data_root)
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


@dataclass(frozen=True)
class VideoMetadata:
    """한 raw SUST-DDD 영상의 이식 가능한 container metadata다."""

    video_id: str
    relative_path: str
    extension: str
    label: str
    fps: float
    frame_count: int
    duration_seconds: float | None
    width: int
    height: int
    is_readable: bool

    def to_dict(self) -> dict[str, str | float | int | bool | None]:
        return asdict(self)


def probe_video(path: str | Path, data_root: str | Path) -> VideoMetadata:
    """영상을 변환하거나 sequence를 추출하지 않고 container metadata만 읽는다."""

    import cv2

    root = resolve_data_root(data_root)
    video_path = Path(path).expanduser().resolve()
    try:
        relative_path = video_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Video is outside the configured data root: {video_path}") from error

    capture = cv2.VideoCapture(str(video_path))
    try:
        readable = capture.isOpened()
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if readable else 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if readable else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if readable else 0
    finally:
        capture.release()

    duration = frame_count / fps if fps > 0 and frame_count >= 0 else None
    return VideoMetadata(
        video_id=video_path.stem,
        relative_path=relative_path,
        extension=video_path.suffix.casefold(),
        label=parse_label(video_path.name),
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        width=width,
        height=height,
        is_readable=readable,
    )


def iter_metadata(paths: Iterable[Path], data_root: str | Path) -> Iterable[VideoMetadata]:
    """입력 경로 순서대로 metadata를 반환하며 저장 방식은 호출자에게 맡긴다."""

    for path in paths:
        yield probe_video(path, data_root)
