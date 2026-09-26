"""원논문 재구성용 20-frame sequence metadata를 만드는 모듈이다.

주요 입력은 영상 metadata 행이며, 주요 출력은 영상별 20개 sampling 행과 anomaly
summary다. 실제 frame/image 추출, resize, normalization, CNN 처리는 하지 않는다.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SEQUENCE_COLUMNS = (
    "video_id",
    "label",
    "slot",
    "target_frame_index",
    "target_timestamp_sec",
    "source_frame_index",
    "source_timestamp_sec",
    "video_path",
)


def uniform_frame_indices(frame_count: int, frames_per_video: int = 20) -> list[int]:
    """첫 frame부터 마지막 frame까지 균등한 정수 index를 반환한다.

    ``np.linspace``로 양 끝점을 포함한 실수 위치를 만든 뒤 ``np.rint``의 가장 가까운
    정수(정확한 .5는 ties-to-even)로 변환한다. 짧은 영상의 중복 index는 제거하지 않아
    항상 요청한 slot 수를 유지하고, 중복 여부는 summary에서 anomaly로 기록한다.
    """

    if frame_count <= 0:
        raise ValueError("frame_count는 1 이상이어야 합니다.")
    if frames_per_video <= 0:
        raise ValueError("frames_per_video는 1 이상이어야 합니다.")
    positions = np.linspace(0, frame_count - 1, num=frames_per_video, endpoint=True)
    return np.rint(positions).astype(np.int64).tolist()


def build_video_sequence(
    metadata_row: Mapping[str, object], frames_per_video: int = 20
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """한 영상의 sequence 행과 상태를 만든다.

    fps 또는 frame_count가 유효하지 않으면 잘못된 index를 만들지 않고 0행과
    ``invalid`` 상태를 반환한다. 짧은 영상은 제외하지 않고 중복 index를 포함한 20행을
    반환해 downstream에서 정책을 명시적으로 선택할 수 있게 한다.
    """

    video_id = str(metadata_row.get("video_id", ""))
    label = str(metadata_row.get("label", ""))
    video_path = str(metadata_row.get("video_path", ""))
    try:
        frame_count = int(metadata_row.get("frame_count", 0))
        fps = float(metadata_row.get("fps", 0))
    except (TypeError, ValueError):
        frame_count, fps = 0, 0.0

    if not video_id or not video_path or frame_count <= 0 or fps <= 0:
        return [], {
            "video_id": video_id,
            "status": "invalid",
            "frame_count": frame_count,
            "duplicate_frame_indices": False,
        }

    indices = uniform_frame_indices(frame_count, frames_per_video)
    has_duplicates = len(indices) != len(set(indices))
    rows = []
    for slot, frame_index in enumerate(indices):
        timestamp = frame_index / fps
        rows.append(
            {
                "video_id": video_id,
                "label": label,
                "slot": slot,
                "target_frame_index": frame_index,
                "target_timestamp_sec": timestamp,
                "source_frame_index": frame_index,
                "source_timestamp_sec": timestamp,
                "video_path": video_path,
            }
        )
    return rows, {
        "video_id": video_id,
        "status": "short" if frame_count < frames_per_video else "ok",
        "frame_count": frame_count,
        "duplicate_frame_indices": has_duplicates,
    }


def build_sequence_metadata(
    metadata_rows: Sequence[Mapping[str, object]], frames_per_video: int = 20
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """전체 metadata 행을 sequence 행과 실행 summary로 변환한다.

    영상 순서를 유지한다. invalid 영상과 short 영상은 ID 목록과 count로 명시하며
    실제 frame을 열거나 추출하지 않는다.
    """

    sequence_rows: list[dict[str, object]] = []
    statuses = []
    frame_counts = []
    for metadata_row in metadata_rows:
        rows, status = build_video_sequence(metadata_row, frames_per_video)
        sequence_rows.extend(rows)
        statuses.append(status)
        if status["frame_count"] >= 0:
            frame_counts.append(int(status["frame_count"]))

    invalid_ids = [status["video_id"] for status in statuses if status["status"] == "invalid"]
    short_ids = [status["video_id"] for status in statuses if status["status"] == "short"]
    duplicate_ids = [
        status["video_id"] for status in statuses if status["duplicate_frame_indices"]
    ]
    summary = {
        "video_count": len(metadata_rows),
        "expected_sequence_rows": len(metadata_rows) * frames_per_video,
        "actual_rows": len(sequence_rows),
        "duplicate_frame_index_detected": bool(duplicate_ids),
        "duplicate_frame_index_video_count": len(duplicate_ids),
        "duplicate_frame_index_video_ids": duplicate_ids,
        "short_video_count": len(short_ids),
        "short_video_ids": short_ids,
        "invalid_video_count": len(invalid_ids),
        "invalid_video_ids": invalid_ids,
        "min_frame_count": min(frame_counts) if frame_counts else None,
        "max_frame_count": max(frame_counts) if frame_counts else None,
        "frames_per_video": frames_per_video,
        "sampling_policy": "uniform endpoints with np.linspace; np.rint ties-to-even",
        "deterministic": True,
        "reconstruction_assumption": (
            "원논문은 영상당 20 frames만 명시하며 정확한 sampling 방식은 공개하지 않았다."
        ),
    }
    return sequence_rows, summary


def write_sequence_outputs(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    csv_path: str | Path,
    summary_path: str | Path,
) -> None:
    """sequence 행과 summary를 지정 경로의 CSV 및 JSON으로 저장한다.

    반환값은 없으며 저장 중 frame을 열거나 sampling 결과를 다시 계산하지 않는다.
    """

    sequence_path = Path(csv_path)
    sequence_summary_path = Path(summary_path)
    sequence_path.parent.mkdir(parents=True, exist_ok=True)
    sequence_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with sequence_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SEQUENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    sequence_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
