"""Deterministic 20-frame sampling과 anomaly 정책을 synthetic metadata로 검증한다.

실제 frame을 열거나 이미지를 추출하지 않으며 CNN 전처리도 수행하지 않는다.
"""

import pytest

from drowsiness_fusion.data.paper_sequence import (
    build_sequence_metadata,
    build_video_sequence,
    uniform_frame_indices,
)


def _row(video_id: str, frame_count: int, fps: float = 30.0) -> dict[str, object]:
    return {
        "video_id": video_id,
        "label": "drowsy",
        "video_path": rf"C:\dataset\{video_id}.mp4",
        "frame_count": frame_count,
        "fps": fps,
    }


def test_uniform_sampling_is_deterministic_and_has_twenty_unique_indices() -> None:
    first = uniform_frame_indices(300, 20)
    second = uniform_frame_indices(300, 20)

    assert first == second
    assert len(first) == 20
    assert len(set(first)) == 20
    assert first[0] == 0
    assert first[-1] == 299


def test_normal_video_creates_exactly_twenty_rows() -> None:
    rows, status = build_video_sequence(_row("d_1", 300), frames_per_video=20)

    assert len(rows) == 20
    assert [row["slot"] for row in rows] == list(range(20))
    assert status["status"] == "ok"
    assert status["duplicate_frame_indices"] is False


def test_short_video_keeps_twenty_slots_and_reports_duplicates() -> None:
    rows, status = build_video_sequence(_row("d_short", 3), frames_per_video=20)

    assert len(rows) == 20
    assert status["status"] == "short"
    assert status["duplicate_frame_indices"] is True


@pytest.mark.parametrize("frame_count,fps", [(0, 30.0), (100, 0.0), (-1, 30.0)])
def test_invalid_video_creates_no_fake_sequence_rows(frame_count: int, fps: float) -> None:
    rows, status = build_video_sequence(_row("d_invalid", frame_count, fps))

    assert rows == []
    assert status["status"] == "invalid"


def test_sequence_summary_reports_short_invalid_and_duplicate_videos() -> None:
    rows, summary = build_sequence_metadata(
        [_row("d_ok", 300), _row("d_short", 3), _row("d_invalid", 0)],
        frames_per_video=20,
    )

    assert len(rows) == 40
    assert summary["expected_sequence_rows"] == 60
    assert summary["actual_rows"] == 40
    assert summary["short_video_count"] == 1
    assert summary["invalid_video_count"] == 1
    assert summary["duplicate_frame_index_detected"] is True
