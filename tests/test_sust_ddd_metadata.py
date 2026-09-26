"""SUST-DDD label parsing, metadata 조회, universe 검증을 mock 입력으로 확인한다.

실제 데이터셋 scan, 영상 decode, metadata 전체 생성은 하지 않는다.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from drowsiness_fusion.data.sust_ddd import parse_label
from drowsiness_fusion.data.sust_ddd_metadata import (
    read_video_metadata,
    validate_expected_universe,
)


def test_parse_drowsy_and_not_drowsy_labels() -> None:
    assert parse_label("d_101.mp4") == "drowsy"
    assert parse_label("n_202.mp4") == "not_drowsy"


def test_parse_label_handles_windows_path() -> None:
    assert parse_label(Path(r"C:\dataset\drowsiness\d_7.mp4")) == "drowsy"


def test_parse_label_rejects_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_label("x_1.mp4")


def test_read_video_metadata_uses_container_values_without_decoding(monkeypatch) -> None:
    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return {1: 300, 2: 30.0, 3: 720, 4: 1280}[property_id]

        def release(self) -> None:
            pass

    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda _: FakeCapture(),
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_FPS=2,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    row = read_video_metadata(Path(r"C:\dataset\d_10.mp4"))

    assert row["label"] == "drowsy"
    assert row["label_id"] == 1
    assert row["frame_count"] == 300
    assert row["duration_sec"] == 10.0


def test_expected_universe_validation_reports_mismatch_without_fixing_rows() -> None:
    rows = [
        {"video_id": "d_1", "label": "drowsy"},
        {"video_id": "n_1", "label": "not_drowsy"},
    ]
    validation = validate_expected_universe(rows)

    assert validation["valid"] is False
    assert validation["total"] == 2
    assert len(rows) == 2
    assert validation["errors"]


def test_expected_universe_validation_accepts_supplied_fixture_counts() -> None:
    rows = [
        {"video_id": "d_1", "label": "drowsy"},
        {"video_id": "n_1", "label": "not_drowsy"},
    ]
    validation = validate_expected_universe(
        rows, {"total": 2, "drowsy": 1, "not_drowsy": 1}
    )
    assert validation["valid"] is True
