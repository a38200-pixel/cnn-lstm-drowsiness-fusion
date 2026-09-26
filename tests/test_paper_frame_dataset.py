"""Sequence ordering과 frame Dataset 출력을 synthetic CSV/mock loader로 검증한다.

실제 SUST-DDD 영상 decode, resize 대량 처리, VGG inference는 하지 않는다.
"""

import io

import torch

from drowsiness_fusion.data.paper_frame_dataset import (
    PaperFrameDataset,
    load_sequence_groups,
)


def _sequence_csv_text() -> str:
    lines = [
        "video_id,label,slot,target_frame_index,target_timestamp_sec,"
        "source_frame_index,source_timestamp_sec,video_path"
    ]
    for slot in reversed(range(20)):
        lines.append(
            f"d_1,drowsy,{slot},{slot * 10},{slot / 3},{slot * 10},{slot / 3},"
            "C:\\dataset\\drowsiness\\d_1.mp4"
        )
    return "\n".join(lines) + "\n"


def _mock_sequence_file(monkeypatch) -> None:
    monkeypatch.setattr("pathlib.Path.is_file", lambda _: True)
    monkeypatch.setattr(
        "pathlib.Path.open",
        lambda *args, **kwargs: io.StringIO(_sequence_csv_text()),
    )


def test_sequence_metadata_is_sorted_by_slot_and_keeps_windows_path(monkeypatch) -> None:
    _mock_sequence_file(monkeypatch)

    sequences = load_sequence_groups("sequence.csv")

    assert sequences[0]["source_frame_indices"] == [slot * 10 for slot in range(20)]
    assert sequences[0]["video_path"] == "C:\\dataset\\drowsiness\\d_1.mp4"
    assert sequences[0]["label_id"] == 1


def test_frame_dataset_returns_expected_shape_and_label(monkeypatch) -> None:
    _mock_sequence_file(monkeypatch)

    def mock_loader(video_path, indices, input_size, normalization):
        assert video_path.endswith("d_1.mp4")
        assert indices == [slot * 10 for slot in range(20)]
        return torch.zeros(20, 3, *input_size)

    dataset = PaperFrameDataset("sequence.csv", frame_loader=mock_loader)
    item = dataset[0]

    assert item["frames"].shape == (20, 3, 224, 224)
    assert item["label"] == 1
    assert item["video_id"] == "d_1"
