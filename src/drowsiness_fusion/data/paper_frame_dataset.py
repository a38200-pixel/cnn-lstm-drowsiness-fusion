"""20-frame sequence metadata를 실제 RGB frame tensor로 읽는 Dataset 모듈이다.

주요 입력은 sequence CSV와 video ID 목록이고, 주요 출력은 ``[20, 3, 224, 224]``
frame tensor, 정수 label, video ID다. augmentation, face crop, feature extraction은 하지 않는다.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .sust_ddd_metadata import LABEL_IDS

SEQUENCE_COLUMNS = {
    "video_id",
    "label",
    "slot",
    "source_frame_index",
    "video_path",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameDecodeError(RuntimeError):
    """요청한 원본 frame을 정확히 읽지 못했을 때 발생하는 오류다."""


def load_sequence_groups(
    sequence_metadata: str | Path,
    sequence_length: int = 20,
) -> list[dict[str, object]]:
    """Sequence CSV를 읽어 video별 slot 순서로 묶는다.

    입력은 CSV 경로와 기대 길이다. 반환값의 각 항목에는 video ID, label, video path,
    정렬된 source frame index가 있다. 누락/중복 slot은 임의 보정하지 않고 오류로 처리한다.
    """

    path = Path(sequence_metadata)
    if not path.is_file():
        raise FileNotFoundError(f"20-frame sequence metadata가 없습니다: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = SEQUENCE_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"sequence metadata 필수 컬럼이 없습니다: {sorted(missing)}")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            grouped[row["video_id"]].append(row)

    sequences = []
    for video_id in sorted(grouped):
        rows = sorted(grouped[video_id], key=lambda row: int(row["slot"]))
        slots = [int(row["slot"]) for row in rows]
        if slots != list(range(sequence_length)):
            raise ValueError(
                f"{video_id}의 slot은 0..{sequence_length - 1}이어야 합니다: {slots}"
            )
        labels = {row["label"] for row in rows}
        video_paths = {row["video_path"] for row in rows}
        if len(labels) != 1 or len(video_paths) != 1:
            raise ValueError(f"{video_id}의 label 또는 video_path가 slot 사이에서 다릅니다.")
        label = next(iter(labels))
        if label not in LABEL_IDS:
            raise ValueError(f"{video_id}의 label을 지원하지 않습니다: {label}")
        sequences.append(
            {
                "video_id": video_id,
                "label": label,
                "label_id": LABEL_IDS[label],
                "video_path": next(iter(video_paths)),
                "source_frame_indices": [int(row["source_frame_index"]) for row in rows],
            }
        )
    return sequences


def preprocess_rgb_frame(
    rgb_frame: np.ndarray,
    input_size: tuple[int, int] = (224, 224),
    normalization: str = "imagenet",
) -> torch.Tensor:
    """RGB uint8 frame을 결정적으로 resize하고 ``[3, H, W]`` tensor로 바꾼다.

    bilinear resize와 0..1 변환을 사용한다. ``normalization=imagenet``이면 torchvision
    pretrained VGG의 표준 mean/std를 적용한다. augmentation은 적용하지 않는다.
    """

    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError(f"RGB frame shape가 올바르지 않습니다: {rgb_frame.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb_frame)).permute(2, 0, 1).float() / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0), size=input_size, mode="bilinear", align_corners=False
    ).squeeze(0)
    if normalization == "imagenet":
        mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
    elif normalization != "none":
        raise ValueError(f"지원하지 않는 normalization입니다: {normalization}")
    return tensor


def decode_video_frames(
    video_path: str | Path,
    frame_indices: Sequence[int],
    input_size: tuple[int, int] = (224, 224),
    normalization: str = "imagenet",
) -> torch.Tensor:
    """원본 영상에서 지정된 frame index만 읽어 ``[T, 3, H, W]``로 반환한다.

    OpenCV의 BGR 결과를 RGB로 바꾼다. seek/read 실패 시 이전 frame이나 임의 frame으로
    대체하면 재구성 입력이 달라지므로 즉시 ``FrameDecodeError``를 발생시킨다.
    """

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FrameDecodeError(f"영상을 열 수 없습니다: {path}")
    frames = []
    try:
        for frame_index in frame_indices:
            if frame_index < 0:
                raise FrameDecodeError(f"음수 frame index입니다: {path}, {frame_index}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, bgr_frame = capture.read()
            if not success or bgr_frame is None:
                raise FrameDecodeError(f"frame decode 실패: {path}, index={frame_index}")
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess_rgb_frame(rgb_frame, input_size, normalization))
    finally:
        capture.release()
    return torch.stack(frames)


class PaperFrameDataset(Dataset):
    """Paper reconstruction sequence를 video 단위 frame tensor로 제공한다.

    ``sequence_metadata``와 선택적 ``video_ids``를 받아 Dataset을 구성한다. 각 item은
    frames, label, video_id dict다. video ID가 없거나 sequence가 20행이 아니면 실패한다.
    """

    def __init__(
        self,
        sequence_metadata: str | Path,
        video_ids: Sequence[str] | None = None,
        sequence_length: int = 20,
        input_size: tuple[int, int] = (224, 224),
        normalization: str = "imagenet",
        frame_loader: Callable[..., torch.Tensor] = decode_video_frames,
    ) -> None:
        sequences = load_sequence_groups(sequence_metadata, sequence_length)
        by_id = {str(item["video_id"]): item for item in sequences}
        ordered_ids = list(video_ids) if video_ids is not None else sorted(by_id)
        missing_ids = [video_id for video_id in ordered_ids if video_id not in by_id]
        if missing_ids:
            raise ValueError(f"sequence metadata에 없는 video_id입니다: {missing_ids[:5]}")
        self.sequences = [by_id[video_id] for video_id in ordered_ids]
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.normalization = normalization
        self.frame_loader = frame_loader

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.sequences[index]
        frames = self.frame_loader(
            item["video_path"],
            item["source_frame_indices"],
            self.input_size,
            self.normalization,
        )
        expected_shape = (self.sequence_length, 3, *self.input_size)
        if tuple(frames.shape) != expected_shape:
            raise ValueError(
                f"{item['video_id']} frame tensor shape 불일치: "
                f"actual={tuple(frames.shape)}, expected={expected_shape}"
            )
        return {
            "frames": frames,
            "label": int(item["label_id"]),
            "video_id": str(item["video_id"]),
        }
