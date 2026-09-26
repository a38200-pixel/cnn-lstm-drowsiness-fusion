"""Offline VGG feature cache를 검증하고 LSTM 학습에 제공하는 Dataset 모듈이다.

주요 입력은 feature manifest, fold CSV, video별 ``.npy``이고 주요 출력은
``[20, 4096]`` feature tensor, label, video ID다. frame decode와 CNN inference는 하지 않는다.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .sust_ddd_metadata import LABEL_IDS

MANIFEST_COLUMNS = (
    "video_id",
    "label",
    "feature_path",
    "sequence_length",
    "feature_dim",
    "backbone",
)


def validate_feature_array(
    features: np.ndarray,
    sequence_length: int = 20,
    feature_dim: int = 4096,
) -> None:
    """Feature array의 shape와 유한값을 검증한다.

    입력 array가 정확히 ``[sequence_length, feature_dim]``인지 확인하며 반환값은 없다.
    NaN/Inf 또는 shape 오류를 보정하지 않고 즉시 오류로 처리한다.
    """

    expected_shape = (sequence_length, feature_dim)
    if features.shape != expected_shape:
        raise ValueError(
            f"feature shape 불일치: actual={features.shape}, expected={expected_shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("feature에 NaN 또는 Inf가 있습니다.")


def read_feature_manifest(path: str | Path) -> list[dict[str, str]]:
    """Feature manifest를 읽고 필수 컬럼과 video ID 유일성을 검증해 반환한다."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"feature manifest가 없습니다. 먼저 feature extraction을 실행하세요: {manifest_path}"
        )
    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"feature manifest 필수 컬럼이 없습니다: {sorted(missing)}")
        rows = list(reader)
    video_ids = [row["video_id"] for row in rows]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("feature manifest에 중복 video_id가 있습니다.")
    return rows


def read_split_rows(path: str | Path) -> list[dict[str, str]]:
    """Fold train/holdout CSV를 읽어 순서를 유지한 행 목록을 반환한다."""

    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"fold split CSV가 없습니다: {split_path}")
    with split_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"video_id", "label"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"fold split CSV schema가 올바르지 않습니다: {split_path}")
        return list(reader)


def write_feature_manifest(rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    """검증된 feature cache 위치를 고정 컬럼의 UTF-8 manifest로 저장한다."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class PaperFeatureDataset(Dataset):
    """Fold 목록에 맞춰 video별 VGG feature를 읽는 Dataset이다.

    Manifest와 split CSV를 입력받아 split 순서로 구성한다. 각 item은 features, label,
    video_id dict이며 cache shape, 유한값, label 일치를 읽을 때마다 검증한다.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split_path: str | Path,
        project_root: str | Path,
        sequence_length: int = 20,
        feature_dim: int = 4096,
        backbone: str | None = None,
    ) -> None:
        manifest_rows = read_feature_manifest(manifest_path)
        split_rows = read_split_rows(split_path)
        manifest_by_id = {row["video_id"]: row for row in manifest_rows}
        records = []
        for split_row in split_rows:
            video_id = split_row["video_id"]
            if video_id not in manifest_by_id:
                raise ValueError(f"feature manifest에 없는 video_id입니다: {video_id}")
            record = dict(manifest_by_id[video_id])
            if record["label"] != split_row["label"]:
                raise ValueError(f"split과 manifest label이 다릅니다: {video_id}")
            if record["label"] not in LABEL_IDS:
                raise ValueError(f"지원하지 않는 label입니다: {record['label']}")
            if int(record["sequence_length"]) != sequence_length:
                raise ValueError(f"manifest sequence_length 불일치: {video_id}")
            if int(record["feature_dim"]) != feature_dim:
                raise ValueError(f"manifest feature_dim 불일치: {video_id}")
            if backbone is not None and record["backbone"] != backbone:
                raise ValueError(
                    f"manifest backbone 불일치: {video_id}, "
                    f"actual={record['backbone']}, expected={backbone}"
                )
            records.append(record)
        self.records = records
        self.project_root = Path(project_root)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        feature_path = Path(record["feature_path"])
        if not feature_path.is_absolute():
            feature_path = self.project_root / feature_path
        if not feature_path.is_file():
            raise FileNotFoundError(f"feature cache가 없습니다: {feature_path}")
        features = np.load(feature_path, allow_pickle=False)
        validate_feature_array(features, self.sequence_length, self.feature_dim)
        return {
            "features": torch.from_numpy(features.astype(np.float32, copy=False)),
            "label": LABEL_IDS[record["label"]],
            "video_id": record["video_id"],
        }
