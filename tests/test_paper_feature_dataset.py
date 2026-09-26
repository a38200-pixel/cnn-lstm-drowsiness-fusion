"""4096D offline cache Dataset의 shape, label, NaN/Inf 방어를 synthetic 파일로 검증한다.

실제 VGG feature cache나 전체 fold CSV는 읽거나 생성하지 않는다.
"""

import numpy as np
import pytest

import drowsiness_fusion.data.paper_feature_dataset as feature_module
from drowsiness_fusion.data.paper_feature_dataset import (
    PaperFeatureDataset,
    validate_feature_array,
)


def _mock_manifest_and_split(monkeypatch, shape=(20, 4096)) -> None:
    manifest = [
        {
            "video_id": "n_1",
            "label": "not_drowsy",
            "feature_path": r"C:\cache\n_1.npy",
            "sequence_length": str(shape[0]),
            "feature_dim": str(shape[1]),
            "backbone": "vgg19",
        }
    ]
    split = [{"video_id": "n_1", "label": "not_drowsy"}]
    monkeypatch.setattr(feature_module, "read_feature_manifest", lambda _: manifest)
    monkeypatch.setattr(feature_module, "read_split_rows", lambda _: split)


def test_cached_feature_load_has_expected_shape_and_label(monkeypatch) -> None:
    _mock_manifest_and_split(monkeypatch)
    monkeypatch.setattr("pathlib.Path.is_file", lambda _: True)
    monkeypatch.setattr(
        feature_module.np,
        "load",
        lambda *args, **kwargs: np.zeros((20, 4096), dtype=np.float32),
    )
    dataset = PaperFeatureDataset("manifest.csv", "fold.csv", project_root=r"C:\project")

    item = dataset[0]

    assert item["features"].shape == (20, 4096)
    assert item["label"] == 0
    assert item["video_id"] == "n_1"


def test_manifest_feature_dim_mismatch_is_error(monkeypatch) -> None:
    _mock_manifest_and_split(monkeypatch, shape=(20, 128))
    with pytest.raises(ValueError, match="feature_dim"):
        PaperFeatureDataset("manifest.csv", "fold.csv", project_root=r"C:\project")


def test_manifest_sequence_length_mismatch_is_error(monkeypatch) -> None:
    _mock_manifest_and_split(monkeypatch, shape=(19, 4096))
    with pytest.raises(ValueError, match="sequence_length"):
        PaperFeatureDataset("manifest.csv", "fold.csv", project_root=r"C:\project")


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_feature_are_rejected(invalid_value) -> None:
    features = np.zeros((20, 4096), dtype=np.float32)
    features[0, 0] = invalid_value
    with pytest.raises(ValueError, match="NaN 또는 Inf"):
        validate_feature_array(features)
