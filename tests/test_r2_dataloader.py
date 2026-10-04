"""Bounded spawn tests with deterministic mock frames; no real dataset scan."""
import csv
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from drowsiness_fusion.data.paper_frame_dataset import PaperFrameDataset
from drowsiness_fusion.data.r2_dataloader import loader_settings, config_with_effective_loader_settings


def deterministic_frames(video_path, frame_indices, input_size, normalization):
    value = int(Path(video_path).stem) + sum(frame_indices)
    assert normalization == "imagenet"
    return torch.full((20, 3, *input_size), float(value))


@pytest.mark.parametrize("workers,persistent,prefetch", [(0, False, None), (2, True, 2), (4, True, 4)])
def test_loader_settings_and_creation(workers, persistent, prefetch):
    settings = loader_settings({"num_workers": workers, "pin_memory": True,
                                "persistent_workers": True, "prefetch_factor": prefetch})
    loader = DataLoader([1, 2], **settings)
    assert loader.num_workers == workers
    assert loader.pin_memory is True
    assert loader.persistent_workers is persistent
    assert loader.prefetch_factor == prefetch


def test_invalid_prefetch():
    with pytest.raises(ValueError, match="prefetch_factor"):
        loader_settings({"num_workers": 2, "prefetch_factor": None})


def test_explicit_defaults_preserve_checkpoint_config_compatibility():
    old = {"training": {"num_workers": 0, "seed": 42}}
    explicit = {"training": {**old["training"], "pin_memory": True,
                             "persistent_workers": False, "prefetch_factor": None}}
    assert config_with_effective_loader_settings(old) == config_with_effective_loader_settings(explicit)
    changed = {"training": {**explicit["training"], "seed": 43}}
    assert config_with_effective_loader_settings(old) != config_with_effective_loader_settings(changed)


def test_config_parse():
    path = Path(__file__).resolve().parents[1] / "configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = loader_settings(config["training"])
    assert settings["num_workers"] >= 0
    assert settings["pin_memory"] is True
    assert config["training"]["batch_size"] == 16
    assert config["training"]["epochs"] == 30
    assert config["training"]["seed"] == 42


def test_worker_spawn_preserves_frames_labels_and_shuffled_identity(tmp_path):
    path = tmp_path / "sequences.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "label", "slot", "source_frame_index", "video_path"])
        for video in range(4):
            for slot in range(20):
                writer.writerow([f"v{video}", "drowsy" if video % 2 else "not_drowsy",
                                 slot, slot*10, f"{video}.mp4"])
    dataset = PaperFrameDataset(path, input_size=(2, 2), frame_loader=deterministic_frames)
    outputs = []
    for workers in (0, 2):
        settings = loader_settings({"num_workers": workers, "pin_memory": False,
                                    "persistent_workers": True, "prefetch_factor": 2})
        loader = DataLoader(dataset, batch_size=2, shuffle=True,
                            generator=torch.Generator().manual_seed(42), **settings)
        iterator = iter(loader)
        try:
            batches = list(iterator)
            assert all(batch["frames"].shape == (2, 20, 3, 2, 2) for batch in batches)
            outputs.append(batches)
        finally:
            if workers:
                iterator._shutdown_workers()
    for zero, multi in zip(*outputs):
        assert zero["video_id"] == multi["video_id"]
        assert torch.equal(zero["label"], multi["label"])
        assert torch.equal(zero["frames"], multi["frames"])
