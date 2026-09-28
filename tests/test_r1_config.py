"""R0/R1 config가 한 가지 refinement 차이만 갖는지 검증한다."""

from pathlib import Path

import yaml

from drowsiness_fusion.models.paper_lstm import PaperLSTMBaseline, parameter_counts
from drowsiness_fusion.training.paper_trainer import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    path = PROJECT_ROOT / "configs" / "paper_reconstruction" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_r0_and_r1_config_contracts() -> None:
    r0 = load_yaml("vgg19_lstm.yaml")
    r1 = load_yaml("vgg19_lstm_fc2.yaml")

    assert r0["cnn"]["feature_layer"] == "fc1"
    assert r1["experiment"]["condition_id"] == "R1"
    assert r1["cnn"]["feature_point"] == "fc2"
    assert "vgg19/fc2" in r1["cnn"]["feature_cache_dir"]
    assert r0["cnn"]["feature_dim"] == r1["cnn"]["feature_dim"] == 4096
    assert r0["sequence"] == r1["sequence"]
    assert r0["lstm"] == r1["lstm"]
    assert r0["classifier"] == r1["classifier"]

    unchanged_training = (
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "device",
        "num_workers",
        "seed",
    )
    for key in unchanged_training:
        assert r0["training"][key] == r1["training"][key]

    assert isinstance(build_model(r0), PaperLSTMBaseline)
    r1_model = build_model(r1)
    assert parameter_counts(r1_model)["total"] == 9_734_850
