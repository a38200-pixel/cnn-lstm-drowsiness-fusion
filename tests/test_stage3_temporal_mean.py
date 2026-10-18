"""Bounded synthetic/meta S3-1 checks; no dataset, pretrained download or full training."""
import copy
import importlib
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torchvision.models import vgg19

from drowsiness_fusion.models.paper_lstm import KerasCompatibleLSTM
from drowsiness_fusion.models.r2_vgg19_lstm import R2VGG19LSTM, parameter_summary
from drowsiness_fusion.models.stage3_temporal_mean import Stage3TemporalMean
from drowsiness_fusion.training.r2_execution import save_checkpoint, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/context_model/stage3_s3_1_temporal_mean.yaml"


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyVGG19Compatible(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Identity() for _ in range(37)]
        layers[0] = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Conv2d(3, 3, 1))
        for start in (5, 10, 19, 28):
            layers[start] = nn.Conv2d(3, 3, 1)
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(3, 4096), nn.ReLU())


def test_optional_sequence_matches_single_bias_pytorch_and_legacy_last():
    torch.manual_seed(42)
    custom = KerasCompatibleLSTM(4, 3)
    reference = nn.LSTM(4, 3, batch_first=True)
    with torch.no_grad():
        reference.weight_ih_l0.copy_(custom.weight_input)
        reference.weight_hh_l0.copy_(custom.weight_hidden)
        reference.bias_ih_l0.copy_(custom.bias)
        reference.bias_hh_l0.zero_()
    x = torch.randn(2, 20, 4)
    sequence = custom(x, return_sequences=True)
    expected, _ = reference(x)
    torch.testing.assert_close(sequence, expected)
    assert torch.equal(custom(x), sequence[:, -1])
    assert list(custom.state_dict()) == ["weight_input", "weight_hidden", "bias"]
    sequence.mean().backward()
    expected.mean().backward()
    torch.testing.assert_close(custom.weight_input.grad, reference.weight_ih_l0.grad)
    torch.testing.assert_close(custom.weight_hidden.grad, reference.weight_hh_l0.grad)
    torch.testing.assert_close(custom.bias.grad, reference.bias_ih_l0.grad)


def test_real_architecture_counts_equal_r2_without_allocating_weights():
    with torch.device("meta"):
        r2 = R2VGG19LSTM(pretrained=False, vgg_model=vgg19(weights=None))
        s3 = Stage3TemporalMean(pretrained=False, vgg_model=vgg19(weights=None))
    expected = {"total": 132_523_778, "trainable": 19_174_082, "frozen": 113_349_696}
    assert parameter_summary(r2) == parameter_summary(s3) == expected
    assert list(r2.state_dict()) == list(s3.state_dict())
    assert {k: v.shape for k, v in r2.state_dict().items()} == {
        k: v.shape for k, v in s3.state_dict().items()
    }


def test_mock_frame_smoke_mean_readout_gradients_and_initialization():
    torch.manual_seed(42)
    r2 = R2VGG19LSTM(pretrained=False, vgg_model=TinyVGG19Compatible())
    torch.manual_seed(42)
    s3 = Stage3TemporalMean(pretrained=False, vgg_model=TinyVGG19Compatible())
    assert all(torch.equal(v, s3.state_dict()[k]) for k, v in r2.state_dict().items())
    assert parameter_summary(r2) == parameter_summary(s3)
    captured = {}

    def hidden_hook(module, inputs, output):
        captured["hidden"] = output
        output.retain_grad()

    def head_hook(module, inputs):
        captured["pooled"] = inputs[0]

    handles = [s3.lstm.register_forward_hook(hidden_hook),
               s3.classifier_fc1.register_forward_pre_hook(head_hook)]
    try:
        logits = s3(torch.randn(1, 20, 3, 224, 224))
        assert logits.shape == (1, 2)
        assert captured["hidden"].shape == (1, 20, 512)
        assert captured["pooled"].shape == (1, 512)
        torch.testing.assert_close(captured["pooled"], captured["hidden"].mean(dim=1))
        assert not torch.equal(captured["pooled"], captured["hidden"][:, -1])
        nn.CrossEntropyLoss()(logits, torch.tensor([1])).backward()
    finally:
        for handle in handles:
            handle.remove()
    for block in (s3.vgg_block1, s3.vgg_block2, s3.vgg_block3, s3.vgg_block4, s3.fc1):
        assert all(not p.requires_grad and p.grad is None for p in block.parameters())
    for module in (s3.vgg_block5, s3.lstm, s3.classifier_fc1,
                   s3.classifier_fc2, s3.classifier_fc3):
        assert all(p.requires_grad and p.grad is not None for p in module.parameters())
        assert any(torch.count_nonzero(p.grad) for p in module.parameters())
    gradient = captured["hidden"].grad
    assert torch.count_nonzero(gradient)
    torch.testing.assert_close(gradient, gradient[:, :1].expand_as(gradient))
    cnn = list(s3.vgg_block5.parameters())
    temporal = list(s3.temporal_parameters())
    assert not {id(p) for p in cnn} & {id(p) for p in temporal}
    assert {id(p) for p in cnn + temporal} == {
        id(p) for p in s3.parameters() if p.requires_grad
    }


def test_config_only_changes_identity_readout_and_execution_metadata(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    script = importlib.import_module("train_stage3_s3_1")
    s3 = script.load_config(CONFIG)
    r2 = yaml.safe_load((ROOT / "configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml")
                        .read_text(encoding="utf-8"))
    for key in ("data", "cnn", "sequence", "lstm", "classifier", "cross_validation"):
        assert s3[key] == r2[key]
    expected_training = {**r2["training"], "num_workers": 4, "prefetch_factor": 1, "amp": False}
    assert s3["training"] == expected_training
    assert s3["temporal_readout"] == "mean"
    assert s3["mlflow"]["experiment_name"] == "context-model-improvement"
    assert s3["output"]["root"] == "outputs/context_model/S3_1_temporal_mean"


@pytest.mark.parametrize("section,key,value", [
    ("training", "epochs", 15), ("training", "amp", True),
    ("training", "batch_size", 8), ("training", "seed", 43),
    ("training", "num_workers", 8), ("cnn", "normalization", "none"),
    ("data", "split_root", "different"), ("experiment", "condition_id", "R2"),
])
def test_config_rejects_changed_contract(section, key, value, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    script = importlib.import_module("train_stage3_s3_1")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config[section][key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        script.load_config(path)


class TinyTemporalMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = KerasCompatibleLSTM(4, 3)
        self.head = nn.Linear(3, 2)

    def forward(self, x):
        return self.head(self.lstm(x, return_sequences=True).mean(dim=1))


def test_stage3_resume_exact_next_update_and_reject_r2_or_changed_readout(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(42)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = TinyTemporalMean()
    optimizer = Adam(model.parameters(), lr=0.0001)
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, 20, 4)
    model(x).sum().backward()
    optimizer.step()
    path = tmp_path / "s3.pt"
    save_checkpoint(path, model, optimizer, 1, config, 1, [{"epoch": 1}], generator)
    assert torch.load(path, weights_only=False)["condition_id"] == "S3_1_temporal_mean"
    expected_rng = torch.rand(3)
    expected_order = torch.randperm(8, generator=generator)
    optimizer.zero_grad()
    model(x).sum().backward()
    optimizer.step()
    restored = TinyTemporalMean()
    restored_optimizer = Adam(restored.parameters(), lr=0.5)
    start, history = load_checkpoint(path, restored, restored_optimizer, config, 1, generator)
    assert start == 2 and history == [{"epoch": 1}]
    assert restored_optimizer.param_groups[0]["lr"] == 0.0001
    assert torch.equal(torch.rand(3), expected_rng)
    assert torch.equal(torch.randperm(8, generator=generator), expected_order)
    restored_optimizer.zero_grad()
    restored(x).sum().backward()
    restored_optimizer.step()
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    for wrong_config, wrong_fold in ((config, 2),
                                      ({**config, "temporal_readout": "last"}, 1)):
        with pytest.raises(ValueError, match="does not match"):
            load_checkpoint(path, restored, restored_optimizer, wrong_config, wrong_fold, generator)
    r2 = copy.deepcopy(config)
    r2["experiment"]["condition_id"] = "R2"
    r2_path = tmp_path / "r2.pt"
    save_checkpoint(r2_path, model, optimizer, 1, r2, 1, [{"epoch": 1}], generator)
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(r2_path, restored, restored_optimizer, config, 1, generator)
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, restored, restored_optimizer, r2, 1, generator)
