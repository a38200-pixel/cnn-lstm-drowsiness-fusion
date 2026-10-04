"""작은 VGG19-compatible mock으로 R2 shape, gradient, optimizer 정책을 검증한다.

Pretrained weight 다운로드, 실제 224×224 VGG19 연산, dataset scan과 epoch training은
수행하지 않는다.
"""

from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Adam

from drowsiness_fusion.models.r2_vgg19_lstm import R2VGG19LSTM, parameter_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyVGG19Compatible(nn.Module):
    """R2의 torchvision block 경계와 fc1 계약만 재현하는 작은 test double이다."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Identity() for _ in range(37)]
        for block_start in (0, 5, 10, 19, 28):
            layers[block_start] = nn.Conv2d(3, 3, kernel_size=1)
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(3, 4096), nn.ReLU())


def test_r2_forward_gradient_and_optimizer_groups() -> None:
    torch.manual_seed(42)
    model = R2VGG19LSTM(
        pretrained=False,
        sequence_length=20,
        vgg_model=TinyVGG19Compatible(),
    )

    frozen_blocks = (
        model.vgg_block1,
        model.vgg_block2,
        model.vgg_block3,
        model.vgg_block4,
    )
    assert all(
        not parameter.requires_grad
        for block in frozen_blocks
        for parameter in block.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.vgg_block5.parameters())
    assert all(not parameter.requires_grad for parameter in model.fc1.parameters())

    cnn_parameters = [
        parameter for parameter in model.vgg_block5.parameters() if parameter.requires_grad
    ]
    temporal_parameters = list(model.temporal_parameters())
    optimizer = Adam(
        [
            {"params": cnn_parameters, "lr": 0.00001},
            {"params": temporal_parameters, "lr": 0.0001},
        ],
        weight_decay=0.00001,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [0.00001, 0.0001]
    cnn_ids = {id(parameter) for parameter in cnn_parameters}
    temporal_ids = {id(parameter) for parameter in temporal_parameters}
    assert not cnn_ids & temporal_ids

    frames = torch.randn(1, 20, 3, 2, 2)
    logits = model(frames)
    assert logits.shape == (1, 2)
    nn.CrossEntropyLoss()(logits, torch.tensor([1])).backward()

    assert any(parameter.grad is not None for parameter in model.vgg_block5.parameters())
    assert all(
        parameter.grad is None
        for block in frozen_blocks
        for parameter in block.parameters()
    )
    assert all(parameter.grad is None for parameter in model.fc1.parameters())
    assert any(parameter.grad is not None for parameter in model.lstm.parameters())
    assert any(parameter.grad is not None for parameter in model.classifier_fc1.parameters())
    assert parameter_summary(model)["trainable"] > 0


def test_r2_config_contract() -> None:
    path = (
        PROJECT_ROOT
        / "configs"
        / "paper_reconstruction"
        / "vgg19_lstm_r2_lastblock_finetune.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["experiment"]["condition_id"] == "R2"
    assert config["cnn"]["feature_point"] == "fc1"
    assert config["cnn"]["cnn_training_policy"] == "last_block_finetune"
    assert config["cnn"]["frozen_blocks"] == [1, 2, 3, 4]
    assert config["cnn"]["trainable_blocks"] == [5]
    assert config["cnn"]["fc1_trainable"] is False
    assert config["training"]["cnn_learning_rate"] == 0.00001
    assert config["training"]["temporal_learning_rate"] == 0.0001
    assert config["training"]["batch_size"] == 16
    assert config["training"]["epochs"] == 30
