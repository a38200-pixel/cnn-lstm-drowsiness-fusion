"""R2 batch-size 16 CUDA forward/backward와 gradient policy를 한 batch로 검증한다.

입력은 R2 YAML과 fold 1의 첫 train batch이며 출력은 parameter 수, logits shape,
gradient 및 optimizer LR 확인 결과다. Optimizer step, epoch training, MLflow 기록,
checkpoint 저장은 하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_fusion.data.paper_feature_dataset import read_split_rows  # noqa: E402
from drowsiness_fusion.data.paper_frame_dataset import PaperFrameDataset  # noqa: E402
from drowsiness_fusion.models.r2_vgg19_lstm import (  # noqa: E402
    R2VGG19LSTM,
    parameter_summary,
)
from drowsiness_fusion.utils.reproducibility import seed_everything  # noqa: E402


def main() -> None:
    """사전 확정 batch 16을 바꾸지 않고 CUDA OOM과 gradient 범위를 확인한다."""

    parser = argparse.ArgumentParser(description="R2 batch16 CUDA smoke test")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    if int(training["batch_size"]) != 16:
        raise ValueError("R2 smoke test는 사전 확정 batch_size=16만 검증합니다.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA를 사용할 수 없어 batch16 OOM smoke test를 실행하지 않았습니다. "
            "R2 전체 학습 전에 CUDA 환경에서 다시 실행하세요."
        )

    seed_everything(int(training["seed"]))
    device = torch.device("cuda")
    split_root = PROJECT_ROOT / config["data"]["split_root"]
    train_rows = read_split_rows(split_root / "fold_1_train.csv")
    dataset = PaperFrameDataset(
        sequence_metadata=PROJECT_ROOT / config["data"]["sequence_metadata"],
        video_ids=[row["video_id"] for row in train_rows],
        sequence_length=int(config["sequence"]["length"]),
        input_size=tuple(int(value) for value in config["cnn"]["input_size"]),
        normalization=str(config["cnn"]["normalization"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
    )

    model = R2VGG19LSTM(
        pretrained=bool(config["cnn"]["pretrained"]),
        sequence_length=int(config["sequence"]["length"]),
        lstm_hidden_size=int(config["lstm"]["hidden_size"]),
    ).to(device)
    cnn_parameters = [
        parameter for parameter in model.vgg_block5.parameters() if parameter.requires_grad
    ]
    temporal_parameters = [
        parameter for parameter in model.temporal_parameters() if parameter.requires_grad
    ]
    optimizer = Adam(
        [
            {"params": cnn_parameters, "lr": float(training["cnn_learning_rate"])},
            {
                "params": temporal_parameters,
                "lr": float(training["temporal_learning_rate"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )

    batch = next(iter(loader))
    frames = batch["frames"].to(device)
    labels = batch["label"].to(device)
    try:
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
    except torch.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "R2 batch_size=16 CUDA OOM이 발생했습니다. batch size를 자동 변경하지 않았습니다."
        ) from error

    block5_has_gradient = any(
        parameter.grad is not None for parameter in model.vgg_block5.parameters()
    )
    frozen_has_gradient = any(
        parameter.grad is not None
        for block in (
            model.vgg_block1,
            model.vgg_block2,
            model.vgg_block3,
            model.vgg_block4,
        )
        for parameter in block.parameters()
    )
    fc1_has_gradient = any(parameter.grad is not None for parameter in model.fc1.parameters())
    temporal_has_gradient = any(parameter.grad is not None for parameter in temporal_parameters)
    gradient_policy_failed = (
        not block5_has_gradient
        or frozen_has_gradient
        or fc1_has_gradient
        or not temporal_has_gradient
    )
    if gradient_policy_failed:
        raise RuntimeError("R2 gradient policy smoke test가 실패했습니다.")

    print(f"device={device} input={tuple(frames.shape)} logits={tuple(logits.shape)}")
    print(f"parameters={parameter_summary(model)}")
    print(f"optimizer_lrs={[group['lr'] for group in optimizer.param_groups]}")
    print("gradient_policy=OK, batch16_cuda_oom=NOT_OBSERVED")


if __name__ == "__main__":
    main()
