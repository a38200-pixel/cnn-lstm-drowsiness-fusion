"""Cached 4096D feature로 단일 fold LSTM을 학습·평가하는 모듈이다.

주요 입력은 baseline config, fold 번호, feature manifest/cache이며 주요 출력은 fold metric,
history, 사용자 실행 시 checkpoint다. CNN inference, 별도 test split, early stopping은 하지 않는다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import DataLoader

from drowsiness_fusion.data.paper_feature_dataset import PaperFeatureDataset
from drowsiness_fusion.evaluation.classification_metrics import calculate_classification_metrics
from drowsiness_fusion.models.paper_lstm import PaperLSTMBaseline
from drowsiness_fusion.utils.reproducibility import seed_everything


def resolve_device(device_name: str) -> torch.device:
    """``auto`` 또는 명시적 device 문자열을 실제 PyTorch device로 반환한다."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device를 요청했지만 CUDA를 사용할 수 없습니다.")
    return device


def build_model(config: Mapping[str, Any]) -> PaperLSTMBaseline:
    """Config의 고정 paper architecture 값을 검증하고 LSTM baseline을 반환한다."""

    sequence = config["sequence"]
    lstm = config["lstm"]
    classifier = config["classifier"]
    cnn = config["cnn"]
    if int(cnn["feature_dim"]) != 4096:
        raise ValueError("Paper reconstruction cnn.feature_dim은 4096이어야 합니다.")
    if int(lstm["input_size"]) != 4096:
        raise ValueError("Paper reconstruction lstm.input_size는 4096이어야 합니다.")
    if int(lstm["hidden_size"]) != 512 or int(lstm["num_layers"]) != 1:
        raise ValueError("Paper reconstruction LSTM은 hidden_size=512, num_layers=1입니다.")
    if lstm.get("batch_first") is not True:
        raise ValueError("Paper reconstruction LSTM은 batch_first=true여야 합니다.")
    if int(sequence["length"]) != 20:
        raise ValueError("Paper reconstruction sequence.length는 20이어야 합니다.")
    classifier_shape = (
        int(classifier["hidden_1"]),
        int(classifier["hidden_2"]),
        int(classifier["num_classes"]),
    )
    if classifier_shape != (512, 64, 2):
        raise ValueError("Paper reconstruction classifier는 512→64→2여야 합니다.")
    if (classifier.get("activation_1"), classifier.get("activation_2")) != (
        "relu",
        "sigmoid",
    ):
        raise ValueError("초기 reconstruction activation은 ReLU→Sigmoid입니다.")
    return PaperLSTMBaseline(
        input_size=int(lstm["input_size"]),
        hidden_size=int(lstm["hidden_size"]),
        num_layers=int(lstm["num_layers"]),
        hidden_1=int(classifier["hidden_1"]),
        hidden_2=int(classifier["hidden_2"]),
        num_classes=int(classifier["num_classes"]),
        sequence_length=int(sequence["length"]),
    )


def build_optimizer(model: nn.Module, training_config: Mapping[str, Any]) -> Optimizer:
    """Config의 Adam 또는 SGD 초기값으로 optimizer를 생성해 반환한다."""

    name = str(training_config["optimizer"]).lower()
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])
    if name == "adam":
        return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"지원하지 않는 optimizer입니다: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """한 epoch를 학습하고 sample-weighted 평균 CrossEntropy loss를 반환한다."""

    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("train dataset이 비어 있습니다.")
    return loss_sum / sample_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    """Holdout에서 loss와 drowsy-positive classification metric을 반환한다."""

    model.eval()
    loss_sum = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        loss_sum += float(loss) * labels.shape[0]
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if not targets:
        raise ValueError("holdout dataset이 비어 있습니다.")
    return {
        "loss": loss_sum / len(targets),
        **calculate_classification_metrics(targets, predictions),
    }


def run_fold_training(
    config: Mapping[str, Any],
    fold: int,
    project_root: str | Path,
) -> dict[str, object]:
    """한 fold의 feature dataset, model, optimizer를 구성해 전체 epoch를 실행한다.

    사용자 실행 시 fold output 폴더에 history, metrics, final checkpoint를 저장하고 metric을
    반환한다. Holdout은 evaluation에만 사용하며 early stopping이나 scheduler는 적용하지 않는다.
    """

    root = Path(project_root)
    folds = int(config["cross_validation"]["folds"])
    if fold < 1 or fold > folds:
        raise ValueError(f"fold는 1..{folds} 범위여야 합니다: {fold}")
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    device = resolve_device(str(training["device"]))

    split_root = root / config["data"]["split_root"]
    manifest_path = root / config["data"]["feature_manifest"]
    dataset_kwargs = {
        "manifest_path": manifest_path,
        "project_root": root,
        "sequence_length": int(config["sequence"]["length"]),
        "feature_dim": int(config["cnn"]["feature_dim"]),
        "backbone": str(config["cnn"]["backbone"]),
    }
    train_dataset = PaperFeatureDataset(
        split_path=split_root / f"fold_{fold}_train.csv", **dataset_kwargs
    )
    holdout_dataset = PaperFeatureDataset(
        split_path=split_root / f"fold_{fold}_holdout.csv", **dataset_kwargs
    )
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    holdout_loader = DataLoader(holdout_dataset, shuffle=False, **loader_kwargs)

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, training)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        holdout = evaluate(model, holdout_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **holdout})
        print(
            f"fold={fold} epoch={epoch}/{training['epochs']} "
            f"train_loss={train_loss:.6f} holdout_loss={holdout['loss']:.6f} "
            f"accuracy={holdout['accuracy']:.4f} f1={holdout['f1']:.4f}"
        )

    final_metrics = {"fold": fold, **history[-1]}
    output_dir = root / config["output"]["root"] / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "fold": fold, "config": dict(config)},
        output_dir / "final_model.pt",
    )
    return final_metrics
