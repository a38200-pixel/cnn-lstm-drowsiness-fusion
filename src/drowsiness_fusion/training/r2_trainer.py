"""R2 frame-to-logit 모델의 한 epoch 학습과 평가 연산을 제공한다.

입력은 R2 model, frame DataLoader, optimizer와 device이고 출력은 loss 및 binary
classification metric이다. Fold orchestration, MLflow run, checkpoint 저장은 하지 않는다.
"""

from __future__ import annotations

import torch
from time import perf_counter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from drowsiness_fusion.evaluation.classification_metrics import calculate_classification_metrics


def profile_batches(model, loader, optimizer, criterion, device, batches, training=True):
    """Bounded diagnostic; synchronization is only used in this explicit profile path."""
    model.train(training)
    records = []
    iterator = iter(loader)
    for index in range(min(batches, len(loader))):
        torch.cuda.synchronize(device)
        start = perf_counter()
        batch = next(iterator)
        loaded = perf_counter()
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        torch.cuda.synchronize(device)
        transferred = perf_counter()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(frames)
            loss = criterion(logits, labels)
        torch.cuda.synchronize(device)
        forwarded = perf_counter()
        if training:
            loss.backward()
        torch.cuda.synchronize(device)
        backwarded = perf_counter()
        if training:
            optimizer.step()
        torch.cuda.synchronize(device)
        stepped = perf_counter()
        record = dict(zip(
            ("data_loading", "h2d", "forward", "backward", "optimizer", "batch_total"),
            (loaded-start, transferred-loaded, forwarded-transferred,
             backwarded-forwarded, stepped-backwarded, stepped-start),
        ))
        records.append(record)
        if (index + 1) % 5 == 0 or index + 1 == min(batches, len(loader)):
            print(f"batch={index+1}/{min(batches, len(loader))} training={training} {record}", flush=True)
    if not records:
        raise ValueError("Profile dataset is empty.")
    averages = {f"average_{key}_ms": sum(r[key] for r in records)/len(records)*1000
                for key in records[0]}
    total = averages["average_batch_total_ms"]
    averages["percentages"] = {
        key: averages[f"average_{key}_ms"] / total * 100
        for key in ("data_loading", "h2d", "forward", "backward", "optimizer")
    }
    averages["percentages"]["other"] = max(0, 100-sum(averages["percentages"].values()))
    return {"batches": len(records), "summary": averages, "batch_timings_seconds": records}


def train_r2_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """R2 한 epoch를 학습하고 sample-weighted 평균 loss를 반환한다."""

    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("R2 train dataset이 비어 있습니다.")
    return loss_sum / sample_count


@torch.no_grad()
def evaluate_r2(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    """R2 held-out test fold의 loss와 drowsy-positive metric을 반환한다."""

    model.eval()
    loss_sum = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)
        logits = model(frames)
        loss = criterion(logits, labels)
        loss_sum += float(loss) * labels.shape[0]
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if not targets:
        raise ValueError("R2 held-out test dataset이 비어 있습니다.")
    return {
        "loss": loss_sum / len(targets),
        **calculate_classification_metrics(targets, predictions),
    }
