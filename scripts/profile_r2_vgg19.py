"""Bounded real-data R2 profile; no full epoch, checkpoint or MLflow training run."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from train_r2_vgg19 import load_config
from drowsiness_fusion.data.paper_feature_dataset import read_split_rows
from drowsiness_fusion.data.paper_frame_dataset import PaperFrameDataset
from drowsiness_fusion.data.r2_dataloader import loader_settings
from drowsiness_fusion.models.r2_vgg19_lstm import R2VGG19LSTM
from drowsiness_fusion.training.r2_execution import require_cuda, environment, atomic_json, gpu_memory
from drowsiness_fusion.training.r2_trainer import profile_batches
from drowsiness_fusion.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--test-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batches <= 30 or not 0 <= args.test_batches <= 5:
        parser.error("train batches must be 1..30; test batches 0..5")
    if args.num_workers is not None and args.num_workers not in (0, 2, 4, 8):
        parser.error("worker diagnostic candidates: 0, 2, 4, 8")
    if args.output.exists():
        raise FileExistsError("Choose a new profile output file.")
    device = require_cuda()
    config = load_config(args.config)
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
        config["training"]["persistent_workers"] = args.num_workers > 0
        config["training"]["prefetch_factor"] = 2 if args.num_workers else None
    seed_everything(int(config["training"]["seed"]))
    env = environment(config, args.fold, device)
    print(json.dumps(env, ensure_ascii=False), flush=True)
    loaders = []
    fold_batch_counts = {}
    for split, count in (("train", args.batches), ("holdout", args.test_batches)):
        if not count:
            loaders.append(None)
            continue
        rows = read_split_rows(PROJECT_ROOT / config["data"]["split_root"] /
                               f"fold_{args.fold}_{split}.csv")
        fold_batch_counts[split] = len(DataLoader(range(len(rows)), batch_size=16, drop_last=False))
        rows = rows[:count * 16]
        dataset = PaperFrameDataset(
            PROJECT_ROOT / config["data"]["sequence_metadata"],
            video_ids=[row["video_id"] for row in rows],
            sequence_length=int(config["sequence"]["length"]),
            input_size=tuple(config["cnn"]["input_size"]),
            normalization=config["cnn"]["normalization"],
        )
        loaders.append(DataLoader(dataset, batch_size=16, shuffle=False,
                                  **loader_settings(config["training"])))
    torch.cuda.reset_peak_memory_stats(device)
    model = R2VGG19LSTM(pretrained=True).to(device)
    optimizer = Adam([
        {"params": model.vgg_block5.parameters(), "lr": 0.00001},
        {"params": model.temporal_parameters(), "lr": 0.0001},
    ], weight_decay=0.00001)
    result = {"result_role": "diagnostic_only", "environment": env}
    result["train"] = profile_batches(model, loaders[0], optimizer, nn.CrossEntropyLoss(),
                                     device, args.batches)
    result["gradient_policy"] = {
        "block5_gradient": any(p.grad is not None for p in model.vgg_block5.parameters()),
        "temporal_gradient": any(p.grad is not None for p in model.temporal_parameters()),
        "frozen_gradient_absent": all(p.grad is None for block in (
            model.vgg_block1, model.vgg_block2, model.vgg_block3, model.vgg_block4, model.fc1
        ) for p in block.parameters()),
    }
    if not all(result["gradient_policy"].values()):
        raise RuntimeError("R2 gradient policy failed.")
    if loaders[1] is not None:
        result["test"] = profile_batches(model, loaders[1], optimizer, nn.CrossEntropyLoss(),
                                        device, args.test_batches, training=False)
    result["memory"] = gpu_memory(device)
    if "test" in result:
        train_seconds = result["train"]["summary"]["average_batch_total_ms"] / 1000 * fold_batch_counts["train"]
        test_seconds = result["test"]["summary"]["average_batch_total_ms"] / 1000 * fold_batch_counts["holdout"]
        result["epoch_time_estimate"] = {
            "role": "ESTIMATE ONLY", "fold_batch_counts": fold_batch_counts,
            "estimated_train_seconds": train_seconds, "estimated_test_seconds": test_seconds,
            "estimated_epoch_seconds": train_seconds + test_seconds,
            "limitations": "Bounded cold-start subset, queue effects, shorter final batch and "
                           "checkpoint/logging costs prevent a measured epoch-time claim.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("train", "test")},
                     ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({key: result[key]["summary"] for key in ("train", "test") if key in result},
                     indent=2), flush=True)
    print("Cold-start bounded profile; timings include synchronization overhead, not an epoch estimate.")


if __name__ == "__main__":
    main()
