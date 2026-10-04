"""CUDA environment, atomic epoch snapshots and trusted local checkpoint resume."""
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from drowsiness_fusion.data.r2_dataloader import loader_settings, config_with_effective_loader_settings


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("R2 fine-tuning requires CUDA. CPU fallback is disabled for this experiment.")
    return torch.device("cuda")


def environment(config, fold, device):
    training = config["training"]
    return {
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(), "mixed_precision": False,
        "batch_size": int(training["batch_size"]), **loader_settings(training),
        "condition_id": "R2", "fold": fold, "seed": int(training["seed"]),
    }


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, model, optimizer, epoch, config, fold, history, generator):
    state = {
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": None,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_rng": generator.get_state(), "config": config,
        "fold": fold, "condition_id": "R2", "history": history,
    }
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, config, fold, generator):
    # Only load user-selected, trusted local checkpoints (contains Python/NumPy RNG objects).
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["condition_id"] != "R2" or state["fold"] != fold
            or config_with_effective_loader_settings(state["config"])
            != config_with_effective_loader_settings(config)):
        raise ValueError("Resume checkpoint fold/config/condition does not match this experiment.")
    epoch = int(state["epoch"])
    if not 1 <= epoch <= int(config["training"]["epochs"]):
        raise ValueError("Invalid checkpoint epoch.")
    if [row["epoch"] for row in state["history"]] != list(range(1, epoch + 1)):
        raise ValueError("Checkpoint history is not contiguous through its saved epoch.")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    generator.set_state(state["loader_rng"])
    return epoch + 1, state["history"]


def gpu_memory(device):
    return {
        "allocated_memory_gib": torch.cuda.memory_allocated(device) / 2**30,
        "reserved_memory_gib": torch.cuda.memory_reserved(device) / 2**30,
        "gpu_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
