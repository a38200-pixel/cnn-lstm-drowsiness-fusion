"""Bounded real-frame DataLoader benchmark. No model, optimizer or full epoch."""
import argparse
import ctypes
import gc
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from train_r2_vgg19 import load_config
from drowsiness_fusion.data.paper_feature_dataset import read_split_rows
from drowsiness_fusion.data.paper_frame_dataset import PaperFrameDataset
from drowsiness_fusion.data.r2_dataloader import loader_settings
from drowsiness_fusion.training.r2_execution import require_cuda, environment, atomic_json
from drowsiness_fusion.utils.reproducibility import seed_everything


def cpu_info():
    info = {"cpu_model": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "physical_core_count": None, "parent_torch_threads": torch.get_num_threads(),
            "worker_torch_threads": 1}
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            info["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        pass
    return info


def host_memory():
    """System RAM counters from Windows; no extra dependency or process enumeration."""
    if os.name != "nt":
        return {}
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "total_pagefile", "available_pagefile",
                "total_virtual", "available_virtual", "available_extended_virtual")]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {}
    return {"system_ram_total_gib": status.total_physical/2**30,
            "system_ram_available_gib": status.available_physical/2**30}


def benchmark(dataset, config, workers, prefetch, warmup, batches, mode, device):
    settings = loader_settings({"num_workers": workers, "pin_memory": True,
                                "persistent_workers": workers > 0,
                                "prefetch_factor": prefetch})
    generator = torch.Generator().manual_seed(int(config["training"]["seed"]))
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False,
                        generator=generator, **settings)
    started = perf_counter()
    iterator = None
    result = {**settings, "mode": mode, "warmup_batches": warmup,
              "measured_batches": batches, "shuffle": True, "drop_last": False,
              "sampler": "RandomSampler", "collate_fn": "default_collate"}
    try:
        iterator = iter(loader)
        durations, loading, transfers, identities, available_ram = [], [], [], [], []
        measured_start = None
        for index in range(warmup + batches):
            start = perf_counter()
            if index == warmup:
                measured_start = start
                result["warmup_seconds"] = start-started
            batch = next(iterator)
            loaded = perf_counter()
            if tuple(batch["frames"].shape) != (16, 20, 3, 224, 224):
                raise RuntimeError("Unexpected benchmark batch shape")
            if mode == "loader_plus_h2d":
                frames = batch["frames"].to(device)
                labels = batch["label"].to(device)
                torch.cuda.synchronize(device)
                del frames, labels
            end = perf_counter()
            available = host_memory().get("system_ram_available_gib")
            if available is not None:
                available_ram.append(available)
            if index >= warmup:
                loading.append(loaded-start)
                transfers.append(end-loaded)
                durations.append(end-start)
                identities.extend(zip(batch["video_id"], batch["label"].tolist()))
            print(f"workers={workers} prefetch={prefetch} batch={index+1}/{warmup+batches} "
                  f"warmup={index < warmup} data_seconds={loaded-start:.3f}", flush=True)
            del batch
        measured_seconds = perf_counter()-measured_start
        result.update({
            "status": "completed", "measured_seconds": measured_seconds,
            "average_data_loading_ms": float(np.mean(loading)*1000),
            "median_data_loading_ms": float(np.median(loading)*1000),
            "p95_data_loading_ms": float(np.percentile(loading, 95)*1000),
            "average_h2d_ms": float(np.mean(transfers)*1000) if mode == "loader_plus_h2d" else None,
            "average_batch_total_ms": float(np.mean(durations)*1000),
            "sequences_per_second": batches*16/measured_seconds,
            "samples_per_second": batches*16/measured_seconds,
            "frames_per_second": batches*16*20/measured_seconds,
            "data_loading_seconds": loading, "batch_total_seconds": durations,
            "measured_sequence_labels": identities,
            "min_system_ram_available_gib": min(available_ram) if available_ram else None,
        })
    finally:
        # Persistent workers otherwise remain alive across candidate runs on Windows.
        if iterator is not None and workers:
            iterator._shutdown_workers()
        del iterator, loader
        gc.collect()
    result["condition_wall_seconds"] = perf_counter()-started
    return result


def recommend(conditions):
    completed = [c for c in conditions if c.get("status") == "completed"]
    if not completed:
        return {"status": "NO_SUCCESSFUL_CANDIDATE"}
    best = min(completed, key=lambda c: c["average_data_loading_ms"])
    # Prefer fewer processes when mean latency is within 3% and tails are comparable.
    near = [c for c in completed
            if c["average_data_loading_ms"] <= best["average_data_loading_ms"]*1.03
            and c["p95_data_loading_ms"] <= best["p95_data_loading_ms"]*1.10]
    chosen = min(near, key=lambda c: (c["num_workers"], c["prefetch_factor"] or 0))
    return {"status": "RECOMMENDATION_ONLY", **{key: chosen[key] for key in (
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
        "average_data_loading_ms", "median_data_loading_ms", "p95_data_loading_ms",
        "sequences_per_second")},
        "measured_batches": chosen["measured_batches"],
        "measurement_window_exceeds_prefetch": chosen["measured_batches"] >
            chosen["num_workers"] * (chosen["prefetch_factor"] or 0),
        "failed_candidates": sum(c.get("status") == "failed" for c in conditions),
        "reason": "Measured mean loader wait; prefer fewer workers within 3% with comparable p95. "
                  "Bounded sequential trials are cache/order sensitive; review before changing config. "
                  "A window shorter than the prefetch queue is insufficient to establish steady throughput."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", choices=(0, 2, 4, 8), default=[0, 2, 4, 8])
    parser.add_argument("--prefetch-workers", type=int, nargs="*", choices=(2, 4, 8), default=[])
    parser.add_argument("--mode", choices=("loader_only", "loader_plus_h2d"), default="loader_plus_h2d")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batches <= 20 or not 1 <= args.warmup_batches <= 3:
        parser.error("batches must be 1..20; warmup batches 1..3")
    if len(args.prefetch_workers) > 2:
        parser.error("Compare prefetch=4 on at most two promising worker counts")
    if args.output.exists():
        raise FileExistsError("Choose a new benchmark output file.")
    device = require_cuda()
    config = load_config(args.config)
    seed_everything(int(config["training"]["seed"]))
    pairs = [(w, 2 if w else None) for w in dict.fromkeys(args.workers)]
    pairs += [(w, 4) for w in dict.fromkeys(args.prefetch_workers)]
    rows = read_split_rows(PROJECT_ROOT / config["data"]["split_root"] /
                           f"fold_{args.fold}_train.csv")
    # Keep additional bounded rows so queues do not drain at the measured subset's end.
    budget = (args.warmup_batches + args.batches + max(w*(p or 0) for w, p in pairs))*16
    ids = [row["video_id"] for row in rows[:min(budget, len(rows))]]
    dataset = PaperFrameDataset(
        PROJECT_ROOT / config["data"]["sequence_metadata"], video_ids=ids,
        sequence_length=int(config["sequence"]["length"]),
        input_size=tuple(config["cnn"]["input_size"]),
        normalization=config["cnn"]["normalization"],
    )
    result = {"environment": {**environment(config, args.fold, device), **cpu_info(), **host_memory()},
              "result_role": "diagnostic_only", "config_snapshot": config,
              "subset_sequences": len(dataset), "fold_train_sequences": len(rows),
              "notes": "Same bounded subset and shuffle seed for all candidates. Prefetch may decode "
                       "ahead by workers*prefetch batches; only warmup/measured batches consumed. "
                       "Worker startup excluded; filesystem caches not flushed. No model computation. "
                       "PyTorch workers use one intra-op thread by default; parent threads unchanged.",
              "conditions": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    expected_identities = None
    print(json.dumps(result["environment"], ensure_ascii=False), flush=True)
    for workers, prefetch in pairs:
        try:
            condition = benchmark(dataset, config, workers, prefetch, args.warmup_batches,
                                  args.batches, args.mode, device)
            if expected_identities is None:
                expected_identities = condition["measured_sequence_labels"]
            if condition["measured_sequence_labels"] != expected_identities:
                raise RuntimeError("Sequence/label order differs between worker candidates")
        except Exception as error:
            condition = {"num_workers": workers, "prefetch_factor": prefetch,
                         "status": "failed", "error": f"{type(error).__name__}: {error}"}
            print(json.dumps(condition), flush=True)
        result["conditions"].append(condition)
        result["recommendation"] = recommend(result["conditions"])
        result["benchmark_wall_seconds"] = perf_counter()-started
        atomic_json(args.output, result)
        print(json.dumps({k: v for k, v in condition.items()
                          if k not in ("measured_sequence_labels", "data_loading_seconds",
                                       "batch_total_seconds")}, indent=2), flush=True)
    print(json.dumps(result["recommendation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
