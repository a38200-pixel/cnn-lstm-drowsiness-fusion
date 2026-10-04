"""Shared R2 execution settings; sampling and preprocessing are not changed."""


def loader_settings(training):
    workers = int(training.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    persistent = bool(training.get("persistent_workers", False)) if workers else False
    prefetch = training.get("prefetch_factor", 2) if workers else None
    if workers and (not isinstance(prefetch, int) or prefetch < 1):
        raise ValueError("prefetch_factor must be a positive integer with workers > 0")
    return {
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True)),
        "persistent_workers": persistent,
        "prefetch_factor": prefetch,
    }


def config_with_effective_loader_settings(config):
    """Explicit execution defaults must not invalidate otherwise identical checkpoints."""
    return {**config, "training": {**config["training"], **loader_settings(config["training"])}}
