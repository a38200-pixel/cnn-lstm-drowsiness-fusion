"""Random seed and deterministic execution helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
