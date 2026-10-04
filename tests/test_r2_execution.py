"""Small CPU tests for CUDA policy and exact epoch checkpoint continuation."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import Adam

from drowsiness_fusion.training.r2_execution import require_cuda, save_checkpoint, load_checkpoint


def test_cuda_required(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        require_cuda()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_r2_vgg19
    monkeypatch.setattr(sys, "argv", ["train_r2_vgg19.py", "--config", "missing.yaml"])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        train_r2_vgg19.main()


def test_checkpoint_resume_optimizer_rng_and_next_update(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = nn.Linear(2, 1)
    optimizer = Adam(model.parameters(), lr=0.0001)
    generator = torch.Generator().manual_seed(42)
    x = torch.ones(1, 2)
    model(x).sum().backward()
    optimizer.step()
    config = {"training": {"epochs": 30}}
    history = [{"epoch": epoch} for epoch in range(1, 26)]
    path = tmp_path / "last_checkpoint.pt"
    save_checkpoint(path, model, optimizer, 25, config, 1, history, generator)
    expected_rng = (random.random(), np.random.rand(), torch.rand(1),
                    torch.rand(1, generator=generator))
    optimizer.zero_grad()
    model(x).sum().backward()
    optimizer.step()
    expected_weights = [p.detach().clone() for p in model.parameters()]
    restored = nn.Linear(2, 1)
    restored_optimizer = Adam(restored.parameters(), lr=0.5)
    start, restored_history = load_checkpoint(path, restored, restored_optimizer, config, 1, generator)
    assert start == 26
    assert restored_history == history
    assert restored_optimizer.param_groups[0]["lr"] == 0.0001
    assert all(state["step"].item() == 1 for state in restored_optimizer.state.values())
    assert random.random() == expected_rng[0]
    assert np.random.rand() == expected_rng[1]
    assert torch.equal(torch.rand(1), expected_rng[2])
    assert torch.equal(torch.rand(1, generator=generator), expected_rng[3])
    restored_optimizer.zero_grad()
    restored(x).sum().backward()
    restored_optimizer.step()
    assert all(torch.equal(actual, expected)
               for actual, expected in zip(restored.parameters(), expected_weights))
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, restored, restored_optimizer, config, 2, generator)
