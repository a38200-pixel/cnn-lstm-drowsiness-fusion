"""Small CPU tests for CUDA policy and exact epoch checkpoint continuation."""
import random
import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import Adam

from drowsiness_fusion.training.r2_execution import require_cuda, save_checkpoint, load_checkpoint


@pytest.mark.parametrize("script", ["train_r2_vgg19", "profile_r2_vgg19"])
@pytest.mark.parametrize("overrides", [[], ["--num-workers", "4", "--pin-memory",
                                        "--persistent-workers", "--prefetch-factor", "1"],
                                      ["--num-workers", "0", "--no-pin-memory",
                                       "--no-persistent-workers"]])
def test_loader_cli_overrides_preserve_research_config(script, overrides, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module(script)
    import train_r2_vgg19
    from drowsiness_fusion.data.r2_dataloader import loader_settings
    path = Path(__file__).resolve().parents[1] / "configs/paper_reconstruction/vgg19_lstm_r2_lastblock_finetune.yaml"
    original = train_r2_vgg19.load_config(path)
    effective = copy.deepcopy(original)
    monkeypatch.setattr(module, "load_config", lambda _: effective)
    monkeypatch.setattr(module, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(module, "seed_everything", lambda _: None)

    def inspect_config(config, fold, device):
        expected = copy.deepcopy(original)
        if overrides:
            expected["training"].update(num_workers=int(overrides[1]),
                                       pin_memory=overrides[1] == "4",
                                       persistent_workers=overrides[1] == "4")
            if overrides[1] == "4":
                expected["training"]["prefetch_factor"] = 1
        assert config == expected
        settings = loader_settings(config["training"])
        if overrides and overrides[1] == "0":
            assert settings["prefetch_factor"] is None
            assert settings["persistent_workers"] is False
        raise RuntimeError("configuration inspected before execution")

    monkeypatch.setattr(module, "environment", inspect_config)
    output_flag = "--output-root" if script == "train_r2_vgg19" else "--output"
    monkeypatch.setattr(sys, "argv", [script, "--config", str(path), "--fold", "1",
                                     output_flag, str(tmp_path / "new-output"), *overrides])
    with pytest.raises(RuntimeError, match="configuration inspected"):
        module.main()
    assert not (tmp_path / "new-output").exists()


@pytest.mark.parametrize("script", ["train_r2_vgg19", "profile_r2_vgg19"])
def test_cli_rejects_nonpositive_prefetch_before_execution(script, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module(script)
    monkeypatch.setattr(sys, "argv", [script, "--config", "missing.yaml",
                                     "--prefetch-factor", "0", *(
                                         ["--output", "unused.json"]
                                         if script == "profile_r2_vgg19" else [])])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


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
