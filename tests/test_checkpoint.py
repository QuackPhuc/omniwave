"""Tests for atomic checkpointing and exact resume."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from omniweave.models.registry import create_model_from_config
from omniweave.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from scripts.export_checkpoint import export_portable_checkpoint


def test_round_trip_checkpoint(tmp_path: Path) -> None:
    """Save and load a model+optimizer checkpoint, verify exact match."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Train one step
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    # Save
    ckpt_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=100,
        config={"lr": 0.01},
    )
    assert ckpt_path.exists()
    assert not ckpt_path.with_suffix(".pt.tmp").exists()

    # Load into fresh objects
    model2 = nn.Linear(10, 5)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    checkpoint = load_checkpoint(
        ckpt_path,
        model=model2,
        optimizer=optimizer2,
        restore_rng=False,
    )

    # Verify parameters match
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=False):
        torch.testing.assert_close(p1, p2)

    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 100
    assert checkpoint["config"]["lr"] == 0.01


def test_rng_state_round_trip() -> None:
    """Capture and restore RNG state, verify reproducible random output."""
    torch.manual_seed(42)
    state = capture_rng_state()
    val1 = torch.randn(5)

    # Generate different values
    _ = torch.randn(100)

    # Restore and regenerate — should match
    restore_rng_state(state)
    val2 = torch.randn(5)
    torch.testing.assert_close(val1, val2)


def test_checkpoint_missing_file(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    import pytest
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "nonexistent.pt", model)


def test_atomic_write_no_leftover(tmp_path: Path) -> None:
    """Verify no .tmp file remains after successful save."""
    model = nn.Linear(2, 2)
    path = tmp_path / "test.pt"
    save_checkpoint(path, model)
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()


def test_export_uses_checkpoint_model_config(tmp_path: Path) -> None:
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump({
            "model": {
                "name": "omniweave_t",
                "widths": [8, 16, 32, 64],
                "depths": [1, 1, 1, 1],
                "spatial_tile": 8,
                "channel_tiles": [8, 8, 8, 8],
                "expansion": 2,
                "layer_scale_init": 1e-5,
                "num_classes": 3,
                "anchor_stages": [],
                "backend": "reference",
            },
        }),
        encoding="utf-8",
    )
    model = create_model_from_config(str(model_config), backend="reference")
    checkpoint_path = tmp_path / "custom.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        config={"model_config": str(model_config)},
        extra={
            "dataset_manifest": {"class_to_idx": {"a": 0}},
            "environment": {"git": {"revision": "test"}},
        },
    )

    output_path = tmp_path / "portable.pt"
    export_portable_checkpoint(checkpoint_path, output_path)
    portable = torch.load(output_path, map_location="cpu", weights_only=False)
    assert portable["backend"] == "reference"
    assert portable["class_to_idx"] == {"a": 0}
    assert portable["source_revision"] == {"revision": "test"}
