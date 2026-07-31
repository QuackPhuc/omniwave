"""Atomic checkpointing and exact resume."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and all CUDA RNG states."""
    import random

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore previously captured RNG states."""
    import random

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])

    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    ema_state: dict[str, Any] | None = None,
    epoch: int = 0,
    global_step: int = 0,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a checkpoint atomically (write to .tmp then replace).

    Returns the final checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "rng": capture_rng_state(),
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if ema_state is not None:
        payload["ema"] = ema_state
    if config is not None:
        payload["config"] = config
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint with strict model loading.

    Returns the full checkpoint dict (includes epoch, global_step, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if restore_rng and "rng" in checkpoint:
        restore_rng_state(checkpoint["rng"])

    return cast(dict[str, Any], checkpoint)
