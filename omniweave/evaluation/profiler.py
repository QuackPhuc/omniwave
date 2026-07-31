"""Profiling utilities for training and inference."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def cuda_timer() -> Generator[dict[str, float], None, None]:
    """Context manager that measures elapsed CUDA time.

    Usage::

        with cuda_timer() as timer:
            model(x)
        print(f"elapsed: {timer['elapsed_ms']:.2f} ms")
    """
    result: dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield result
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    result["elapsed_ms"] = (t1 - t0) * 1000
    result["elapsed_seconds"] = t1 - t0


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def module_parameter_summary(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Per-module parameter counts for debugging."""
    summary: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            summary.append({"name": name, "parameters": params})
    return summary
