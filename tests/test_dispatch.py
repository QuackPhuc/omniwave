"""Tests for backend dispatch."""

from __future__ import annotations

import logging

import pytest
import torch

from omniweave.ops.dispatch import select_backend


def test_auto_falls_back_on_cpu(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        backend, reason = select_backend(
            torch.randn(1),
            "auto",
            lambda _: (False, "CUDA is unavailable"),
        )
    assert backend == "reference"
    assert reason == "CUDA is unavailable"
    assert "CUDA is unavailable" in caplog.text


def test_forced_triton_fails() -> None:
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        select_backend(
            torch.randn(1),
            "triton",
            lambda _: (False, "CUDA is unavailable"),
        )


def test_reference_passthrough() -> None:
    backend, reason = select_backend(
        torch.randn(1),
        "reference",
        lambda _: (True, None),
    )
    assert backend == "reference"
    assert reason is None


def test_compile_passthrough() -> None:
    backend, reason = select_backend(
        torch.randn(1),
        "compile",
        lambda _: (True, None),
    )
    assert backend == "compile"
    assert reason is None


def test_auto_selects_triton_when_supported() -> None:
    backend, reason = select_backend(
        torch.randn(1),
        "auto",
        lambda _: (True, None),
    )
    assert backend == "triton"
    assert reason is None


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        select_backend(torch.randn(1), "cuda_graphs", lambda _: (True, None))
