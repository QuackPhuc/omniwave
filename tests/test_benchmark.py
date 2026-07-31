"""Tests for benchmark infrastructure."""

from __future__ import annotations

import time

import torch

from omniweave.evaluation.benchmark import BenchmarkResult, benchmark_fn
from omniweave.models.routing import build_route_plan
from scripts.benchmark import _make_operator_tensors, _run_bigemm
from scripts.check_gates import evaluate_operator_gates


def test_percentile_ordering() -> None:
    """Verify p50 ≤ p90 ≤ p99."""
    def fn():
        time.sleep(0.0001)

    result = benchmark_fn(fn, warmup=5, iterations=20, cuda_sync=False)
    assert result.p50_seconds <= result.p90_seconds
    assert result.p90_seconds <= result.p99_seconds


def test_mean_positive() -> None:
    def fn():
        _ = sum(range(100))

    result = benchmark_fn(fn, warmup=5, iterations=50, cuda_sync=False)
    assert result.mean_seconds > 0
    assert result.throughput_per_second > 0


def test_result_fields() -> None:
    result = benchmark_fn(lambda: None, warmup=2, iterations=10, cuda_sync=False)
    assert isinstance(result, BenchmarkResult)
    assert result.warmup == 2
    assert result.iterations == 10
    assert result.peak_allocated_bytes >= 0
    assert result.peak_reserved_bytes >= 0


def test_known_mean() -> None:
    """Verify mean for a known distribution."""

    # Use a deterministic function with near-zero variance
    def fn():
        pass

    result = benchmark_fn(fn, warmup=5, iterations=100, cuda_sync=False)
    # Mean should be very small for no-op
    assert result.mean_seconds < 0.01


def test_gate_rejects_missing_numerical_evidence() -> None:
    rows = [
        {
            "shape": f"shape-{index}",
            "baseline_ms": 2.0,
            "triton_ms": 1.0,
            "baseline_memory": 100,
            "triton_memory": 90,
        }
        for index in range(2)
    ]
    assert evaluate_operator_gates(rows)["passed"] is False


def test_gate_requires_numerical_and_performance_pass() -> None:
    rows = [
        {
            "shape": f"shape-{index}",
            "baseline_ms": 2.0,
            "triton_ms": 1.0,
            "baseline_memory": 100,
            "triton_memory": 90,
            "forward_max_abs_error": 1e-3,
            "backward_max_abs_error": 2e-3,
            "numerical_tolerance": 5e-3,
        }
        for index in range(2)
    ]
    assert evaluate_operator_gates(rows)["passed"] is True


def test_gate_rejects_large_numerical_error() -> None:
    rows = [
        {
            "shape": f"shape-{index}",
            "baseline_ms": 2.0,
            "triton_ms": 1.0,
            "baseline_memory": 100,
            "triton_memory": 90,
            "forward_max_abs_error": 1.0,
            "backward_max_abs_error": 1.0,
        }
        for index in range(2)
    ]
    assert evaluate_operator_gates(rows)["passed"] is False


def test_gate_requires_distinct_passing_shapes() -> None:
    row = {
        "shape": "56x56x128",
        "baseline_ms": 2.0,
        "triton_ms": 1.0,
        "baseline_memory": 100,
        "triton_memory": 90,
        "forward_max_abs_error": 1e-3,
        "backward_max_abs_error": 2e-3,
        "numerical_tolerance": 5e-3,
    }
    result = evaluate_operator_gates([row, dict(row)])
    assert result["passed"] is False
    assert result["passing_shapes"] == ["56x56x128"]


def test_operator_benchmark_initialization_is_finite() -> None:
    tensors = _make_operator_tensors(
        batch_size=1,
        height=8,
        width=8,
        channels=16,
        spatial_tile=4,
        channel_tile=8,
        expansion=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    plan = build_route_plan(8, 8, 16, 4, 8, "local")
    output = _run_bigemm(plan, "reference", tensors)
    assert torch.isfinite(output).all()
    assert torch.equal(tensors["gamma"], torch.full((16,), 1e-5))
