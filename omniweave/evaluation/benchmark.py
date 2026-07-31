"""Benchmark harness — synchronized timing with CUDA synchronization."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class BenchmarkResult:
    """Performance measurement result."""

    p50_seconds: float
    p90_seconds: float
    p99_seconds: float
    mean_seconds: float
    throughput_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    warmup: int
    iterations: int


def benchmark_fn(
    fn: Callable[[], Any],
    warmup: int = 100,
    iterations: int = 1000,
    cuda_sync: bool = True,
) -> BenchmarkResult:
    """Run a function with synchronized timing.

    Parameters
    ----------
    fn : callable to benchmark (no arguments)
    warmup : warm-up iterations before measurement
    iterations : measured iterations
    cuda_sync : synchronize CUDA before and after each iteration
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    # Warm up
    for _ in range(warmup):
        fn()

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        if cuda_sync:
            torch.cuda.synchronize()

    # Measure
    times: list[float] = []
    for _ in range(iterations):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times)

    peak_alloc = 0
    peak_reserved = 0
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

    return BenchmarkResult(
        p50_seconds=float(np.percentile(arr, 50)),
        p90_seconds=float(np.percentile(arr, 90)),
        p99_seconds=float(np.percentile(arr, 99)),
        mean_seconds=float(arr.mean()),
        throughput_per_second=1.0 / max(float(arr.mean()), 1e-12),
        peak_allocated_bytes=peak_alloc,
        peak_reserved_bytes=peak_reserved,
        warmup=warmup,
        iterations=iterations,
    )


def benchmark_operator(
    fn: Callable[[], Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Benchmark an operator with configuration metadata."""
    result = benchmark_fn(
        fn,
        warmup=config.get("warmup", 100),
        iterations=config.get("iterations", 1000),
    )
    return {
        **config,
        "p50_ms": result.p50_seconds * 1000,
        "p90_ms": result.p90_seconds * 1000,
        "p99_ms": result.p99_seconds * 1000,
        "mean_ms": result.mean_seconds * 1000,
        "throughput_per_second": result.throughput_per_second,
        "peak_allocated_mb": result.peak_allocated_bytes / (1024 ** 2),
        "peak_reserved_mb": result.peak_reserved_bytes / (1024 ** 2),
    }
