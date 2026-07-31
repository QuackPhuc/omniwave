"""CLI entry point for benchmarking."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


def _bigemm_with_gradients(
    backend: str,
    plan: Any,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    names = ("x", "a_u", "a_v", "a_o", "b_u", "b_v", "b_o", "gamma")
    values = [
        tensors[name].detach().clone().requires_grad_(True)
        for name in names
    ]
    kwargs = dict(zip(names, values, strict=True))

    from omniweave.ops.bigemm import bigemm

    output = bigemm(plan=plan, backend=backend, **kwargs)
    gradients = torch.autograd.grad(output.float().sum(), values)
    return output.detach(), [gradient.detach() for gradient in gradients]


def _numerical_errors(
    backend: str,
    plan: Any,
    tensors: dict[str, torch.Tensor],
) -> tuple[float, float]:
    if backend == "reference":
        return 0.0, 0.0

    reference_output, reference_gradients = _bigemm_with_gradients(
        "reference", plan, tensors
    )
    output, gradients = _bigemm_with_gradients(backend, plan, tensors)
    forward_error = (output - reference_output).abs().max().item()
    backward_error = max(
        (actual - expected).abs().max().item()
        for actual, expected in zip(gradients, reference_gradients, strict=True)
    )
    return forward_error, backward_error


def _make_operator_tensors(
    *,
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    spatial_tile: int,
    channel_tile: int,
    expansion: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Initialize a benchmark case with the same scales as OmniWeaveBlock."""
    expanded_tile = channel_tile * expansion
    channel_groups = math.ceil(channels / channel_tile)

    def initialized(
        shape: tuple[int, ...],
        initializer: Any,
    ) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=torch.float32, device=device)
        initializer(tensor)
        return tensor.to(dtype)

    return {
        "x": torch.randn(
            batch_size,
            height,
            width,
            channels,
            dtype=dtype,
            device=device,
        ),
        "a_u": initialized((spatial_tile, spatial_tile), nn.init.orthogonal_),
        "a_v": initialized((spatial_tile, spatial_tile), nn.init.orthogonal_),
        "a_o": initialized((spatial_tile, spatial_tile), nn.init.eye_),
        "b_u": initialized(
            (channel_groups, channel_tile, expanded_tile),
            nn.init.xavier_uniform_,
        ),
        "b_v": initialized(
            (channel_groups, channel_tile, expanded_tile),
            nn.init.xavier_uniform_,
        ),
        "b_o": initialized(
            (channel_groups, expanded_tile, channel_tile),
            nn.init.xavier_uniform_,
        ),
        "gamma": torch.full(
            (channels,),
            1e-5,
            dtype=dtype,
            device=device,
        ),
    }


@torch.inference_mode()
def _run_bigemm(plan: Any, backend: str, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    from omniweave.ops.bigemm import bigemm

    return bigemm(plan=plan, backend=backend, **tensors)


@torch.inference_mode()
def _run_model(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model(inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OmniWeave benchmarks")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to benchmark on",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")

    results: list[dict[str, Any]] = []

    # Operator benchmark
    if "shapes" in config:
        from omniweave.evaluation.benchmark import benchmark_fn
        from omniweave.models.routing import build_route_plan

        for shape in config["shapes"]:
            for batch_size in config.get("batch_sizes", [1]):
                for dtype_name in config.get("dtypes", ["float32"]):
                    dtype = getattr(torch, dtype_name)
                    for backend in config.get("backends", ["reference"]):
                        h, w = shape["height"], shape["width"]
                        c = shape["channels"]
                        d = shape["channel_tile"]
                        g = 16

                        try:
                            plan = build_route_plan(h, w, c, g, d, "local")
                            tensors = _make_operator_tensors(
                                batch_size=batch_size,
                                height=h,
                                width=w,
                                channels=c,
                                spatial_tile=g,
                                channel_tile=d,
                                expansion=2,
                                dtype=dtype,
                                device=device,
                            )
                            probe = _run_bigemm(plan, backend, tensors)
                            if not torch.isfinite(probe).all():
                                raise RuntimeError(
                                    "operator produced non-finite output"
                                )
                            forward_error, backward_error = _numerical_errors(
                                backend, plan, tensors
                            )
                            run = partial(_run_bigemm, plan, backend, tensors)

                            result = benchmark_fn(
                                run,
                                warmup=config.get("warmup", 100),
                                iterations=config.get("iterations", 1000),
                            )

                            results.append({
                                "shape": f"{h}x{w}x{c}",
                                "batch_size": batch_size,
                                "dtype": dtype_name,
                                "backend": backend,
                                "p50_ms": result.p50_seconds * 1000,
                                "p90_ms": result.p90_seconds * 1000,
                                "p99_ms": result.p99_seconds * 1000,
                                "mean_ms": result.mean_seconds * 1000,
                                "throughput": (
                                    batch_size * result.throughput_per_second
                                ),
                                "peak_allocated_mb": (
                                    result.peak_allocated_bytes / (1024**2)
                                ),
                                "peak_reserved_mb": (
                                    result.peak_reserved_bytes / (1024**2)
                                ),
                                "forward_max_abs_error": forward_error,
                                "backward_max_abs_error": backward_error,
                                "numerical_tolerance": (
                                    1e-4 if dtype == torch.float32 else 5e-3
                                ),
                            })
                        except (ValueError, RuntimeError) as e:
                            results.append({
                                "shape": f"{h}x{w}x{c}",
                                "batch_size": batch_size,
                                "dtype": dtype_name,
                                "backend": backend,
                                "error": str(e),
                            })

    if "resolutions" in config:
        from omniweave.evaluation.benchmark import benchmark_fn
        from omniweave.models.registry import create_model

        latency_batches = config.get("latency_batch_sizes", [1, 8])
        throughput_batches = config.get("throughput_batch_sizes", [32, 64])
        benchmark_batches = [
            ("latency", batch_size) for batch_size in latency_batches
        ] + [
            ("throughput", batch_size) for batch_size in throughput_batches
        ]

        for model_name in config.get("models", ["omniweave_t"]):
            for resolution in config["resolutions"]:
                for backend in config.get("backends", ["reference"]):
                    for mode, batch_size in benchmark_batches:
                        try:
                            model = create_model(
                                model_name,
                                backend=backend,
                                input_size=resolution,
                            ).to(device).eval()
                            inputs = torch.randn(
                                batch_size,
                                3,
                                resolution,
                                resolution,
                                device=device,
                            )
                            run = partial(_run_model, model, inputs)
                            result = benchmark_fn(
                                run,
                                warmup=config.get("warmup", 100),
                                iterations=config.get("iterations", 1000),
                            )
                            results.append({
                                "kind": "model",
                                "model": model_name,
                                "resolution": resolution,
                                "performance_only": resolution != 224,
                                "mode": mode,
                                "batch_size": batch_size,
                                "backend": backend,
                                "p50_ms": result.p50_seconds * 1000,
                                "p90_ms": result.p90_seconds * 1000,
                                "p99_ms": result.p99_seconds * 1000,
                                "mean_ms": result.mean_seconds * 1000,
                                "images_per_second": (
                                    batch_size * result.throughput_per_second
                                ),
                                "peak_allocated_mb": (
                                    result.peak_allocated_bytes / (1024**2)
                                ),
                                "peak_reserved_mb": (
                                    result.peak_reserved_bytes / (1024**2)
                                ),
                            })
                        except (ValueError, RuntimeError) as exc:
                            results.append({
                                "kind": "model",
                                "model": model_name,
                                "resolution": resolution,
                                "mode": mode,
                                "batch_size": batch_size,
                                "backend": backend,
                                "error": str(exc),
                            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
