"""Gate evaluation script — checks operator gates A/B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def evaluate_operator_gates(
    rows: list[dict[str, Any]],
    speedup_threshold: float = 1.3,
    min_passing_shapes: int = 2,
) -> dict[str, Any]:
    """Evaluate operator gates from benchmark results.

    Gate A: numerical forward/backward tolerance.
    Gate B: ≥1.3× speedup on at least 2 shapes, no peak-memory regression.

    Parameters
    ----------
    rows : list of benchmark result dicts, each with:
        - shape: str
        - baseline_ms: float
        - triton_ms: float
        - baseline_memory: int
        - triton_memory: int

    Returns
    -------
    gate result dict with ``passed``, ``passing_shapes``, ``details``
    """
    passing_shapes: set[str] = set()
    details: list[dict[str, Any]] = []

    for row in rows:
        speedup = row["baseline_ms"] / max(row["triton_ms"], 1e-12)
        memory_ok = row["triton_memory"] <= row["baseline_memory"]
        tolerance = float(row.get("numerical_tolerance", 5e-3))
        forward_error = row.get("forward_max_abs_error")
        backward_error = row.get("backward_max_abs_error")
        numerical_ok = (
            forward_error is not None
            and backward_error is not None
            and float(forward_error) <= tolerance
            and float(backward_error) <= tolerance
        )
        shape_pass = (
            numerical_ok
            and speedup >= speedup_threshold
            and memory_ok
        )

        detail = {
            "shape": row["shape"],
            "speedup": round(speedup, 3),
            "memory_ok": memory_ok,
            "numerical_ok": numerical_ok,
            "forward_max_abs_error": forward_error,
            "backward_max_abs_error": backward_error,
            "numerical_tolerance": tolerance,
            "passed": shape_pass,
        }
        details.append(detail)

        if shape_pass:
            passing_shapes.add(row["shape"])

    passed = len(passing_shapes) >= min_passing_shapes

    return {
        "passed": passed,
        "passing_shapes": sorted(passing_shapes),
        "total_shapes": len({row["shape"] for row in rows}),
        "speedup_threshold": speedup_threshold,
        "min_passing_shapes": min_passing_shapes,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check operator gates A/B")
    parser.add_argument("--operator-results", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--speedup-threshold", type=float, default=1.3)
    parser.add_argument("--min-passing-shapes", type=int, default=2)
    args = parser.parse_args()

    with open(args.operator_results, encoding="utf-8") as f:
        results = json.load(f)

    # Pair baseline (reference) and triton results by shape
    by_shape: dict[str, dict[str, Any]] = {}
    for row in results:
        shape = row.get("shape", "")
        if "error" in row:
            continue
        backend = row.get("backend", "")
        if backend in ("reference", "compile"):
            key = f"{shape}_{row.get('batch_size')}_{row.get('dtype')}"
            if key not in by_shape:
                by_shape[key] = {}
            candidate_ms = row.get("mean_ms", row.get("p50_ms", 0))
            if (
                "baseline_ms" not in by_shape[key]
                or candidate_ms < by_shape[key]["baseline_ms"]
            ):
                by_shape[key]["baseline_ms"] = candidate_ms
                by_shape[key]["baseline_memory"] = row.get(
                    "peak_allocated_mb", float("inf")
                )
            by_shape[key]["shape"] = shape
        elif backend == "triton":
            key = f"{shape}_{row.get('batch_size')}_{row.get('dtype')}"
            if key not in by_shape:
                by_shape[key] = {}
            by_shape[key]["triton_ms"] = row.get("mean_ms", row.get("p50_ms", 0))
            by_shape[key]["triton_memory"] = row.get(
                "peak_allocated_mb", float("inf")
            )
            by_shape[key]["forward_max_abs_error"] = row.get(
                "forward_max_abs_error"
            )
            by_shape[key]["backward_max_abs_error"] = row.get(
                "backward_max_abs_error"
            )
            by_shape[key]["numerical_tolerance"] = row.get(
                "numerical_tolerance", 5e-3
            )
            by_shape[key]["shape"] = shape

    # Filter to rows that have both baseline and triton
    paired = [
        v for v in by_shape.values()
        if "baseline_ms" in v and "triton_ms" in v
    ]

    gate_result = evaluate_operator_gates(
        paired,
        speedup_threshold=args.speedup_threshold,
        min_passing_shapes=args.min_passing_shapes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gate_result, f, indent=2)

    if gate_result["passed"]:
        print(f"✓ Gate B PASSED: {len(gate_result['passing_shapes'])} shapes meet threshold")
        sys.exit(0)
    else:
        print(f"✗ Gate B FAILED: only {len(gate_result['passing_shapes'])} shapes meet threshold")
        sys.exit(2)


if __name__ == "__main__":
    main()
