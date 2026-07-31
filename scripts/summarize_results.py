"""Summarize experiment results from multiple runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_runs(runs_root: Path) -> dict[str, Any]:
    """Collect and summarize results from all experiment runs."""
    summary: dict[str, Any] = {
        "runs": [],
        "gates": {},
    }

    if not runs_root.exists():
        return summary

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_info: dict[str, Any] = {"name": run_dir.name}

        # Collect train logs
        log_path = run_dir / "metrics.jsonl"
        if log_path.exists():
            lines = log_path.read_text().strip().split("\n")
            records = [json.loads(line) for line in lines if line.strip()]
            if records:
                run_info["final_metrics"] = records[-1]
                run_info["total_steps"] = len(records)

        # Collect gate results
        gates_path = run_dir / "gates.json"
        if gates_path.exists():
            run_info["gates"] = json.loads(gates_path.read_text())

        # Collect benchmark results
        results_path = run_dir / "results.json"
        if results_path.exists():
            run_info["benchmark"] = json.loads(results_path.read_text())

        summary["runs"].append(run_info)

    return summary


def generate_markdown_tables(summary: dict[str, Any]) -> str:
    """Generate markdown benchmark tables from summary."""
    lines: list[str] = ["# OmniWeave Benchmark Results\n"]

    for run in summary.get("runs", []):
        lines.append(f"## {run['name']}\n")

        if "final_metrics" in run:
            metrics = run["final_metrics"]
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in sorted(metrics.items()):
                lines.append(f"| {k} | {v} |")
            lines.append("")

        if "gates" in run:
            gates = run["gates"]
            status = "✓ PASSED" if gates.get("passed") else "✗ FAILED"
            lines.append(f"**Gate Status:** {status}\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--runs-root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary", type=str, default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    summary = summarize_runs(runs_root)

    # Write JSON summary
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")

    # Write markdown
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md = generate_markdown_tables(summary)
    output_path.write_text(md, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
