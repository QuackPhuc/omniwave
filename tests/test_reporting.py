"""Tests for result collection and report generation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_results import summarize_runs


def test_summarize_runs_reads_training_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    records = [
        {"epoch": 0, "train_loss": 2.0},
        {"epoch": 0, "val_top1": 0.5},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    summary = summarize_runs(tmp_path)

    assert summary["runs"][0]["final_metrics"] == records[-1]
    assert summary["runs"][0]["total_steps"] == 2
