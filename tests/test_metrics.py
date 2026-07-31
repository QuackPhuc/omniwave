"""Tests for evaluation metrics and experiment hashing."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from omniweave.evaluation.metrics import expected_calibration_error, topk_accuracy
from omniweave.utils.environment import collect_environment
from omniweave.utils.logging import JsonlLogger, experiment_id


def test_topk_accuracy_perfect() -> None:
    logits = torch.eye(5)  # Perfect one-hot predictions
    targets = torch.arange(5)
    result = topk_accuracy(logits, targets, topk=(1, 5))
    assert result["top1"] == 1.0
    assert result["top5"] == 1.0


def test_topk_accuracy_known() -> None:
    # logit[0] predicts class 1, but target is 0 → top1 miss, top2 hit
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0]])
    targets = torch.tensor([0, 0])
    result = topk_accuracy(logits, targets, topk=(1, 2))
    assert result["top1"] == 0.5
    assert result["top2"] == 1.0


def test_topk_accuracy_empty() -> None:
    logits = torch.zeros(0, 5)
    targets = torch.zeros(0, dtype=torch.long)
    result = topk_accuracy(logits, targets, topk=(1,))
    assert result["top1"] == 0.0


def test_topk_accuracy_clamps_to_available_classes() -> None:
    logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    targets = torch.tensor([1, 0])
    result = topk_accuracy(logits, targets, topk=(1, 5))
    assert result == {"top1": 1.0, "top5": 1.0}


def test_ece_near_perfect() -> None:
    # Confident correct predictions → low ECE
    logits = torch.zeros(100, 10)
    targets = torch.zeros(100, dtype=torch.long)
    logits[:, 0] = 10.0  # Very confident in class 0
    ece = expected_calibration_error(logits, targets)
    assert ece < 0.05


def test_ece_bad_calibration() -> None:
    # Confident but wrong predictions → high ECE
    logits = torch.zeros(100, 10)
    targets = torch.ones(100, dtype=torch.long)
    logits[:, 0] = 10.0  # Very confident in wrong class
    ece = expected_calibration_error(logits, targets)
    assert ece > 0.5


def test_jsonl_logger(tmp_path: Path) -> None:
    log_path = tmp_path / "test.jsonl"
    with JsonlLogger(log_path) as logger:
        logger.write({"epoch": 1, "loss": 2.5})
        logger.write({"epoch": 2, "loss": 1.8})

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "epoch" in record
        assert "loss" in record


def test_experiment_id_deterministic() -> None:
    cfg = {"model": "test", "lr": 0.001}
    id1 = experiment_id(cfg, "abc123", "hash456")
    id2 = experiment_id(cfg, "abc123", "hash456")
    assert id1 == id2
    assert len(id1) == 16


def test_experiment_id_changes_with_input() -> None:
    cfg = {"model": "test"}
    id1 = experiment_id(cfg, "rev1", "hash1")
    id2 = experiment_id(cfg, "rev2", "hash1")
    assert id1 != id2


def test_environment_collection() -> None:
    env = collect_environment()
    assert "python" in env
    assert "pytorch" in env
    assert "cuda" in env
    assert "git" in env
