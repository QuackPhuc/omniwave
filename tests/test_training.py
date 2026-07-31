"""Tests for the training engine."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from omniweave.training.engine import (
    EMAModel,
    apply_batch_augment,
    evaluate_model,
    train,
    train_one_epoch,
)


def _make_loader(n_samples: int = 32, n_classes: int = 10) -> DataLoader:
    """Create a tiny DataLoader for testing."""
    images = torch.randn(n_samples, 3, 32, 32)
    targets = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(images, targets), batch_size=8)


def _make_model(n_classes: int = 10) -> nn.Module:
    """Simple conv model for training tests."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes),
    )


def test_train_one_epoch_changes_params() -> None:
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_loader()

    params_before = {k: v.clone() for k, v in model.state_dict().items()}

    metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=0,
    )

    # Parameters should change
    changed = False
    for k, v in model.state_dict().items():
        if not torch.equal(v, params_before[k]):
            changed = True
            break
    assert changed, "parameters did not change after training"

    assert "train_loss" in metrics
    assert "images_per_second" in metrics
    assert metrics["images_per_second"] > 0


@pytest.mark.parametrize(
    ("mixup_alpha", "cutmix_alpha"),
    [(0.8, 0.0), (0.0, 1.0), (0.8, 1.0)],
)
def test_batch_augment_preserves_shape_and_targets(
    mixup_alpha: float,
    cutmix_alpha: float,
) -> None:
    images = torch.randn(4, 3, 16, 16)
    targets = torch.tensor([0, 1, 2, 3])
    mixed, target_a, target_b, lam = apply_batch_augment(
        images,
        targets,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
    )
    assert mixed.shape == images.shape
    assert target_a.shape == targets.shape
    assert target_b.shape == targets.shape
    assert 0.0 <= lam <= 1.0


def test_train_one_epoch_checkpoint_callback() -> None:
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    callbacks: list[tuple[int, int]] = []
    train_one_epoch(
        model,
        _make_loader(n_samples=16),
        optimizer,
        torch.device("cpu"),
        epoch=2,
        checkpoint_interval_seconds=1e-9,
        checkpoint_callback=lambda batch_idx, steps: callbacks.append(
            (batch_idx, steps)
        ),
    )
    assert callbacks
    assert callbacks[-1][0] > 0
    assert callbacks[-1][1] > 0


def test_train_accumulation_matches_single_batch() -> None:
    """Two micro-batches with accumulation ≈ one full batch."""
    torch.manual_seed(99)
    model_single = _make_model()
    model_accum = _make_model()
    model_accum.load_state_dict(model_single.state_dict())

    lr = 0.01

    # Single batch
    optimizer_s = torch.optim.SGD(model_single.parameters(), lr=lr)
    images = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    loader_single = DataLoader(TensorDataset(images, targets), batch_size=16)

    train_one_epoch(
        model_single, loader_single, optimizer_s,
        torch.device("cpu"), epoch=0,
    )

    # Two micro-batches with accumulation
    optimizer_a = torch.optim.SGD(model_accum.parameters(), lr=lr)
    loader_accum = DataLoader(TensorDataset(images, targets), batch_size=8)

    train_one_epoch(
        model_accum, loader_accum, optimizer_a,
        torch.device("cpu"), epoch=0, accumulation_steps=2,
    )

    # Gradients should be approximately equal (SGD is linear)
    for p1, p2 in zip(model_single.parameters(), model_accum.parameters(), strict=False):
        torch.testing.assert_close(p1, p2, atol=1e-5, rtol=1e-5)


def test_train_accumulation_flushes_short_final_group() -> None:
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_loader(n_samples=17)

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        epoch=0,
        accumulation_steps=2,
    )

    assert metrics["steps"] == 2


def test_train_fails_fast_on_non_finite_loss() -> None:
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.full((1, 3, 32, 32), float("nan"))
    loader = DataLoader(TensorDataset(images, torch.tensor([0])), batch_size=1)

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train_one_epoch(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            epoch=0,
        )


def test_train_one_epoch_uses_label_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = nn.functional.cross_entropy
    observed: list[float] = []

    def capture_label_smoothing(
        input: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        observed.append(float(kwargs.get("label_smoothing", 0.0)))
        return original(input, target, **kwargs)

    monkeypatch.setattr(nn.functional, "cross_entropy", capture_label_smoothing)
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_one_epoch(
        model,
        _make_loader(n_samples=8),
        optimizer,
        torch.device("cpu"),
        epoch=0,
        label_smoothing=0.1,
    )
    assert observed
    assert all(value == 0.1 for value in observed)


def test_evaluate_model() -> None:
    model = _make_model()
    loader = _make_loader()
    metrics = evaluate_model(model, loader, torch.device("cpu"))
    assert "val_loss" in metrics
    assert "val_top1" in metrics
    assert "val_top5" in metrics
    assert 0 <= metrics["val_top1"] <= 1


def test_evaluate_model_reduces_distributed_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = list(_make_loader(n_samples=8))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def duplicate_across_two_ranks(
        tensor: torch.Tensor,
        op: object = None,
    ) -> None:
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", duplicate_across_two_ranks)
    model = _make_model()
    metrics = evaluate_model(model, loader, torch.device("cpu"))
    assert metrics["val_samples"] == 16


def test_ema_model() -> None:
    model = _make_model()
    ema = EMAModel(model, decay=0.9)

    # Record original EMA state for the first parameter
    first_param_name = "1.weight"
    ema_before = ema.state[first_param_name].clone()

    # Modify model parameters directly (not via state_dict copy)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.0)

    ema.update(model)

    # EMA should be: decay * old + (1 - decay) * new = 0.9 * old + 0.1 * 0 = 0.9 * old
    ema_after = ema.state[first_param_name]
    expected = ema_before * 0.9
    torch.testing.assert_close(ema_after, expected, atol=1e-6, rtol=1e-6)

    # State dict round-trip
    state = ema.state_dict()
    ema2 = EMAModel(model)
    ema2.load_state_dict(state)
    for k in state:
        torch.testing.assert_close(ema2.state[k], state[k])


def test_train_runs_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data_root = tmp_path / "data"
    for split in ("train", "val"):
        for class_index in range(2):
            class_dir = data_root / split / f"class-{class_index}"
            class_dir.mkdir(parents=True)
            for image_index in range(2):
                Image.new(
                    "RGB",
                    (40, 40),
                    color=(class_index * 64, image_index * 64, 0),
                ).save(class_dir / f"{image_index}.png")

    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        """
model:
  name: omniweave_t
  widths: [8, 16, 32, 64]
  depths: [1, 1, 1, 1]
  spatial_tile: 16
  channel_tiles: [8, 8, 16, 16]
  expansion: 2
  layer_scale_init: 1.0e-5
  num_classes: 2
  anchor_stages: []
  backend: reference
""".strip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    result = train({
        "seed": 7,
        "model_config": str(model_config),
        "data": {
            "root": str(data_root),
            "image_size": 32,
            "expected_classes": 2,
            "num_workers": 0,
        },
        "train": {
            "epochs": 1,
            "global_batch_size": 2,
            "learning_rate": 1e-3,
            "amp": False,
            "ema": True,
        },
        "output_dir": str(output_dir),
    })

    assert result["global_step"] == 2
    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "metrics.jsonl").exists()
