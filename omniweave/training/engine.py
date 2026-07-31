"""Training engine — finite-safe loops with AMP, EMA, accumulation, and DDP."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from omniweave.evaluation.metrics import topk_accuracy
from omniweave.training.checkpoint import load_checkpoint, save_checkpoint
from omniweave.utils.logging import JsonlLogger

logger = logging.getLogger(__name__)


def _is_finite(t: Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _distributed_sum(values: Tensor) -> Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values)
    return values


def _distributed_max(value: Tensor) -> Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return value


def _resolve_amp_dtype(device: torch.device, enabled: bool) -> torch.dtype:
    if not enabled or device.type != "cuda":
        return torch.float32
    return (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )


def apply_batch_augment(
    images: Tensor,
    targets: Tensor,
    *,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Apply Mixup or CutMix and return two hard-target streams.

    Returning two target streams keeps the loss compatible with label
    smoothing without allocating a full one-hot tensor for every batch.
    """
    if mixup_alpha < 0 or cutmix_alpha < 0:
        raise ValueError("mixup_alpha and cutmix_alpha must be non-negative")
    if images.size(0) < 2 or (mixup_alpha == 0 and cutmix_alpha == 0):
        return images, targets, targets, 1.0

    use_cutmix = (
        cutmix_alpha > 0
        and (mixup_alpha == 0 or bool(torch.rand((), device=images.device) < 0.5))
    )
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    if alpha <= 0:
        return images, targets, targets, 1.0

    permutation = torch.randperm(images.size(0), device=images.device)
    lam = float(
        torch.distributions.Beta(alpha, alpha)
        .sample()
        .to(device=images.device)
        .item()
    )
    if not use_cutmix:
        mixed = images * lam + images[permutation] * (1.0 - lam)
        return mixed, targets, targets[permutation], lam

    _, _, height, width = images.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_height = int(height * cut_ratio)
    cut_width = int(width * cut_ratio)
    center_y = int(torch.randint(height, (), device=images.device).item())
    center_x = int(torch.randint(width, (), device=images.device).item())
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + cut_height // 2, height)
    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    area = max(y2 - y1, 0) * max(x2 - x1, 0)
    lam = 1.0 - area / max(height * width, 1)
    return mixed, targets, targets[permutation], lam


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Store parameter names for efficient iteration
        self.state: dict[str, Tensor] = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }

    def update(self, model: nn.Module) -> None:
        source = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            for k, param in source.named_parameters():
                if k in self.state:
                    self.state[k].lerp_(param.data.to(self.state[k].device), 1 - self.decay)
            for k, buf in source.named_buffers():
                if k in self.state:
                    self.state[k].copy_(buf.to(self.state[k].device))

    def state_dict(self) -> dict[str, Tensor]:
        return {k: v.clone() for k, v in self.state.items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.state = {k: v.clone() for k, v in state.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    ema: EMAModel | None = None,
    gradient_clip_norm: float = 1.0,
    accumulation_steps: int = 1,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    start_batch_idx: int = 0,
    checkpoint_interval_seconds: float | None = None,
    checkpoint_callback: Callable[[int, int], None] | None = None,
    jsonl_logger: JsonlLogger | None = None,
) -> dict[str, float]:
    """Train for one epoch. Returns metrics dict."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")
    if start_batch_idx < 0:
        raise ValueError("start_batch_idx must be non-negative")
    if checkpoint_interval_seconds is not None and checkpoint_interval_seconds <= 0:
        raise ValueError("checkpoint_interval_seconds must be positive")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    step_count = 0
    non_finite_count = 0
    t_start = time.perf_counter()
    last_checkpoint = t_start
    num_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx < start_batch_idx:
            continue
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, targets_a, targets_b, target_mix = apply_batch_augment(
            images,
            targets,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        )

        group_start = (batch_idx // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, num_batches - group_start)
        is_accumulating = (
            (batch_idx + 1) % accumulation_steps != 0
            and batch_idx + 1 != num_batches
        )

        # Forward with AMP
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(images)
            loss_a = nn.functional.cross_entropy(
                logits, targets_a, label_smoothing=label_smoothing
            )
            loss_b = nn.functional.cross_entropy(
                logits, targets_b, label_smoothing=label_smoothing
            )
            raw_loss = target_mix * loss_a + (1.0 - target_mix) * loss_b
            loss = raw_loss / group_size

        # Check finite loss
        if not _is_finite(loss):
            non_finite_count += 1
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"non-finite loss at epoch {epoch} batch {batch_idx}"
            )

        # Backward
        sync_context = (
            model.no_sync()
            if is_accumulating and hasattr(model, "no_sync")
            else nullcontext()
        )
        with sync_context:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if not is_accumulating:
            # Unscale and check gradients
            if scaler is not None:
                scaler.unscale_(optimizer)

            # Check finite gradients
            grad_finite = all(
                _is_finite(p.grad) for p in model.parameters() if p.grad is not None
            )
            if not grad_finite:
                non_finite_count += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()
                raise FloatingPointError(
                    f"non-finite gradient at epoch {epoch} batch {batch_idx}"
                )

            # Clip gradients
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # EMA update
            if ema is not None:
                ema.update(model)

            step_count += 1
            if (
                checkpoint_callback is not None
                and checkpoint_interval_seconds is not None
                and time.perf_counter() - last_checkpoint
                >= checkpoint_interval_seconds
            ):
                checkpoint_callback(batch_idx + 1, step_count)
                last_checkpoint = time.perf_counter()

        batch_size = images.size(0)
        total_loss += raw_loss.item() * batch_size
        total_samples += batch_size

    elapsed = time.perf_counter() - t_start
    totals = _distributed_sum(torch.tensor(
        [total_loss, total_samples],
        dtype=torch.float64,
        device=device,
    ))
    global_elapsed = _distributed_max(torch.tensor(
        elapsed,
        dtype=torch.float64,
        device=device,
    )).item()
    global_loss, global_samples = totals.tolist()
    avg_loss = global_loss / max(global_samples, 1)
    throughput = global_samples / max(global_elapsed, 1e-6)

    metrics = {
        "epoch": epoch,
        "train_loss": avg_loss,
        "images_per_second": throughput,
        "elapsed_seconds": global_elapsed,
        "steps": step_count,
        "non_finite_events": non_finite_count,
    }

    if jsonl_logger is not None:
        jsonl_logger.write(metrics)

    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Evaluate model on validation set. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_top1 = 0.0
    all_top5 = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        acc = topk_accuracy(logits, targets, topk=(1, 5))
        all_top1 += acc["top1"] * batch_size
        all_top5 += acc["top5"] * batch_size

    totals = _distributed_sum(torch.tensor(
        [total_loss, total_samples, all_top1, all_top5],
        dtype=torch.float64,
        device=device,
    ))
    total_loss, total_samples, all_top1, all_top5 = totals.tolist()
    n = max(total_samples, 1)
    return {
        "val_loss": total_loss / n,
        "val_top1": all_top1 / n,
        "val_top5": all_top5 / n,
        "val_samples": int(total_samples),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Full training loop from config dict.

    Config structure:
    - ``seed``: random seed
    - ``model_config``: path to model YAML
    - ``data.root``: ImageNet root
    - ``data.image_size``: resolution
    - ``train.epochs``: number of epochs
    - ``train.global_batch_size``: effective batch size
    - ``train.learning_rate``: peak LR
    - ``train.weight_decay``: AdamW weight decay
    - ``train.gradient_clip_norm``: gradient clipping
    - ``train.amp``: enable AMP
    - ``output_dir``: output directory
    """
    from omniweave.data.imagenet import build_imagenet_loaders
    from omniweave.models.registry import create_model_from_config
    from omniweave.training.distributed import cleanup_distributed, initialize_distributed
    from omniweave.utils.environment import collect_environment

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ctx = initialize_distributed()
    jsonl_logger: JsonlLogger | None = None
    try:
        device = ctx.device
        train_cfg = config.get("train", {})
        data_cfg = config.get("data", {})
        epochs = int(train_cfg.get("epochs", 100))
        accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
        global_batch_size = int(train_cfg.get("global_batch_size", 64))
        batch_divisor = ctx.world_size * accumulation_steps
        if global_batch_size % batch_divisor:
            raise ValueError(
                "global_batch_size must be divisible by world_size * accumulation_steps"
            )
        device_batch_size = global_batch_size // batch_divisor

        model_config_path = config.get(
            "model_config", "configs/model/omniweave_t.yaml"
        )
        model_overrides: dict[str, Any] = {
            "num_classes": int(data_cfg.get("expected_classes", 1000)),
            "input_size": int(data_cfg.get("image_size", 224)),
        }
        if "backend" in config:
            model_overrides["backend"] = str(config["backend"])
        if "drop_path_rate" in train_cfg:
            model_overrides["drop_path_rate"] = float(
                train_cfg["drop_path_rate"]
            )
        raw_model = create_model_from_config(
            model_config_path,
            **model_overrides,
        ).to(device)

        lr = float(train_cfg.get("learning_rate", 1e-3))
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        )

        warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1)
        )
        if warmup_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=warmup_epochs
            )
            scheduler: Any = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine

        amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
        amp_dtype = _resolve_amp_dtype(device, amp_enabled)
        scaler = (
            torch.amp.GradScaler("cuda")
            if amp_enabled and amp_dtype == torch.float16
            else None
        )
        ema = EMAModel(raw_model) if train_cfg.get("ema", True) else None

        output_dir = Path(config.get("output_dir", "runs/default"))
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / "checkpoint.pt"
        environment = collect_environment()

        start_epoch = 0
        resume_batch_idx = 0
        resumed_epoch: int | None = None
        global_step = 0
        if ckpt_path.exists():
            checkpoint = load_checkpoint(
                ckpt_path,
                raw_model,
                optimizer,
                scheduler,
                scaler,
                map_location=device,
            )
            resumed_epoch = int(checkpoint["epoch"])
            checkpoint_extra = checkpoint.get("extra", {})
            resume_batch_idx = int(checkpoint_extra.get("next_batch_idx", 0))
            if "next_batch_idx" not in checkpoint_extra:
                start_epoch = resumed_epoch + 1
            global_step = int(checkpoint.get("global_step", 0))
            if ema is not None and "ema" in checkpoint:
                ema.load_state_dict(checkpoint["ema"])
            logger.info("resumed from epoch %d", start_epoch)

        model: nn.Module = raw_model
        if ctx.is_distributed:
            model = nn.parallel.DistributedDataParallel(
                raw_model, device_ids=[ctx.local_rank]
            )

        train_loader, val_loader, manifest = build_imagenet_loaders(
            config,
            distributed=ctx.is_distributed,
            batch_size=device_batch_size,
        )

        if ctx.is_primary:
            (output_dir / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
            )
            jsonl_logger = JsonlLogger(output_dir / "metrics.jsonl")

        if resumed_epoch is not None and "next_batch_idx" in checkpoint_extra:
            if resume_batch_idx >= len(train_loader):
                start_epoch = resumed_epoch + 1
                resume_batch_idx = 0
            else:
                start_epoch = resumed_epoch

        checkpoint_minutes = float(train_cfg.get("checkpoint_minutes", 0))
        checkpoint_interval_seconds = (
            checkpoint_minutes * 60 if checkpoint_minutes > 0 else None
        )
        dataset_manifest = {
            "train_samples": manifest.train_samples,
            "val_samples": manifest.val_samples,
            "class_to_idx": manifest.class_to_idx,
            "relative_paths_sha256": manifest.relative_paths_sha256,
        }

        final_metrics: dict[str, float] = {}
        for epoch in range(start_epoch, epochs):
            sampler = getattr(train_loader, "sampler", None)
            set_epoch = getattr(sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            epoch_start_batch = resume_batch_idx if epoch == start_epoch else 0

            def save_interval_checkpoint(
                next_batch_idx: int,
                step_count: int,
                checkpoint_epoch: int = epoch,
                checkpoint_global_step: int = global_step,
            ) -> None:
                if not ctx.is_primary:
                    return
                save_checkpoint(
                    ckpt_path,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema_state=ema.state_dict() if ema is not None else None,
                    epoch=checkpoint_epoch,
                    global_step=checkpoint_global_step + step_count,
                    config=config,
                    extra={
                        "environment": environment,
                        "dataset_manifest": dataset_manifest,
                        "next_batch_idx": next_batch_idx,
                    },
                )

            try:
                train_metrics = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    scaler=scaler,
                    ema=ema,
                    gradient_clip_norm=float(
                        train_cfg.get("gradient_clip_norm", 1.0)
                    ),
                    accumulation_steps=accumulation_steps,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    label_smoothing=float(
                        train_cfg.get("label_smoothing", 0.0)
                    ),
                    mixup_alpha=float(train_cfg.get("mixup_alpha", 0.0)),
                    cutmix_alpha=float(train_cfg.get("cutmix_alpha", 0.0)),
                    start_batch_idx=epoch_start_batch,
                    checkpoint_interval_seconds=checkpoint_interval_seconds,
                    checkpoint_callback=save_interval_checkpoint,
                    jsonl_logger=jsonl_logger,
                )
            except FloatingPointError as exc:
                if ctx.is_primary:
                    save_checkpoint(
                        ckpt_path,
                        model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        ema_state=ema.state_dict() if ema is not None else None,
                        epoch=epoch,
                        global_step=global_step,
                        config=config,
                        extra={
                            "environment": environment,
                            "failure": {
                                "type": "non_finite",
                                "epoch": epoch,
                                "message": str(exc),
                            },
                        },
                    )
                raise
            global_step += int(train_metrics["steps"])
            val_metrics = evaluate_model(
                raw_model,
                val_loader,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            final_metrics = {**train_metrics, **val_metrics}
            if jsonl_logger is not None:
                jsonl_logger.write({"epoch": epoch, **val_metrics})

            scheduler.step()
            if ctx.is_primary:
                save_checkpoint(
                    ckpt_path,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema_state=ema.state_dict() if ema is not None else None,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                    extra={
                        "environment": environment,
                        "dataset_manifest": dataset_manifest,
                        "next_batch_idx": len(train_loader),
                    },
                )
            resume_batch_idx = 0

        return {
            "device": str(device),
            "start_epoch": start_epoch,
            "epochs": epochs,
            "global_step": global_step,
            "metrics": final_metrics,
            "checkpoint": str(ckpt_path),
        }
    finally:
        if jsonl_logger is not None:
            jsonl_logger.close()
        if ctx.is_distributed:
            cleanup_distributed()
