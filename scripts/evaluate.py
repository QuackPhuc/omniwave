"""CLI entry point for model evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from omniweave.data.imagenet import build_imagenet_loaders
from omniweave.models.registry import create_model_from_config
from omniweave.training.checkpoint import load_checkpoint
from omniweave.training.engine import _resolve_amp_dtype, evaluate_model
from omniweave.utils.environment import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OmniWeave model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from omniweave.utils.config import load_config
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    model = create_model_from_config(
        config.get("model_config", "configs/model/omniweave_t.yaml"),
        backend="reference",
        num_classes=int(data_cfg.get("expected_classes", 1000)),
        input_size=int(data_cfg.get("image_size", 224)),
    )

    checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
    model = model.to(device)
    _, val_loader, manifest = build_imagenet_loaders(config)
    metrics = evaluate_model(
        model,
        val_loader,
        device,
        amp_enabled=bool(config.get("train", {}).get("amp", False)),
        amp_dtype=_resolve_amp_dtype(
            device,
            bool(config.get("train", {}).get("amp", False)),
        ),
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": metrics,
        "dataset_manifest": {
            "val_samples": manifest.val_samples,
            "relative_paths_sha256": manifest.relative_paths_sha256,
        },
        "environment": collect_environment(),
    }
    print(json.dumps(result, indent=2, default=str))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
