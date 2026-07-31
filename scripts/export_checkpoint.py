"""Export a portable checkpoint suitable for inference without Triton."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from omniweave.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from omniweave.models.registry import create_model, create_model_from_config


def export_portable_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    use_ema: bool = False,
    model_name: str = "omniweave_t",
) -> Path:
    """Strip training state and export inference-ready checkpoint.

    The exported checkpoint loads with the reference backend,
    no Triton import required.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    model_config_path = config.get("model_config")
    if model_config_path and Path(model_config_path).exists():
        model = create_model_from_config(
            model_config_path,
            backend="reference",
        )
    else:
        model = create_model(model_name, backend="reference")
    model.load_state_dict(checkpoint["model"], strict=True)

    if use_ema and "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"], strict=True)

    portable = {
        "model": model.state_dict(),
        "config": config,
        "epoch": checkpoint.get("epoch", 0),
        "backend": "reference",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
    }

    # Add benchmark summary if available
    extra = checkpoint.get("extra", {})
    if "benchmark" in extra:
        portable["benchmark_summary"] = extra["benchmark"]
    if "dataset_manifest" in extra:
        portable["class_to_idx"] = extra["dataset_manifest"].get("class_to_idx", {})
    if "environment" in extra:
        portable["source_revision"] = extra["environment"].get("git", {})

    torch.save(portable, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export portable checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--model", type=str, default="omniweave_t")
    args = parser.parse_args()

    path = export_portable_checkpoint(
        args.checkpoint, args.output,
        use_ema=args.use_ema, model_name=args.model,
    )
    print(f"Portable checkpoint exported to {path}")


if __name__ == "__main__":
    main()
