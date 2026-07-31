"""CLI entry point for training."""

from __future__ import annotations

import argparse

from omniweave.training.engine import train
from omniweave.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OmniWeave model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    result = train(config)
    print(f"Training complete: {result}")


if __name__ == "__main__":
    main()
