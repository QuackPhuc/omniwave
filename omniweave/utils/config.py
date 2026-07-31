"""Strict YAML configuration loading and model-config validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_REQUIRED_MODEL_KEYS: frozenset[str] = frozenset({
    "name",
    "widths",
    "depths",
    "spatial_tile",
    "channel_tiles",
    "expansion",
    "layer_scale_init",
    "num_classes",
})

_VALID_SPATIAL_TILES: frozenset[int] = frozenset({8, 16, 32})
_STAGE_COUNT = 4


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Validates the model section if present."""
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("configuration root must be a mapping")
    if "model" in cfg:
        validate_model_config(cfg)
    return cfg


def validate_model_config(config: Mapping[str, Any]) -> None:
    """Validate the ``model`` section of a configuration mapping.

    Raises ``ValueError`` on any structural or semantic inconsistency.
    """
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("configuration must contain a model mapping")

    missing = sorted(_REQUIRED_MODEL_KEYS - set(model))
    if missing:
        raise ValueError(f"missing model keys: {missing}")

    widths = list(model["widths"])
    depths = list(model["depths"])
    tiles = list(model["channel_tiles"])

    if not (len(widths) == len(depths) == len(tiles) == _STAGE_COUNT):
        raise ValueError(
            "widths, depths, and channel_tiles must have the same length of four"
        )

    if int(model["spatial_tile"]) not in _VALID_SPATIAL_TILES:
        raise ValueError("spatial_tile must be one of 8, 16, or 32")

    for width, tile in zip(widths, tiles, strict=True):
        if int(width) % int(tile) != 0:
            raise ValueError(
                f"stage width {width} is not divisible by channel tile {tile}"
            )
