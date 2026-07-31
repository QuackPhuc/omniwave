"""Tests for configuration loading and model-config validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniweave.utils.config import load_config, validate_model_config

# Resolve repo root from this file's location (tests/ is one level below root)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_canonical_model_config() -> None:
    cfg = load_config(_REPO_ROOT / "configs" / "model" / "omniweave_t.yaml")
    assert cfg["model"]["widths"] == [128, 256, 512, 1024]
    assert cfg["model"]["depths"] == [2, 3, 8, 2]
    assert cfg["model"]["channel_tiles"] == [64, 64, 128, 128]
    assert cfg["model"]["spatial_tile"] == 16
    assert cfg["model"]["expansion"] == 2


def test_rejects_inconsistent_stage_lists() -> None:
    cfg = {
        "model": {
            "name": "bad",
            "widths": [128, 256],
            "depths": [2],
            "channel_tiles": [64, 64],
            "spatial_tile": 16,
            "expansion": 2,
            "layer_scale_init": 1e-5,
            "num_classes": 1000,
        }
    }
    with pytest.raises(ValueError, match="same length"):
        validate_model_config(cfg)


def test_rejects_missing_keys() -> None:
    cfg = {"model": {"name": "incomplete"}}
    with pytest.raises(ValueError, match="missing model keys"):
        validate_model_config(cfg)


def test_rejects_invalid_spatial_tile() -> None:
    cfg = {
        "model": {
            "name": "bad",
            "widths": [128, 256, 512, 1024],
            "depths": [2, 3, 8, 2],
            "channel_tiles": [64, 64, 128, 128],
            "spatial_tile": 7,
            "expansion": 2,
            "layer_scale_init": 1e-5,
            "num_classes": 1000,
        }
    }
    with pytest.raises(ValueError, match="spatial_tile must be one of"):
        validate_model_config(cfg)


def test_rejects_indivisible_channel_tile() -> None:
    cfg = {
        "model": {
            "name": "bad",
            "widths": [128, 256, 512, 1024],
            "depths": [2, 3, 8, 2],
            "channel_tiles": [64, 64, 128, 100],
            "spatial_tile": 16,
            "expansion": 2,
            "layer_scale_init": 1e-5,
            "num_classes": 1000,
        }
    }
    with pytest.raises(ValueError, match="not divisible"):
        validate_model_config(cfg)


def test_rejects_non_mapping_root(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(bad)


def test_rejects_missing_model_section() -> None:
    with pytest.raises(ValueError, match="model mapping"):
        validate_model_config({"training": {}})
