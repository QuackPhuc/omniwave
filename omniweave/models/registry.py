"""Model registry — create OmniWeave models by name."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from omniweave.models.backbone import OmniWeaveBackbone
from omniweave.utils.config import load_config

_REGISTRY: dict[str, dict[str, Any]] = {
    "omniweave_t": {
        "widths": [128, 256, 512, 1024],
        "depths": [2, 3, 8, 2],
        "channel_tiles": [64, 64, 128, 128],
        "expansion": 2,
        "layer_scale_init": 1e-5,
        "anchor_stages": [3, 4],
    },
}

_TIMM_INTERNAL_KWARGS = {
    "cache_dir",
    "pretrained_cfg",
    "pretrained_cfg_overlay",
}


def create_model(
    name: str,
    pretrained: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Create an OmniWeave model by registered name.

    Parameters
    ----------
    name : registered model name (e.g. ``"omniweave_t"``)
    pretrained : if True, load pretrained weights (not yet available)
    **kwargs : overrides forwarded to the backbone constructor

    Raises
    ------
    ValueError : unknown model name
    NotImplementedError : pretrained requested but no checkpoint URI
    """
    if pretrained:
        raise NotImplementedError(
            "pretrained weights are not yet available; "
            "supply an explicit checkpoint path instead"
        )

    if name not in _REGISTRY:
        raise ValueError(
            f"unknown model: {name!r}; available: {sorted(_REGISTRY)}"
        )

    defaults = _REGISTRY[name].copy()
    defaults.update(kwargs)
    return OmniWeaveBackbone(**defaults)


def omniweave_t(pretrained: bool = False, **kwargs: Any) -> nn.Module:
    """Timm-compatible OmniWeave-T entry point."""
    for key in _TIMM_INTERNAL_KWARGS:
        kwargs.pop(key, None)
    return create_model("omniweave_t", pretrained=pretrained, **kwargs)


def create_model_from_config(
    config_path: str,
    **kwargs: Any,
) -> nn.Module:
    """Create a model from a YAML config file."""
    cfg = load_config(config_path)
    model_cfg = cfg["model"].copy()
    model_cfg.pop("name")

    model_cfg.update(kwargs)
    return OmniWeaveBackbone(**model_cfg)


try:
    from timm.models import register_model
except ImportError:
    register_model = None

if register_model is not None:
    omniweave_t = register_model(omniweave_t)
