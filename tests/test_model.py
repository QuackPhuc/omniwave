"""Tests for OmniWeaveBlock and OmniWeaveBackbone."""

from __future__ import annotations

import pytest
import timm
import torch

from omniweave.models.backbone import OmniWeaveBackbone
from omniweave.models.block import OmniWeaveBlock


def test_block_preserves_shape() -> None:
    block = OmniWeaveBlock(
        height=8,
        width=8,
        dim=128,
        channel_tile=64,
        route="shifted",
        radix_level=0,
        expansion=2,
        backend="reference",
    )
    x = torch.randn(2, 8, 8, 128)
    assert block(x).shape == x.shape


def test_block_local_route() -> None:
    block = OmniWeaveBlock(
        height=8,
        width=8,
        dim=128,
        channel_tile=64,
        route="local",
        radix_level=0,
        expansion=2,
        backend="reference",
    )
    x = torch.randn(1, 8, 8, 128)
    out = block(x)
    assert out.shape == x.shape
    # Output should differ from input (non-trivial transform)
    assert not torch.equal(out, x)


def test_block_radix_route() -> None:
    block = OmniWeaveBlock(
        height=8,
        width=8,
        dim=128,
        channel_tile=64,
        route="radix",
        radix_level=1,
        expansion=2,
        backend="reference",
    )
    x = torch.randn(1, 8, 8, 128)
    assert block(x).shape == x.shape


def test_backbone_feature_shapes() -> None:
    model = OmniWeaveBackbone(
        widths=[128, 256, 512, 1024],
        depths=[2, 3, 8, 2],
        channel_tiles=[64, 64, 128, 128],
        num_classes=1000,
        backend="reference",
    )
    outputs = model.forward_features(torch.randn(1, 3, 224, 224))
    assert [tuple(x.shape) for x in outputs.values()] == [
        (1, 56, 56, 128),
        (1, 28, 28, 256),
        (1, 14, 14, 512),
        (1, 7, 7, 1024),
    ]


def test_backbone_classification() -> None:
    model = OmniWeaveBackbone(
        widths=[128, 256, 512, 1024],
        depths=[2, 3, 8, 2],
        channel_tiles=[64, 64, 128, 128],
        num_classes=1000,
        backend="reference",
    )
    logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 1000)


def test_parameter_count_target() -> None:
    model = OmniWeaveBackbone(
        widths=[128, 256, 512, 1024],
        depths=[2, 3, 8, 2],
        channel_tiles=[64, 64, 128, 128],
        num_classes=1000,
        backend="reference",
    )
    count = sum(p.numel() for p in model.parameters())
    assert 10_000_000 <= count <= 20_000_000, (
        f"parameter count {count:,} is outside the target range [10M, 20M]"
    )


def test_anchor_stages_can_be_disabled() -> None:
    model = OmniWeaveBackbone(
        widths=[32, 64, 128, 256],
        depths=[1, 1, 1, 1],
        channel_tiles=[32, 32, 64, 64],
        anchor_stages=[],
        input_size=32,
    )
    assert all(stage.anchor is None for stage in model.stages)


def test_timm_model_creation() -> None:
    model = timm.create_model(
        "omniweave_t",
        pretrained=False,
        widths=[32, 64, 128, 256],
        depths=[1, 1, 1, 1],
        channel_tiles=[32, 32, 64, 64],
        input_size=32,
    )
    assert isinstance(model, OmniWeaveBackbone)


def test_backbone_gradient_flows() -> None:
    """Verify gradients propagate through the full backbone."""
    model = OmniWeaveBackbone(
        widths=[128, 256, 512, 1024],
        depths=[2, 3, 8, 2],
        channel_tiles=[64, 64, 128, 128],
        num_classes=10,
        backend="reference",
    )
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_block_preserves_amp_input_dtype(
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, torch.dtype] = {}

    def capture_dtype(*, x: torch.Tensor, **_: object) -> torch.Tensor:
        observed["dtype"] = x.dtype
        return torch.zeros_like(x)

    monkeypatch.setattr("omniweave.models.block.bigemm", capture_dtype)
    block = OmniWeaveBlock(
        height=8,
        width=8,
        dim=16,
        channel_tile=8,
        route="local",
        radix_level=0,
    )
    x = torch.randn(1, 8, 8, 16, dtype=dtype)
    assert block(x).dtype == dtype
    assert observed["dtype"] == dtype


def test_drop_path_schedule_is_linear() -> None:
    model = OmniWeaveBackbone(
        widths=[16, 32, 64, 128],
        depths=[1, 2, 1, 1],
        channel_tiles=[8, 8, 16, 16],
        input_size=32,
        drop_path_rate=0.2,
    )
    rates = [
        block.drop_path_rate
        for stage in model.stages
        for block in stage.blocks
    ]
    torch.testing.assert_close(
        torch.tensor(rates),
        torch.linspace(0.0, 0.2, len(rates)),
    )
