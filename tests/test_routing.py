"""Tests for deterministic route plans."""

from __future__ import annotations

import torch

from omniweave.models.routing import build_route_plan, build_stage_schedule


def test_local_route_covers_tokens_once() -> None:
    plan = build_route_plan(8, 8, 128, 16, 64, "local")
    valid = plan.token_indices[plan.token_mask]
    assert torch.equal(torch.sort(valid).values, torch.arange(64))
    assert plan.token_indices.shape == (4, 16)
    assert plan.channel_indices.shape == (2, 64)


def test_shifted_route_is_a_permutation() -> None:
    local = build_route_plan(8, 8, 128, 16, 64, "local")
    shifted = build_route_plan(8, 8, 128, 16, 64, "shifted")
    assert not torch.equal(local.token_indices, shifted.token_indices)
    assert torch.equal(
        torch.sort(shifted.token_indices[shifted.token_mask]).values,
        torch.arange(64),
    )


def test_stage_schedule() -> None:
    assert build_stage_schedule(8) == [
        ("local", 0),
        ("shifted", 0),
        ("radix", 0),
        ("radix", 1),
        ("radix", 2),
        ("local", 0),
        ("shifted", 0),
        ("radix", 0),
    ]


def test_padding_masks_non_divisible_shapes() -> None:
    plan = build_route_plan(7, 7, 70, 16, 64, "radix", radix_level=1)
    assert plan.token_mask.sum().item() == 49
    assert plan.channel_mask.sum().item() == 70
    # Pad index for tokens is N=49, for channels is C=70
    assert plan.token_indices.max().item() == 49
    assert plan.channel_indices.max().item() == 70


def test_radix_route_covers_tokens() -> None:
    plan = build_route_plan(8, 8, 128, 16, 64, "radix", radix_level=0)
    valid = plan.token_indices[plan.token_mask]
    assert torch.equal(torch.sort(valid).values, torch.arange(64))


def test_local_route_small_spatial() -> None:
    plan = build_route_plan(4, 4, 32, 16, 32, "local")
    valid = plan.token_indices[plan.token_mask]
    assert torch.equal(torch.sort(valid).values, torch.arange(16))
    assert plan.token_indices.shape == (1, 16)


def test_schedule_depth_one() -> None:
    assert build_stage_schedule(1) == [("local", 0)]


def test_schedule_depth_five() -> None:
    assert build_stage_schedule(5) == [
        ("local", 0),
        ("shifted", 0),
        ("radix", 0),
        ("radix", 1),
        ("radix", 2),
    ]
