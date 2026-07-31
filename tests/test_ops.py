"""Tests for pack/unpack and reference BiGEMM operator."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from omniweave.models.routing import build_route_plan
from omniweave.ops.reference import bigemm_delta_reference, pack_tiles, unpack_tiles


def test_pack_unpack_identity() -> None:
    x = torch.arange(2 * 7 * 7 * 70, dtype=torch.float32).reshape(2, 7, 7, 70)
    plan = build_route_plan(7, 7, 70, 16, 64, "radix", radix_level=1)
    tiles, mask = pack_tiles(x, plan)
    assert tiles.shape == (2, 4, 2, 16, 64)
    assert mask.shape == (1, 4, 2, 16, 64)
    torch.testing.assert_close(unpack_tiles(tiles, plan, 2), x)


def test_padding_values_are_zero() -> None:
    x = torch.ones(1, 7, 7, 70)
    plan = build_route_plan(7, 7, 70, 16, 64, "local")
    tiles, mask = pack_tiles(x, plan)
    assert torch.count_nonzero(tiles.masked_select(~mask)) == 0


def test_pack_unpack_exact_shapes() -> None:
    """Exact-divisible shapes should round-trip perfectly."""
    x = torch.randn(3, 8, 8, 128)
    plan = build_route_plan(8, 8, 128, 16, 64, "local")
    tiles, mask = pack_tiles(x, plan)
    assert tiles.shape == (3, 4, 2, 16, 64)
    assert mask.all()
    torch.testing.assert_close(unpack_tiles(tiles, plan, 3), x)


def test_pack_unpack_shifted() -> None:
    x = torch.randn(2, 8, 8, 128)
    plan = build_route_plan(8, 8, 128, 16, 64, "shifted")
    tiles, _ = pack_tiles(x, plan)
    torch.testing.assert_close(unpack_tiles(tiles, plan, 2), x)


def test_single_tile_matches_equation() -> None:
    """Verify the reference BiGEMM matches the direct mathematical definition."""
    torch.manual_seed(7)
    g, d, rd = 16, 4, 8
    x = torch.randn(1, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    plan = build_route_plan(4, 4, 4, g, d, "local")

    a_u = torch.randn(g, g, dtype=torch.float64)
    a_v = torch.randn(g, g, dtype=torch.float64)
    a_o = torch.randn(g, g, dtype=torch.float64)
    b_u = torch.randn(1, d, rd, dtype=torch.float64)
    b_v = torch.randn(1, d, rd, dtype=torch.float64)
    b_o = torch.randn(1, rd, d, dtype=torch.float64)
    gamma = torch.ones(d, dtype=torch.float64)

    actual = bigemm_delta_reference(
        x=x, plan=plan,
        a_u=a_u, a_v=a_v, a_o=a_o,
        b_u=b_u, b_v=b_v, b_o=b_o,
        gamma=gamma,
    )

    # Direct equation: tile is [g, d]
    tile = x.detach().reshape(g, d)
    expected = (
        a_o @ (F.silu(a_u @ tile @ b_u[0]) * (a_v @ tile @ b_v[0])) @ b_o[0]
    ).reshape_as(x)
    torch.testing.assert_close(actual, expected)


def test_bigemm_gradcheck() -> None:
    """FP64 gradient check for the reference BiGEMM."""
    torch.manual_seed(42)
    g, d, rd = 16, 4, 8
    x = torch.randn(1, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    plan = build_route_plan(4, 4, 4, g, d, "local")

    a_u = torch.randn(g, g, dtype=torch.float64, requires_grad=True)
    a_v = torch.randn(g, g, dtype=torch.float64, requires_grad=True)
    a_o = torch.randn(g, g, dtype=torch.float64, requires_grad=True)
    b_u = torch.randn(1, d, rd, dtype=torch.float64, requires_grad=True)
    b_v = torch.randn(1, d, rd, dtype=torch.float64, requires_grad=True)
    b_o = torch.randn(1, rd, d, dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(d, dtype=torch.float64, requires_grad=True)

    def func(x_, a_u_, a_v_, a_o_, b_u_, b_v_, b_o_, gamma_):
        return bigemm_delta_reference(
            x=x_, plan=plan,
            a_u=a_u_, a_v=a_v_, a_o=a_o_,
            b_u=b_u_, b_v=b_v_, b_o=b_o_,
            gamma=gamma_,
        )

    torch.autograd.gradcheck(
        func,
        (x, a_u, a_v, a_o, b_u, b_v, b_o, gamma),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_bigemm_multi_channel_groups() -> None:
    """Multiple channel groups with non-trivial routing."""
    torch.manual_seed(17)
    h, w, c = 8, 8, 128
    g, d, rd = 16, 64, 128
    q_c = c // d  # 2

    x = torch.randn(2, h, w, c, dtype=torch.float64)
    plan = build_route_plan(h, w, c, g, d, "radix", radix_level=0)

    a_u = torch.randn(g, g, dtype=torch.float64)
    a_v = torch.randn(g, g, dtype=torch.float64)
    a_o = torch.eye(g, dtype=torch.float64)
    b_u = torch.randn(q_c, d, rd, dtype=torch.float64)
    b_v = torch.randn(q_c, d, rd, dtype=torch.float64)
    b_o = torch.randn(q_c, rd, d, dtype=torch.float64)
    gamma = torch.ones(c, dtype=torch.float64) * 1e-5

    delta = bigemm_delta_reference(
        x=x, plan=plan,
        a_u=a_u, a_v=a_v, a_o=a_o,
        b_u=b_u, b_v=b_v, b_o=b_o,
        gamma=gamma,
    )
    assert delta.shape == x.shape
    # Delta should be small due to LayerScale
    assert delta.abs().max().item() < 1.0
