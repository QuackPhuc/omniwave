"""Deterministic weave routing — local, shifted, and radix route plans.

Route plans are immutable, cacheable, and derived only from spatial shape,
channel count, tile sizes, and route type.  No dense permutation matrices
are ever allocated.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutePlan:
    """Immutable routing specification for one block invocation.

    All index tensors are on CPU and int64.  Pad index for tokens is ``N``
    (one past valid), for channels is ``C``.
    """

    token_indices: torch.Tensor   # [Q_s, g], pad index = N
    channel_indices: torch.Tensor  # [Q_c, d], pad index = C
    token_mask: torch.Tensor       # [Q_s, g], bool
    channel_mask: torch.Tensor     # [Q_c, d], bool
    height: int
    width: int
    channels: int
    route: str
    radix_level: int


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _pad_and_mask(
    indices: list[int],
    group_size: int,
    pad_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad *indices* to groups of *group_size* and return (indices, mask)."""
    n = len(indices)
    n_groups = math.ceil(n / group_size)
    total = n_groups * group_size
    padded = indices + [pad_value] * (total - n)
    idx = torch.tensor(padded, dtype=torch.long).reshape(n_groups, group_size)
    mask = torch.ones(n_groups, group_size, dtype=torch.bool)
    if total > n:
        mask.reshape(-1)[n:] = False
    return idx, mask


def _local_token_order(height: int, width: int, window: int = 4) -> list[int]:
    """Contiguous ``window × window`` blocks in raster order."""
    order: list[int] = []
    for row_base in range(0, height, window):
        for col_base in range(0, width, window):
            for r in range(row_base, min(row_base + window, height)):
                for c in range(col_base, min(col_base + window, width)):
                    order.append(r * width + c)
    return order


def _shifted_token_order(
    height: int,
    width: int,
    window: int = 4,
    shift: int = 2,
) -> list[int]:
    """Shifted-window grouping: apply cyclic offset before local grouping."""
    order: list[int] = []
    for row_base in range(-shift, height - shift, window):
        for col_base in range(-shift, width - shift, window):
            for r in range(row_base, row_base + window):
                for c in range(col_base, col_base + window):
                    actual_r = r % height
                    actual_c = c % width
                    order.append(actual_r * width + actual_c)
    # Deduplicate while preserving order (shifted windows may overlap at edges)
    seen: set[int] = set()
    deduped: list[int] = []
    for idx in order:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def radix_order(token_count: int, group_size: int, level: int) -> list[int]:
    """Mixed-radix reorder at *level* — groups tokens varying in digit ``a_level``."""
    stride = group_size ** level
    block = stride * group_size
    order: list[int] = []
    for base in range(0, token_count, block):
        for offset in range(stride):
            for digit in range(group_size):
                index = base + digit * stride + offset
                if index < token_count:
                    order.append(index)
    return order


def _channel_order(channels: int, channel_tile: int, radix_level: int) -> list[int]:
    """Deterministic channel group rotation keyed by *radix_level*."""
    n_groups = math.ceil(channels / channel_tile)
    # Perfect-shuffle schedule: rotate group assignment by radix_level
    indices = list(range(channels))
    if radix_level == 0 or n_groups <= 1:
        return indices
    # Apply group-level rotation
    groups: list[list[int]] = [[] for _ in range(n_groups)]
    for i, idx in enumerate(indices):
        groups[i % n_groups].append(idx)
    # Rotate groups
    rotation = radix_level % n_groups
    rotated = groups[rotation:] + groups[:rotation]
    return [idx for group in rotated for idx in group]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def build_route_plan(
    height: int,
    width: int,
    channels: int,
    spatial_tile: int,
    channel_tile: int,
    route: str,
    radix_level: int = 0,
) -> RoutePlan:
    """Build a deterministic, cached route plan.

    Parameters
    ----------
    height, width : spatial dimensions
    channels : channel count
    spatial_tile : ``g`` — number of tokens per spatial group
    channel_tile : ``d`` — number of channels per channel group
    route : one of ``"local"``, ``"shifted"``, ``"radix"``
    radix_level : radix digit level (only for route ``"radix"``)
    """
    n_tokens = height * width

    # --- Token routing ---
    if route == "local":
        token_order = _local_token_order(height, width)
    elif route == "shifted":
        token_order = _shifted_token_order(height, width)
    elif route == "radix":
        token_order = radix_order(n_tokens, spatial_tile, radix_level)
    else:
        raise ValueError(f"unknown route type: {route!r}")

    token_indices, token_mask = _pad_and_mask(token_order, spatial_tile, n_tokens)

    # --- Channel routing ---
    channel_order = _channel_order(channels, channel_tile, radix_level)
    channel_indices, channel_mask = _pad_and_mask(
        channel_order, channel_tile, channels,
    )

    return RoutePlan(
        token_indices=token_indices,
        channel_indices=channel_indices,
        token_mask=token_mask,
        channel_mask=channel_mask,
        height=height,
        width=width,
        channels=channels,
        route=route,
        radix_level=radix_level,
    )


def build_stage_schedule(depth: int) -> list[tuple[str, int]]:
    """Build the canonical route schedule for a stage of given *depth*.

    Cycles through: local → shifted → radix-0 → radix-1 → radix-2,
    truncated or repeated to match *depth*.  Every stage begins with local.
    """
    cycle: list[tuple[str, int]] = [
        ("local", 0),
        ("shifted", 0),
        ("radix", 0),
        ("radix", 1),
        ("radix", 2),
    ]
    schedule: list[tuple[str, int]] = []
    for i in range(depth):
        schedule.append(cycle[i % len(cycle)])
    return schedule
