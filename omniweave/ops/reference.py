"""Canonical pack/unpack and reference BiGEMM implementation.

This module defines the numerical oracle for the two-sided tiled GEMM.
It must run on CPU and CUDA without Triton installed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from omniweave.models.routing import RoutePlan


def pack_tiles(
    x: Tensor,
    plan: RoutePlan,
) -> tuple[Tensor, Tensor]:
    """Pack an NHWC tensor into route-planned tiles.

    Returns
    -------
    tiles : ``[B, Q_s, Q_c, g, d]``
    mask  : ``[1, Q_s, Q_c, g, d]`` — broadcast over batch
    """
    b, h, w, c = x.shape
    n = h * w
    flat = x.reshape(b, n, c)  # [B, N, C]

    # Append one zero-padded row for spatial padding
    flat = torch.cat([flat, flat.new_zeros(b, 1, c)], dim=1)  # [B, N+1, C]
    # Append one zero-padded channel for channel padding
    flat = torch.cat([flat, flat.new_zeros(b, n + 1, 1)], dim=2)  # [B, N+1, C+1]

    # Gather spatial: token_indices has pad index = N (valid in N+1 dim)
    tok_idx = plan.token_indices  # [Q_s, g]
    q_s, g = tok_idx.shape
    tok_idx_expanded = tok_idx.reshape(1, q_s * g, 1).expand(b, -1, c + 1)
    tok_idx_expanded = tok_idx_expanded.to(flat.device)
    gathered = torch.gather(flat, 1, tok_idx_expanded)  # [B, Q_s*g, C+1]
    gathered = gathered.reshape(b, q_s, g, c + 1)

    # Gather channel: channel_indices has pad index = C (valid in C+1 dim)
    ch_idx = plan.channel_indices  # [Q_c, d]
    q_c, d = ch_idx.shape
    ch_idx_expanded = ch_idx.reshape(1, 1, 1, q_c * d).expand(b, q_s, g, -1)
    ch_idx_expanded = ch_idx_expanded.to(gathered.device)
    tiles = torch.gather(gathered, 3, ch_idx_expanded)
    tiles = tiles.reshape(b, q_s, g, q_c, d).permute(0, 1, 3, 2, 4)  # [B, Q_s, Q_c, g, d]

    # Build broadcast mask
    tok_mask = plan.token_mask.to(x.device)   # [Q_s, g]
    ch_mask = plan.channel_mask.to(x.device)  # [Q_c, d]
    mask = (
        tok_mask.unsqueeze(1).unsqueeze(-1)  # [Q_s, 1, g, 1]
        & ch_mask.unsqueeze(0).unsqueeze(2)  # [1, Q_c, 1, d]
    )  # [Q_s, Q_c, g, d]
    mask = mask.unsqueeze(0)  # [1, Q_s, Q_c, g, d]

    # Zero out padding positions
    tiles = tiles * mask.to(tiles.dtype)

    return tiles, mask


def unpack_tiles(
    tiles: Tensor,
    plan: RoutePlan,
    batch_size: int,
) -> Tensor:
    """Scatter route-planned tiles back to NHWC layout.

    Parameters
    ----------
    tiles : ``[B, Q_s, Q_c, g, d]``
    plan  : the same ``RoutePlan`` used to pack
    batch_size : B

    Returns
    -------
    output : ``[B, H, W, C]``
    """
    b = batch_size
    h, w, c = plan.height, plan.width, plan.channels
    n = h * w
    q_s, g = plan.token_indices.shape
    q_c, d = plan.channel_indices.shape

    # Initialize output + padding row/channel
    output = tiles.new_zeros(b, n + 1, c + 1)

    # Scatter channel first: tiles -> [B, Q_s, g, Q_c, d]
    tiles_reorder = tiles.permute(0, 1, 3, 2, 4)  # [B, Q_s, g, Q_c, d]

    # Scatter channels back
    ch_idx = plan.channel_indices.to(tiles.device)  # [Q_c, d]
    ch_flat = ch_idx.reshape(q_c * d)

    # For each spatial group, scatter channels
    spatial_gathered = tiles_reorder.reshape(b, q_s, g, q_c * d)

    # Build intermediate: [B, Q_s, g, C+1]
    intermediate = tiles.new_zeros(b, q_s, g, c + 1)
    ch_scatter = ch_flat.reshape(1, 1, 1, q_c * d).expand(b, q_s, g, -1)
    intermediate.scatter_add_(3, ch_scatter, spatial_gathered)

    # Scatter tokens back
    tok_idx = plan.token_indices.to(tiles.device)  # [Q_s, g]
    tok_flat = tok_idx.reshape(q_s * g)

    intermediate_flat = intermediate.reshape(b, q_s * g, c + 1)
    tok_scatter = tok_flat.reshape(1, q_s * g, 1).expand(b, -1, c + 1)
    output.scatter_add_(1, tok_scatter, intermediate_flat)

    # Trim padding
    return output[:, :n, :c].reshape(b, h, w, c)


def bigemm_delta_reference(
    *,
    x: Tensor,
    plan: RoutePlan,
    a_u: Tensor,
    a_v: Tensor,
    a_o: Tensor,
    b_u: Tensor,
    b_v: Tensor,
    b_o: Tensor,
    gamma: Tensor,
) -> Tensor:
    """Reference two-sided BiGEMM producing a residual delta.

    Parameters
    ----------
    x     : ``[B, H, W, C]`` — input features (pre-normalized by caller)
    plan  : route plan
    a_u, a_v, a_o : ``[g, g]`` — shared spatial matrices
    b_u, b_v : ``[Q_c, d, rd]`` — per-channel-group right projection
    b_o   : ``[Q_c, rd, d]`` — per-channel-group output projection
    gamma : ``[C]`` — LayerScale vector

    Returns
    -------
    delta : ``[B, H, W, C]``
    """
    tiles, mask = pack_tiles(x, plan)
    b, q_s, q_c, g, d = tiles.shape

    # Reshape for batched einsum: merge B and Q_s
    flat = tiles.reshape(b * q_s, q_c, g, d)

    # Two-sided projections: A @ T @ B for U and V branches
    # einsum: spatial mix (left) then channel project (right)
    u = torch.einsum("ij,bqjk,qke->bqie", a_u, flat, b_u)  # [B*Q_s, Q_c, g, rd]
    v = torch.einsum("ij,bqjk,qke->bqie", a_v, flat, b_v)  # [B*Q_s, Q_c, g, rd]

    # Gated activation
    z = F.silu(u) * v  # [B*Q_s, Q_c, g, rd]

    # Output projection
    delta = torch.einsum("ij,bqje,qed->bqid", a_o, z, b_o)  # [B*Q_s, Q_c, g, d]

    # Reshape back and mask padding
    delta = delta.reshape(b, q_s, q_c, g, d) * mask.to(delta.dtype)

    # Apply LayerScale per channel
    ch_idx = plan.channel_indices.clamp_max(plan.channels - 1).to(gamma.device)
    ch_mask = plan.channel_mask.to(gamma.device, gamma.dtype)
    grouped_gamma = gamma[ch_idx] * ch_mask  # [Q_c, d]
    delta = delta * grouped_gamma[None, None, :, None, :]

    return unpack_tiles(delta, plan, b)
