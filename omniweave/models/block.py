"""OmniWeave block — single residual with RMSNorm + BiGEMM delta."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from omniweave.models.routing import RoutePlan, build_route_plan
from omniweave.ops.bigemm import bigemm


class OmniWeaveBlock(nn.Module):
    """One OmniWeave block: RMSNorm → BiGEMM → residual.

    The block owns *one* route plan and returns ``x + delta``.
    """

    def __init__(
        self,
        height: int,
        width: int,
        dim: int,
        channel_tile: int,
        route: str,
        radix_level: int,
        expansion: int = 2,
        layer_scale_init: float = 1e-5,
        spatial_tile: int = 16,
        backend: str = "reference",
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        self.height = height
        self.width = width
        self.dim = dim
        self.backend = backend
        self.drop_path_rate = drop_path_rate

        g = spatial_tile
        d = channel_tile
        rd = d * expansion
        q_c = math.ceil(dim / d)

        # Build and cache the route plan
        self.plan: RoutePlan = build_route_plan(
            height, width, dim, g, d, route, radix_level,
        )

        # RMSNorm weight
        self.norm_weight = nn.Parameter(torch.ones(dim))

        # Spatial mixing matrices (shared across all tiles)
        self.a_u = nn.Parameter(torch.empty(g, g))
        self.a_v = nn.Parameter(torch.empty(g, g))
        self.a_o = nn.Parameter(torch.empty(g, g))

        # Channel projection matrices (per channel group)
        self.b_u = nn.Parameter(torch.empty(q_c, d, rd))
        self.b_v = nn.Parameter(torch.empty(q_c, d, rd))
        self.b_o = nn.Parameter(torch.empty(q_c, rd, d))

        # LayerScale
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.a_u)
        nn.init.orthogonal_(self.a_v)
        nn.init.eye_(self.a_o)
        nn.init.xavier_uniform_(self.b_u)
        nn.init.xavier_uniform_(self.b_v)
        nn.init.xavier_uniform_(self.b_o)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``[B, H, W, C]`` → ``[B, H, W, C]``."""
        # RMSNorm
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance.to(x.dtype) + 1e-6)
        normalized = (normalized * self.norm_weight).to(x.dtype)

        delta = bigemm(
            x=normalized,
            plan=self.plan,
            a_u=self.a_u,
            a_v=self.a_v,
            a_o=self.a_o,
            b_u=self.b_u,
            b_v=self.b_v,
            b_o=self.b_o,
            gamma=self.gamma,
            backend=self.backend,
        )
        if self.training and self.drop_path_rate:
            keep_probability = 1.0 - self.drop_path_rate
            sample_shape = (delta.shape[0],) + (1,) * (delta.ndim - 1)
            keep_mask = torch.empty(
                sample_shape,
                dtype=delta.dtype,
                device=delta.device,
            ).bernoulli_(keep_probability)
            delta = delta * keep_mask / keep_probability
        return x + delta
