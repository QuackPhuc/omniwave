"""OmniWeave hierarchical backbone — stem, four stages, classifier."""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
from torch import Tensor

from omniweave.models.block import OmniWeaveBlock
from omniweave.models.routing import build_stage_schedule


def space_to_depth_nhwc(x: Tensor, factor: int) -> Tensor:
    """Rearrange spatial dimensions into channels (NHWC layout).

    ``[B, H, W, C] → [B, H/f, W/f, C * f²]``
    """
    b, h, w, c = x.shape
    if h % factor or w % factor:
        raise ValueError(
            f"height {h} and width {w} must be divisible by factor {factor}"
        )
    x = x.reshape(b, h // factor, factor, w // factor, factor, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(
        b, h // factor, w // factor, c * factor * factor
    )


class _Stage(nn.Module):
    """One backbone stage: optional anchor → sequence of OmniWeaveBlocks."""

    def __init__(
        self,
        height: int,
        width: int,
        dim: int,
        depth: int,
        channel_tile: int,
        expansion: int,
        layer_scale_init: float,
        spatial_tile: int,
        use_anchor: bool,
        backend: str,
        drop_path_rates: list[float],
    ) -> None:
        super().__init__()
        self.anchor: nn.Linear | None = None
        if use_anchor:
            self.anchor = nn.Linear(dim, dim, bias=False)

        schedule = build_stage_schedule(depth)
        self.blocks = nn.ModuleList()
        for (route, radix_level), drop_path_rate in zip(
            schedule,
            drop_path_rates,
            strict=True,
        ):
            self.blocks.append(
                OmniWeaveBlock(
                    height=height,
                    width=width,
                    dim=dim,
                    channel_tile=channel_tile,
                    route=route,
                    radix_level=radix_level,
                    expansion=expansion,
                    layer_scale_init=layer_scale_init,
                    spatial_tile=spatial_tile,
                    backend=backend,
                    drop_path_rate=drop_path_rate,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.anchor is not None:
            x = self.anchor(x)
        for block in self.blocks:
            x = block(x)
        return x


class OmniWeaveBackbone(nn.Module):
    """Four-stage OmniWeave-T classification backbone.

    Input:  ``[B, 3, H, W]``  (NCHW — standard PyTorch image convention)
    Output: ``[B, num_classes]``

    ``forward_features`` returns an ``OrderedDict`` of four NHWC feature maps.
    """

    def __init__(
        self,
        widths: list[int] | None = None,
        depths: list[int] | None = None,
        channel_tiles: list[int] | None = None,
        num_classes: int = 1000,
        expansion: int = 2,
        layer_scale_init: float = 1e-5,
        anchor_stages: list[int] | None = None,
        spatial_tile: int = 16,
        backend: str = "reference",
        input_size: int = 224,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        widths = widths or [128, 256, 512, 1024]
        depths = depths or [2, 3, 8, 2]
        channel_tiles = channel_tiles or [64, 64, 128, 128]
        if anchor_stages is None:
            anchor_stages = [3, 4]

        if not (len(widths) == len(depths) == len(channel_tiles) == 4):
            raise ValueError(
                "widths, depths, and channel_tiles must each have length 4"
            )
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")

        # Stem: space-to-depth(4) → Linear(48, C1)
        self.stem_proj = nn.Linear(3 * 4 * 4, widths[0], bias=False)

        # Compute resolutions per stage
        res0 = input_size // 4  # After stem
        resolutions = [res0 // (2 ** i) for i in range(4)]

        # Downsample modules (between stages)
        self.downsamples = nn.ModuleList()
        for i in range(3):
            self.downsamples.append(
                nn.Linear(widths[i] * 4, widths[i + 1], bias=False)
            )

        # Four stages
        self.stages = nn.ModuleList()
        total_depth = sum(depths)
        drop_path_rates = [
            drop_path_rate * index / max(total_depth - 1, 1)
            for index in range(total_depth)
        ]
        block_offset = 0
        for i in range(4):
            stage_rates = drop_path_rates[block_offset:block_offset + depths[i]]
            self.stages.append(
                _Stage(
                    height=resolutions[i],
                    width=resolutions[i],
                    dim=widths[i],
                    depth=depths[i],
                    channel_tile=channel_tiles[i],
                    expansion=expansion,
                    layer_scale_init=layer_scale_init,
                    spatial_tile=spatial_tile,
                    use_anchor=(i + 1) in anchor_stages,
                    backend=backend,
                    drop_path_rates=stage_rates,
                )
            )
            block_offset += depths[i]

        # Classifier: global average pool → linear
        self.head = nn.Linear(widths[-1], num_classes)
        self._widths = widths

    def forward_features(self, x: Tensor) -> OrderedDict[str, Tensor]:
        """Extract four-scale feature maps.

        Input:  ``[B, 3, H, W]`` (NCHW)
        Output: ``OrderedDict`` with keys ``stage1``..``stage4``,
                values in NHWC layout.
        """
        # NCHW → NHWC
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 3]

        # Stem: space-to-depth(4) → project
        x = space_to_depth_nhwc(x, 4)  # [B, H/4, W/4, 48]
        x = self.stem_proj(x)          # [B, H/4, W/4, C1]

        features = OrderedDict()
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features[f"stage{i + 1}"] = x

            # Downsample between stages (not after the last)
            if i < 3:
                x = space_to_depth_nhwc(x, 2)
                x = self.downsamples[i](x)

        return features

    def forward(self, x: Tensor) -> Tensor:
        """Classify: ``[B, 3, H, W]`` → ``[B, num_classes]``."""
        features = self.forward_features(x)
        stage4 = features["stage4"]
        # Global average pool over spatial dims (NHWC)
        pooled = stage4.mean(dim=(1, 2))
        return self.head(pooled)
