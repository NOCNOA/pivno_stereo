"""Isolated DispNO stereo model with PIVNO's implicit SR decoder.

This module deliberately does not register or modify the repository's existing
PIVNO/DEFOM model paths.  It contains only the DispNO image encoder, Galerkin
operator, internal SepConvGRU, and coordinate-query super-resolution head.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .galerkin import simple_attn
from .sronet import BasicEncoder256, SepConvGRU


def make_coord_grid(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized pixel-centre coordinates in ``(y, x)`` order."""
    if len(shape) != 2 or any(int(size) <= 0 for size in shape):
        raise ValueError(f"shape must contain two positive sizes, got {shape}")
    axes = []
    for size in map(int, shape):
        radius = 1.0 / size
        axes.append(
            -1.0
            + radius
            + 2.0
            * radius
            * torch.arange(size, device=device, dtype=dtype)
        )
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


class DispNOSR(nn.Module):
    """Predict full-resolution scalar disparity using only DispNO and SR.

    Images stay at their native resolution.  ``BasicEncoder256`` performs the
    learned 1/4 downsampling used by the current stereo DispNO path, while the
    original four-neighbour implicit decoder queries every output pixel.
    """

    MODEL_NAME = "dispno_sr_only"
    TRAINABLE_ROOTS = (
        "snet",
        "conv0",
        "conv1",
        "gru",
        "sr_fc1",
        "sr_fc2",
    )
    SR_HALO = 2  # two 3x3 convolutions in the SR decoder

    def __init__(
        self,
        *,
        width: int = 128,
        blocks: int = 16,
        iters: int = 5,
        input_channels: int = 3,
        sr_tile_height: int = 80,
        checkpoint_sr: bool = True,
    ) -> None:
        super().__init__()
        if int(width) != 128:
            raise ValueError(
                "DispNOSR currently requires width=128 because two 64-channel "
                f"stereo features are concatenated, got {width}"
            )
        if int(iters) < 1:
            raise ValueError(f"iters must be positive, got {iters}")
        if int(input_channels) not in (1, 3):
            raise ValueError(
                f"input_channels must be 1 or 3, got {input_channels}"
            )
        if int(sr_tile_height) < 0:
            raise ValueError(
                f"sr_tile_height must be non-negative, got {sr_tile_height}"
            )

        self.width = int(width)
        self.blocks = int(blocks)
        self.iters = int(iters)
        self.input_channels = int(input_channels)
        self.sr_tile_height = int(sr_tile_height)
        self.checkpoint_sr = bool(checkpoint_sr)

        self.snet = BasicEncoder256(
            output_dim=64,
            norm_fn="instance",
            dropout=0.0,
            input_channels=self.input_channels,
        )
        self.conv0 = simple_attn(self.width, self.blocks)
        self.conv1 = simple_attn(self.width, self.blocks)
        self.gru = SepConvGRU(hidden_dim=self.width, input_dim=self.width)

        sr_input_channels = self.width * 4 + 4 * 2 + 2
        self.sr_fc1 = nn.Conv2d(
            sr_input_channels, 64, kernel_size=3, stride=1, padding=1
        )
        self.sr_fc2 = nn.Conv2d(
            64, 1, kernel_size=3, stride=1, padding=1
        )

    def model_config(self) -> dict:
        return {
            "model": self.MODEL_NAME,
            "width": self.width,
            "blocks": self.blocks,
            "iters": self.iters,
            "input_channels": self.input_channels,
            "sr_tile_height": self.sr_tile_height,
            "checkpoint_sr": self.checkpoint_sr,
            "output": "full_resolution_scalar_disparity_pixels",
            "components": list(self.TRAINABLE_ROOTS),
        }

    def _operator(self, stereo_feature: torch.Tensor) -> torch.Tensor:
        # Retain the current stereo PIVNO numerical policy: the two Galerkin
        # projections can overflow FP16 on full-resolution validation inputs.
        with torch.cuda.amp.autocast(enabled=False):
            hidden = self.conv0(stereo_feature.float(), 0)
            hidden = self.conv1(hidden, 1)
        return hidden

    @staticmethod
    def _normalise_queries(
        coord: torch.Tensor,
        cell: torch.Tensor,
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coord = coord.to(device=feature.device, dtype=feature.dtype)
        cell = cell.to(device=feature.device, dtype=feature.dtype)
        if coord.ndim == 3:
            coord = coord.unsqueeze(0)
        if cell.ndim == 1:
            cell = cell.unsqueeze(0)
        if coord.ndim != 4 or coord.shape[-1] != 2:
            raise ValueError(
                f"coord must be [B,H,W,2] or [H,W,2], got {tuple(coord.shape)}"
            )
        if cell.ndim != 2 or cell.shape[-1] != 2:
            raise ValueError(
                f"cell must be [B,2] or [2], got {tuple(cell.shape)}"
            )

        batch = feature.shape[0]
        if coord.shape[0] == 1 and batch > 1:
            coord = coord.expand(batch, -1, -1, -1)
        if cell.shape[0] == 1 and batch > 1:
            cell = cell.expand(batch, -1)
        if coord.shape[0] != batch or cell.shape[0] != batch:
            raise ValueError(
                "coord/cell batch dimensions must match the feature batch"
            )
        return coord, cell

    def spatial_interpolation(
        self,
        feature: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """Build the official four-neighbour implicit-SR query tensor."""
        coord, cell = self._normalise_queries(coord, cell, feature)
        feature_height, feature_width = feature.shape[-2:]
        feature_coord = make_coord_grid(
            (feature_height, feature_width),
            device=feature.device,
            dtype=feature.dtype,
        )
        feature_coord = feature_coord.permute(2, 0, 1).unsqueeze(0)
        feature_coord = feature_coord.expand(feature.shape[0], -1, -1, -1)

        radius_y = 1.0 / feature_height
        radius_x = 1.0 / feature_width
        relative_coords = []
        sampled_features = []
        areas = []
        for offset_y in (-1, 1):
            for offset_x in (-1, 1):
                shifted_coord = coord.clone()
                shifted_coord[..., 0] += offset_y * radius_y + 1e-6
                shifted_coord[..., 1] += offset_x * radius_x + 1e-6
                shifted_coord.clamp_(-1.0 + 1e-6, 1.0 - 1e-6)
                sampling_grid = shifted_coord.flip(-1)

                sampled_feature = F.grid_sample(
                    feature,
                    sampling_grid,
                    mode="nearest",
                    align_corners=False,
                )
                sampled_coord = F.grid_sample(
                    feature_coord,
                    sampling_grid,
                    mode="nearest",
                    align_corners=False,
                )
                relative_coord = coord.permute(0, 3, 1, 2) - sampled_coord
                relative_coord[:, 0] *= feature_height
                relative_coord[:, 1] *= feature_width
                area = (
                    relative_coord[:, 0] * relative_coord[:, 1]
                ).abs()

                relative_coords.append(relative_coord)
                sampled_features.append(sampled_feature)
                areas.append(area + 1e-9)

        total_area = torch.stack(areas, dim=0).sum(dim=0)
        # The official local ensemble uses the opposite corner's area.
        opposite_areas = (areas[3], areas[2], areas[1], areas[0])
        sampled_features = [
            sampled * (area / total_area).unsqueeze(1)
            for sampled, area in zip(sampled_features, opposite_areas)
        ]

        relative_cell = cell.clone()
        relative_cell[:, 0] *= feature_height
        relative_cell[:, 1] *= feature_width
        relative_cell = relative_cell[:, :, None, None].expand(
            -1, -1, coord.shape[1], coord.shape[2]
        )
        return torch.cat(
            [*relative_coords, *sampled_features, relative_cell], dim=1
        )

    def _decode_query_tile(
        self,
        hidden: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        query = self.spatial_interpolation(hidden, coord, cell)
        return self.sr_fc2(F.gelu(self.sr_fc1(query)))

    def _decode_full_resolution(
        self,
        hidden: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        output_height = coord.shape[1]
        tile_height = self.sr_tile_height or output_height
        if tile_height >= output_height:
            if self.training and self.checkpoint_sr and hidden.requires_grad:
                return checkpoint(
                    self._decode_query_tile,
                    hidden,
                    coord,
                    cell,
                    use_reentrant=False,
                )
            return self._decode_query_tile(hidden, coord, cell)

        tiles = []
        for top in range(0, output_height, tile_height):
            bottom = min(top + tile_height, output_height)
            extended_top = max(0, top - self.SR_HALO)
            extended_bottom = min(output_height, bottom + self.SR_HALO)
            extended_coord = coord[:, extended_top:extended_bottom]
            if self.training and self.checkpoint_sr and hidden.requires_grad:
                extended_prediction = checkpoint(
                    self._decode_query_tile,
                    hidden,
                    extended_coord,
                    cell,
                    use_reentrant=False,
                )
            else:
                extended_prediction = self._decode_query_tile(
                    hidden, extended_coord, cell
                )
            crop_top = top - extended_top
            crop_bottom = crop_top + (bottom - top)
            tiles.append(extended_prediction[:, :, crop_top:crop_bottom])
        return torch.cat(tiles, dim=2)

    def _dense_queries(
        self,
        batch: int,
        output_size: Tuple[int, int],
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = output_size
        coord = make_coord_grid(
            output_size,
            device=feature.device,
            dtype=feature.dtype,
        ).unsqueeze(0)
        coord = coord.expand(batch, -1, -1, -1)
        cell = feature.new_tensor((2.0 / height, 2.0 / width))
        cell = cell.unsqueeze(0).expand(batch, -1)
        return coord, cell

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        coord: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        if image1.ndim != 4 or image2.ndim != 4:
            raise ValueError(
                "DispNOSR inputs must be [B,C,H,W], got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
        if image1.shape != image2.shape:
            raise ValueError(
                f"stereo input shapes must match, got {tuple(image1.shape)} "
                f"and {tuple(image2.shape)}"
            )
        if image1.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, "
                f"got {image1.shape[1]}"
            )
        if (coord is None) != (cell is None):
            raise ValueError("coord and cell must be provided together")

        left_feature, right_feature = self.snet([image1, image2])
        operator_input = torch.cat((left_feature, right_feature), dim=1)
        stereo_feature = self._operator(operator_input)
        if coord is None:
            coord, cell = self._dense_queries(
                image1.shape[0], image1.shape[-2:], stereo_feature
            )
        else:
            coord, cell = self._normalise_queries(coord, cell, stereo_feature)

        hidden = stereo_feature.clone()
        predictions = []
        for _ in range(self.iters):
            hidden = self.gru(hidden, stereo_feature)
            predictions.append(
                self._decode_full_resolution(hidden, coord, cell)
            )
        return predictions

