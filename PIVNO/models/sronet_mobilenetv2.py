"""MobileNetV2-backed PIVNO image feature encoder.

This module intentionally leaves the original :mod:`sronet` implementation
unchanged.  The MobileNetV2 backbone exposes features at four spatial scales
and fuses them back to PIVNO's required quarter-resolution, 64-channel stereo
feature interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

from .sronet import PIVNO


class MobileNetV2FeatureEncoder(nn.Module):
    """Shared RGB encoder producing ``[B, 64, H/4, W/4]`` features.

    A complete MobileNetV2 is initialized from scratch.  Features from output
    strides 4, 8, 16, and 32 are projected to 16 channels each, resized to the
    stride-4 grid, and fused.  No ImageNet weights are requested, so model
    construction never downloads external files and the stereo input contract
    remains RGB in ``[0, 1]``.
    """

    ENDPOINT_CHANNELS = {
        3: 24,    # output stride 4
        6: 32,    # output stride 8
        13: 96,   # output stride 16
        18: 1280, # output stride 32
    }
    OUTPUT_DIM = 64

    def __init__(self, input_channels: int = 3):
        super().__init__()
        if int(input_channels) != 3:
            raise ValueError(
                "MobileNetV2 PIVNO supports RGB input only, got "
                f"{input_channels} channels"
            )

        # weights=None is deliberate: this is an isolated from-scratch
        # backbone experiment and must not trigger an implicit download.
        self.input_channels = 3
        self.backbone = mobilenet_v2(weights=None).features
        lateral_dim = self.OUTPUT_DIM // len(self.ENDPOINT_CHANNELS)
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, lateral_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(lateral_dim),
                nn.ReLU6(inplace=True),
            )
            for channels in self.ENDPOINT_CHANNELS.values()
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(self.OUTPUT_DIM, self.OUTPUT_DIM, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.OUTPUT_DIM),
            nn.ReLU6(inplace=True),
        )

    def _encode_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != self.input_channels:
            raise ValueError(
                "MobileNetV2 encoder expects [B,3,H,W], got "
                f"{tuple(image.shape)}"
            )

        endpoints = []
        feature = image
        for index, layer in enumerate(self.backbone):
            feature = layer(feature)
            if index in self.ENDPOINT_CHANNELS:
                endpoints.append(feature)

        if len(endpoints) != len(self.ENDPOINT_CHANNELS):
            raise RuntimeError(
                f"expected {len(self.ENDPOINT_CHANNELS)} MobileNetV2 endpoints, "
                f"got {len(endpoints)}"
            )
        target_size = endpoints[0].shape[-2:]
        projected = []
        for endpoint, projection in zip(endpoints, self.lateral):
            endpoint = projection(endpoint)
            if endpoint.shape[-2:] != target_size:
                endpoint = F.interpolate(
                    endpoint,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(endpoint)
        return self.fuse(torch.cat(projected, dim=1))

    def forward(self, image):
        is_pair = isinstance(image, (tuple, list))
        if not is_pair:
            return self._encode_tensor(image)
        if len(image) != 2:
            raise ValueError(
                f"stereo feature input must contain two images, got {len(image)}"
            )
        batch_size = image[0].shape[0]
        if image[1].shape[0] != batch_size:
            raise ValueError(
                "left and right image batches must match, got "
                f"{image[0].shape[0]} and {image[1].shape[0]}"
            )
        encoded = self._encode_tensor(torch.cat(image, dim=0))
        return torch.split(encoded, [batch_size, batch_size], dim=0)


class PIVNOMobileNetV2(PIVNO):
    """PIVNO with only its shared image encoder replaced by MobileNetV2."""

    FEATURE_BACKBONE = "mobilenet_v2_multiscale_stride4"

    def __init__(self, width=128, blocks=16, iters=5, input_channels=3):
        if int(input_channels) != 3:
            raise ValueError(
                "PIVNOMobileNetV2 supports RGB input only, got "
                f"{input_channels} channels"
            )
        super().__init__(
            width=width,
            blocks=blocks,
            iters=iters,
            input_channels=input_channels,
        )
        self.snet = MobileNetV2FeatureEncoder(input_channels=input_channels)
