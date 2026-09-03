"""PACT2-only MobileNet feature fusion and context encoders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from core.extractor import ConvBlock, MultiBasicEncoder, ResidualBlock
from core.submodules import BasicConv_IN, Conv2x_IN


def _build_pretrained_mobilenetv2() -> nn.Module:
    """Load official weights locally when possible, without network retries."""
    filename = "mobilenetv2_100_ra-b33bc2c4.pth"
    candidates = [
        Path(__file__).resolve().parents[1] / "checkpoints" / filename,
        Path(torch.hub.get_dir()) / "checkpoints" / filename,
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename,
    ]
    explicit_checkpoint = os.environ.get("PACT2_MOBILENET_CHECKPOINT")
    if explicit_checkpoint:
        candidates.insert(0, Path(explicit_checkpoint).expanduser())
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "PACT2 requires local MobileNetV2 weights named "
            f"{filename}; checked: {checked}. Set "
            "PACT2_MOBILENET_CHECKPOINT to an explicit file if needed."
        )
    # The cached file includes the classifier, so load the full model and
    # retain only the backbone modules below.
    return timm.create_model(
        "mobilenetv2_100", pretrained=False,
        checkpoint_path=str(checkpoint), features_only=False,
    )


def _drop_legacy_keys(
    state_dict: MutableMapping[str, torch.Tensor], prefixes: Sequence[str]
) -> None:
    """Discard removed PACT compatibility branches during strict loading."""
    for key in tuple(state_dict):
        if any(key.startswith(prefix) for prefix in prefixes):
            state_dict.pop(key)


def _remove_unused_convblock_norms(block: ConvBlock) -> None:
    """ConvBlock.forward uses norm1 only; remove its dormant norm aliases."""
    for name in ("norm2", "norm3"):
        if hasattr(block, name):
            delattr(block, name)


class PACT2FeatureEncoder(nn.Module):
    """IGEV++ MobileNetV2 pyramid fused with per-view monocular features.

    MobileNet follows the official IGEV++ encoder/decoder stages.  Its decoded
    1/4 feature is concatenated with a projected Depth-Anything feature; the
    1/8 and 1/16 MobileNet outputs remain stereo-only so the coarse matching
    path is not directly dominated by the monocular prior.
    """

    def __init__(self, d_dim: int, output_dim: Sequence[int],
                 norm_fn: str = "batch", downsample: int = 2) -> None:
        super().__init__()
        if len(output_dim) != 3:
            raise ValueError(
                f"PACT2FeatureEncoder expects 3 output dims, got {output_dim}"
            )
        if downsample != 2:
            raise ValueError(
                "PACT2 MobileNet features are fixed at 1/4, 1/8 and 1/16; "
                f"n_downsample must be 2, got {downsample}"
            )
        if norm_fn != "instance":
            raise ValueError(
                "PACT2 MobileNet decoder uses InstanceNorm; "
                f"got norm_fn={norm_fn!r}"
            )

        model = _build_pretrained_mobilenetv2()
        layers = (1, 2, 3, 5, 6)
        channels = (16, 24, 32, 96, 160)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        # timm>=1.0 folds the activation into BatchNormAct2d; older timm
        # exposes the separate act1 used by the official IGEV++ source.
        self.act1 = getattr(model, "act1", nn.Identity())
        self.block0 = nn.Sequential(*model.blocks[:layers[0]])
        self.block1 = nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2x_IN(
            channels[4], channels[3], deconv=True, concat=True
        )
        self.deconv16_8 = Conv2x_IN(
            channels[3] * 2, channels[2], deconv=True, concat=True
        )
        self.deconv8_4 = Conv2x_IN(
            channels[2] * 2, channels[1], deconv=True, concat=True
        )
        self.mobile_4 = BasicConv_IN(
            channels[1] * 2, channels[1] * 2,
            kernel_size=3, stride=1, padding=1,
        )

        # IGEV++ concatenates a separate 1/4 stem with its decoded MobileNet
        # feature.  Here the per-view monocular feature takes that role.
        self.mono_4 = BasicConv_IN(
            d_dim, channels[1] * 2, kernel_size=1, stride=1, padding=0,
        )
        self.fuse_4 = BasicConv_IN(
            channels[1] * 4, output_dim[0],
            kernel_size=3, stride=1, padding=1,
        )
        self.output_8 = BasicConv_IN(
            channels[2] * 2, output_dim[1],
            kernel_size=1, stride=1, padding=0,
        )
        self.output_16 = (
            nn.Identity()
            if channels[3] * 2 == output_dim[2]
            else nn.Conv2d(channels[3] * 2, output_dim[2], kernel_size=1)
        )

    def forward(self, images, depth_features):
        split_lr = isinstance(images, (tuple, list))
        if split_lr:
            batch_size = images[0].shape[0]
            images = torch.cat(images, dim=0)
        if isinstance(depth_features, (tuple, list)):
            depth_features = torch.cat(depth_features, dim=0)

        features = self.act1(self.bn1(self.conv_stem(images)))
        features_2 = self.block0(features)
        features_4 = self.block1(features_2)
        features_8 = self.block2(features_4)
        features_16 = self.block3(features_8)
        features_32 = self.block4(features_16)

        features_16 = self.deconv32_16(features_32, features_16)
        features_8 = self.deconv16_8(features_16, features_8)
        features_4 = self.mobile_4(self.deconv8_4(features_8, features_4))
        mono_4 = self.mono_4(depth_features)
        if mono_4.shape[-2:] != features_4.shape[-2:]:
            mono_4 = F.interpolate(
                mono_4, size=features_4.shape[-2:], mode="bilinear",
                align_corners=True,
            )
        pyramid = [
            self.fuse_4(torch.cat((features_4, mono_4), dim=1)),
            self.output_8(features_8),
            self.output_16(features_16),
        ]

        if split_lr:
            return (
                [feature[:batch_size] for feature in pyramid],
                [feature[batch_size:] for feature in pyramid],
            )
        return pyramid


class PACT2ContextEncoder(MultiBasicEncoder):
    """Three-level context pyramid with one recurrent-state head per level."""

    def __init__(self, d_dim: int, output_dim: Sequence[int],
                 norm_fn: str = "batch", downsample: int = 2) -> None:
        super().__init__(
            d_dim,
            output_dim=[output_dim],
            norm_fn=norm_fn,
            downsample=downsample,
        )
        for block in (self.conv08, self.conv16, self.conv32):
            _remove_unused_convblock_norms(block)

    def forward(self, image, depth_features):
        return super().forward(
            image, depth_features, num_layers=3, output_counts=(1, 1, 1)
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        removed_prefixes = [
            f"{prefix}{block_name}.{norm_name}."
            for block_name in ("conv08", "conv16", "conv32")
            for norm_name in ("norm2", "norm3")
        ]
        for output_name in ("outputs08", "outputs16", "outputs32"):
            output_prefix = f"{prefix}{output_name}."
            for key in tuple(state_dict):
                if not key.startswith(output_prefix):
                    continue
                output_index = key[len(output_prefix):].split(".", 1)[0]
                if output_index != "0":
                    removed_prefixes.append(f"{output_prefix}{output_index}.")
        _drop_legacy_keys(state_dict, tuple(set(removed_prefixes)))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


__all__ = ["PACT2FeatureEncoder", "PACT2ContextEncoder"]
