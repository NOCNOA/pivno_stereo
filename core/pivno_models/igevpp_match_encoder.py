"""IGEV++ quarter-resolution descriptor without any cost-volume modules."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    """Match the module/key layout used by the upstream IGEV++ checkpoint."""

    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        IN=True,
        relu=True,
        **kwargs,
    ):
        super().__init__()
        if is_3d:
            conv_cls = nn.ConvTranspose3d if deconv else nn.Conv3d
            norm_cls = nn.InstanceNorm3d
        else:
            conv_cls = nn.ConvTranspose2d if deconv else nn.Conv2d
            norm_cls = nn.InstanceNorm2d
        self.conv = conv_cls(in_channels, out_channels, bias=False, **kwargs)
        self.IN = norm_cls(out_channels)
        self.use_in = bool(IN)
        self.relu = bool(relu)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = F.leaky_relu(x)
        return x


class Conv2x(nn.Module):
    """Exact 2-D upsample/fuse block used by the IGEV++ feature pyramid."""

    def __init__(self, in_channels, out_channels, deconv=True, concat=True):
        super().__init__()
        self.concat = bool(concat)
        self.conv1 = BasicConv(
            in_channels,
            out_channels,
            deconv=deconv,
            kernel_size=4 if deconv else 3,
            stride=2,
            padding=1,
        )
        conv2_in = 2 * out_channels if self.concat else out_channels
        self.conv2 = BasicConv(
            conv2_in,
            2 * out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, residual):
        x = self.conv1(x)
        if x.shape[-2:] != residual.shape[-2:]:
            x = F.interpolate(x, size=residual.shape[-2:], mode="nearest")
        x = torch.cat((x, residual), dim=1) if self.concat else x + residual
        return self.conv2(x)


class MobileNetV2FPN(nn.Module):
    """The MobileNetV2/FPN portion of the released IGEV++ model."""

    def __init__(self):
        super().__init__()
        # Never contact the network here. The stereo checkpoint is loaded by
        # ``IGEVPlusPlusMatchEncoder.load_pretrained``.
        model = timm.create_model(
            "mobilenetv2_100", pretrained=False, features_only=True
        )
        boundaries = (1, 2, 3, 5, 6)
        channels = (16, 24, 32, 96, 160)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = getattr(model, "act1", nn.Identity())
        self.block0 = nn.Sequential(*model.blocks[0:boundaries[0]])
        self.block1 = nn.Sequential(*model.blocks[boundaries[0]:boundaries[1]])
        self.block2 = nn.Sequential(*model.blocks[boundaries[1]:boundaries[2]])
        self.block3 = nn.Sequential(*model.blocks[boundaries[2]:boundaries[3]])
        self.block4 = nn.Sequential(*model.blocks[boundaries[3]:boundaries[4]])
        self.deconv32_16 = Conv2x(channels[4], channels[3])
        self.deconv16_8 = Conv2x(2 * channels[3], channels[2])
        self.deconv8_4 = Conv2x(2 * channels[2], channels[1])
        self.conv4 = BasicConv(
            2 * channels[1],
            2 * channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, image):
        image = self.act1(self.bn1(self.conv_stem(image)))
        feat2 = self.block0(image)
        feat4 = self.block1(feat2)
        feat8 = self.block2(feat4)
        feat16 = self.block3(feat8)
        feat32 = self.block4(feat16)
        feat16 = self.deconv32_16(feat32, feat16)
        feat8 = self.deconv16_8(feat16, feat8)
        feat4 = self.deconv8_4(feat8, feat4)
        feat4 = self.conv4(feat4)
        return feat4


class IGEVPlusPlusMatchEncoder(nn.Module):
    """Only the released IGEV++ image-to-96-channel matching descriptor."""

    CHECKPOINT_PREFIXES = ("feature.", "stem_2.", "stem_4.", "conv.", "desc.")
    OUTPUT_CHANNELS = 96

    def __init__(self):
        super().__init__()
        self.feature = MobileNetV2FPN()
        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
        self.conv = BasicConv(
            96, 96, kernel_size=3, padding=1, stride=1
        )
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

    @staticmethod
    def _strip_module(state):
        return {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state.items()
        }

    def load_pretrained(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        state = self._strip_module(state)
        selected = {
            key: value
            for key, value in state.items()
            if key.startswith(self.CHECKPOINT_PREFIXES)
        }
        expected = set(self.state_dict())
        if set(selected) != expected:
            raise ValueError(
                "IGEV++ descriptor checkpoint mismatch: "
                f"missing={sorted(expected - set(selected))}, "
                f"unexpected={sorted(set(selected) - expected)}"
            )
        self.load_state_dict(selected, strict=True)
        return len(selected)

    def forward_one(self, image):
        feature4 = self.feature(image)
        stem2 = self.stem_2(image)
        stem4 = self.stem_4(stem2)
        return self.desc(self.conv(torch.cat((feature4, stem4), dim=1)))

    def forward(self, left, right):
        if left.shape != right.shape:
            raise ValueError(
                "IGEV++ stereo inputs must match, got "
                f"{tuple(left.shape)} and {tuple(right.shape)}"
            )
        batch = left.shape[0]
        descriptor = self.forward_one(torch.cat((left, right), dim=0))
        left_descriptor, right_descriptor = descriptor.split(batch, dim=0)
        return left_descriptor, right_descriptor
