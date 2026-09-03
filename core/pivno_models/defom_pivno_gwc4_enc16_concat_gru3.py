"""PIVNO-DEFOM with directly concatenated GWC4/enc16 matches and 3x3 GRUs."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.pivno_models.defom_pivno import DEFOMStereo as BaseDEFOMStereo
from core.pivno_models.update_gru3 import IGEVStyleBasicMultiUpdateBlock
from core.submodules import (
    encode_sampled_right_features,
    sample_right_feature_pyramid,
)


class DEFOMStereo(BaseDEFOMStereo):
    """Encode every sampled match, then concatenate all scales without gating."""

    MODEL_VARIANT = "defom_pivno_gwc4_enc16_concat_gru3"
    FUSION_MODE = "gwc4_enc16_direct_concat_no_gate_no_left"
    GRU_KERNEL_SIZE = IGEVStyleBasicMultiUpdateBlock.GRU_KERNEL_SIZE
    LOW_FEATURE_DIM = 32
    MATCH_NUM_GROUPS = 4
    MATCH_ENCODED_CHANNELS = 16
    RIGHT_SAMPLE_ENCODING = "residual_gwc4_conv16_no_left_concat"
    AMP_POLICY = "fp16_compute_fp32_corr_attention"

    def __init__(self, args):
        super().__init__(args)

        low_feature_dim = int(self.low_channel.out_channels)
        if low_feature_dim != self.LOW_FEATURE_DIM:
            raise ValueError(
                f"expected {self.LOW_FEATURE_DIM} low feature channels, "
                f"got {low_feature_dim}"
            )
        if low_feature_dim % self.MATCH_NUM_GROUPS != 0:
            raise ValueError(
                f"low feature channels ({low_feature_dim}) must be divisible "
                f"by GWC groups ({self.MATCH_NUM_GROUPS})"
            )

        raw_encoded_sample_dim = (
            low_feature_dim + self.MATCH_NUM_GROUPS
        )
        self.sample_match_encoder = nn.Sequential(
            nn.Conv2d(
                raw_encoded_sample_dim,
                self.MATCH_ENCODED_CHANNELS,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(4, self.MATCH_ENCODED_CHANNELS),
            nn.GELU(),
        )

        num_scales = len(self.compression_ratios)
        num_offsets = int(self.warp_offsets.numel())
        encoded_fuse_channels = (
            num_scales * num_offsets * self.MATCH_ENCODED_CHANNELS
        )
        self.refine_right_fuse = self._make_right_fuser(
            encoded_fuse_channels
        )
        self.update_block = IGEVStyleBasicMultiUpdateBlock(
            self.args,
            hidden_dims=args.hidden_dims,
        )

    def _encode_direct_concat(self, fmap1_low, sampled_right):
        """Return all per-sample enc16 matches as one unweighted tensor."""
        if sampled_right.ndim != 6:
            raise ValueError(
                "sampled_right must be [B,S,K,C,H,W], got "
                f"{tuple(sampled_right.shape)}"
            )

        batch, num_scales, sample_count, channels, height, width = (
            sampled_right.shape
        )
        if num_scales != len(self.compression_ratios):
            raise ValueError(
                f"expected {len(self.compression_ratios)} scales, "
                f"got {num_scales}"
            )
        if tuple(fmap1_low.shape) != (
            batch,
            channels,
            height,
            width,
        ):
            raise ValueError(
                "left/sample shape mismatch: "
                f"left={tuple(fmap1_low.shape)} "
                f"sampled={tuple(sampled_right.shape)}"
            )

        encoded_right = encode_sampled_right_features(
            fmap1_low,
            sampled_right,
            num_groups=self.MATCH_NUM_GROUPS,
        )
        encoded_channels = int(encoded_right.shape[3])
        encoded_right = self.sample_match_encoder(
            encoded_right.reshape(
                batch * num_scales * sample_count,
                encoded_channels,
                height,
                width,
            )
        ).reshape(
            batch,
            num_scales,
            sample_count,
            self.MATCH_ENCODED_CHANNELS,
            height,
            width,
        )
        return encoded_right.flatten(1, 3)

    def _refine_warp_feature(
        self,
        fmap1_low,
        right_feature_pyramid,
        disp,
    ):
        sampled_right = sample_right_feature_pyramid(
            right_feature_pyramid,
            disp,
            offsets=self.warp_offsets,
            compression_ratios=self.compression_ratios,
            padding_mode="zeros",
            align_corners=True,
        )
        batch, num_samples, channels, height, width = sampled_right.shape
        num_scales = len(self.compression_ratios)
        if num_samples % num_scales != 0:
            raise RuntimeError(
                f"cannot split {num_samples} samples into "
                f"{num_scales} scales"
            )
        sampled_right = sampled_right.reshape(
            batch,
            num_scales,
            num_samples // num_scales,
            channels,
            height,
            width,
        )
        encoded_for_fusion = self._encode_direct_concat(
            fmap1_low,
            sampled_right,
        )
        fused_feature = self.refine_right_fuse(encoded_for_fusion)
        with torch.cuda.amp.autocast(enabled=False):
            return self.refine_rope(fused_feature.float())
