"""Joint raw-dilated stereo model trained from scratch except MobileNetV2."""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.pivno_models.defom_pivno_mobilenet.defom_pivno_gated_gru3_mobilenetv2_convex_bnfreeze import (
    DEFOMStereo as MobileNetV2ConvexBNFreezeStereo,
)
from core.submodules import (
    ParallelRightWidthCompressor,
    encode_sampled_right_features,
    sample_right_feature_dilated_scales,
)


class StateCompatibleRawRightAdapter(ParallelRightWidthCompressor):
    """Keep legacy compressor keys while bypassing inactive compression."""

    def forward(self, right_feat):
        if right_feat.ndim != 4:
            raise ValueError(
                f"expected right_feat [B,C,H,W], got {tuple(right_feat.shape)}"
            )
        return right_feat, right_feat


class LearnedGroupAggregation(nn.Module):
    """Candidate-independent group reliability scorer."""

    def __init__(self, num_candidates=9, hidden_dim=8):
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.scorer = nn.Sequential(
            nn.Linear(self.num_candidates, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(self, group_correlation):
        if group_correlation.ndim != 6:
            raise ValueError(
                "group_correlation must be [B,S,K,G,H,W], got "
                f"{tuple(group_correlation.shape)}"
            )
        if group_correlation.shape[2] != self.num_candidates:
            raise ValueError(
                f"expected K={self.num_candidates}, got "
                f"{group_correlation.shape[2]}"
            )
        profile = group_correlation.permute(0, 1, 3, 4, 5, 2)
        logits = self.scorer(profile).squeeze(-1)
        weights = torch.softmax(logits.float(), dim=2)
        correlation = (
            group_correlation.float() * weights[:, :, None]
        ).sum(dim=3)
        return correlation, weights


class DEFOMStereo(MobileNetV2ConvexBNFreezeStereo):
    """Train raw lookup, local aggregation and learned groups jointly."""

    MODEL_VARIANT = (
        "defom_pivno_gated_gru3_mobilenetv2_raw_dilated_joint"
    )
    SAMPLING_DILATIONS = (1, 2, 4)
    LOCAL_AGG_CHANNELS = 32
    ENCODER_CONFIG = dict(
        MobileNetV2ConvexBNFreezeStereo.ENCODER_CONFIG,
        pivno_right_sampling=(
            "raw_quarter_feature_dilated_1_2_4_same_physical_candidates"
        ),
        pivno_right_sampling_dilations=list(SAMPLING_DILATIONS),
        pivno_right_compression_active=False,
        pivno_right_local_aggregation=(
            "shared_residual_horizontal_dwconv_1x5_zero_init"
        ),
        pivno_right_local_aggregation_scales=["mid", "coarse"],
        pivno_scale_gate_group_reduction=(
            "pixel_scale_dependent_shared_mlp_k9_h8_softmax_g4"
        ),
        pivno_scale_gate_group_candidate_dependent=False,
        pivno_training_initialization="scratch_except_imagenet_mobilenetv2",
        pivno_training_scope="all_active_modules",
    )

    def __init__(self, args):
        super().__init__(args)
        if self.corr_radius != 4 or int(self.warp_offsets.numel()) != 9:
            raise ValueError("joint raw-dilated model requires corr_radius=4")
        channels = self.low_channel.out_channels
        if channels != self.LOCAL_AGG_CHANNELS:
            raise ValueError(
                f"expected matching channels={self.LOCAL_AGG_CHANNELS}, got {channels}"
            )

        compressor = self.right_width_compressor
        adapter = StateCompatibleRawRightAdapter(
            compressor.in_channels,
            compressor.out_channels,
            mode=compressor.mode,
        )
        adapter.load_state_dict(compressor.state_dict(), strict=True)
        self.right_width_compressor = adapter

        self.horizontal_dwconv = nn.Conv2d(
            channels, channels, kernel_size=(1, 5), stride=1,
            padding=(0, 2), groups=channels, bias=False,
        )
        nn.init.zeros_(self.horizontal_dwconv.weight)
        self.group_aggregator = LearnedGroupAggregation(
            num_candidates=int(self.warp_offsets.numel()), hidden_dim=8
        )
        self.last_group_aggregation_stats = None

        for parameter in self.parameters():
            parameter.requires_grad_(True)
        for parameter in self.right_width_compressor.parameters():
            parameter.requires_grad_(False)

    def _build_right_feature_pyramid(self, fmap2_low):
        local_delta = self.horizontal_dwconv(fmap2_low)
        right_local = fmap2_low + local_delta
        return fmap2_low, right_local, right_local

    def _apply_scale_gate(self, fmap1_low, group_correlation):
        batch, num_scales, _, num_groups, height, width = group_correlation.shape
        if num_scales != 3 or num_groups != self.MATCH_NUM_GROUPS:
            raise ValueError(
                f"expected S=3/G={self.MATCH_NUM_GROUPS}, got "
                f"S={num_scales}/G={num_groups}"
            )
        with torch.cuda.amp.autocast(enabled=False):
            left_unit = F.normalize(fmap1_low.float(), dim=1, eps=1e-6)
            correlation, group_weights = self.group_aggregator(
                group_correlation.float()
            )
        gate_dtype = fmap1_low.dtype
        gate_input = torch.cat(
            [left_unit.to(gate_dtype), correlation.flatten(1, 2).to(gate_dtype)],
            dim=1,
        )
        scale_logits = self.scale_gate(gate_input)
        with torch.cuda.amp.autocast(enabled=False):
            scale_weights = torch.softmax(scale_logits.float(), dim=1)
        detached = scale_weights.detach()
        self.last_scale_weight_mean = detached.mean(dim=(0, 2, 3))
        entropy = -(
            detached * detached.clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        self.last_scale_gate_entropy = entropy / math.log(num_scales)
        self.last_group_aggregation_stats = group_weights.detach()
        return scale_weights

    def _refine_warp_feature(self, fmap1_low, right_feature_pyramid, disp):
        fine = sample_right_feature_dilated_scales(
            right_feature_pyramid[0], disp, offsets=self.warp_offsets,
            dilation_rates=(1,), padding_mode="zeros", align_corners=True,
        )
        mid_coarse = sample_right_feature_dilated_scales(
            right_feature_pyramid[1], disp, offsets=self.warp_offsets,
            dilation_rates=(2, 4), padding_mode="zeros", align_corners=True,
        )
        sampled_right = torch.cat((fine, mid_coarse), dim=1)
        batch, num_samples, channels, height, width = sampled_right.shape
        num_scales = len(self.SAMPLING_DILATIONS)
        sample_count = num_samples // num_scales
        sampled_right = sampled_right.reshape(
            batch, num_scales, sample_count, channels, height, width
        )
        encoded_right = encode_sampled_right_features(
            fmap1_low, sampled_right, num_groups=self.MATCH_NUM_GROUPS
        )
        group_correlation = encoded_right[:, :, :, -self.MATCH_NUM_GROUPS:]
        scale_weights = self._apply_scale_gate(fmap1_low, group_correlation)
        encoded_channels = encoded_right.shape[3]
        encoded_right = self.sample_match_encoder(
            encoded_right.reshape(
                batch * num_scales * sample_count,
                encoded_channels, height, width,
            )
        ).reshape(
            batch, num_scales, sample_count,
            self.MATCH_ENCODED_CHANNELS, height, width,
        )
        encoded_right = encoded_right * (
            float(num_scales) * scale_weights
        ).to(encoded_right.dtype)[:, :, None, None]
        fused = self.refine_right_fuse(encoded_right.flatten(1, 3))
        with torch.cuda.amp.autocast(enabled=False):
            return self.refine_rope(fused.float())

    def log_parameter_contract(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        frozen = [
            name for name, parameter in self.named_parameters()
            if not parameter.requires_grad
        ]
        invalid = [
            name for name in frozen
            if not name.startswith("right_width_compressor.")
        ]
        if invalid:
            raise RuntimeError(f"unexpected frozen active parameters: {invalid}")
        logging.info(
            "Parameters: total=%d trainable=%d inactive_frozen=%d",
            total, trainable, total - trainable,
        )
        logging.info(
            "Joint additions: horizontal_dwconv=%d group_aggregator=%d "
            "scale_gate=%d",
            sum(p.numel() for p in self.horizontal_dwconv.parameters()),
            sum(p.numel() for p in self.group_aggregator.parameters()),
            sum(p.numel() for p in self.scale_gate.parameters()),
        )
