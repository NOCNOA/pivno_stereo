"""PACT2-GEV-only aggregation modules.

The full 1/16 GWC volume is aggregated with the matching 1/16 left feature
before it is used by either coarse regression or recurrent sampling.  The
optional ``dual`` mode applies the same idea to the nine dynamically sampled
1/4 candidates with the matching 1/4 left feature.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.adaptive_tube import (
    GlobalCoarseGwcAggregator,
    _check_feature_pair,
    _check_groups,
    _sample_right_at_disparities,
)
from core.adaptive_tube_pact2 import (
    PACT2FixedRadiusMidScaleRefiner,
    _groupwise_candidate_correlation,
    _sample_coarse_gwc,
)
from core.galerkin import simple_cross_attn_rope_2d


Tensor = torch.Tensor


class FeatureAttention(nn.Module):
    """Generate an IGEV-style channel gate from the matching left feature."""

    def __init__(self, volume_channels: int, feature_channels: int) -> None:
        super().__init__()
        hidden_channels = max(feature_channels // 2, volume_channels)
        groups = math.gcd(hidden_channels, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, volume_channels, 1),
        )

    def forward(self, volume: Tensor, left_feature: Tensor) -> Tensor:
        if volume.ndim != 5:
            raise ValueError("feature-attention volume must be [B,C,D,H,W]")
        if left_feature.shape[0] != volume.shape[0] or left_feature.shape[-2:] != volume.shape[-2:]:
            raise ValueError("feature-attention volume and left feature do not align")
        attention = torch.sigmoid(self.gate(left_feature.float())).unsqueeze(2)
        return volume.float() * attention


class SameResolutionGevBlock(nn.Module):
    """Aggregate disparity and image neighborhoods without changing resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = math.gcd(channels, 4)
        self.disparity = nn.Sequential(
            nn.Conv3d(
                channels, channels, kernel_size=(3, 1, 1),
                padding=(1, 0, 0), groups=channels, bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(
                channels, channels, kernel_size=(1, 3, 3),
                padding=(0, 1, 1), groups=channels, bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.group_mix = nn.Conv3d(channels, channels, 1, bias=False)

    def forward(self, volume: Tensor, valid: Optional[Tensor] = None) -> Tensor:
        residual = self.group_mix(self.spatial(self.disparity(volume.float())))
        output = volume.float() + residual
        if valid is not None:
            output = output * valid.to(output.dtype)
        return output


class SameResolutionGev(nn.Module):
    """Two lightweight same-resolution GEV residual blocks."""

    def __init__(self, channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SameResolutionGevBlock(channels) for _ in range(num_blocks)
        )

    def forward(self, volume: Tensor, valid: Optional[Tensor] = None) -> Tensor:
        output = volume
        if valid is not None:
            output = output * valid.to(output.dtype)
        for block in self.blocks:
            output = block(output, valid)
        return output


class PACT2GevGlobalCoarseAggregator(GlobalCoarseGwcAggregator):
    """Feature-attended 1/16 GEV used by regression, init and recurrence."""

    def __init__(
        self,
        max_disp: int,
        num_groups: int = 8,
        feature_channels: int = 192,
        **kwargs,
    ) -> None:
        super().__init__(max_disp=max_disp, num_groups=num_groups, **kwargs)
        # Do not use the inherited regularizer: this branch deliberately uses
        # two explicit same-resolution GEV blocks and no hourglass.
        del self.regularizer
        self.feature_attention = FeatureAttention(num_groups, feature_channels)
        self.gev = SameResolutionGev(num_groups, num_blocks=2)
        self.geometry_head = nn.Sequential(
            nn.Conv2d(2 * num_groups, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )

    def forward(self, fmap1: Tensor, fmap2: Tensor) -> Dict[str, Tensor]:
        raw_gwc, valid_broadcast = self._build_volume(fmap1, fmap2)
        valid_5d = valid_broadcast.expand(
            raw_gwc.shape[0], 1, self.num_bins,
            raw_gwc.shape[-2], raw_gwc.shape[-1],
        )
        attended = self.feature_attention(raw_gwc, fmap1)
        coarse_gev = self.gev(attended, valid_5d)

        raw_logits = self.classifier(coarse_gev).squeeze(1)
        valid = valid_5d.squeeze(1)
        logits = raw_logits.float().masked_fill(~valid, self.invalid_logit)

        posterior = torch.softmax(logits / self.temperature, dim=1)
        posterior = posterior * valid.float()
        posterior = posterior / posterior.sum(
            dim=1, keepdim=True
        ).clamp_min(self.eps)

        peak1 = logits.argmax(dim=1, keepdim=True)
        bins = torch.arange(
            self.num_bins, device=logits.device
        ).view(1, -1, 1, 1)
        suppressed = (bins - peak1).abs() <= self.nms_radius
        second_valid = valid & ~suppressed
        has_second = second_valid.any(dim=1, keepdim=True)
        peak2 = logits.masked_fill(
            ~second_valid, self.invalid_logit
        ).argmax(dim=1, keepdim=True)
        peak2 = torch.where(has_second, peak2, peak1)
        peaks = torch.cat((peak1, peak2), dim=1)

        refined_bins = self._local_softargmax(logits, valid, peaks)
        coarse_disp = refined_bins[:, 0:1] * self.bin_to_output
        peak_scores = torch.gather(posterior, 1, peaks)
        peak_scores = torch.cat(
            (
                peak_scores[:, 0:1],
                peak_scores[:, 1:2] * has_second.float(),
            ),
            dim=1,
        )

        disparity_bins = bins.float()
        mean_bins = (posterior * disparity_bins).sum(dim=1, keepdim=True)
        variance_bins = (
            posterior * (disparity_bins - mean_bins).square()
        ).sum(dim=1, keepdim=True)
        std = variance_bins.clamp_min(self.eps).sqrt() * self.bin_to_output
        entropy_raw = -(
            posterior * posterior.clamp_min(self.eps).log()
        ).sum(dim=1, keepdim=True)
        valid_count = valid.sum(dim=1, keepdim=True).float()
        entropy = torch.where(
            valid_count > 1.0,
            entropy_raw / valid_count.clamp_min(2.0).log(),
            torch.zeros_like(entropy_raw),
        ).clamp(0.0, 1.0)
        margin = (
            peak_scores[:, 0:1] - peak_scores[:, 1:2]
        ).clamp(0.0, 1.0)

        posterior_geometry = (
            coarse_gev * posterior.unsqueeze(1).to(coarse_gev.dtype)
        ).sum(dim=2)
        peak_index = peak1.unsqueeze(1).expand(
            -1, self.num_groups, -1, -1, -1
        )
        peak_geometry = torch.gather(
            coarse_gev, dim=2, index=peak_index
        ).squeeze(2)
        coarse_geometry_2d = self.geometry_head(
            torch.cat((posterior_geometry, peak_geometry), dim=1)
        )

        return {
            "logits": logits,
            "posterior": posterior,
            "valid": valid,
            "coarse_disp": coarse_disp,
            "std": std,
            "entropy": entropy,
            "margin": margin,
            "coarse_geometry_2d": coarse_geometry_2d,
            # This is intentionally the aggregated volume. Recurrent sampling
            # must consume the same geometry used for coarse classification.
            "gwc_volume": coarse_gev,
        }


class PACT2GevDualScaleGwcBlock(nn.Module):
    """Fixed-radius dual-scale GWC with optional 1/4 candidate GEV."""

    FIXED_RADIUS_QUARTER = 4.0
    MODES = ("coarse", "dual")

    def __init__(
        self,
        max_disp: int,
        gev_mode: str = "dual",
        num_groups: int = 8,
        feature_channels: int = 64,
        feature_stride: int = 4,
        galerkin_heads: int = 4,
        match_temperature: float = 0.1,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if gev_mode not in self.MODES:
            raise ValueError(
                f"pact_gev_mode must be one of {self.MODES}, got {gev_mode!r}"
            )
        self.gev_mode = gev_mode
        self.max_disp_feature = float(max_disp) / float(feature_stride)
        self.num_groups = int(num_groups)
        self.feature_channels = int(feature_channels)
        self.num_candidates = 9
        self.match_temperature = float(match_temperature)
        self.eps = float(eps)
        self.output_channels = 2 * self.num_candidates * self.num_groups
        self.register_buffer(
            "candidate_offsets", torch.arange(-4.0, 5.0, dtype=torch.float32)
        )
        self.rope_galerkin = simple_cross_attn_rope_2d(
            feature_channels, galerkin_heads
        )
        if self.gev_mode == "dual":
            self.feature_attention = FeatureAttention(
                num_groups, feature_channels
            )
            self.local_gev = SameResolutionGev(num_groups, num_blocks=2)
            self.candidate_classifier = nn.Conv3d(
                num_groups, 1, kernel_size=1
            )

    def forward(
        self,
        fmap1_4: Tensor,
        fmap2_4: Tensor,
        disp: Tensor,
        coarse_gwc: Tensor,
        coarse_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch, channels, height, width = _check_feature_pair(
            fmap1_4, fmap2_4, self.__class__.__name__
        )
        if channels != self.feature_channels:
            raise ValueError(
                f"expected {self.feature_channels} feature channels, got {channels}"
            )
        if coarse_gwc.shape[:2] != (batch, self.num_groups):
            raise ValueError("coarse GWC group count does not match local GWC")
        _check_groups(channels, self.num_groups, self.__class__.__name__)

        offsets = self.candidate_offsets.to(
            device=disp.device, dtype=disp.dtype
        ).view(1, self.num_candidates, 1, 1)
        candidates = disp + offsets
        sampled_right, local_valid = _sample_right_at_disparities(
            fmap2_4, candidates
        )
        local_valid = local_valid & (candidates < self.max_disp_feature)

        local_gwc = _groupwise_candidate_correlation(
            fmap1_4, sampled_right, self.num_groups, self.eps
        )
        local_valid_5d = local_valid.unsqueeze(1)
        local_volume = local_gwc.permute(0, 2, 1, 3, 4).contiguous()
        if self.gev_mode == "dual":
            local_volume = self.feature_attention(local_volume, fmap1_4)
            local_volume = self.local_gev(local_volume, local_valid_5d)
            match_logits = self.candidate_classifier(
                local_volume
            ).squeeze(1)
        else:
            local_volume = local_volume * local_valid_5d.to(local_volume.dtype)
            match_logits = local_volume.mean(dim=1)

        # Restore candidate-major layout to retain the existing 72-channel
        # recurrent-correlation contract.
        local_gwc = local_volume.permute(0, 2, 1, 3, 4).contiguous()
        local_gwc = local_gwc * local_valid.unsqueeze(2).to(local_gwc.dtype)
        local_channels = local_gwc.reshape(
            batch, self.num_candidates * self.num_groups, height, width
        )

        coarse_channels = _sample_coarse_gwc(
            coarse_gwc, coarse_valid, candidates
        )
        coarse_channels = coarse_channels * local_valid.repeat_interleave(
            self.num_groups, dim=1
        ).to(coarse_channels.dtype)

        match_logits = match_logits.float().masked_fill(~local_valid, -1.0e4)
        match_weights = torch.softmax(
            match_logits / self.match_temperature, dim=1
        ) * local_valid.float()
        match_weights = match_weights / match_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(self.eps)
        aligned_right = (
            match_weights.unsqueeze(2).to(sampled_right.dtype) * sampled_right
        ).sum(dim=1)
        aligned_right = aligned_right * local_valid.any(
            dim=1, keepdim=True
        ).to(aligned_right.dtype)
        galerkin_feat = self.rope_galerkin(
            fmap1_4.float(), aligned_right.float()
        )

        corr = torch.cat(
            (local_channels.float(), coarse_channels.float()), dim=1
        ).contiguous()
        return corr, galerkin_feat


__all__ = [
    "FeatureAttention",
    "SameResolutionGevBlock",
    "SameResolutionGev",
    "PACT2FixedRadiusMidScaleRefiner",
    "PACT2GevGlobalCoarseAggregator",
    "PACT2GevDualScaleGwcBlock",
]
