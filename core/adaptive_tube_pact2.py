"""PACT2-only GWC modules for the recurrent correlation input."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.adaptive_tube import (
    AdaptiveLocalCorrBlock,
    GlobalCoarseGwcAggregator,
    _check_feature_pair,
    _check_groups,
    _sample_right_at_disparities,
)
from core.galerkin import simple_cross_attn_rope_2d


Tensor = torch.Tensor


class PACT2FixedRadiusMidScaleRefiner(nn.Module):
    """One-channel 1/8 correction with a fixed radius of four quarter pixels."""

    FIXED_RADIUS_QUARTER = 4.0

    def __init__(self, max_disp: int, feature_channels: int = 128,
                 hidden_channels: int = 64, num_groups: int = 8) -> None:
        super().__init__()
        self.max_disp_quarter = float(max_disp) / 4.0
        self.corr = AdaptiveLocalCorrBlock(
            max_disp=max_disp,
            num_groups=num_groups,
            feature_channels=feature_channels,
            feature_stride=8,
            sampling_layout="legacy9",
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.corr.output_channels + feature_channels,
                hidden_channels, 3, padding=1, bias=False,
            ),
            nn.GroupNorm(math.gcd(hidden_channels, 8), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, fmap1_8: Tensor, fmap2_8: Tensor, disp_q: Tensor,
                coarse_std_q: Tensor, coarse_entropy: Tensor,
                coarse_margin: Tensor, return_aux: bool = False):
        radius_q = torch.full_like(disp_q, self.FIXED_RADIUS_QUARTER)
        corr, galerkin, corr_aux = self.corr(
            fmap1_8, fmap2_8, disp_q / 2.0, radius_q / 2.0,
            coarse_std_q / 2.0, coarse_entropy, coarse_margin,
            return_aux=True,
        )
        raw_delta = self.head(torch.cat((corr, galerkin), dim=1))
        delta_q = self.FIXED_RADIUS_QUARTER * torch.tanh(raw_delta.float())
        proposed_disp_q = (disp_q.float() + delta_q).clamp(
            0.0, self.max_disp_quarter - 1.0e-3
        )
        has_valid = corr_aux["valid"].any(dim=1, keepdim=True)
        x_limit_q = 2.0 * torch.arange(
            fmap1_8.shape[-1], device=disp_q.device, dtype=disp_q.dtype
        ).view(1, 1, 1, -1)
        fallback_q = torch.minimum(
            disp_q.float().clamp(0.0, self.max_disp_quarter - 1.0e-3),
            x_limit_q.float(),
        )
        refined_disp_q = torch.where(has_valid, proposed_disp_q, fallback_q)
        if not return_aux:
            return refined_disp_q
        return refined_disp_q, {
            "corr": corr,
            "galerkin": galerkin,
            "delta_q": delta_q,
            "has_valid": has_valid,
            **corr_aux,
        }


class PACT2GlobalCoarseGwcAggregator(GlobalCoarseGwcAggregator):
    """Coarse matcher that also exposes its already-computed 1/16 GWC volume."""

    def forward(self, fmap1: Tensor, fmap2: Tensor) -> Dict[str, Tensor]:
        gwc_volume, valid_broadcast = self._build_volume(fmap1, fmap2)
        aggregated = self.regularizer(gwc_volume)
        raw_logits = self.classifier(aggregated).squeeze(1)
        valid = valid_broadcast.expand(
            raw_logits.shape[0], 1, self.num_bins,
            raw_logits.shape[2], raw_logits.shape[3],
        ).squeeze(1)
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

        return {
            "logits": logits,
            "valid": valid,
            "coarse_disp": coarse_disp,
            "std": std,
            "entropy": entropy,
            "margin": margin,
            "gwc_volume": gwc_volume,
        }


def _groupwise_candidate_correlation(
    fmap1: Tensor,
    sampled2: Tensor,
    num_groups: int,
    eps: float,
) -> Tensor:
    """Compute only the groupwise correlation required by PACT2."""
    batch, channels, height, width = fmap1.shape
    sampled_batch, candidates, sampled_channels, sampled_height, sampled_width = (
        sampled2.shape
    )
    if (
        sampled_batch,
        sampled_channels,
        sampled_height,
        sampled_width,
    ) != (batch, channels, height, width):
        raise ValueError("left and sampled-right feature shapes do not match")
    _check_groups(channels, num_groups, "PACT2 candidate GWC")

    channels_per_group = channels // num_groups
    left = fmap1.float().reshape(
        batch, num_groups, channels_per_group, height, width
    )
    right = sampled2.float().reshape(
        batch, candidates, num_groups, channels_per_group, height, width
    )
    left = F.normalize(left, p=2, dim=2, eps=eps).unsqueeze(1)
    right = F.normalize(right, p=2, dim=3, eps=eps)
    return (left * right).sum(dim=3).float()


def _normalise_grid_coordinate(coordinate: Tensor, size: int) -> Tensor:
    if size > 1:
        return 2.0 * coordinate / float(size - 1) - 1.0
    return torch.zeros_like(coordinate)


def _sample_coarse_gwc(
    volume: Tensor,
    valid_volume: Tensor,
    candidates_quarter: Tensor,
    disparity_divisor: float = 4.0,
) -> Tensor:
    """Trilinearly sample a 1/16 GWC volume on a 1/4 candidate grid."""
    if volume.ndim != 5:
        raise ValueError("coarse GWC volume must be [B,G,D,H,W]")
    if valid_volume.ndim != 4:
        raise ValueError("coarse validity volume must be [B,D,H,W]")
    batch, groups, disparities, _, _ = volume.shape
    candidate_batch, candidates, height, width = candidates_quarter.shape
    if candidate_batch != batch or valid_volume.shape != (
        batch, disparities, volume.shape[-2], volume.shape[-1]
    ):
        raise ValueError("coarse GWC, validity and candidate shapes do not match")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=volume.device, dtype=torch.float32),
        torch.arange(width, device=volume.device, dtype=torch.float32),
        indexing="ij",
    )
    x_grid = _normalise_grid_coordinate(xx, width).view(
        1, 1, height, width
    ).expand(batch, candidates, -1, -1)
    y_grid = _normalise_grid_coordinate(yy, height).view(
        1, 1, height, width
    ).expand(batch, candidates, -1, -1)
    disparity_grid = _normalise_grid_coordinate(
        candidates_quarter.float() / float(disparity_divisor), disparities
    )
    grid = torch.stack((x_grid, y_grid, disparity_grid), dim=-1)

    sampled = F.grid_sample(
        volume.float(), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )
    sampled_valid = F.grid_sample(
        valid_volume.unsqueeze(1).float(), grid, mode="nearest",
        padding_mode="zeros", align_corners=True,
    ) > 0.5
    sampled = sampled * sampled_valid.to(sampled.dtype)
    sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()
    return sampled.reshape(batch, candidates * groups, height, width)


class PACT2DualScaleGwcBlock(nn.Module):
    """Concatenate 1/4 and 1/16 GWC at fixed offsets ``[-4, ..., 4]``."""

    FIXED_RADIUS_QUARTER = 4.0

    def __init__(
        self,
        max_disp: int,
        num_groups: int = 8,
        feature_channels: int = 64,
        feature_stride: int = 4,
        galerkin_heads: int = 4,
        match_temperature: float = 0.1,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.max_disp_feature = float(max_disp) / float(feature_stride)
        self.num_groups = int(num_groups)
        self.feature_channels = int(feature_channels)
        self.num_candidates = 9
        self.match_temperature = float(match_temperature)
        self.eps = float(eps)
        self.output_channels = 2 * self.num_candidates * self.num_groups
        self.register_buffer(
            "candidate_offsets",
            torch.arange(-4.0, 5.0, dtype=torch.float32),
        )
        self.rope_galerkin = simple_cross_attn_rope_2d(
            feature_channels, galerkin_heads
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

        match_logits = local_gwc.mean(dim=2).masked_fill(
            ~local_valid, -1.0e4
        )
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
    "PACT2FixedRadiusMidScaleRefiner",
    "PACT2GlobalCoarseGwcAggregator",
    "PACT2DualScaleGwcBlock",
]
