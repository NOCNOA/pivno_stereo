"""Memory-bounded stereo cost representations for large disparity ranges.

The modules in this file deliberately avoid a dense cost volume at 1/4
resolution.  Disparities passed between modules use *quarter-resolution feature
pixels* (``full_resolution_disparity / 4``), which is also the convention used
by the recurrent update in DEFOM-Stereo.

The active anchor-free PACT path uses:

``GlobalCoarseGwcAggregator``
    Builds and lightly regularizes a small, full-range 1/16 GWC volume.

``AdaptiveMidScaleRefiner``
    Samples one nine-point neighborhood at 1/8 resolution and corrects the
    single coarse disparity state without introducing anchors.

``AdaptiveLocalCorrBlock``
    Computes nine correlations and a matching-weighted Cross-RoPE feature at
    1/8 or 1/4 resolution without a disparity-sized persistent allocation.

The older tube classes remain in this file only for checkpoint compatibility.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.galerkin import simple_cross_attn_rope_2d


Tensor = torch.Tensor


def confidence_to_radius(
    confidence: Tensor,
    min_radius: float = 1.0,
    max_radius: float = 8.0,
) -> Tensor:
    """Map high confidence to a small search radius, and vice versa."""
    if min_radius <= 0 or max_radius < min_radius:
        raise ValueError("invalid confidence-to-radius bounds")
    confidence = confidence.float().clamp(0.0, 1.0)
    return min_radius + (1.0 - confidence) * (max_radius - min_radius)


def _check_feature_pair(fmap1: Tensor, fmap2: Tensor, name: str) -> Tuple[int, int, int, int]:
    if fmap1.ndim != 4 or fmap2.ndim != 4:
        raise ValueError(
            f"{name} expects fmap1/fmap2 as [B,C,H,W], got "
            f"{tuple(fmap1.shape)} and {tuple(fmap2.shape)}"
        )
    if fmap1.shape != fmap2.shape:
        raise ValueError(
            f"{name} requires equal feature shapes, got "
            f"{tuple(fmap1.shape)} and {tuple(fmap2.shape)}"
        )
    if fmap1.device != fmap2.device:
        raise ValueError(f"{name} requires both features on the same device")
    return tuple(fmap1.shape)  # type: ignore[return-value]


def _check_groups(channels: int, num_groups: int, name: str) -> None:
    if num_groups <= 0 or channels % num_groups != 0:
        raise ValueError(
            f"{name}: feature channels ({channels}) must be divisible by "
            f"num_groups ({num_groups})"
        )


def _resize_field(field: Tensor, size: Tuple[int, int]) -> Tensor:
    if field.shape[-2:] == size:
        return field.float()
    return F.interpolate(field.float(), size=size, mode="bilinear", align_corners=True)


def _normalise_pixel_coordinate(coord: Tensor, size: int) -> Tensor:
    if size > 1:
        return 2.0 * coord / float(size - 1) - 1.0
    return torch.zeros_like(coord)


def _sample_right_at_disparities(
    fmap2: Tensor,
    disparities: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Sample a right feature map at ``x_right = x_left - disparity``.

    Args:
        fmap2: Right feature map ``[B,C,H,W]``.
        disparities: Candidate disparities in fmap2 pixels, ``[B,K,H,W]``.

    Returns:
        sampled: ``[B,K,C,H,W]``. Out-of-bounds values are zero.
        valid: ``[B,K,H,W]``. Negative disparities are considered invalid.

    A single grid_sample is used for all K hypotheses.  K is folded into the
    output width rather than the input batch, so the source feature map is not
    replicated K times.
    """
    if fmap2.ndim != 4:
        raise ValueError(f"fmap2 must be [B,C,H,W], got {tuple(fmap2.shape)}")
    if disparities.ndim != 4:
        raise ValueError(
            f"disparities must be [B,K,H,W], got {tuple(disparities.shape)}"
        )
    b, c, h, w = fmap2.shape
    bd, k, hd, wd = disparities.shape
    if (bd, hd, wd) != (b, h, w):
        raise ValueError(
            "feature/candidate shape mismatch: "
            f"fmap2={tuple(fmap2.shape)}, disparities={tuple(disparities.shape)}"
        )
    if fmap2.device != disparities.device:
        raise ValueError("fmap2 and disparities must be on the same device")

    # CPU grid_sample does not implement fp16/bfloat16.  CUDA can retain the
    # feature dtype, avoiding an unnecessary fp32 candidate tensor under AMP.
    if fmap2.device.type == "cpu" and fmap2.dtype in (torch.float16, torch.bfloat16):
        source = fmap2.float()
    else:
        source = fmap2
    grid_dtype = source.dtype

    disp = disparities.to(dtype=grid_dtype)
    xx = torch.arange(w, device=fmap2.device, dtype=grid_dtype).view(1, 1, 1, w)
    yy = torch.arange(h, device=fmap2.device, dtype=grid_dtype).view(1, 1, h, 1)
    right_x = xx - disp
    valid = (disparities >= 0.0) & (right_x.float() >= 0.0) & (right_x.float() <= float(w - 1))

    x_norm = _normalise_pixel_coordinate(right_x, w)
    y_norm = _normalise_pixel_coordinate(yy, h).expand(b, k, h, w)

    # [B,K,H,W] -> [B,H,K*W], preserving a simple [K,W] reshape.
    x_grid = x_norm.permute(0, 2, 1, 3).reshape(b, h, k * w)
    y_grid = y_norm.permute(0, 2, 1, 3).reshape(b, h, k * w)
    grid = torch.stack((x_grid, y_grid), dim=-1)
    sampled = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.reshape(b, c, h, k, w).permute(0, 3, 1, 2, 4).contiguous()
    return sampled, valid


def _candidate_correlations(
    fmap1: Tensor,
    sampled2: Tensor,
    num_groups: int,
    eps: float,
) -> Tuple[Tensor, Tensor]:
    """Return groupwise and whole-feature cosine correlations."""
    if sampled2.ndim != 5:
        raise ValueError(f"sampled2 must be [B,K,C,H,W], got {tuple(sampled2.shape)}")
    b, c, h, w = fmap1.shape
    bs, k, cs, hs, ws = sampled2.shape
    if (bs, cs, hs, ws) != (b, c, h, w):
        raise ValueError(
            f"correlation shape mismatch: fmap1={tuple(fmap1.shape)}, "
            f"sampled2={tuple(sampled2.shape)}"
        )
    _check_groups(c, num_groups, "candidate correlation")
    channels_per_group = c // num_groups

    # FP16 normalization with eps=1e-6 has a backward factor near 1e6 for
    # zero/OOB samples.  With several candidates that overflows before
    # GradScaler can unscale the gradient.  Training callers therefore must
    # create their feature maps in FP32 (not cast them after an FP16 encoder).
    if torch.is_grad_enabled() and (
        fmap1.dtype != torch.float32 or sampled2.dtype != torch.float32
    ):
        raise RuntimeError(
            "candidate correlation training requires FP32 feature maps; "
            f"got fmap1={fmap1.dtype}, sampled2={sampled2.dtype}"
        )
    left_g = fmap1.float().reshape(
        b, num_groups, channels_per_group, h, w
    )
    right_g = sampled2.float().reshape(
        b, k, num_groups, channels_per_group, h, w
    )
    left_g = F.normalize(left_g, p=2, dim=2, eps=eps).unsqueeze(1)
    right_g = F.normalize(right_g, p=2, dim=3, eps=eps)
    gwc = (left_g * right_g).sum(dim=3).float()

    left_dot = F.normalize(fmap1.float(), p=2, dim=1, eps=eps).unsqueeze(1)
    right_dot = F.normalize(sampled2.float(), p=2, dim=2, eps=eps)
    dot = (left_dot * right_dot).sum(dim=2).float()
    return gwc, dot


def _safe_mask(mask: Tensor, candidates: Tensor, x_unit_scale: float = 1.0) -> Tensor:
    """Ensure that every pixel has one finite fallback candidate.

    The returned mask is used only for numerically safe regression.  Callers
    still return the original validity mask so losses can ignore stereo-OOB
    samples.  The fallback is the candidate closest to the feasible interval
    ``[0, x_left]``.
    """
    if mask.shape != candidates.shape:
        raise ValueError("mask and candidates must have the same shape")
    if x_unit_scale <= 0:
        raise ValueError(f"x_unit_scale must be positive, got {x_unit_scale}")
    has_valid = mask.any(dim=1, keepdim=True)
    if bool(has_valid.all()):
        return mask

    w = candidates.shape[-1]
    xx = (
        torch.arange(w, device=candidates.device, dtype=candidates.dtype)
        .mul(float(x_unit_scale))
        .view(1, 1, 1, w)
    )
    violation = F.relu(-candidates) + F.relu(candidates - xx)
    fallback_index = violation.argmin(dim=1, keepdim=True)
    fallback = torch.zeros_like(mask).scatter_(1, fallback_index, True)
    return mask | (fallback & ~has_valid)


class _Separable3DBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        groups = math.gcd(hidden_channels, 4)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class GlobalCoarseGwcAggregator(nn.Module):
    """
    输出
  - coarse_disp：主匹配峰附近回归的唯一粗视差，单位是 1/4 像素；
  - std：视差概率分布的标准差；
  - entropy：整体匹配不确定性；
  - margin：第一峰和第二峰的置信度差；
  - logits：48 个粗视差分类分数；
  - posterior：48 个视差档位的概率分布。
    """

    def __init__(
        self,
        max_disp: int,
        num_groups: int = 8,
        hidden_channels: int = 16,
        feature_scale: int = 16,
        output_disp_scale: int = 4,
        nms_radius: int = 2,
        refine_radius: int = 1,
        temperature: float = 1.0,
        invalid_logit: float = -1.0e4,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if max_disp <= 0:
            raise ValueError(f"max_disp must be positive, got {max_disp}")
        if feature_scale <= 0 or output_disp_scale <= 0:
            raise ValueError("feature_scale and output_disp_scale must be positive")
        if feature_scale % output_disp_scale != 0:
            raise ValueError(
                "feature_scale must be divisible by output_disp_scale for an exact unit conversion"
            )
        if nms_radius < 0 or refine_radius < 0:
            raise ValueError("NMS/local refinement radii must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.max_disp = int(max_disp)
        self.num_groups = int(num_groups)
        self.feature_scale = int(feature_scale)
        self.output_disp_scale = int(output_disp_scale)
        self.bin_to_output = float(feature_scale) / float(output_disp_scale)
        self.num_bins = int(math.ceil(float(max_disp) / float(feature_scale)))
        self.nms_radius = int(nms_radius)
        self.refine_radius = int(refine_radius)
        self.temperature = float(temperature)
        self.invalid_logit = float(invalid_logit)
        self.eps = float(eps)

        self.regularizer = _Separable3DBlock(num_groups, hidden_channels)
        self.classifier = nn.Conv3d(num_groups, 1, kernel_size=3, padding=1)

    def _build_volume(self, fmap1: Tensor, fmap2: Tensor) -> Tuple[Tensor, Tensor]:
        b, c, h, w = _check_feature_pair(fmap1, fmap2, self.__class__.__name__)
        _check_groups(c, self.num_groups, self.__class__.__name__)
        cpg = c // self.num_groups
        left = fmap1.reshape(b, self.num_groups, cpg, h, w)
        right = fmap2.reshape(b, self.num_groups, cpg, h, w)
        left = F.normalize(left, p=2, dim=2, eps=self.eps)
        right = F.normalize(right, p=2, dim=2, eps=self.eps)

        costs = []
        valid = torch.zeros(
            1, 1, self.num_bins, 1, w, device=fmap1.device, dtype=torch.bool
        )
        for disparity_bin in range(self.num_bins):
            if disparity_bin == 0:
                cost = (left * right).sum(dim=2)
                valid[:, :, disparity_bin] = True
            elif disparity_bin < w:
                overlap = (left[..., disparity_bin:] * right[..., :-disparity_bin]).sum(dim=2)
                cost = overlap.new_zeros((b, self.num_groups, h, w))
                cost[..., disparity_bin:] = overlap
                valid[:, :, disparity_bin, :, disparity_bin:] = True
            else:
                cost = left.new_zeros((b, self.num_groups, h, w))
            costs.append(cost)
        return torch.stack(costs, dim=2).contiguous(), valid

    def _local_softargmax(
        self,
        logits: Tensor,
        valid: Tensor,
        peaks: Tensor,
    ) -> Tensor:
        b, d, h, w = logits.shape
        mode_count = peaks.shape[1]
        offsets = torch.arange(
            -self.refine_radius,
            self.refine_radius + 1,
            device=logits.device,
        ).view(1, 1, -1, 1, 1)
        local_bins = peaks.unsqueeze(2) + offsets
        local_in_range = (local_bins >= 0) & (local_bins < d)
        gather_idx = local_bins.clamp(0, d - 1).long()
        expanded_logits = logits.unsqueeze(1).expand(b, mode_count, d, h, w)
        expanded_valid = valid.unsqueeze(1).expand(b, mode_count, d, h, w)
        local_logits = torch.gather(expanded_logits, 2, gather_idx)
        local_valid = local_in_range & torch.gather(expanded_valid, 2, gather_idx)
        local_logits = local_logits.masked_fill(~local_valid, self.invalid_logit)
        probability = torch.softmax(local_logits.float() / self.temperature, dim=2)
        probability = probability * local_valid.float()
        probability = probability / probability.sum(dim=2, keepdim=True).clamp_min(self.eps)
        return (probability * local_bins.float()).sum(dim=2)

    def forward(self, fmap1: Tensor, fmap2: Tensor) -> Dict[str, Tensor]:
        volume, valid_broadcast = self._build_volume(fmap1, fmap2)
        aggregated = self.regularizer(volume)
        raw_logits = self.classifier(aggregated).squeeze(1)
        valid = valid_broadcast.expand(
            raw_logits.shape[0], 1, self.num_bins, raw_logits.shape[2], raw_logits.shape[3]
        ).squeeze(1)
        logits = raw_logits.float().masked_fill(~valid, self.invalid_logit)

        posterior = torch.softmax(logits / self.temperature, dim=1)
        posterior = posterior * valid.float()
        posterior = posterior / posterior.sum(dim=1, keepdim=True).clamp_min(self.eps)

        peak1 = logits.argmax(dim=1, keepdim=True)
        bins = torch.arange(self.num_bins, device=logits.device).view(1, -1, 1, 1)
        suppressed = (bins - peak1).abs() <= self.nms_radius
        second_valid = valid & ~suppressed
        has_second = second_valid.any(dim=1, keepdim=True)
        second_logits = logits.masked_fill(~second_valid, self.invalid_logit)
        peak2 = second_logits.argmax(dim=1, keepdim=True)
        peak2 = torch.where(has_second, peak2, peak1)
        peaks = torch.cat((peak1, peak2), dim=1)

        refined_bins = self._local_softargmax(logits, valid, peaks)
        coarse_disp = refined_bins[:, 0:1] * self.bin_to_output
        peak_scores = torch.gather(posterior, 1, peaks)
        peak_scores = torch.cat(
            (peak_scores[:, 0:1], peak_scores[:, 1:2] * has_second.float()), dim=1
        )
        anchor_valid = torch.cat((torch.ones_like(has_second), has_second), dim=1)

        disparity_bins = bins.float()
        mean_bins = (posterior * disparity_bins).sum(dim=1, keepdim=True)
        variance_bins = (
            posterior * (disparity_bins - mean_bins).square()
        ).sum(dim=1, keepdim=True)
        # sqrt has an infinite derivative at exactly zero.  Boundary pixels
        # often have a single valid disparity, so keep the uncertainty finite
        # in both the forward and backward passes.
        std = variance_bins.clamp_min(self.eps).sqrt() * self.bin_to_output

        entropy_raw = -(posterior * posterior.clamp_min(self.eps).log()).sum(dim=1, keepdim=True)
        valid_count = valid.sum(dim=1, keepdim=True).float()
        entropy_norm = torch.where(
            valid_count > 1.0,
            entropy_raw / valid_count.clamp_min(2.0).log(),
            torch.zeros_like(entropy_raw),
        ).clamp(0.0, 1.0)
        margin = (peak_scores[:, 0:1] - peak_scores[:, 1:2]).clamp(0.0, 1.0)

        return {
            "logits": logits,
            "raw_logits": raw_logits,
            "valid": valid,
            "posterior": posterior,
            "coarse_disp": coarse_disp,
            "primary_score": peak_scores[:, 0:1],
            "peak_bin": peaks[:, 0:1],
            "refined_peak_bin": refined_bins[:, 0:1],
            # Preserve both separated coarse modes for diagnostics.  Values
            # use the same quarter-disparity unit as ``coarse_disp``; callers
            # that expose full-resolution disparities multiply them by four.
            "refined_peak_bins": refined_bins * self.bin_to_output,
            "peak_scores": peak_scores,
            "peak_valid": anchor_valid,
            "posterior_mean_bins": mean_bins,
            "std": std,
            "entropy": entropy_norm,
            "margin": margin,
        }


class _TubeAggregator(nn.Module):
    """Shared, factorised aggregator for a K-sample disparity tube."""

    def __init__(self, num_groups: int, geometry_channels: int) -> None:
        super().__init__()
        groups = math.gcd(geometry_channels, 4)
        self.spatial = nn.Sequential(
            nn.Conv3d(
                num_groups,
                geometry_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(groups, geometry_channels),
            nn.ReLU(inplace=True),
        )
        self.disparity = nn.Sequential(
            nn.Conv3d(
                geometry_channels,
                geometry_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                groups=geometry_channels,
                bias=False,
            ),
            nn.Conv3d(geometry_channels, geometry_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, geometry_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv3d(geometry_channels, 1, kernel_size=1)

    def forward(self, tube: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.spatial(tube)
        features = features + self.disparity(features)
        logits = self.classifier(features).squeeze(1)
        return features, logits


class AdaptiveTubeRefiner(nn.Module):
    """Refine two coarse modes with fixed-memory 2 x 9 tubes at 1/8 scale.

    Input ``coarse_anchors`` and ``coarse_std`` are in quarter units, even
    though their spatial resolution is normally 1/16.  Candidate sampling
    converts them to 1/8 feature pixels internally.  Returned disparities and
    candidates are again in quarter units.

    AdaptiveTubeRefiner 使用 coarse anchor 和不确定性，在 1/8 分辨率建立两条固定长度的自适应候选 Tube，将低精度全局视差精修成 PRU 可用
    的初始视差、双 anchor、几何信息和搜索半径。
    """

    DEFAULT_OFFSETS: Tuple[float, ...] = (
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )

    def __init__(
        self,
        num_groups: int = 8,
        geometry_channels: int = 16,
        max_disp: Optional[int] = None,
        min_radius_8: float = 2.0,
        max_radius_8: float = 16.0,
        uncertainty_scale: float = 1.0,
        offsets: Sequence[float] = DEFAULT_OFFSETS,
        temperature: float = 1.0,
        prior_strength: float = 1.0,
        invalid_logit: float = -1.0e4,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if len(offsets) != 9:
            raise ValueError(f"AdaptiveTubeRefiner requires exactly 9 offsets, got {len(offsets)}")
        if min_radius_8 <= 0 or max_radius_8 < min_radius_8:
            raise ValueError("invalid 1/8 tube radius bounds")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_disp is not None and max_disp <= 0:
            raise ValueError(f"max_disp must be positive, got {max_disp}")
        self.num_groups = int(num_groups)
        self.geometry_channels = int(geometry_channels)
        self.max_disp_quarter = (
            None if max_disp is None else float(max_disp) / 4.0
        )
        self.min_radius_8 = float(min_radius_8)
        self.max_radius_8 = float(max_radius_8)
        self.uncertainty_scale = float(uncertainty_scale)
        self.temperature = float(temperature)
        self.prior_strength = float(prior_strength)
        self.invalid_logit = float(invalid_logit)
        self.eps = float(eps)
        self.register_buffer("offsets", torch.tensor(tuple(offsets), dtype=torch.float32))
        self.aggregator = _TubeAggregator(num_groups, geometry_channels)

    def forward(
        self,
        fmap1_8: Tensor,
        fmap2_8: Tensor,
        coarse_anchors: Tensor,
        coarse_std: Tensor,
        coarse_scores: Optional[Tensor] = None,
        coarse_anchor_valid: Optional[Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Tensor]:
        b, c, h, w = _check_feature_pair(
            fmap1_8, fmap2_8, self.__class__.__name__
        )
        _check_groups(c, self.num_groups, self.__class__.__name__)
        if coarse_anchors.ndim != 4 or coarse_anchors.shape[0] != b or coarse_anchors.shape[1] != 2:
            raise ValueError(
                "coarse_anchors must be [B,2,Hc,Wc] in quarter units, got "
                f"{tuple(coarse_anchors.shape)}"
            )
        if coarse_std.ndim != 4 or coarse_std.shape[0] != b or coarse_std.shape[1] not in (1, 2):
            raise ValueError(
                "coarse_std must be [B,1|2,Hc,Wc] in quarter units, got "
                f"{tuple(coarse_std.shape)}"
            )
        if coarse_anchors.device != fmap1_8.device or coarse_std.device != fmap1_8.device:
            raise ValueError("coarse outputs and 1/8 features must be on the same device")

        anchors_q_8 = _resize_field(coarse_anchors, (h, w))
        std_q_8 = _resize_field(coarse_std, (h, w))
        if std_q_8.shape[1] == 1:
            std_q_8 = std_q_8.expand(-1, 2, -1, -1)
        if coarse_anchor_valid is None:
            coarse_mode_valid_8 = torch.ones(
                b, 2, h, w, device=fmap1_8.device, dtype=torch.bool
            )
        else:
            if coarse_anchor_valid.ndim != 4 or coarse_anchor_valid.shape[:2] != (b, 2):
                raise ValueError(
                    "coarse_anchor_valid must be [B,2,Hc,Wc], got "
                    f"{tuple(coarse_anchor_valid.shape)}"
                )
            if coarse_anchor_valid.device != fmap1_8.device:
                raise ValueError("coarse_anchor_valid must be on the feature device")
            coarse_mode_valid_8 = F.interpolate(
                coarse_anchor_valid.float(), size=(h, w), mode="nearest"
            ).bool()

        # quarter disparity unit / 2 == one 1/8 feature-pixel disparity.
        anchors_8 = anchors_q_8 / 2.0
        radius_8 = (std_q_8 / 2.0 * self.uncertainty_scale).clamp(
            self.min_radius_8, self.max_radius_8
        )
        offsets = self.offsets.to(device=fmap1_8.device).view(1, 1, 9, 1, 1)
        candidates_8 = anchors_8.unsqueeze(2) + radius_8.unsqueeze(2) * offsets
        candidates_q = candidates_8 * 2.0

        sampled, valid_flat = _sample_right_at_disparities(
            fmap2_8, candidates_8.reshape(b, 18, h, w)
        )
        valid_flat = valid_flat.reshape(b, 2, 9, h, w)
        valid_flat = valid_flat & coarse_mode_valid_8.unsqueeze(2)
        if self.max_disp_quarter is not None:
            valid_flat = valid_flat & (
                candidates_q < self.max_disp_quarter
            )
        valid_flat = valid_flat.reshape(b, 18, h, w)
        gwc_flat, _ = _candidate_correlations(
            fmap1_8, sampled, self.num_groups, self.eps
        )
        gwc_flat = gwc_flat * valid_flat.unsqueeze(2).to(gwc_flat.dtype)
        gwc = gwc_flat.reshape(b, 2, 9, self.num_groups, h, w).permute(
            0, 1, 3, 2, 4, 5
        ).contiguous()

        tube_features, raw_logits = self.aggregator(
            gwc.reshape(b * 2, self.num_groups, 9, h, w)
        )
        tube_features = tube_features.reshape(
            b, 2, self.geometry_channels, 9, h, w
        )
        raw_logits = raw_logits.reshape(b, 2, 9, h, w).float()
        valid = valid_flat.reshape(b, 2, 9, h, w)

        if coarse_scores is None:
            prior = raw_logits.new_full((b, 2, h, w), 0.5)
        else:
            if coarse_scores.ndim != 4 or coarse_scores.shape[:2] != (b, 2):
                raise ValueError(
                    f"coarse_scores must be [B,2,Hc,Wc], got {tuple(coarse_scores.shape)}"
                )
            prior = _resize_field(coarse_scores, (h, w)).clamp_min(self.eps)
            prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(self.eps)
        logits_with_prior = raw_logits + self.prior_strength * prior.log().unsqueeze(2)

        flat_logits = logits_with_prior.reshape(b, 18, h, w)
        flat_valid = valid.reshape(b, 18, h, w)
        flat_candidates_q = candidates_q.reshape(b, 18, h, w)
        safe_valid = _safe_mask(flat_valid, flat_candidates_q, x_unit_scale=2.0)
        masked_logits = flat_logits.masked_fill(~safe_valid, self.invalid_logit)
        probability = torch.softmax(masked_logits.float() / self.temperature, dim=1)
        probability = probability * safe_valid.float()
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(self.eps)
        init_disp_8 = (probability * flat_candidates_q.float()).sum(dim=1, keepdim=True)

        # Per-mode estimates preserve two alternatives for PRU global rescue.
        # An invalid second coarse mode remains fully masked; it is not revived
        # merely to make softmax finite.
        refined_anchor_valid_8 = valid.any(dim=2)
        mode_logits = logits_with_prior.masked_fill(~valid, self.invalid_logit)
        mode_probability = torch.softmax(mode_logits.float() / self.temperature, dim=2)
        mode_probability = mode_probability * valid.float()
        mode_probability = mode_probability / mode_probability.sum(dim=2, keepdim=True).clamp_min(self.eps)
        refined_from_tube = (
            mode_probability * candidates_q.float()
        ).sum(dim=2)
        refined_anchors_8 = torch.where(
            refined_anchor_valid_8, refined_from_tube, anchors_q_8
        )

        global_probability = probability.reshape(b, 2, 9, h, w)
        anchor_scores_8 = global_probability.sum(dim=2)
        geometry_8 = (
            tube_features.float() * global_probability.unsqueeze(2)
        ).sum(dim=(1, 3))

        variance_q = (
            probability
            * (flat_candidates_q.float() - init_disp_8).square()
        ).sum(dim=1, keepdim=True)
        initial_radius_8 = variance_q.clamp_min(self.eps).sqrt().clamp(1.0, 16.0)

        entropy_raw = -(
            probability * probability.clamp_min(self.eps).log()
        ).sum(dim=1, keepdim=True)
        valid_count = flat_valid.sum(dim=1, keepdim=True).float()
        entropy_8 = torch.where(
            valid_count > 1.0,
            entropy_raw / valid_count.clamp_min(2.0).log(),
            torch.zeros_like(entropy_raw),
        ).clamp(0.0, 1.0)
        top_prob = probability.topk(k=2, dim=1).values
        margin_8 = (top_prob[:, 0:1] - top_prob[:, 1:2]).clamp(0.0, 1.0)

        if output_size is None:
            output_size = (2 * h, 2 * w)
        if len(output_size) != 2 or output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError(f"invalid output_size {output_size}")

        return {
            # PRU-facing tensors at 1/4 spatial resolution, all disparity-like
            # values in quarter units.
            "init_disp": _resize_field(init_disp_8, output_size),
            "anchors": _resize_field(refined_anchors_8, output_size),
            "anchor_scores": _resize_field(anchor_scores_8, output_size),
            "anchor_valid": F.interpolate(
                refined_anchor_valid_8.float(), size=output_size, mode="nearest"
            ).bool(),
            "geometry": _resize_field(geometry_8, output_size),
            "initial_radius": _resize_field(initial_radius_8, output_size),
            "entropy": _resize_field(entropy_8, output_size),
            "margin": _resize_field(margin_8, output_size),
            # Loss/diagnostic tensors remain at the native 1/8 spatial scale.
            "tube_logits": flat_logits.masked_fill(
                ~flat_valid, self.invalid_logit
            ).reshape(b, 2, 9, h, w),
            "tube_regression_logits": masked_logits.reshape(b, 2, 9, h, w),
            "tube_raw_logits": raw_logits,
            "tube_candidates": candidates_q,
            "tube_candidates_full": candidates_q * 4.0,
            "tube_valid": valid,
            "tube_probability": global_probability,
            "tube_radius_8": radius_8,
            "init_disp_8": init_disp_8,
            "refined_anchors_8": refined_anchors_8,
            "anchor_valid_8": refined_anchor_valid_8,
        }


class AdaptiveTubeCorrBlock(nn.Module):
    """On-demand nine-candidate correlation for PRU (exactly 119 channels).

    Candidate layout is seven local hypotheses around ``disp`` followed by the
    two refined global anchors.  The returned channel layout is:

    - 72 groupwise correlations: 9 candidates x 8 groups;
    - 9 whole-feature cosine correlations;
    - 9 stereo validity flags;
    - 9 offsets relative to the current disparity, normalised by ``max_disp/4``;
    - 16 geometry descriptor channels;
    - 4 statistics: anchor score 1/2, entropy, and posterior margin.
    """

    LOCAL_MULTIPLIERS: Tuple[float, ...] = (
        -1.0,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        1.0,
    )

    def __init__(
        self,
        max_disp: int,
        num_groups: int = 8,
        geometry_channels: int = 16,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if max_disp <= 0:
            raise ValueError(f"max_disp must be positive, got {max_disp}")
        if num_groups != 8:
            raise ValueError(
                "The fixed 119-channel PRU interface requires num_groups=8; "
                f"got {num_groups}"
            )
        if geometry_channels != 16:
            raise ValueError(
                "The fixed 119-channel PRU interface requires geometry_channels=16; "
                f"got {geometry_channels}"
            )
        self.max_disp = int(max_disp)
        self.max_disp_quarter = float(max_disp) / 4.0
        self.num_groups = int(num_groups)
        self.geometry_channels = int(geometry_channels)
        self.eps = float(eps)
        self.output_channels = 119
        self.register_buffer(
            "local_multipliers",
            torch.tensor(self.LOCAL_MULTIPLIERS, dtype=torch.float32),
        )

    @staticmethod
    def _one_channel_field(
        value: Union[Tensor, float],
        batch: int,
        size: Tuple[int, int],
        device: torch.device,
        name: str,
    ) -> Tensor:
        if isinstance(value, (float, int)):
            return torch.full((batch, 1, *size), float(value), device=device)
        if value.ndim != 4 or value.shape[0] != batch or value.shape[1] != 1:
            raise ValueError(f"{name} must be scalar or [B,1,H,W], got {tuple(value.shape)}")
        if value.device != device:
            raise ValueError(f"{name} must be on the feature device")
        return _resize_field(value, size)

    def forward(
        self,
        fmap1_4: Tensor,
        fmap2_4: Tensor,
        disp: Tensor,
        radius: Union[Tensor, float],
        anchors: Tensor,
        geometry: Tensor,
        anchor_scores: Tensor,
        entropy: Union[Tensor, float],
        margin: Union[Tensor, float],
        anchor_valid: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        b, c, h, w = _check_feature_pair(
            fmap1_4, fmap2_4, self.__class__.__name__
        )
        _check_groups(c, self.num_groups, self.__class__.__name__)
        if disp.ndim != 4 or disp.shape[:2] != (b, 1):
            raise ValueError(f"disp must be [B,1,H,W] in quarter units, got {tuple(disp.shape)}")
        if anchors.ndim != 4 or anchors.shape[:2] != (b, 2):
            raise ValueError(
                f"anchors must be [B,2,H,W] in quarter units, got {tuple(anchors.shape)}"
            )
        if anchor_scores.ndim != 4 or anchor_scores.shape[:2] != (b, 2):
            raise ValueError(
                f"anchor_scores must be [B,2,H,W], got {tuple(anchor_scores.shape)}"
            )
        if geometry.ndim != 4 or geometry.shape[:2] != (b, self.geometry_channels):
            raise ValueError(
                f"geometry must be [B,{self.geometry_channels},H,W], got {tuple(geometry.shape)}"
            )
        for name, value in (
            ("disp", disp),
            ("anchors", anchors),
            ("geometry", geometry),
            ("anchor_scores", anchor_scores),
        ):
            if value.device != fmap1_4.device:
                raise ValueError(f"{name} must be on the feature device")

        size = (h, w)
        disp = _resize_field(disp, size)
        radius_field = self._one_channel_field(
            radius, b, size, fmap1_4.device, "radius"
        ).clamp_min(self.eps)
        anchors = _resize_field(anchors, size)
        geometry = _resize_field(geometry, size)
        anchor_scores = _resize_field(anchor_scores, size)
        entropy_field = self._one_channel_field(
            entropy, b, size, fmap1_4.device, "entropy"
        )
        margin_field = self._one_channel_field(
            margin, b, size, fmap1_4.device, "margin"
        )
        if anchor_valid is None:
            anchor_valid_field = torch.ones(
                b, 2, h, w, device=fmap1_4.device, dtype=torch.bool
            )
        else:
            if (
                anchor_valid.ndim != 4
                or anchor_valid.shape[:2] != (b, 2)
                or anchor_valid.device != fmap1_4.device
            ):
                raise ValueError(
                    "anchor_valid must be [B,2,H,W] on the feature device, "
                    f"got {tuple(anchor_valid.shape)}"
                )
            anchor_valid_field = F.interpolate(
                anchor_valid.float(), size=size, mode="nearest"
            ).bool()

        multipliers = self.local_multipliers.to(device=fmap1_4.device).view(1, 7, 1, 1)
        local_candidates = disp + radius_field * multipliers
        candidates = torch.cat((local_candidates, anchors), dim=1)
        sampled, valid = _sample_right_at_disparities(fmap2_4, candidates)
        semantic_valid = torch.cat(
            (
                torch.ones(
                    b, 7, h, w,
                    device=fmap1_4.device,
                    dtype=torch.bool,
                ),
                anchor_valid_field,
            ),
            dim=1,
        )
        valid = (
            valid
            & semantic_valid
            & (candidates < self.max_disp_quarter)
        )
        gwc, dot = _candidate_correlations(
            fmap1_4, sampled, self.num_groups, self.eps
        )
        gwc = gwc * valid.unsqueeze(2).to(gwc.dtype)
        dot = dot * valid.to(dot.dtype)

        # Candidate-major layout: [candidate0/group0..7, candidate1/...].
        gwc_channels = gwc.reshape(b, 9 * self.num_groups, h, w)
        offsets = (
            (candidates.float() - disp) / self.max_disp_quarter
        ).clamp(-1.0, 1.0)
        offsets = offsets * valid.float()
        stats = torch.cat(
            (
                anchor_scores.float(),
                entropy_field.float(),
                margin_field.float(),
            ),
            dim=1,
        )
        corr = torch.cat(
            (
                gwc_channels.float(),
                dot.float(),
                valid.float(),
                offsets,
                geometry.float(),
                stats,
            ),
            dim=1,
        ).contiguous()
        if corr.shape != (b, self.output_channels, h, w):
            raise RuntimeError(
                f"AdaptiveTubeCorrBlock internal channel error: got {tuple(corr.shape)}, "
                f"expected {(b, self.output_channels, h, w)}"
            )

        if not return_aux:
            return corr
        return corr, {
            "candidates": candidates,
            "valid": valid,
            "gwc": gwc,
            "dot": dot,
            "offsets": offsets,
        }


class AdaptiveLocalCorrBlock(nn.Module):
    """
    Anchor-free correlations sampled only around the current disparity.
    """

    LEGACY_MULTIPLIERS: Tuple[float, ...] = (
        -1.0,
        -0.75,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    # Keep nine candidates and the original correlation width, but reserve
    # four samples for far recovery when the current disparity is unreliable.
    WIDE_MULTIPLIERS: Tuple[float, ...] = (
        -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0,
    )
    SAMPLING_LAYOUTS = {
        "legacy9": LEGACY_MULTIPLIERS,
        "wide9": WIDE_MULTIPLIERS,
    }

    def __init__(self, max_disp: int, num_groups: int = 8, feature_channels: int = 64,
                 feature_stride: int = 4, galerkin_heads: int = 4,
                 match_temperature: float = 0.1, eps: float = 1.0e-6,
                 sampling_layout: str = "legacy9") -> None:
        super().__init__()
        if max_disp <= 0:
            raise ValueError(f"max_disp must be positive, got {max_disp}")
        if num_groups <= 0 or feature_channels % num_groups != 0:
            raise ValueError(
                f"feature_channels={feature_channels} must be divisible by "
                f"positive num_groups={num_groups}"
            )
        if match_temperature <= 0:
            raise ValueError(f"match_temperature must be positive, got {match_temperature}")
        if feature_stride <= 0:
            raise ValueError(f"feature_stride must be positive, got {feature_stride}")
        self.max_disp = int(max_disp)
        self.feature_stride = int(feature_stride)
        self.max_disp_feature = float(max_disp) / float(feature_stride)
        self.num_groups = int(num_groups)
        self.feature_channels = int(feature_channels)
        self.match_temperature = float(match_temperature)
        self.eps = float(eps)
        self.sampling_layout = str(sampling_layout)
        multipliers = self.SAMPLING_LAYOUTS.get(
            self.sampling_layout, self.LEGACY_MULTIPLIERS
        )
        self.num_candidates = len(multipliers)
        # K*G GWC + K cosine + K valid + K offset + four state statistics.
        self.output_channels = self.num_candidates * (num_groups + 3) + 4
        self.register_buffer("local_multipliers", torch.tensor(multipliers, dtype=torch.float32))
        self.rope_galerkin = simple_cross_attn_rope_2d(feature_channels, galerkin_heads)

    @staticmethod
    def _field(value: Tensor, batch: int, size: Tuple[int, int],
               device: torch.device, name: str) -> Tensor:
        if value.ndim != 4 or value.shape[:2] != (batch, 1):
            raise ValueError(
                f"{name} must be [B,1,H,W], got {tuple(value.shape)}"
            )
        if value.device != device:
            raise ValueError(f"{name} must be on the feature device")
        return _resize_field(value, size)

    def forward(self, fmap1_4: Tensor, fmap2_4: Tensor, disp: Tensor,
                radius: Tensor, coarse_std: Tensor, coarse_entropy: Tensor,
                coarse_margin: Tensor, return_aux: bool = False):
        b, c, h, w = _check_feature_pair(
            fmap1_4, fmap2_4, self.__class__.__name__
        )
        _check_groups(c, self.num_groups, self.__class__.__name__)
        if c != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} feature channels, got {c}")
        size = (h, w)
        # disp = self._field(disp, b, size, fmap1_4.device, "disp")
        # radius = self._field(radius, b, size, fmap1_4.device, "radius").clamp_min(self.eps)
        # coarse_std = self._field(coarse_std, b, size, fmap1_4.device, "coarse_std")
        # coarse_entropy = self._field(coarse_entropy, b, size, fmap1_4.device, "coarse_entropy")
        # coarse_margin = self._field(coarse_margin, b, size, fmap1_4.device, "coarse_margin")

        multipliers = self.local_multipliers.to(
            device=fmap1_4.device, dtype=disp.dtype
        ).view(1, self.num_candidates, 1, 1)
        candidates = disp + radius * multipliers
        sampled, valid = _sample_right_at_disparities(fmap2_4, candidates)
        valid = valid & (candidates < self.max_disp_feature)
        gwc, dot = _candidate_correlations(fmap1_4, sampled, self.num_groups, self.eps)
        gwc = gwc * valid.unsqueeze(2).to(gwc.dtype)
        dot = dot * valid.to(dot.dtype)

        match_logits = (dot + gwc.mean(dim=2)).masked_fill(~valid, -1.0e4)
        match_weights = torch.softmax(match_logits / self.match_temperature, dim=1) * valid.float()
        match_weights = match_weights / match_weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        aligned_right = (match_weights.unsqueeze(2).to(sampled.dtype) * sampled).sum(dim=1)
        aligned_right = aligned_right * valid.any(dim=1, keepdim=True).to(aligned_right.dtype)
        galerkin_feat = self.rope_galerkin(fmap1_4.float(), aligned_right.float())

        gwc_channels = gwc.reshape(
            b, self.num_candidates * self.num_groups, h, w
        )
        offsets = ((candidates.float() - disp.float()) /
                   self.max_disp_feature).clamp(-1.0, 1.0)
        offsets = offsets * valid.float()
        stats = torch.cat(
            (
                coarse_std.float() / self.max_disp_feature,
                coarse_entropy.float(),
                coarse_margin.float(),
                radius.float() / self.max_disp_feature,
            ),
            dim=1,
        )
        corr = torch.cat((gwc_channels.float(), dot.float(), valid.float(), offsets, stats,), dim=1).contiguous()
        # expected = (b, self.output_channels, h, w)
        # if corr.shape != expected:
        #     raise RuntimeError(
        #         f"AdaptiveLocalCorrBlock channel error: got {tuple(corr.shape)}, "
        #         f"expected {expected}"
        #     )
        if not return_aux:
            return corr, galerkin_feat
        return corr, galerkin_feat, {
            "candidates": candidates,
            "valid": valid,
            "gwc": gwc,
            "dot": dot,
            "offsets": offsets,
            "match_weights": match_weights,
            "aligned_right": aligned_right,
        }


class AdaptiveMidScaleRefiner(nn.Module):
    """Use 1/8 stereo features to correct a single coarse disparity state.

    Inputs and outputs keep the model-wide quarter-pixel disparity convention.
    Candidate disparities are converted to 1/8 feature pixels only while
    sampling the right feature map.
    """

    def __init__(self, max_disp: int, feature_channels: int = 128,
                 hidden_channels: int = 64, num_groups: int = 8,
                 min_radius: float = 1.0, max_radius: float = 8.0,
                 sampling_layout: str = "legacy9", delta_scale: float = 1.0) -> None:
        super().__init__()
        if min_radius <= 0 or max_radius < min_radius:
            raise ValueError("invalid 1/8 refinement radius bounds")
        self.max_disp_quarter = float(max_disp) / 4.0
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.delta_scale = float(delta_scale)
        self.corr = AdaptiveLocalCorrBlock(
            max_disp=max_disp,
            num_groups=num_groups,
            feature_channels=feature_channels,
            feature_stride=8,
            sampling_layout=sampling_layout,
        )
        self.head = nn.Sequential(
            nn.Conv2d(self.corr.output_channels + feature_channels,
                      hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(hidden_channels, 8), hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.normal_(self.head[-1].weight[:1], mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.head[-1].weight[1:])
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, fmap1_8: Tensor, fmap2_8: Tensor, disp_q: Tensor,
                radius_q: Tensor, coarse_std_q: Tensor,
                coarse_entropy: Tensor, coarse_margin: Tensor,
                return_aux: bool = False):
        corr, galerkin, corr_aux = self.corr(
            fmap1_8, fmap2_8, disp_q / 2.0, radius_q / 2.0,
            coarse_std_q / 2.0, coarse_entropy, coarse_margin,
            return_aux=True,
        )
        raw_delta, raw_confidence = self.head(
            torch.cat((corr, galerkin), dim=1)
        ).chunk(2, dim=1)
        confidence = torch.sigmoid(raw_confidence.float())
        # Wide9 exposes far candidates without adding channels.  Its mid
        # correction needs the matching jump range to match that visibility.
        delta_q = (
            confidence * self.delta_scale * radius_q.float()
            * torch.tanh(raw_delta.float())
        )
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
        confidence = confidence * has_valid.to(confidence.dtype)
        refined_radius_q = confidence_to_radius(
            confidence, self.min_radius, self.max_radius
        )
        if not return_aux:
            return refined_disp_q, refined_radius_q, confidence
        return refined_disp_q, refined_radius_q, confidence, {
            "corr": corr,
            "galerkin": galerkin,
            "delta_q": delta_q,
            "has_valid": has_valid,
            **corr_aux,
        }


# Short aliases are kept for readable model wiring.
GlobalCoarseGWC = GlobalCoarseGwcAggregator


__all__ = [
    "confidence_to_radius",
    "GlobalCoarseGwcAggregator",
    "GlobalCoarseGWC",
    "AdaptiveTubeRefiner",
    "AdaptiveTubeCorrBlock",
    "AdaptiveLocalCorrBlock",
    "AdaptiveMidScaleRefiner",
]
