"""Correlation modules used only by the BiLap-GRU PACT variant."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.adaptive_tube import (
    AdaptiveLocalCorrBlock,
    _candidate_correlations,
    _check_feature_pair,
    _check_groups,
    _normalise_pixel_coordinate,
    _sample_right_at_disparities,
)


class BiLapLocalCorrBlock(AdaptiveLocalCorrBlock):
    """Local matching with only GWC and dot scores exposed to the update block.

    Candidate validity and radius remain internal geometric controls.  Unlike
    the base PACT block, they are not concatenated into the recurrent matching
    feature together with offsets or coarse posterior statistics.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_channels = self.num_candidates * (self.num_groups + 1)

    def forward(self, fmap1_4, fmap2_4, disp, radius, return_aux=False):
        batch, channels, height, width = _check_feature_pair(fmap1_4, fmap2_4, self.__class__.__name__)
        _check_groups(channels, self.num_groups, self.__class__.__name__)
        if channels != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} feature channels, got {channels}")

        multipliers = self.local_multipliers.to(device=fmap1_4.device, dtype=disp.dtype).view(1, self.num_candidates, 1, 1)
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

        gwc_channels = gwc.reshape(batch, self.num_candidates * self.num_groups, height, width)
        corr = torch.cat((gwc_channels.float(), dot.float()), dim=1).contiguous()
        if not return_aux:
            return corr, galerkin_feat
        return corr, galerkin_feat, {
            "candidates": candidates,
            "valid": valid,
            "gwc": gwc,
            "dot": dot,
            "match_weights": match_weights,
            "aligned_right": aligned_right,
        }


class RightWidthPyramid(nn.Module):
    """Project to 32 channels, then create W and W/2 widths."""

    def __init__(self, input_channels=64, channels=32) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.to_full = nn.Sequential(nn.Conv2d(input_channels, channels, kernel_size=1, bias=False), nn.GroupNorm(groups, channels), nn.GELU())
        self.to_half = self._make_stage(channels, groups)

    @staticmethod
    def _make_stage(channels, groups):
        return nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), groups=channels, bias=False), nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.GroupNorm(groups, channels), nn.GELU())

    def forward(self, right_input):
        right_full = self.to_full(right_input)
        right_half = self.to_half(right_full)
        return right_full, right_half


class BiLapMultiScaleRightCorrBlock(AdaptiveLocalCorrBlock):
    """Fuse 32-channel local correlations from W and W/2."""

    WIDTH_LEVELS = 2

    def __init__(self, *args, input_channels=64, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_channels = int(input_channels)
        self.branch_channels = self.num_candidates * (self.num_groups + 1)
        self.output_channels = self.branch_channels
        groups = min(8, self.feature_channels)
        while self.feature_channels % groups != 0:
            groups -= 1
        self.left_projection = nn.Sequential(nn.Conv2d(self.input_channels, self.feature_channels, kernel_size=1, bias=False), nn.GroupNorm(groups, self.feature_channels), nn.GELU())
        self.right_width_pyramid = RightWidthPyramid(self.input_channels, self.feature_channels)
        self.corr_fusion = nn.Conv2d(self.WIDTH_LEVELS * self.branch_channels, self.branch_channels, kernel_size=1)
        self.aligned_fusion = nn.Conv2d(self.WIDTH_LEVELS * self.feature_channels, self.feature_channels, kernel_size=1)
        nn.init.zeros_(self.corr_fusion.weight)
        nn.init.zeros_(self.corr_fusion.bias)
        nn.init.zeros_(self.aligned_fusion.weight)
        nn.init.zeros_(self.aligned_fusion.bias)

    def build_right_pyramid(self, right_full):
        return self.right_width_pyramid(right_full)

    def build_left_feature(self, left_input):
        return self.left_projection(left_input)

    def _sample_width_level(self, right_feature, disp, radius):
        batch, channels, source_height, source_width = right_feature.shape
        if disp.ndim != 4 or disp.shape[1] != 1 or radius.shape != disp.shape:
            raise ValueError("disp and radius must have equal [B,1,H,W] shapes")
        query_height, query_width = disp.shape[-2:]
        if batch != disp.shape[0] or channels != self.feature_channels or source_height != query_height:
            raise ValueError("compressed right feature does not align with the left query grid")
        if query_width <= 1 or source_width <= 1:
            raise ValueError("BiLap multi-width correlation requires feature widths greater than one")

        coordinate_scale = float(source_width - 1) / float(query_width - 1)
        multipliers = self.local_multipliers.to(device=disp.device, dtype=disp.dtype).view(1, self.num_candidates, 1, 1)
        compressed_disp = coordinate_scale * disp + radius * multipliers
        effective_candidates = disp + radius * multipliers / coordinate_scale

        source = right_feature.float() if right_feature.device.type == "cpu" and right_feature.dtype in (torch.float16, torch.bfloat16) else right_feature
        grid_dtype = source.dtype
        xx = torch.arange(query_width, device=disp.device, dtype=grid_dtype).view(1, 1, 1, query_width)
        yy = torch.arange(query_height, device=disp.device, dtype=grid_dtype).view(1, 1, query_height, 1)
        right_x = coordinate_scale * xx - compressed_disp.to(grid_dtype)
        valid = (effective_candidates >= 0.0) & (effective_candidates < self.max_disp_feature) & (right_x.float() >= 0.0) & (right_x.float() <= float(source_width - 1))

        x_norm = _normalise_pixel_coordinate(right_x, source_width)
        y_norm = _normalise_pixel_coordinate(yy, source_height).expand(batch, self.num_candidates, query_height, query_width)
        x_grid = x_norm.permute(0, 2, 1, 3).reshape(batch, query_height, self.num_candidates * query_width)
        y_grid = y_norm.permute(0, 2, 1, 3).reshape(batch, query_height, self.num_candidates * query_width)
        grid = torch.stack((x_grid, y_grid), dim=-1)
        sampled = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.reshape(batch, channels, query_height, self.num_candidates, query_width).permute(0, 3, 1, 2, 4).contiguous()
        return sampled, valid, effective_candidates

    def _correlate_width_level(self, left_feature, right_feature, disp, radius):
        sampled, valid, candidates = self._sample_width_level(right_feature, disp, radius)
        gwc, dot = _candidate_correlations(left_feature, sampled, self.num_groups, self.eps)
        gwc = gwc * valid.unsqueeze(2).to(gwc.dtype)
        dot = dot * valid.to(dot.dtype)
        match_logits = (dot + gwc.mean(dim=2)).masked_fill(~valid, -1.0e4)
        match_weights = torch.softmax(match_logits / self.match_temperature, dim=1) * valid.float()
        match_weights = match_weights / match_weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        aligned_right = (match_weights.unsqueeze(2).to(sampled.dtype) * sampled).sum(dim=1)
        aligned_right = aligned_right * valid.any(dim=1, keepdim=True).to(aligned_right.dtype)
        batch, _, height, width = dot.shape
        corr = torch.cat((gwc.reshape(batch, self.num_candidates * self.num_groups, height, width).float(), dot.float()), dim=1).contiguous()
        return corr, aligned_right, {"candidates": candidates, "valid": valid, "gwc": gwc, "dot": dot, "match_weights": match_weights}

    def _sample_width_level_modes(self, right_feature, means, radius):
        """Sample all modes and candidates in one grid_sample call.

        ``means`` and ``radius`` are ``[B,M,H,W]``. The right source is not
        repeated per mode; modes and candidates are folded into the output
        width of a single sampling grid.
        """
        batch, channels, source_height, source_width = right_feature.shape
        if means.ndim != 4 or radius.shape != means.shape:
            raise ValueError("means and radius must have equal [B,M,H,W] shapes")
        modes, query_height, query_width = means.shape[1:]
        if batch != means.shape[0] or channels != self.feature_channels or source_height != query_height:
            raise ValueError("compressed right feature does not align with the modal query grid")
        coordinate_scale = float(source_width - 1) / float(query_width - 1)
        multipliers = self.local_multipliers.to(device=means.device, dtype=means.dtype).view(1, 1, self.num_candidates, 1, 1)
        compressed_disp = coordinate_scale * means.unsqueeze(2) + radius.unsqueeze(2) * multipliers
        effective_candidates = means.unsqueeze(2) + radius.unsqueeze(2) * multipliers / coordinate_scale
        source = right_feature.float() if right_feature.device.type == "cpu" and right_feature.dtype in (torch.float16, torch.bfloat16) else right_feature
        grid_dtype = source.dtype
        xx = torch.arange(query_width, device=means.device, dtype=grid_dtype).view(1, 1, 1, 1, query_width)
        yy = torch.arange(query_height, device=means.device, dtype=grid_dtype).view(1, 1, 1, query_height, 1)
        right_x = coordinate_scale * xx - compressed_disp.to(grid_dtype)
        valid = (effective_candidates >= 0.0) & (effective_candidates < self.max_disp_feature) & (right_x.float() >= 0.0) & (right_x.float() <= float(source_width - 1))
        x_norm = _normalise_pixel_coordinate(right_x, source_width)
        y_norm = _normalise_pixel_coordinate(yy, source_height).expand(batch, modes, self.num_candidates, query_height, query_width)
        x_grid = x_norm.permute(0, 3, 1, 2, 4).reshape(batch, query_height, modes * self.num_candidates * query_width)
        y_grid = y_norm.permute(0, 3, 1, 2, 4).reshape(batch, query_height, modes * self.num_candidates * query_width)
        grid = torch.stack((x_grid, y_grid), dim=-1)
        sampled = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.reshape(batch, channels, query_height, modes, self.num_candidates, query_width).permute(0, 3, 4, 1, 2, 5).contiguous()
        return sampled, valid, effective_candidates

    def _correlate_width_level_modes(self, left_feature, right_feature, means, radius):
        sampled, valid, candidates = self._sample_width_level_modes(right_feature, means, radius)
        batch, modes, candidates_count, channels, height, width = sampled.shape
        left = left_feature.unsqueeze(1).expand(-1, modes, -1, -1, -1).reshape(batch * modes, channels, height, width)
        sampled_flat = sampled.reshape(batch * modes, candidates_count, channels, height, width)
        valid_flat = valid.reshape(batch * modes, candidates_count, height, width)
        gwc, dot = _candidate_correlations(left, sampled_flat, self.num_groups, self.eps)
        gwc = gwc * valid_flat.unsqueeze(2).to(gwc.dtype)
        dot = dot * valid_flat.to(dot.dtype)
        match_logits = (dot + gwc.mean(dim=2)).masked_fill(~valid_flat, -1.0e4)
        match_weights = torch.softmax(match_logits / self.match_temperature, dim=1) * valid_flat.float()
        match_weights = match_weights / match_weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        aligned = (match_weights.unsqueeze(2).to(sampled_flat.dtype) * sampled_flat).sum(dim=1)
        aligned = aligned * valid_flat.any(dim=1, keepdim=True).to(aligned.dtype)
        corr = torch.cat((gwc.reshape(batch * modes, self.num_candidates * self.num_groups, height, width).float(), dot.float()), dim=1)
        return corr.reshape(batch, modes, self.branch_channels, height, width), aligned.reshape(batch, modes, channels, height, width), {"candidates": candidates, "valid": valid, "gwc": gwc.reshape(batch, modes, self.num_candidates, self.num_groups, height, width), "dot": dot.reshape(batch, modes, self.num_candidates, height, width), "match_weights": match_weights.reshape(batch, modes, self.num_candidates, height, width)}

    def forward_modes(self, left_feature, right_full, right_half, means, radius, return_aux=False):
        """Return per-mode fused correlation and aligned Galerkin features."""
        batch, channels, height, width = _check_feature_pair(left_feature, right_full, self.__class__.__name__)
        _check_groups(channels, self.num_groups, self.__class__.__name__)
        mode_count = means.shape[1]
        branch_corrs, branch_aligned, branch_aux = [], [], []
        for right_feature in (right_full, right_half):
            corr, aligned, aux = self._correlate_width_level_modes(left_feature, right_feature, means, radius)
            branch_corrs.append(corr)
            branch_aligned.append(aligned)
            branch_aux.append(aux)
        flat_corr = torch.cat(branch_corrs, dim=2).reshape(batch * mode_count, self.WIDTH_LEVELS * self.branch_channels, height, width)
        flat_aligned = torch.cat(branch_aligned, dim=2).reshape(batch * mode_count, self.WIDTH_LEVELS * channels, height, width)
        corr_fused = branch_corrs[0].reshape(batch * mode_count, self.branch_channels, height, width) + self.corr_fusion(flat_corr)
        aligned_fused = branch_aligned[0].reshape(batch * mode_count, channels, height, width) + self.aligned_fusion(flat_aligned)
        left_modes = left_feature.unsqueeze(1).expand(-1, mode_count, -1, -1, -1).reshape(batch * mode_count, channels, height, width)
        galerkin = self.rope_galerkin(left_modes.float(), aligned_fused.float())
        corr_fused = corr_fused.reshape(batch, mode_count, self.branch_channels, height, width)
        galerkin = galerkin.reshape(batch, mode_count, channels, height, width)
        if not return_aux:
            return corr_fused, galerkin
        return corr_fused, galerkin, {"candidates": torch.cat([item["candidates"] for item in branch_aux], dim=2), "valid": torch.cat([item["valid"] for item in branch_aux], dim=2), "match_weights": torch.stack([item["match_weights"] for item in branch_aux], dim=2), "corr_by_width": torch.stack(branch_corrs, dim=2), "aligned_right_by_width": torch.stack(branch_aligned, dim=2)}

    def forward(self, left_feature, right_full, right_half, disp, radius, return_aux=False):
        batch, channels, height, width = _check_feature_pair(left_feature, right_full, self.__class__.__name__)
        _check_groups(channels, self.num_groups, self.__class__.__name__)
        if channels != self.feature_channels:
            raise ValueError(f"expected {self.feature_channels} feature channels, got {channels}")

        branch_corrs = []
        branch_aligned = []
        branch_aux = []
        for right_feature in (right_full, right_half):
            corr, aligned_right, aux = self._correlate_width_level(left_feature, right_feature, disp, radius)
            branch_corrs.append(corr)
            branch_aligned.append(aligned_right)
            if return_aux:
                branch_aux.append(aux)

        corr_fused = branch_corrs[0] + self.corr_fusion(torch.cat(branch_corrs, dim=1))
        aligned_fused = branch_aligned[0] + self.aligned_fusion(torch.cat(branch_aligned, dim=1))
        galerkin_feat = self.rope_galerkin(left_feature.float(), aligned_fused.float())
        if not return_aux:
            return corr_fused, galerkin_feat
        return corr_fused, galerkin_feat, {
            "candidates": torch.cat([item["candidates"] for item in branch_aux], dim=1),
            "valid": torch.cat([item["valid"] for item in branch_aux], dim=1),
            "gwc": torch.cat([item["gwc"] for item in branch_aux], dim=1),
            "dot": torch.cat([item["dot"] for item in branch_aux], dim=1),
            "match_weights": torch.stack([item["match_weights"] for item in branch_aux], dim=1),
            "corr_by_width": torch.stack(branch_corrs, dim=1),
            "aligned_right_by_width": torch.stack(branch_aligned, dim=1),
            "aligned_right": aligned_fused,
        }


__all__ = ["BiLapLocalCorrBlock", "RightWidthPyramid", "BiLapMultiScaleRightCorrBlock"]
