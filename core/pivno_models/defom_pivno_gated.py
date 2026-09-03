"""PIVNO-DEFOM with checkpoint-compatible per-pixel scale gating."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.pivno_models.defom_pivno import DEFOMStereo as BaseDEFOMStereo
from core.submodules import sample_right_feature_pyramid


class DEFOMStereo(BaseDEFOMStereo):
    """Reweight scale branches using stereo matching evidence before concat."""

    MODEL_VARIANT = "defom_pivno_gated"
    SCALE_GATE_MODE = "correlation_softmax_weighted_concat"

    def __init__(self, args):
        super().__init__(args)
        num_scales = len(self.compression_ratios)
        num_offsets = int(self.warp_offsets.numel())
        low_feature_dim = int(self.low_channel.out_channels)
        gate_in_channels = low_feature_dim + num_scales * num_offsets
        self.scale_gate = nn.Sequential(
            nn.Conv2d(gate_in_channels, low_feature_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(low_feature_dim, num_scales, 1),
        )
        # Uniform softmax weights followed by multiplication by S reproduce the
        # original unweighted concatenation at initialization.
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)
        self.last_scale_weight_mean = None
        self.last_scale_gate_entropy = None

    def _apply_scale_gate(self, fmap1_low, sampled_right):
        if sampled_right.ndim != 6:
            raise ValueError(
                "sampled_right must be [B,S,K,C,H,W], got "
                f"{tuple(sampled_right.shape)}"
            )
        batch, num_scales, _, channels, height, width = sampled_right.shape
        if num_scales != len(self.compression_ratios):
            raise ValueError(
                f"expected {len(self.compression_ratios)} scales, got {num_scales}"
            )
        if tuple(fmap1_low.shape) != (batch, channels, height, width):
            raise ValueError(
                "left/sample shape mismatch: "
                f"left={tuple(fmap1_low.shape)} sampled={tuple(sampled_right.shape)}"
            )

        # Cosine matching curves make gate logits comparable across the three
        # independently compressed right-feature branches.
        with torch.cuda.amp.autocast(enabled=False):
            left_unit = F.normalize(fmap1_low.float(), dim=1, eps=1e-6)
            right_unit = F.normalize(sampled_right.float(), dim=3, eps=1e-6)
            correlation = (
                left_unit[:, None, None] * right_unit
            ).sum(dim=3)
            gate_input = torch.cat(
                [left_unit, correlation.flatten(1, 2)],
                dim=1,
            )
            scale_logits = self.scale_gate(gate_input)
            scale_weights = torch.softmax(scale_logits, dim=1)

        scaled_weights = (
            float(num_scales) * scale_weights
        ).to(sampled_right.dtype)
        weighted_right = sampled_right * scaled_weights[:, :, None, None]

        detached_weights = scale_weights.detach()
        self.last_scale_weight_mean = detached_weights.mean(dim=(0, 2, 3))
        entropy = -(
            detached_weights
            * detached_weights.clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        self.last_scale_gate_entropy = entropy / math.log(num_scales)
        return weighted_right, scale_weights

    def scale_gate_metrics(self):
        if self.last_scale_weight_mean is None:
            return {}
        metrics = {
            f"scale_gate_weight_r{ratio}": float(weight)
            for ratio, weight in zip(
                self.compression_ratios,
                self.last_scale_weight_mean.cpu().tolist(),
            )
        }
        metrics["scale_gate_entropy"] = float(
            self.last_scale_gate_entropy.cpu()
        )
        return metrics

    def optimizer_parameter_groups(self, args):
        gate_parameters = [
            parameter for parameter in self.scale_gate.parameters()
            if parameter.requires_grad
        ]
        gate_ids = {id(parameter) for parameter in gate_parameters}
        base_parameters = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in gate_ids
        ]
        gate_lr = getattr(args, "pivno_gate_lr", None)
        gate_lr = float(args.lr if gate_lr is None else gate_lr)
        return [
            {"params": base_parameters, "lr": float(args.lr)},
            {"params": gate_parameters, "lr": gate_lr},
        ]

    def _refine_warp_feature(self, fmap1_low, right_feature_pyramid, disp):
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
        sampled_right = sampled_right.reshape(
            batch,
            num_scales,
            num_samples // num_scales,
            channels,
            height,
            width,
        )
        weighted_right, _ = self._apply_scale_gate(fmap1_low, sampled_right)
        right_for_fusion = weighted_right.flatten(1, 3)
        fused_input = torch.cat(
            [fmap1_low.to(right_for_fusion.dtype), right_for_fusion],
            dim=1,
        )
        fused_feature = self.refine_right_fuse(fused_input)
        with torch.cuda.amp.autocast(enabled=False):
            return self.refine_rope(fused_feature.float())
