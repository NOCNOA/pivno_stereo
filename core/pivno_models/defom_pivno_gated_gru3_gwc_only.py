"""Standalone RGB DEFOM-PIVNO using only GWC for sampled-right matches."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIVNO.models.sronet import PIVNO
from core.extractor import MultiBasicEncoder2
from core.galerkin import simple_attn_rope_2d
from core.pivno_models.update_gru3 import IGEVStyleBasicMultiUpdateBlock
from core.submodules import (
    ParallelRightWidthCompressor,
    sample_right_feature_pyramid,
)


def encode_group_correlations(
    left_feature,
    sampled_right,
    num_groups=8,
    eps=1e-4,
):
    """Encode each sampled-right candidate with group cosine correlation only.

    Args:
        left_feature: reference feature ``[B,C,H,W]``.
        sampled_right: sampled right features ``[B,S,K,C,H,W]``.
        num_groups: number of channel groups used by normalized GWC.
        eps: lower bound for group norms. Zero/OOB samples receive zero
            correlation and zero correlation gradient.

    Returns:
        Group correlations ``[B,S,K,G,H,W]``.
    """
    if left_feature.ndim != 4:
        raise ValueError(
            "left_feature must be [B,C,H,W], got "
            f"{tuple(left_feature.shape)}"
        )
    if sampled_right.ndim != 6:
        raise ValueError(
            "sampled_right must be [B,S,K,C,H,W], got "
            f"{tuple(sampled_right.shape)}"
        )

    batch, channels, height, width = left_feature.shape
    right_batch, _, _, right_channels, right_height, right_width = (
        sampled_right.shape
    )
    if (right_batch, right_channels, right_height, right_width) != (
        batch,
        channels,
        height,
        width,
    ):
        raise ValueError(
            "left/sample shape mismatch: "
            f"left={tuple(left_feature.shape)} "
            f"sampled={tuple(sampled_right.shape)}"
        )
    if num_groups <= 0 or channels % num_groups != 0:
        raise ValueError(
            f"feature channels ({channels}) must be divisible by "
            f"num_groups ({num_groups})"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    channels_per_group = channels // num_groups
    with torch.cuda.amp.autocast(enabled=False):
        left_groups = left_feature.float().reshape(
            batch,
            num_groups,
            channels_per_group,
            height,
            width,
        )
        left_squared_norm = left_groups.square().sum(dim=2)
        correlations = []
        for right_scale in sampled_right.unbind(dim=1):
            sample_count = right_scale.shape[1]
            right_groups = right_scale.float().reshape(
                batch,
                sample_count,
                num_groups,
                channels_per_group,
                height,
                width,
            )
            numerator = (
                left_groups[:, None] * right_groups
            ).sum(dim=3)
            right_squared_norm = right_groups.square().sum(dim=3)
            squared_denominator = (
                left_squared_norm[:, None] * right_squared_norm
            )
            valid_group = squared_denominator > eps * eps
            denominator = torch.sqrt(
                squared_denominator.clamp_min(eps * eps)
            )
            correlation = numerator / denominator
            correlations.append(torch.where(
                valid_group,
                correlation,
                torch.zeros_like(correlation),
            ))

    return torch.stack(correlations, dim=1).to(sampled_right.dtype)


class DEFOMStereo(nn.Module):
    """Independent GWC-only PIVNO model with scale gating and 3x3 GRUs."""

    MODEL_VARIANT = "defom_pivno_gated_gru3_gwc_only"
    SCALE_GATE_MODE = "gwc8_mean_softmax_weighted_encoded_concat"
    GRU_KERNEL_SIZE = IGEVStyleBasicMultiUpdateBlock.GRU_KERNEL_SIZE
    MATCH_NUM_GROUPS = 8
    MATCH_ENCODED_CHANNELS = 16
    RIGHT_SAMPLE_ENCODING = "gwc8_only_conv16_no_left_concat"
    AMP_POLICY = "fp16_compute_fp32_corr_attention_softmax"

    def __init__(self, args):
        super().__init__()
        if int(args.n_downsample) != 2:
            raise ValueError(
                "PIVNO produces quarter-resolution features, so "
                f"n_downsample must be 2, got {args.n_downsample}"
            )
        if len(args.hidden_dims) != 3:
            raise ValueError(
                "The recurrent update requires three hidden dimensions, "
                f"got {args.hidden_dims}"
            )

        self.args = args
        self.mixed_precision = bool(
            getattr(args, "mixed_precision", False)
        )
        self.register_buffer(
            "mean",
            torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255,
        )
        self.register_buffer(
            "std",
            torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255,
        )

        # This standalone variant always uses the RGB PIVNO encoder.
        self.pivno = PIVNO(input_channels=3)

        context_dims = args.hidden_dims
        self.cnet = MultiBasicEncoder2(
            output_dim=[args.hidden_dims, context_dims],
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
        )
        level_context_dims = list(reversed(context_dims))
        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(dim, dim * 3, 3, padding=1)
            for dim in level_context_dims[:args.n_gru_layers]
        ])

        low_feature_dim = 48
        self.corr_radius = int(args.corr_radius)
        self.compression_ratios = (1, 2, 4)
        self.max_delta_disp = float(
            max(self.compression_ratios) * self.corr_radius
        )
        self.register_buffer(
            "warp_offsets",
            torch.arange(
                -self.corr_radius,
                self.corr_radius + 1,
                dtype=torch.float32,
            ),
        )
        self.low_channel = nn.Conv2d(64, low_feature_dim, kernel_size=1)
        self.right_width_compressor = ParallelRightWidthCompressor(
            low_feature_dim,
            low_feature_dim,
            mode="conv",
        )

        if low_feature_dim % self.MATCH_NUM_GROUPS != 0:
            raise ValueError(
                f"low feature channels ({low_feature_dim}) must be divisible "
                f"by GWC groups ({self.MATCH_NUM_GROUPS})"
            )
        raw_encoded_sample_dim = self.MATCH_NUM_GROUPS
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
        gate_in_channels = low_feature_dim + num_scales * num_offsets
        self.scale_gate = nn.Sequential(
            nn.Conv2d(
                gate_in_channels,
                low_feature_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(low_feature_dim, num_scales, kernel_size=1),
        )
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)
        self.last_scale_weight_mean = None
        self.last_scale_gate_entropy = None

        encoded_fuse_channels = (
            num_scales * num_offsets * self.MATCH_ENCODED_CHANNELS
        )
        self.refine_right_fuse = self._make_right_fuser(
            encoded_fuse_channels
        )
        self.refine_rope = simple_attn_rope_2d(128, 4)
        self.update_block = IGEVStyleBasicMultiUpdateBlock(
            self.args,
            hidden_dims=args.hidden_dims,
        )

    @staticmethod
    def _make_right_fuser(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def _apply_scale_gate(self, fmap1_low, group_correlation):
        if group_correlation.ndim != 6:
            raise ValueError(
                "group_correlation must be [B,S,K,G,H,W], got "
                f"{tuple(group_correlation.shape)}"
            )
        batch, num_scales, _, num_groups, height, width = (
            group_correlation.shape
        )
        if num_scales != len(self.compression_ratios):
            raise ValueError(
                f"expected {len(self.compression_ratios)} scales, "
                f"got {num_scales}"
            )
        if num_groups != self.MATCH_NUM_GROUPS:
            raise ValueError(
                f"expected {self.MATCH_NUM_GROUPS} GWC groups, "
                f"got {num_groups}"
            )
        if (
            fmap1_low.shape[0] != batch
            or fmap1_low.shape[-2:] != (height, width)
        ):
            raise ValueError(
                "left/GWC shape mismatch: "
                f"left={tuple(fmap1_low.shape)} "
                f"gwc={tuple(group_correlation.shape)}"
            )

        # GWC is already part of the per-sample encoding. Reuse its mean
        # instead of building a second full-cosine feature solely for gating.
        with torch.cuda.amp.autocast(enabled=False):
            left_unit = F.normalize(
                fmap1_low.float(),
                dim=1,
                eps=1e-6,
            )
            correlation = group_correlation.float().mean(dim=3)

        gate_dtype = fmap1_low.dtype
        gate_input = torch.cat(
            [
                left_unit.to(gate_dtype),
                correlation.flatten(1, 2).to(gate_dtype),
            ],
            dim=1,
        )
        scale_logits = self.scale_gate(gate_input)
        with torch.cuda.amp.autocast(enabled=False):
            scale_weights = torch.softmax(scale_logits.float(), dim=1)

        detached_weights = scale_weights.detach()
        self.last_scale_weight_mean = detached_weights.mean(
            dim=(0, 2, 3)
        )
        entropy = -(
            detached_weights
            * detached_weights.clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        self.last_scale_gate_entropy = entropy / math.log(num_scales)
        return scale_weights

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
            parameter
            for parameter in self.scale_gate.parameters()
            if parameter.requires_grad
        ]
        gate_ids = {id(parameter) for parameter in gate_parameters}
        base_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in gate_ids
        ]
        gate_lr = getattr(args, "pivno_gate_lr", None)
        gate_lr = float(args.lr if gate_lr is None else gate_lr)
        return [
            {"params": base_parameters, "lr": float(args.lr)},
            {"params": gate_parameters, "lr": gate_lr},
        ]

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
        sample_count = num_samples // num_scales
        sampled_right = sampled_right.reshape(
            batch,
            num_scales,
            sample_count,
            channels,
            height,
            width,
        )

        encoded_right = encode_group_correlations(
            fmap1_low,
            sampled_right,
            num_groups=self.MATCH_NUM_GROUPS,
        )
        del sampled_right
        scale_weights = self._apply_scale_gate(
            fmap1_low,
            encoded_right,
        )
        encoded_channels = encoded_right.shape[3]
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
        scaled_weights = (
            float(num_scales) * scale_weights
        ).to(encoded_right.dtype)
        encoded_right = (
            encoded_right
            * scaled_weights[:, :, None, None]
        )

        encoded_for_fusion = encoded_right.flatten(1, 3)
        fused_feature = self.refine_right_fuse(encoded_for_fusion)
        with torch.cuda.amp.autocast(enabled=False):
            return self.refine_rope(fused_feature.float())

    def _clamp_delta_disp(self, delta_disp):
        """Limit one update to the widest sampled search support."""
        return delta_disp.clamp(
            min=-self.max_delta_disp,
            max=self.max_delta_disp,
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def upsample_flow(self, flow, mask):
        """Upsample a quarter-resolution disparity with a convex mask."""
        batch, channels, height, width = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(
            batch,
            1,
            9,
            factor,
            factor,
            height,
            width,
        )
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(
            batch,
            channels,
            9,
            1,
            1,
            height,
            width,
        )
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(
            batch,
            channels,
            factor * height,
            factor * width,
        )

    @staticmethod
    def _to_pivno_rgb(image):
        """Scale RGB input from [0,255] to PIVNO's [0,1] range."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"Expected RGB input [B,3,H,W], got {tuple(image.shape)}"
            )
        return image / 255.0

    def forward(
        self,
        image1,
        image2,
        iters=12,
        scale_iters=None,
        test_mode=False,
    ):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            return self._forward_impl(
                image1,
                image2,
                iters=iters,
                scale_iters=scale_iters,
                test_mode=test_mode,
            )

    def _forward_impl(
        self,
        image1,
        image2,
        iters=12,
        scale_iters=None,
        test_mode=False,
    ):
        del scale_iters
        if image1.shape != image2.shape:
            raise ValueError(
                "Stereo images must have equal shapes, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
        if iters < 1:
            raise ValueError(f"iters must be at least 1, got {iters}")

        pivno_image1 = self._to_pivno_rgb(image1).contiguous().float()
        pivno_image2 = self._to_pivno_rgb(image2).contiguous().float()
        context_image = (
            (image1 - self.mean) / self.std
        ).contiguous().float()

        init_disp_predictions, fmap1_4, fmap2_4, d0 = self.pivno(
            pivno_image1,
            pivno_image2,
            return_imgfeature=True,
        )
        del pivno_image1, pivno_image2
        disp = d0

        cnet_list = self.cnet(
            context_image,
            num_layers=self.args.n_gru_layers,
        )
        net_list = [torch.tanh(item[0]) for item in cnet_list]
        inp_list = [torch.relu(item[1]) for item in cnet_list]
        inp_list = [
            list(conv(context).chunk(3, dim=1))
            for context, conv in zip(
                inp_list,
                self.context_zqr_convs,
            )
        ]
        del context_image, cnet_list

        fmap1_low = self.low_channel(fmap1_4)
        fmap2_low = self.low_channel(fmap2_4)
        del fmap1_4, fmap2_4
        fmap2_half, fmap2_quarter = self.right_width_compressor(
            fmap2_low
        )
        right_feature_pyramid = (
            fmap2_low,
            fmap2_half,
            fmap2_quarter,
        )
        del fmap2_low, fmap2_half, fmap2_quarter

        disp_predictions = []
        for _ in range(iters):
            disp = disp.detach()
            warp_feature = self._refine_warp_feature(
                fmap1_low,
                right_feature_pyramid,
                disp,
            )
            net_list, up_mask, delta_disp = self.update_block(
                net_list,
                inp_list,
                warp_feature,
                disp,
                iter32=self.args.n_gru_layers == 3,
                iter16=self.args.n_gru_layers >= 2,
            )
            delta_disp = self._clamp_delta_disp(delta_disp)
            disp = disp + delta_disp
            disp_up = self.upsample_flow(disp, up_mask)
            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up
        return init_disp_predictions, disp_predictions
