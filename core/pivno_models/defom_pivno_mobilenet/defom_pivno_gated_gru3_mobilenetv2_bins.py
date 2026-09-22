"""Standalone MobileNetV2 bins model with GWC8-only iterative matching."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.extractor import MultiBasicEncoder2
from core.galerkin import simple_attn_rope_2d
from core.pivno_models.defom_pivno_gated_gru3_bins import (
    MultiRangeBinInitializer, PIVNOFeatureExtractor,
)
from core.pivno_models.defom_pivno_mobilenet.defom_pivno_gated_gru3_mobilenetv2 import MobileNetV2PIVNOEncoder
from core.pivno_models.update_gru3 import IGEVStyleBasicMultiUpdateBlock
from core.submodules import ParallelRightWidthCompressor, sample_right_feature_pyramid


def group_correlation(left, sampled_right, groups=8, eps=1e-4):
    """Return normalized GWC [B,S,K,G,H,W], without feature differences."""
    b, c, h, w = left.shape
    s, k = sampled_right.shape[1:3]
    cg = c // groups
    with torch.cuda.amp.autocast(enabled=False):
        left = left.float().reshape(b, 1, 1, groups, cg, h, w)
        right = sampled_right.float().reshape(b, s, k, groups, cg, h, w)
        numerator = (left * right).sum(4)
        squared_denominator = (
            left.square().sum(4) * right.square().sum(4)
        )
        denominator = torch.sqrt(squared_denominator.clamp_min(eps * eps))
        correlation = torch.where(
            squared_denominator > eps * eps,
            numerator / denominator,
            torch.zeros_like(numerator),
        )
    return correlation.to(sampled_right.dtype)


class DEFOMStereo(nn.Module):
    """Multi-range bin initialization plus standalone GWC8 GRU3 refinement."""

    MODEL_VARIANT = "defom_pivno_gated_gru3_mobilenetv2_bins"
    BASE_MODEL_VARIANT = "defom_pivno_gated_gru3_bins"
    INITIALIZATION_MODE = "global_row_corr_bins_1x2x4x_no_mono"
    SCALE_GATE_MODE = "gwc8_mean_softmax_weighted_encoded_concat"
    RIGHT_SAMPLE_ENCODING = "gwc8_only_conv16_no_left_concat"
    MATCH_NUM_GROUPS = 8
    MATCH_ENCODED_CHANNELS = 16
    FINETUNE_STAGES = ("init", "joint")
    GRU_KERNEL_SIZE = IGEVStyleBasicMultiUpdateBlock.GRU_KERNEL_SIZE
    AMP_POLICY = "fp16_compute_fp32_corr_attention_softmax"
    ENCODER_CONFIG = {
        "feature_backbone": "timm_mobilenetv2_100_igevpp_fpn",
        "pivno_feature_encoder": "igevpp_feature_stem_96_project_64",
        "pivno_encoder_normalization": "rgb_0_1_to_minus1_1",
        "pivno_encoder_output_channels": 64,
        "pivno_bn_training_policy": "igevpp_frozen_batchnorm_preserve_activation",
        "pivno_upsampling": "bins_bilinear_init_gru3_convex_refinement",
        "matching_low_channels": 48,
        "matching_encoding": "gwc8_only",
    }

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mixed_precision = bool(getattr(args, "mixed_precision", False))
        self.imagenet_pretrained = bool(
            getattr(args, "pivno_mobilenet_pretrained", True)
        )
        self.register_buffer("mean", torch.tensor(
            [[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer("std", torch.tensor(
            [[0.229, 0.224, 0.225]])[..., None, None] * 255)

        restoring = bool(
            getattr(args, "resume_ckpt", None)
            or getattr(args, "restore_ckpt", None)
        )
        self.pivno = PIVNOFeatureExtractor(input_channels=3)
        self.pivno.snet = MobileNetV2PIVNOEncoder(
            pretrained=self.imagenet_pretrained and not restoring
        )

        context_dims = args.hidden_dims
        self.cnet = MultiBasicEncoder2(
            output_dim=[args.hidden_dims, context_dims],
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
        )
        level_dims = list(reversed(context_dims))
        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(dim, dim * 3, 3, padding=1)
            for dim in level_dims[:args.n_gru_layers]
        ])

        low_dim = 48
        self.corr_radius = int(args.corr_radius)
        self.compression_ratios = (1, 2, 4)
        self.max_delta_disp = float(max(self.compression_ratios) * self.corr_radius)
        self.register_buffer("warp_offsets", torch.arange(
            -self.corr_radius, self.corr_radius + 1, dtype=torch.float32))
        self.low_channel = nn.Conv2d(64, low_dim, 1)
        self.right_width_compressor = ParallelRightWidthCompressor(
            low_dim, low_dim, mode="conv")

        self.sample_match_encoder = nn.Sequential(
            nn.Conv2d(8, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.GELU())
        scales = len(self.compression_ratios)
        candidates = int(self.warp_offsets.numel())
        self.scale_gate = nn.Sequential(
            nn.Conv2d(low_dim + scales * candidates, low_dim, 3, padding=1),
            nn.GELU(), nn.Conv2d(low_dim, scales, 1))
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)

        self.refine_right_fuse = nn.Sequential(
            nn.Conv2d(scales * candidates * 16, 128, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.refine_rope = simple_attn_rope_2d(128, 4)
        self.update_block = IGEVStyleBasicMultiUpdateBlock(
            args, hidden_dims=args.hidden_dims)

        max_offset = getattr(args, "pivno_bins_max_offset", None)
        if max_offset is None:
            max_offset = float(getattr(args, "max_disp", 768.0)) / (
                (2 ** int(args.n_downsample)) * max(self.compression_ratios))
        self.bin_initializer = MultiRangeBinInitializer(
            feature_dim=low_dim,
            num_bins=int(getattr(args, "pivno_num_init_bins", 48)),
            max_offset=float(max_offset),
            corr_scale=float(getattr(args, "pivno_init_corr_scale", 10.0)),
        )
        self.num_init_ranges = len(self.compression_ratios)
        finest_dim = int(args.hidden_dims[-1])
        self.initial_weight_head = nn.Sequential(
            nn.Conv2d(finest_dim + low_dim + 2 * self.num_init_ranges,
                      finest_dim, 3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(finest_dim, self.num_init_ranges, 1))
        nn.init.zeros_(self.initial_weight_head[-1].weight)
        nn.init.zeros_(self.initial_weight_head[-1].bias)

        self.last_scale_weight_mean = None
        self.last_scale_gate_entropy = None
        self.last_hypothesis_weight_mean = None
        self.last_initial_confidence_mean = None
        self.finetune_stage = str(getattr(args, "pivno_bins_stage", "init"))
        self.set_finetune_stage(self.finetune_stage)

    @property
    def _new_module_prefixes(self):
        return ("bin_initializer.", "initial_weight_head.")

    def set_finetune_stage(self, stage):
        if stage not in self.FINETUNE_STAGES:
            raise ValueError(f"unknown bins stage: {stage}")
        self.finetune_stage = stage
        for name, parameter in self.named_parameters():
            parameter.requires_grad = (
                stage == "joint" or name.startswith(self._new_module_prefixes))

    def optimizer_parameter_groups(self, args):
        new, pretrained = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (new if name.startswith(self._new_module_prefixes)
                 else pretrained).append(parameter)
        if self.finetune_stage == "init":
            return [{"params": new, "lr": float(args.lr)}]
        old_lr = getattr(args, "pivno_bins_pretrained_lr", None)
        old_lr = float(0.1 * args.lr if old_lr is None else old_lr)
        return [{"params": pretrained, "lr": old_lr},
                {"params": new, "lr": float(args.lr)}]

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def scale_gate_metrics(self):
        if self.last_scale_weight_mean is None:
            return {}
        result = {
            f"scale_gate_weight_r{r}": float(w)
            for r, w in zip(self.compression_ratios,
                            self.last_scale_weight_mean.cpu().tolist())}
        result["scale_gate_entropy"] = float(self.last_scale_gate_entropy.cpu())
        return result

    def initialization_metrics(self):
        if self.last_hypothesis_weight_mean is None:
            return {}
        result = {
            f"init_weight_r{r}": float(w)
            for r, w in zip(self.compression_ratios,
                            self.last_hypothesis_weight_mean.cpu().tolist())}
        result["init_confidence_mean"] = float(
            self.last_initial_confidence_mean.cpu())
        return result

    def _apply_scale_gate(self, left, correlation):
        with torch.cuda.amp.autocast(enabled=False):
            left_unit = F.normalize(left.float(), dim=1, eps=1e-6)
            mean_corr = correlation.float().mean(dim=3)
        gate_input = torch.cat(
            [left_unit.to(left.dtype), mean_corr.flatten(1, 2).to(left.dtype)], 1)
        logits = self.scale_gate(gate_input)
        with torch.cuda.amp.autocast(enabled=False):
            weights = torch.softmax(logits.float(), dim=1)
        detached = weights.detach()
        self.last_scale_weight_mean = detached.mean(dim=(0, 2, 3))
        entropy = -(detached * detached.clamp_min(1e-8).log()).sum(dim=1)
        self.last_scale_gate_entropy = entropy.mean() / math.log(3)
        return weights

    def _refine_warp_feature(self, left, right_pyramid, disparity):
        sampled = sample_right_feature_pyramid(
            right_pyramid, disparity, offsets=self.warp_offsets,
            compression_ratios=self.compression_ratios,
            padding_mode="zeros", align_corners=True)
        b, _, c, h, w = sampled.shape
        s, k = 3, int(self.warp_offsets.numel())
        sampled = sampled.reshape(b, s, k, c, h, w)
        correlation = group_correlation(left, sampled, groups=8) #[B,S,K,G,H,W]
        weights = self._apply_scale_gate(left, correlation)
        encoded = self.sample_match_encoder(
            correlation.reshape(b * s * k, 8, h, w)
        ).reshape(b, s, k, 16, h, w)
        encoded = encoded * (s * weights[:, :, None, None]).to(encoded.dtype)
        fused = self.refine_right_fuse(encoded.flatten(1, 3))
        with torch.cuda.amp.autocast(enabled=False):
            return self.refine_rope(fused.float())

    def _bound_disparity(self, disparity):
        maximum = float(getattr(self.args, "max_disp", 768.0)) / (
            2 ** int(self.args.n_downsample))
        return disparity.float().clamp(0.0, maximum)

    def _upsample_initial_disparity(self, disparity):
        factor = 2 ** int(self.args.n_downsample)
        return factor * F.interpolate(
            disparity, scale_factor=factor, mode="bilinear", align_corners=False)

    def upsample_flow(self, flow, mask):
        b, c, h, w = flow.shape
        factor = 2 ** int(self.args.n_downsample)
        mask = torch.softmax(mask.view(b, 1, 9, factor, factor, h, w), dim=2)
        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(b, c, 9, 1, 1, h, w)
        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(b, c, factor * h, factor * w)

    def forward(self, image1, image2, iters=12, scale_iters=None,
                test_mode=False):
        del scale_iters
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # Keep the image/matching feature path in FP32. With joint
            # training, rare augmented crops can overflow MobileNet/FPN in
            # FP16 before GradScaler gets a chance to skip the update.
            with torch.cuda.amp.autocast(enabled=False):
                fmap1, fmap2 = self.pivno(
                    (image1.float() / 255.0).contiguous(),
                    (image2.float() / 255.0).contiguous())
                left = self.low_channel(fmap1.float())
                right = self.low_channel(fmap2.float())
                right_half, right_quarter = self.right_width_compressor(right)
            context = self.cnet(
                ((image1 - self.mean) / self.std).float(),
                num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(item[0]) for item in context]
            raw_context = [torch.relu(item[1]) for item in context]

            right_pyramid = (right, right_half, right_quarter)

            init_disps, _, init_confidence, _ = self.bin_initializer(
                left, right_pyramid)
            init_disps = [self._bound_disparity(item) for item in init_disps]
            init_input = torch.cat(
                [raw_context[0], left] + init_disps + [init_confidence], dim=1)
            hypothesis_weights = torch.softmax(
                self.initial_weight_head(init_input).float(), dim=1)
            disparity = sum(
                hypothesis_weights[:, i:i + 1] * branch
                for i, branch in enumerate(init_disps))
            disparity = self._bound_disparity(disparity)
            initial_prediction = self._upsample_initial_disparity(disparity)
            self.last_hypothesis_weight_mean = (
                hypothesis_weights.detach().mean(dim=(0, 2, 3)))
            self.last_initial_confidence_mean = init_confidence.detach().mean()

            if self.training and self.finetune_stage == "init":
                return [initial_prediction], []

            inp_list = [
                list(conv(item).chunk(3, dim=1))
                for item, conv in zip(raw_context, self.context_zqr_convs)]
            predictions = []
            prediction = initial_prediction
            for _ in range(iters):
                disparity = disparity.detach()
                warp_feature = self._refine_warp_feature(
                    left, right_pyramid, disparity)
                net_list, up_mask, delta = self.update_block(
                    net_list, inp_list, warp_feature, disparity,
                    iter32=self.args.n_gru_layers == 3,
                    iter16=self.args.n_gru_layers >= 2)
                disparity = self._bound_disparity(
                    disparity + delta.clamp(
                        -self.max_delta_disp, self.max_delta_disp))
                prediction = self.upsample_flow(disparity, up_mask)
                predictions.append(prediction)

            if test_mode:
                return prediction
            return [initial_prediction], predictions
