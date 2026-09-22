"""Gated-GRU3 PIVNO model with multi-range correlation-bin initialization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIVNO.models.sronet import BasicEncoder256
from core.pivno_models.defom_pivno_gated_gru3 import (
    DEFOMStereo as GatedGRU3DEFOMStereo,
)


class PIVNOFeatureExtractor(nn.Module):
    """The PIVNO image encoder without its attention, GRU, or disparity head.

    Keeping the encoder under ``pivno.snet`` preserves the corresponding key
    names in completed ``defom_pivno_gated_gru3`` checkpoints.
    """

    def __init__(self, input_channels=3):
        super().__init__()
        self.input_channels = int(input_channels)
        self.snet = BasicEncoder256(
            output_dim=64,
            norm_fn="instance",
            dropout=0.0,
            input_channels=self.input_channels,
        )

    def forward(self, image1, image2):
        if image1.shape != image2.shape:
            raise ValueError(
                "PIVNO extractor inputs must have equal shapes, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
        if image1.ndim != 4 or image1.shape[1] != self.input_channels:
            raise ValueError(
                "PIVNO extractor expects [B,C,H,W] with "
                f"C={self.input_channels}, got {tuple(image1.shape)}"
            )
        return self.snet([image1, image2])


class MultiRangeBinInitializer(nn.Module):
    """Regress three disparities from global row correlation bins.

    Unlike the DEFOM CorGA variant, this initializer deliberately has no
    monocular-depth input. Each branch uses the PIVNO features alone.
    """

    COMPRESSION_RATIOS = (1, 2, 4)

    def __init__(
        self,
        feature_dim=48,
        num_bins=48,
        max_offset=48.0,
        corr_scale=10.0,
    ):
        super().__init__()
        if int(num_bins) < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}")
        if float(max_offset) <= 0:
            raise ValueError(
                f"max_offset must be positive, got {max_offset}"
            )

        self.num_bins = int(num_bins)
        self.corr_scale = float(corr_scale)
        self.compression_ratios = self.COMPRESSION_RATIOS
        self.bin_width = float(max_offset) / self.num_bins
        offsets = (
            torch.arange(self.num_bins, dtype=torch.float32)
            * self.bin_width
        )
        ratios = torch.tensor(self.compression_ratios, dtype=torch.float32)
        self.register_buffer("bin_offsets", offsets, persistent=False)
        self.register_buffer(
            "bin_disparities",
            ratios[:, None] * offsets[None, :],
            persistent=False,
        )

        # There is intentionally no extra mono-idepth channel here.
        self.logit_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.num_bins, feature_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, self.num_bins, 3, padding=1),
            )
            for _ in self.compression_ratios
        ])
        for refiner in self.logit_refiners:
            nn.init.zeros_(refiner[-1].weight)
            nn.init.zeros_(refiner[-1].bias)

    @staticmethod
    def _require_finite(name, tensor, branch_index=None):
        if torch.isfinite(tensor).all():
            return
        suffix = "" if branch_index is None else f" in branch {branch_index}"
        raise FloatingPointError(
            f"non-finite {name}{suffix}: "
            f"nan={torch.isnan(tensor).sum().item()}, "
            f"inf={torch.isinf(tensor).sum().item()}"
        )

    def _match_logits(self, left_feature, right_feature, ratio):
        batch, channels, height, width = left_feature.shape
        if right_feature.shape[:3] != (batch, channels, height):
            raise ValueError(
                "incompatible left/right feature shapes: "
                f"{tuple(left_feature.shape)} and "
                f"{tuple(right_feature.shape)}"
            )
        right_width = right_feature.shape[-1]
        with torch.cuda.amp.autocast(enabled=False):
            left = F.normalize(left_feature.float(), dim=1, eps=1e-6)
            right = F.normalize(right_feature.float(), dim=1, eps=1e-6)
            all_pairs = torch.einsum("bchw,bchv->bhwv", left, right)

            x_left = torch.arange(
                width, device=left.device, dtype=torch.float32
            ).view(1, 1, width, 1)
            offsets = self.bin_offsets.view(1, 1, 1, self.num_bins)
            center_offset = (float(ratio) - 1.0) / 2.0
            sample_x = (
                (x_left - center_offset) / float(ratio) - offsets
            )
            valid = (
                (sample_x >= 0.0)
                & (sample_x <= float(right_width - 1))
            )
            sample_x = sample_x.clamp(
                0.0, float(max(right_width - 1, 0))
            )
            index0 = sample_x.floor().long()
            index1 = (index0 + 1).clamp(
                max=max(right_width - 1, 0)
            )
            alpha = sample_x - index0.float()
            gather_shape = (batch, height, width, self.num_bins)
            value0 = torch.gather(
                all_pairs, -1, index0.expand(gather_shape)
            )
            value1 = torch.gather(
                all_pairs, -1, index1.expand(gather_shape)
            )
            logits = (
                (1.0 - alpha) * value0 + alpha * value1
            ).permute(0, 3, 1, 2).contiguous()
        valid = valid.permute(0, 3, 1, 2).expand(
            batch, self.num_bins, height, width
        )
        return logits, valid

    def forward(self, left_feature, right_feature_pyramid):
        if len(right_feature_pyramid) != len(self.compression_ratios):
            raise ValueError(
                f"expected {len(self.compression_ratios)} right features, "
                f"got {len(right_feature_pyramid)}"
            )
        self._require_finite("left_feature", left_feature)

        branch_disps = []
        branch_logits = []
        branch_confidences = []
        branch_valid = []
        with torch.cuda.amp.autocast(enabled=False):
            for branch_index, (right_feature, ratio, refiner) in enumerate(
                zip(
                    right_feature_pyramid,
                    self.compression_ratios,
                    self.logit_refiners,
                )
            ):
                self._require_finite(
                    "right_feature", right_feature, branch_index
                )
                raw_logits, valid = self._match_logits(
                    left_feature, right_feature, ratio
                )
                refiner_input = raw_logits.masked_fill(~valid, -1.0)
                refiner_delta = refiner(refiner_input.float())
                logits = (
                    self.corr_scale * raw_logits + refiner_delta
                ).masked_fill(~valid, -1e4)
                probability = torch.softmax(logits.float(), dim=1)
                probability = probability * valid.float()
                probability = probability / probability.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-6)
                bins = self.bin_disparities[branch_index].view(
                    1, self.num_bins, 1, 1
                )
                branch_disp = torch.sum(
                    probability * bins, dim=1, keepdim=True
                )
                self._require_finite(
                    "branch_disp", branch_disp, branch_index
                )
                branch_disps.append(branch_disp)
                branch_logits.append(logits)
                branch_confidences.append(probability.amax(dim=1))
                branch_valid.append(valid)

        return (
            branch_disps,
            torch.stack(branch_logits, dim=1),
            torch.stack(branch_confidences, dim=1),
            torch.stack(branch_valid, dim=1),
        )


class DEFOMStereo(GatedGRU3DEFOMStereo):
    """Gated GRU3 refinement initialized by multi-range disparity bins."""

    MODEL_VARIANT = "defom_pivno_gated_gru3_bins"
    BASE_MODEL_VARIANT = "defom_pivno_gated_gru3"
    INITIALIZATION_MODE = "global_row_corr_bins_1x2x4x_no_mono"
    FINETUNE_STAGES = ("init", "joint")

    def __init__(self, args):
        super().__init__(args)

        # Discard PIVNO's attention/GRU/disparity decoder. The wrapper keeps
        # only snet and preserves its old checkpoint prefix, pivno.snet.*.
        self.pivno = PIVNOFeatureExtractor(input_channels=3)

        max_offset = getattr(args, "pivno_bins_max_offset", None)
        if max_offset is None:
            full_max_disp = float(getattr(args, "max_disp", 768.0))
            max_offset = full_max_disp / (
                (2 ** int(args.n_downsample))
                * max(self.compression_ratios)
            )
        self.bin_initializer = MultiRangeBinInitializer(
            feature_dim=self.low_channel.out_channels,
            num_bins=int(getattr(args, "pivno_num_init_bins", 48)),
            max_offset=float(max_offset),
            corr_scale=float(getattr(args, "pivno_init_corr_scale", 10.0)),
        )
        self.num_init_ranges = len(
            self.bin_initializer.compression_ratios
        )
        finest_context_dim = int(args.hidden_dims[-1])
        aggregation_channels = (
            finest_context_dim
            + self.low_channel.out_channels
            + 2 * self.num_init_ranges
        )
        self.initial_weight_head = nn.Sequential(
            nn.Conv2d(
                aggregation_channels,
                finest_context_dim,
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                finest_context_dim,
                self.num_init_ranges,
                1,
            ),
        )
        nn.init.zeros_(self.initial_weight_head[-1].weight)
        nn.init.zeros_(self.initial_weight_head[-1].bias)

        self.finetune_stage = str(
            getattr(args, "pivno_bins_stage", "init")
        )
        self.set_finetune_stage(self.finetune_stage)
        self.last_hypothesis_weight_mean = None
        self.last_initial_confidence_mean = None

    @property
    def _new_module_prefixes(self):
        return ("bin_initializer.", "initial_weight_head.")

    def set_finetune_stage(self, stage):
        stage = str(stage)
        if stage not in self.FINETUNE_STAGES:
            raise ValueError(
                f"pivno_bins_stage must be one of {self.FINETUNE_STAGES}, "
                f"got {stage!r}"
            )
        self.finetune_stage = stage
        for name, parameter in self.named_parameters():
            is_new = name.startswith(self._new_module_prefixes)
            parameter.requires_grad = stage == "joint" or is_new

    def optimizer_parameter_groups(self, args):
        new_parameters = []
        pretrained_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(self._new_module_prefixes):
                new_parameters.append(parameter)
            else:
                pretrained_parameters.append(parameter)

        if self.finetune_stage == "init":
            if not new_parameters or pretrained_parameters:
                raise RuntimeError(
                    "init stage must train only the new bin modules"
                )
            return [{"params": new_parameters, "lr": float(args.lr)}]

        pretrained_lr = getattr(
            args, "pivno_bins_pretrained_lr", None
        )
        pretrained_lr = float(
            0.1 * args.lr if pretrained_lr is None else pretrained_lr
        )
        return [
            {"params": pretrained_parameters, "lr": pretrained_lr},
            {"params": new_parameters, "lr": float(args.lr)},
        ]

    def initialization_metrics(self):
        if self.last_hypothesis_weight_mean is None:
            return {}
        metrics = {
            f"init_weight_r{ratio}": float(weight)
            for ratio, weight in zip(
                self.compression_ratios,
                self.last_hypothesis_weight_mean.cpu().tolist(),
            )
        }
        metrics["init_confidence_mean"] = float(
            self.last_initial_confidence_mean.cpu()
        )
        return metrics

    def _bound_disparity(self, disparity):
        max_low_disp = float(getattr(self.args, "max_disp", 768.0)) / (
            2 ** int(self.args.n_downsample)
        )
        return disparity.float().clamp(min=0.0, max=max_low_disp)

    def _upsample_initial_disparity(self, disparity):
        factor = 2 ** int(self.args.n_downsample)
        return factor * F.interpolate(
            disparity,
            scale_factor=factor,
            mode="bilinear",
            align_corners=False,
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
                "stereo images must have equal shapes, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
        if iters < 1:
            raise ValueError(f"iters must be at least 1, got {iters}")

        pivno_image1 = self._to_pivno_rgb(image1).contiguous().float()
        pivno_image2 = self._to_pivno_rgb(image2).contiguous().float()
        context_image = ((image1 - self.mean) / self.std).contiguous().float()

        fmap1_4, fmap2_4 = self.pivno(pivno_image1, pivno_image2)
        del pivno_image1, pivno_image2

        cnet_list = self.cnet(
            context_image,
            num_layers=self.args.n_gru_layers,
        )
        del context_image
        net_list = [torch.tanh(item[0]) for item in cnet_list]
        raw_context_list = [torch.relu(item[1]) for item in cnet_list]

        fmap1_low = self.low_channel(fmap1_4)
        fmap2_low = self.low_channel(fmap2_4)
        del fmap1_4, fmap2_4
        fmap2_half, fmap2_quarter = self.right_width_compressor(fmap2_low)
        right_feature_pyramid = (
            fmap2_low,
            fmap2_half,
            fmap2_quarter,
        )
        del fmap2_low, fmap2_half, fmap2_quarter

        init_disps, init_logits, init_confidence, init_valid = (
            self.bin_initializer(fmap1_low, right_feature_pyramid)
        )
        init_disps = [self._bound_disparity(item) for item in init_disps]
        aggregation_input = torch.cat(
            [raw_context_list[0], fmap1_low]
            + init_disps
            + [init_confidence],
            dim=1,
        )
        hypothesis_logits = self.initial_weight_head(
            aggregation_input
        ).float()
        hypothesis_weights = torch.softmax(hypothesis_logits, dim=1)
        disp = sum(
            hypothesis_weights[:, index:index + 1] * branch_disp
            for index, branch_disp in enumerate(init_disps)
        )
        disp = self._bound_disparity(disp)
        initial_prediction = self._upsample_initial_disparity(disp)

        detached_weights = hypothesis_weights.detach()
        self.last_hypothesis_weight_mean = detached_weights.mean(
            dim=(0, 2, 3)
        )
        self.last_initial_confidence_mean = (
            init_confidence.detach().mean()
        )

        # Stage one optimizes only the fused initialization. Avoid running a
        # frozen recurrent network whose detached updates cannot train it.
        if self.training and self.finetune_stage == "init":
            return [initial_prediction], []

        inp_list = [
            list(conv(context).chunk(3, dim=1))
            for context, conv in zip(
                raw_context_list,
                self.context_zqr_convs,
            )
        ]
        del cnet_list, raw_context_list, aggregation_input
        del hypothesis_logits, init_logits, init_valid

        disp_predictions = []
        disp_up = initial_prediction
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
            disp = self._bound_disparity(disp + delta_disp)
            disp_up = self.upsample_flow(disp, up_mask)
            disp_predictions.append(disp_up)

        if test_mode:
            return disp_up
        return [initial_prediction], disp_predictions
