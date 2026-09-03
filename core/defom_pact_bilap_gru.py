"""Isolated bimodal-Laplace plus recurrent refinement model for portable PACT.

The legacy :mod:`core.defom_pact` implementation is intentionally left
untouched. This variant starts from PACT-SMD, keeps its bimodal initialization,
and adds multi-width right-feature matching to recurrent 1/4 refinement.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from core.adaptive_tube_bilap_gru import BiLapMultiScaleRightCorrBlock
from core.defom_pact import DEFOMStereo as PACTDEFOMStereo, autocast
from core.update_bilap_gru import BiLapUpdateBlock
from core.utils.utils import get_danv2_io_size, upflow


class D0BimodalLaplaceHead(nn.Module):
    """Predict two Laplace disparity modes from one shared 1/4 feature map.

    Disparity-valued inputs and outputs use quarter-resolution pixel units.
    The first mode is a conservative correction around the original PACT d0;
    the second mode has an uncertainty-adaptive range for boundary recovery.
    """

    FEATURE_CHANNELS = 64
    STAT_CHANNELS = 7
    HIDDEN_CHANNELS = 32

    def __init__(
        self,
        max_disp: int,
        local_radius: float = 4.0,
        broad_radius_max: float = 64.0,
        scale_min: float = 0.25,
        scale_max: float = 16.0,
        mode_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.max_disp_quarter = float(max_disp) / 4.0
        self.local_radius = float(local_radius)
        self.broad_radius_max = min(
            float(broad_radius_max), self.max_disp_quarter
        )
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.mode_threshold = float(mode_threshold)

        self.image_branch = nn.Sequential(
            nn.Conv2d(self.FEATURE_CHANNELS, 16, 1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.statistics_branch = nn.Sequential(
            nn.Conv2d(self.STAT_CHANNELS, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.shared_trunk = nn.Sequential(
            nn.Conv2d(32, self.HIDDEN_CHANNELS, 3, padding=1, bias=False),
            nn.GroupNorm(8, self.HIDDEN_CHANNELS),
            nn.GELU(),
            nn.Conv2d(
                self.HIDDEN_CHANNELS,
                self.HIDDEN_CHANNELS,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, self.HIDDEN_CHANNELS),
            nn.GELU(),
        )
        self.mean_head = nn.Conv2d(self.HIDDEN_CHANNELS, 2, 1)
        self.scale_head = nn.Conv2d(self.HIDDEN_CHANNELS, 2, 1)
        self.weight_head = nn.Conv2d(self.HIDDEN_CHANNELS, 1, 1)
        self._initialize_output_heads()

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return math.log(math.expm1(value))

    def _initialize_output_heads(self) -> None:
        # Mode one reproduces the original d0.  Mode two starts very close to
        # it but has independent feature-dependent perturbations, avoiding an
        # exactly symmetric mixture while preserving the safe MAP output.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.normal_(self.mean_head.weight[1:2], mean=0.0, std=1.0e-3)

        nn.init.zeros_(self.scale_head.weight)
        with torch.no_grad():
            self.scale_head.bias.copy_(
                torch.tensor(
                    [self._inverse_softplus(1.0), self._inverse_softplus(4.0)]
                )
            )
        nn.init.zeros_(self.weight_head.weight)
        nn.init.constant_(self.weight_head.bias, math.log(0.9 / 0.1))

    @staticmethod
    def _check_field(name: str, value: torch.Tensor, reference: torch.Tensor) -> None:
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"{name} must be [B,1,H,W], got {tuple(value.shape)}")
        if value.shape != reference.shape:
            raise ValueError(
                f"PACT-SMD field mismatch: d0={tuple(reference.shape)}, "
                f"{name}={tuple(value.shape)}"
            )

    def forward(
        self,
        fmap1_4: torch.Tensor,
        d0_old: torch.Tensor,
        mono_4: torch.Tensor,
        coarse_std_4: torch.Tensor,
        coarse_entropy_4: torch.Tensor,
        coarse_margin_4: torch.Tensor,
        mid_confidence_4: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if fmap1_4.ndim != 4 or fmap1_4.shape[1] != self.FEATURE_CHANNELS:
            raise ValueError(
                "fmap1_4 must be [B,64,H,W], got " f"{tuple(fmap1_4.shape)}"
            )
        for name, value in (
            ("mono_4", mono_4),
            ("coarse_std_4", coarse_std_4),
            ("coarse_entropy_4", coarse_entropy_4),
            ("coarse_margin_4", coarse_margin_4),
            ("mid_confidence_4", mid_confidence_4),
        ):
            self._check_field(name, value, d0_old)
        if fmap1_4.shape[0] != d0_old.shape[0] or fmap1_4.shape[-2:] != d0_old.shape[-2:]:
            raise ValueError("PACT-SMD image/statistics features do not align")

        scale_norm = max(self.max_disp_quarter, 1.0)
        statistics = torch.cat(
            (
                d0_old.float() / scale_norm,
                mono_4.float() / scale_norm,
                (mono_4.float() - d0_old.float()) / scale_norm,
                coarse_std_4.float() / scale_norm,
                coarse_entropy_4.float(),
                coarse_margin_4.float(),
                mid_confidence_4.float(),
            ),
            dim=1,
        )
        image_feature = self.image_branch(fmap1_4.float())
        statistics_feature = self.statistics_branch(statistics)
        shared = self.shared_trunk(
            torch.cat((image_feature, statistics_feature), dim=1)
        )

        raw_means = self.mean_head(shared).float()
        raw_scales = self.scale_head(shared).float()
        alpha = self.weight_head(shared).float()

        broad_radius = (
            self.local_radius
            + 2.0 * coarse_std_4.float()
            + (mono_4.float() - d0_old.float()).abs()
        ).clamp(self.local_radius, self.broad_radius_max)
        mu1 = d0_old.float() + self.local_radius * torch.tanh(raw_means[:, 0:1])
        mu2 = d0_old.float() + broad_radius * torch.tanh(raw_means[:, 1:2])
        means = torch.cat((mu1, mu2), dim=1).clamp(
            0.0, self.max_disp_quarter - 1.0e-3
        )
        scales = F.softplus(raw_scales).clamp(self.scale_min, self.scale_max)
        mixture_logits = torch.cat((alpha, torch.zeros_like(alpha)), dim=1)
        weights = torch.softmax(mixture_logits, dim=1)
        peak_scores = torch.log(weights.clamp_min(1.0e-8)) - torch.log(scales)
        selected_mode = (
            peak_scores[:, 1:2] - peak_scores[:, 0:1] > self.mode_threshold
        )
        d0_new = torch.where(selected_mode, means[:, 1:2], means[:, 0:1])

        return {
            "d0": d0_new,
            "means": means,
            "scales": scales,
            "mixture_logits": mixture_logits,
            "weights": weights,
            "peak_scores": peak_scores,
            "selected_mode": selected_mode,
            "broad_radius": broad_radius,
        }


class DEFOMStereo(PACTDEFOMStereo):
    """PACT with two recurrent Laplace modes and multi-width matching."""

    MODEL_VARIANT = "pact_bilap_gru_v8_two_width_c32_gwc4_edge_nll"

    def __init__(self, args) -> None:
        if int(args.n_downsample) != 2:
            raise ValueError("PACT-BiLap-GRU is defined on the 1/4 disparity grid")
        super().__init__(args)
        self.bilap_ablation = str(getattr(args, "bilap_ablation", "dual_symmetric_interaction"))
        self.bilap_init = str(getattr(args, "bilap_init", "smd"))
        self.bilap_init_delta = float(getattr(args, "bilap_init_delta", 2.0))
        self.bilap_init_scale = float(getattr(args, "bilap_init_scale", 2.0))
        self.bilap_lookup_mode = str(getattr(args, "bilap_lookup_mode", "scale_aware"))
        self.bilap_q_min = float(getattr(args, "bilap_q_min", 1.0))
        self.bilap_q_max = float(getattr(args, "bilap_q_max", 4.0))
        self.bilap_q_scale = float(getattr(args, "bilap_q_scale", 0.5))
        self.checkpoint_bilap_update = bool(getattr(args, "bilap_checkpoint_update", True))
        self.d0_smd_head = D0BimodalLaplaceHead(max_disp=self.max_disp_full, mode_threshold=float(getattr(args, "pact_smd_mode_threshold", 0.5)))
        self.corr_fn = BiLapMultiScaleRightCorrBlock(max_disp=self.max_disp_full, num_groups=4, feature_channels=32, input_channels=64, sampling_layout=self.pact_sampling_layout)
        self.update_block = BiLapUpdateBlock(hidden_dim=args.hidden_dims[0], corr_dim=self.corr_fn.output_channels, aligned_dim=32, context_dim=args.hidden_dims[0], separate_mode_gru=bool(getattr(args, "bilap_separate_mode_gru", False)), interaction=self.bilap_ablation != "dual_no_interaction", up_factor=4, max_disp_quarter=float(self.max_disp))
        self._configure_from_scratch_training()

    def _configure_from_scratch_training(self):
        for parameter in self.parameters():
            parameter.requires_grad = True
        for parameter in self.defomencoder.depth_anything.pretrained.parameters():
            parameter.requires_grad = False
        for parameter in self.defomencoder.depth_anything.depth_head.parameters():
            parameter.requires_grad = False
        self._freeze_unused_compatibility_branches()
        for block in (self.cnet.conv08, self.cnet.conv16, self.cnet.conv32):
            for name in ("norm2", "norm3"):
                module = getattr(block, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
        if self.bilap_ablation == "dual_no_interaction":
            for module in (self.update_block.mode_summary, self.update_block.interaction_encoder):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def _dynamic_corr_modes(self, left_feature, right_full, right_half, means, radius, return_aux=False):
        inputs = (left_feature, right_full, right_half, means, radius)
        if return_aux:
            return self.corr_fn.forward_modes(*inputs, return_aux=True)
        return self.corr_fn.forward_modes(*inputs)

    def _recurrent_update(self, mode_hidden, global8, global16, corr, galerkin_feat, means, log_scales, logits, mono_calibrated, context):
        inputs = (mode_hidden, global8, global16, corr, galerkin_feat, means, log_scales, logits, mono_calibrated, context)
        if self.training and self.checkpoint_bilap_update and torch.is_grad_enabled():
            return checkpoint(self.update_block, *inputs, use_reentrant=False)
        return self.update_block(*inputs)

    @staticmethod
    def _augment_auxiliary(
        auxiliary: Dict[str, object],
        d0_old: torch.Tensor,
        smd: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        auxiliary["init_disp_old"] = d0_old.float() * 4.0
        auxiliary["init_disp"] = smd["d0"].float() * 4.0
        auxiliary["smd_means"] = smd["means"].float() * 4.0
        auxiliary["smd_scales"] = smd["scales"].float() * 4.0
        auxiliary["smd_mixture_logits"] = smd["mixture_logits"].float()
        auxiliary["smd_weights"] = smd["weights"].float()
        auxiliary["smd_peak_scores"] = smd["peak_scores"].float()
        auxiliary["smd_selected_mode"] = smd["selected_mode"].float()
        auxiliary["smd_broad_radius"] = smd["broad_radius"].float() * 4.0
        return auxiliary

    def _upsample_distribution(self, means, log_scales, logits, up_mask):
        scales = log_scales.exp()
        means_up = self.upsample_field(means, up_mask, value_scale=4.0)
        scales_up = self.upsample_field(scales, up_mask, value_scale=4.0).clamp_min(1.0e-4)
        logits_up = self.upsample_field(logits, up_mask, value_scale=1.0)
        logits_up = logits_up - logits_up.mean(dim=1, keepdim=True)
        weights_up = torch.softmax(logits_up.float(), dim=1)
        peak_scores = torch.log(weights_up.clamp_min(1.0e-8)) - torch.log(scales_up)
        selected = peak_scores.argmax(dim=1, keepdim=True)
        disparity = torch.gather(means_up, 1, selected)
        log_scales_up = torch.log(scales_up)
        mixture_mean = (weights_up * means_up).sum(dim=1, keepdim=True)
        entropy = -(weights_up * torch.log(weights_up.clamp_min(1.0e-8))).sum(dim=1, keepdim=True)
        separation = (means_up[:, 0:1] - means_up[:, -1:]).abs()
        distribution = {"means": means_up, "scales": scales_up, "log_scales": log_scales_up, "mixture_logits": logits_up, "weights": weights_up, "peak_scores": peak_scores, "selected_mode": selected, "disp": disparity, "map_disp": disparity, "mixture_mean": mixture_mean, "separation": separation, "entropy": entropy}
        if means_up.shape[1] == 2:
            distribution.update({"mu_1": means_up[:, 0:1], "mu_2": means_up[:, 1:2], "b_1": scales_up[:, 0:1], "b_2": scales_up[:, 1:2], "log_b_1": log_scales_up[:, 0:1], "log_b_2": log_scales_up[:, 1:2], "pi_1": weights_up[:, 0:1], "pi_2": weights_up[:, 1:2]})
        return distribution

    def forward(
        self,
        image1,
        image2,
        iters=12,
        scale_iters=3,
        test_mode=False,
        return_init_disp=False,
        return_mono_disp=False,
        return_aux=False,
        pact_ablation="full",
        return_diagnostics=False,
        return_distribution=True,
    ):
        if iters < 1:
            raise ValueError("PACT-BiLap-GRU requires iters >= 1")
        _, _, height, width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()
        danv2_io_sizes = get_danv2_io_size(height, width, self.args.n_downsample)

        with autocast(enabled=self.args.mixed_precision):
            d_features, dfeat1, dfeat2, mono_disp1, _ = self.defomencoder(
                [image1, image2], danv2_io_sizes, return_idepth=True
            )

        with autocast(enabled=self.args.mixed_precision and not self.fp32_stereo):
            fmap1, fmap2 = self.fnet(
                [image1, image2], [dfeat1.float(), dfeat2.float()], num_layers=3
            )
            fmap1_4, fmap1_8, fmap1_16 = fmap1
            fmap2_4, fmap2_8, fmap2_16 = fmap2
            cnet_list = self.cnet(
                image1,
                [feature.float() for feature in d_features],
                output_counts=(1, 1, 1),
            )
            net_list = [torch.tanh(item[0]).float() for item in cnet_list]
            context = self.context_adapter(d_features[0]).float()
            coarse = self.coarse_volume(fmap1_16, fmap2_16)

        del fmap1_16, fmap2_16, fmap1, fmap2

        state_size = fmap1_4.shape[-2:]
        mid_size = fmap1_8.shape[-2:]
        interp_args = {"size": state_size, "mode": "bilinear", "align_corners": True}
        mid_interp = {"size": mid_size, "mode": "bilinear", "align_corners": True}
        mono_disp_8 = F.interpolate(mono_disp1.float(), **mid_interp)
        coarse_disp_8 = F.interpolate(coarse["coarse_disp"].float(), **mid_interp)
        coarse_std_8 = F.interpolate(coarse["std"].float(), **mid_interp)
        coarse_entropy_8 = F.interpolate(coarse["entropy"].float(), **mid_interp)
        coarse_margin_8 = F.interpolate(coarse["margin"].float(), **mid_interp)
        initialization = self.initializer(
            coarse_disp_8,
            mono_disp_8,
            coarse_std_8,
            coarse_entropy_8,
            coarse_margin_8,
        )
        disp_8 = initialization["disp0"].float()
        radius_8 = initialization["initial_radius"].float().clamp(
            self.min_radius, self.max_radius
        )

        mid_confidence = torch.zeros_like(disp_8)
        mid_match_peak = torch.zeros_like(disp_8)
        refine_rounds = 0 if pact_ablation == "no_mid_refine" else self.mid_refine_iters
        for _ in range(refine_rounds):
            mid_result = self._mid_refine(
                fmap1_8,
                fmap2_8,
                disp_8,
                radius_8,
                coarse_std_8,
                coarse_entropy_8,
                coarse_margin_8,
                return_aux=return_diagnostics,
            )
            if return_diagnostics:
                disp_8, radius_8, mid_confidence, mid_aux = mid_result
                mid_match_peak = mid_aux["match_weights"].max(
                    dim=1, keepdim=True
                ).values
            else:
                disp_8, radius_8, mid_confidence = mid_result

        del fmap1_8, fmap2_8

        d0_old = F.interpolate(disp_8, **interp_args).clamp(
            0.0, float(self.max_disp) - 1.0e-3
        )
        search_radius = F.interpolate(radius_8, **interp_args).clamp(
            self.min_radius, self.max_radius
        )
        mono_calibrated = F.interpolate(
            initialization["mono_calibrated"].float(), **interp_args
        )
        coarse_std = F.interpolate(coarse["std"].float(), **interp_args)
        coarse_entropy = F.interpolate(coarse["entropy"].float(), **interp_args)
        coarse_margin = F.interpolate(coarse["margin"].float(), **interp_args)
        mid_confidence_4 = F.interpolate(mid_confidence.float(), **interp_args)

        smd = self.d0_smd_head(fmap1_4, d0_old, mono_calibrated, coarse_std, coarse_entropy, coarse_margin, mid_confidence_4)
        means = smd["means"].float()
        log_scales = torch.log(smd["scales"].float().clamp_min(1.0e-4))
        logits = smd["mixture_logits"].float()
        if self.bilap_init == "symmetric":
            means = torch.cat((d0_old - self.bilap_init_delta, d0_old + self.bilap_init_delta), dim=1).clamp(0.0, float(self.max_disp) - 1.0e-3)
            log_scales = torch.full_like(means, math.log(self.bilap_init_scale))
            logits = torch.zeros_like(means)
        if self.bilap_ablation == "single_laplace":
            means, log_scales, logits = means[:, :1], log_scales[:, :1], logits[:, :1]

        mono_disp_up = upflow(
            mono_calibrated, factor=2 ** self.args.n_downsample, sacle=True
        )
        init_disp_up = upflow(smd["d0"].float(), factor=2 ** self.args.n_downsample, sacle=True)
        aux = None
        if return_aux:
            base_aux = self._make_auxiliary(coarse, initialization, d0_old, mid_confidence, include_diagnostics=return_diagnostics)
            aux = dict(base_aux)
            aux["base_aux"] = base_aux
            self._augment_auxiliary(aux, d0_old, smd)
            aux["bilap_predictions"] = []
            if return_diagnostics:
                aux["mid_match_peak"] = mid_match_peak.float()

        left_match = self.corr_fn.build_left_feature(fmap1_4)
        right_full, right_half = self.corr_fn.build_right_pyramid(fmap2_4)
        global8, global16 = net_list[1].float(), net_list[2].float()
        mode_hidden = self.update_block.initialize_modes(net_list[0].float(), means, log_scales, logits, mono_calibrated)
        distribution_predictions = []

        for iteration in range(iters):
            means = means.detach()
            log_scales = log_scales.detach()
            logits = logits.detach()
            scales = log_scales.exp()
            search_radius = (self.bilap_q_min + self.bilap_q_scale * scales.detach()).clamp(self.bilap_q_min, self.bilap_q_max)
            if self.bilap_lookup_mode == "fixed":
                search_radius = torch.full_like(search_radius, 2.0)
            corr_result = self._dynamic_corr_modes(left_match, right_full, right_half, means, search_radius, return_aux=return_diagnostics)
            if return_diagnostics:
                corr, galerkin_feat, corr_aux = corr_result
                aux["search_radius_preds"].append(search_radius.float() * 4.0)
                aux["candidate_valid_ratio_preds"].append(corr_aux["valid"].float().mean(dim=2))
            else:
                corr, galerkin_feat = corr_result
            if self.fp32_update:
                mode_hidden, global8, global16, corr, galerkin_feat = mode_hidden.float(), global8.float(), global16.float(), corr.float(), galerkin_feat.float()
            with autocast(enabled=self.args.mixed_precision and not self.fp32_update):
                mode_hidden, global8, global16, raw_update, up_mask = self._recurrent_update(mode_hidden, global8, global16, corr, galerkin_feat, means, log_scales, logits, mono_calibrated, context)
            means = (means + 16.0 * torch.tanh(raw_update[:, :, 0])).clamp(0.0, float(self.max_disp) - 1.0e-3)
            log_scales = (log_scales + 0.5 * torch.tanh(raw_update[:, :, 1])).clamp(math.log(0.25), math.log(16.0))
            logits = logits + 2.0 * torch.tanh(raw_update[:, :, 2])
            logits = (logits - logits.mean(dim=1, keepdim=True)).clamp(-8.0, 8.0)
            if test_mode and iteration < iters - 1 and not return_aux:
                continue
            prediction = self._upsample_distribution(means, log_scales, logits, up_mask)
            distribution_predictions.append(prediction)
            if aux is not None:
                aux["bilap_predictions"].append(prediction)

        if test_mode:
            final_distribution = distribution_predictions[-1] if distribution_predictions else self._upsample_distribution(means, log_scales, logits, up_mask)
            outputs = [final_distribution if return_distribution else final_distribution["disp"]]
            if return_init_disp:
                outputs.append(init_disp_up)
            if return_mono_disp:
                outputs.append(mono_disp_up)
            result = outputs[0] if len(outputs) == 1 else tuple(outputs)
        else:
            result = distribution_predictions
        return (result, aux) if return_aux else result


__all__ = ["D0BimodalLaplaceHead", "DEFOMStereo"]
