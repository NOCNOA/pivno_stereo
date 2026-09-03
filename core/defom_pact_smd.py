"""Isolated SMD-style bimodal initialization for portable PACT.

The legacy :mod:`core.defom_pact` implementation is intentionally left
untouched.  This variant reuses its feature, matching and recurrent modules,
then inserts a lightweight bimodal Laplace head after the 1/8 initializer has
been resized to the 1/4 recurrent grid.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defom_pact import DEFOMStereo as PACTDEFOMStereo, autocast
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
    """PACT with an isolated SMD-style bimodal d0 prediction head."""

    MODEL_VARIANT = "pact_smd_d0_v1"

    def __init__(self, args) -> None:
        if int(args.n_downsample) != 2:
            raise ValueError("PACT-SMD is defined on the 1/4 disparity grid")
        super().__init__(args)
        # Preserve the base PACT trainability contract before stage-specific
        # freezing. This keeps Depth Anything and unused compatibility
        # branches frozen while allowing a true stereo-from-scratch stage.
        self._base_trainable_parameter_names = frozenset(
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )
        self.pact_smd_stage = str(getattr(args, "pact_smd_stage", "joint"))
        if self.pact_smd_stage not in ("head", "joint", "full"):
            raise ValueError("pact_smd_stage must be 'head', 'joint' or 'full'")
        self.pact_smd_grad_iters = int(getattr(args, "pact_smd_grad_iters", 2))
        if self.pact_smd_grad_iters < 0:
            raise ValueError("pact_smd_grad_iters must be non-negative")
        self.d0_smd_head = D0BimodalLaplaceHead(
            max_disp=self.max_disp_full,
            mode_threshold=float(getattr(args, "pact_smd_mode_threshold", 0.5)),
        )
        self.configure_training_stage(self.pact_smd_stage)

    def configure_training_stage(self, stage: str) -> None:
        if stage not in ("head", "joint", "full"):
            raise ValueError(
                "PACT-SMD training stage must be 'head', 'joint' or 'full'"
            )
        self.pact_smd_stage = stage
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.d0_smd_head.parameters():
            parameter.requires_grad = True
        if stage == "full":
            for name, parameter in self.named_parameters():
                if name in self._base_trainable_parameter_names:
                    parameter.requires_grad = True
        elif stage == "joint":
            for module in (self.mid_refiner, self.update_block):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def optimizer_parameter_groups(self, args):
        groups = [
            {
                "params": [
                    parameter
                    for parameter in self.d0_smd_head.parameters()
                    if parameter.requires_grad
                ],
                "lr": float(args.pact_smd_head_lr),
                "name": "smd_head",
            }
        ]
        if self.pact_smd_stage == "full":
            base_parameters = [
                parameter
                for name, parameter in self.named_parameters()
                if name in self._base_trainable_parameter_names
                and parameter.requires_grad
            ]
            groups.append(
                {
                    "params": base_parameters,
                    "lr": float(args.lr),
                    "name": "smd_full_stereo",
                }
            )
        elif self.pact_smd_stage == "joint":
            adapted = [
                parameter
                for module in (self.mid_refiner, self.update_block)
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            groups.append(
                {
                    "params": adapted,
                    "lr": float(args.pact_smd_adapt_lr),
                    "name": "smd_adapted_recurrent",
                }
            )
        return groups

    def _postprocess_final_prediction(
        self,
        net_4: torch.Tensor,
        disp_4: torch.Tensor,
        mono_4: torch.Tensor,
        fmap1_4: torch.Tensor,
        up_mask: torch.Tensor,
        raw_disp_up: torch.Tensor,
    ):
        """Optional last-iteration hook used by isolated derived models.

        The PACT-SMD implementation itself is deliberately an identity here.
        This keeps existing checkpoints and outputs unchanged while allowing a
        post-GRU experiment to reuse the already computed recurrent tensors and
        convex upsampling mask without copying the complete forward method.
        """

        del net_4, disp_4, mono_4, fmap1_4, up_mask
        return raw_disp_up, {}

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
        return_init_only=False,
    ):
        if iters < 1 and not (self.training and self.pact_smd_stage == "head"):
            raise ValueError("PACT-SMD requires iters >= 1 outside head-only training")
        supported_ablations = {
            "full", "no_mid_refine", "no_mono_prompt", "fixed_radius"
        }
        if pact_ablation not in supported_ablations:
            raise ValueError(
                f"unsupported PACT-SMD ablation {pact_ablation!r}; "
                f"choose from {sorted(supported_ablations)}"
            )
        if return_diagnostics and not return_aux:
            raise ValueError("PACT-SMD diagnostics require return_aux=True")
        if image1.ndim != 4 or image1.shape[1] != 3 or image2.shape != image1.shape:
            raise ValueError(
                "image1/image2 must have identical [B,3,H,W] shapes, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
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
            cnet_list = self.cnet(
                image1,
                [feature.float() for feature in d_features],
                output_counts=(1, 1, 1),
            )
            net_list = [torch.tanh(item[0]).float() for item in cnet_list]
            context = self.context_adapter(d_features[0]).float()
            coarse = self.coarse_volume(fmap1[2], fmap2[2])

        state_size = fmap1[0].shape[-2:]
        mid_size = fmap1[1].shape[-2:]
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
                fmap1[1],
                fmap2[1],
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

        detach_head_inputs = self.training and self.pact_smd_stage == "head"
        smd_inputs = (
            fmap1[0],
            d0_old,
            mono_calibrated,
            coarse_std,
            coarse_entropy,
            coarse_margin,
            mid_confidence_4,
        )
        if detach_head_inputs:
            smd_inputs = tuple(value.detach() for value in smd_inputs)
        smd = self.d0_smd_head(*smd_inputs)
        disp = smd["d0"].float()

        mono_disp_up = upflow(
            mono_calibrated, factor=2 ** self.args.n_downsample, sacle=True
        )
        init_disp_up = upflow(
            disp, factor=2 ** self.args.n_downsample, sacle=True
        )
        aux = None
        if return_aux:
            aux = self._make_auxiliary(
                coarse,
                initialization,
                d0_old,
                mid_confidence,
                include_diagnostics=return_diagnostics,
            )
            self._augment_auxiliary(aux, d0_old, smd)
            if return_diagnostics:
                aux["mid_match_peak"] = mid_match_peak.float()

        if return_init_only:
            return (init_disp_up, aux) if return_aux else init_disp_up

        disp_predictions: List[torch.Tensor] = [init_disp_up]
        if self.training and self.pact_smd_stage == "head":
            result = disp_predictions
            return (result, aux) if return_aux else result

        for iteration in range(iters):
            if not self.training or iteration >= self.pact_smd_grad_iters:
                disp = disp.detach()
                search_radius = search_radius.detach()
                net_list = [state.detach() for state in net_list]
            radius_base = search_radius.detach()
            corr_result = self._dynamic_corr(
                fmap1[0],
                fmap2[0],
                disp,
                search_radius,
                coarse_std,
                coarse_entropy,
                coarse_margin,
                return_aux=return_diagnostics,
            )
            if return_diagnostics:
                corr, galerkin_feat, corr_aux = corr_result
                aux["search_radius_preds"].append(search_radius.float() * 4.0)
                aux["candidate_valid_ratio_preds"].append(
                    corr_aux["valid"].float().mean(dim=1, keepdim=True)
                )
            else:
                corr, galerkin_feat = corr_result

            mono_prompt = disp if pact_ablation == "no_mono_prompt" else mono_calibrated
            if self.fp32_update:
                net_list = [state.float() for state in net_list]
                corr = corr.float()
                update_disp = disp.float()
                update_context = context.float()
                update_galerkin = galerkin_feat.float()
                update_mono_prompt = mono_prompt.float()
            else:
                update_disp = disp
                update_context = context
                update_galerkin = galerkin_feat
                update_mono_prompt = mono_prompt
            with autocast(enabled=self.args.mixed_precision and not self.fp32_update):
                net_list, raw_delta, up_mask, delta_info = self.update_block(
                    net_list,
                    corr,
                    update_disp,
                    update_context,
                    update_galerkin,
                    update_mono_prompt,
                )
            local_delta = search_radius * torch.tanh(raw_delta.float())
            disp = (disp + local_delta).clamp(
                0.0, float(self.max_disp) - 1.0e-3
            )
            confidence = torch.softmax(delta_info.float()[:, :2], dim=1)[:, 1:2]
            if return_diagnostics:
                aux["update_confidence_preds"].append(confidence)
            next_radius = self.min_radius + (
                self.max_radius - self.min_radius
            ) * (1.0 - confidence)
            if pact_ablation == "fixed_radius":
                next_radius = radius_base
            search_radius = next_radius

            if test_mode and iteration < iters - 1:
                continue
            if up_mask is None:
                disp_up = upflow(
                    disp, factor=2 ** self.args.n_downsample, sacle=True
                )
                delta_info_up = F.interpolate(
                    delta_info.float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                disp_up = self.upsample_flow(disp, up_mask)
                delta_info_up = self.upsample_field(
                    delta_info, up_mask, value_scale=1.0
                )
            if iteration == iters - 1:
                disp_up, post_aux = self._postprocess_final_prediction(
                    net_list[0],
                    disp,
                    mono_calibrated,
                    fmap1[0],
                    up_mask,
                    disp_up,
                )
                if aux is not None:
                    aux.update(post_aux)
            if aux is not None:
                aux["delta_info_preds"].append(delta_info_up)
            disp_predictions.append(disp_up)

        if test_mode:
            outputs = [disp_up]
            if return_init_disp:
                outputs.append(init_disp_up)
            if return_mono_disp:
                outputs.append(mono_disp_up)
            result = outputs[0] if len(outputs) == 1 else tuple(outputs)
        else:
            result = disp_predictions
        return (result, aux) if return_aux else result


__all__ = ["D0BimodalLaplaceHead", "DEFOMStereo"]
