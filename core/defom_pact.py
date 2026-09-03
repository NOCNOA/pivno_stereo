"""Single-current-disparity PACT DEFOM-Stereo model.

This is the large-disparity, memory-bounded counterpart of
core.defom_cor_ga. It intentionally lives in a separate file so the original
dense-volume model and its checkpoints remain untouched.

All disparity states use quarter-resolution feature-pixel units. A full-range
1/16 volume and the left monocular prior produce one coarse map, 1/8 stereo
features correct it once or a few times, and 1/4 recurrent matching performs
the final local refinement.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from core.adaptive_tube import (
    AdaptiveLocalCorrBlock,
    AdaptiveMidScaleRefiner,
    GlobalCoarseGwcAggregator,
    confidence_to_radius,
)
from core.extractor import BasicEncoder2, DefomEncoder, MultiBasicEncoder
from core.update_pru import MultiPromptUpdateBlock
from core.utils.utils import get_danv2_io_size, upflow

try:
    autocast = torch.cuda.amp.autocast
except AttributeError:
    class autocast:  # pragma: no cover - compatibility for old PyTorch
        def __init__(self, enabled):
            self.enabled = enabled

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False


class MonoCoarseInitializer(nn.Module):
    """Fuse coarse stereo and monocular inverse depth into one disparity map."""

    def __init__(self, max_disp: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.max_disp_quarter = float(max_disp) / 4.0
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            # log scale, shift, fusion confidence and residual
            nn.Conv2d(hidden_channels, 4, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coarse_disp: torch.Tensor, mono_disp: torch.Tensor,
                coarse_std: torch.Tensor, entropy: torch.Tensor,
                margin: torch.Tensor) -> Dict[str, torch.Tensor]:
        scale_norm = max(self.max_disp_quarter, 1.0)
        coarse_norm = coarse_disp.float() / scale_norm
        mono_norm = mono_disp.float() / scale_norm
        inputs = torch.cat(
            (
                coarse_norm,
                mono_norm,
                mono_norm - coarse_norm,
                coarse_std.float() / scale_norm,
                entropy.float(),
                margin.float(),
            ),
            dim=1,
        )
        raw_scale, raw_shift, raw_confidence, raw_residual = self.net(inputs).chunk(4, dim=1)
        mono_scale = torch.exp(0.25 * torch.tanh(raw_scale.float()))
        mono_shift = 0.25 * scale_norm * torch.tanh(raw_shift.float())
        mono_calibrated = (
            mono_disp.float() * mono_scale + mono_shift
        ).clamp(0.0, self.max_disp_quarter - 1.0e-3)

        disagreement = (mono_calibrated - coarse_disp.float()).abs()
        correction_radius = (disagreement + coarse_std.float()).clamp(1.0, min(32.0, self.max_disp_quarter))
        mono_confidence = torch.sigmoid(raw_confidence.float())
        correction = mono_confidence * correction_radius * torch.tanh(raw_residual.float())
        disp0 = (coarse_disp.float() + correction).clamp(0.0, self.max_disp_quarter - 1.0e-3)
        initial_radius = (coarse_std.float() + disagreement).clamp(1.0, 16.0)
        return {
            "disp0": disp0,
            "mono_calibrated": mono_calibrated,
            "mono_confidence": mono_confidence,
            "mono_scale": mono_scale,
            "mono_shift": mono_shift,
            "initial_radius": initial_radius,
        }


class DINOContextAdapter(nn.Module):
    """Lightweight trainable adapter from DINO context to the PRU width."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      groups=out_channels, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        context = self.proj(feature.float())
        return context + self.residual_scale * self.refine(context)


class DEFOMStereo(nn.Module):
    """Large-disparity DEFOM-Stereo with PACT compressed matching."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.register_buffer("mean", torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer("std", torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)

        self.max_disp_full = int(args.max_disp)
        self.max_disp = self.max_disp_full // 4
        # Disparity states use quarter-resolution units.  `wide9` retains
        # nine correlation candidates while placing its outer samples farther
        # away; radius bounds and the mid correction scale are checkpointed.
        self.pact_sampling_layout = str(
            getattr(args, "pact_sampling_layout", "legacy9")
        )
        self.min_radius = float(getattr(args, "pact_min_radius", 1.0))
        self.max_radius = float(getattr(args, "pact_max_radius", 8.0))
        self.mid_delta_scale = float(getattr(args, "pact_mid_delta_scale", 1.0))
        self.checkpoint_dynamic_corr = bool(getattr(args, "pact_checkpoint_corr", True))
        self.debug_finite = bool(getattr(args, "pact_debug_finite", False))
        # Candidate correlation normalizes sampled features with a very small
        # epsilon.  Keeping the trainable stereo path in FP32 prevents its
        # backward derivative from overflowing at zero/OOB samples under AMP.
        # The frozen Depth-Anything encoder remains under autocast, which
        # retains most of AMP's memory and throughput benefit.
        self.fp32_stereo = bool(getattr(args, "pact_fp32_stereo", True))
        self.fp32_update = bool(getattr(args, "pact_fp32_update", True))
        self.mid_refine_iters = int(getattr(args, "pact_mid_refine_iters", 1))

        self.defomencoder = DefomEncoder(args.dinov2_encoder, idepth_scale=args.idepth_scale)
        self.fnet = BasicEncoder2(self.defomencoder.out_dim, output_dim=[64, 128, 192, 256],
                                  norm_fn="instance", downsample=args.n_downsample)
        context_dims = args.hidden_dims
        self.cnet = MultiBasicEncoder(self.defomencoder.out_dim, output_dim=[args.hidden_dims, context_dims],
                                      norm_fn=args.context_norm, downsample=args.n_downsample)
        self.context_adapter = DINOContextAdapter(
            self.defomencoder.out_dim, args.hidden_dims[0]
        )

        self.coarse_volume = GlobalCoarseGwcAggregator(max_disp=self.max_disp_full, num_groups=8, hidden_channels=16)
        self.initializer = MonoCoarseInitializer(self.max_disp_full)
        self.mid_refiner = AdaptiveMidScaleRefiner(
            max_disp=self.max_disp_full,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            sampling_layout=self.pact_sampling_layout,
            delta_scale=self.mid_delta_scale,
        )
        self.corr_fn = AdaptiveLocalCorrBlock(
            max_disp=self.max_disp_full,
            num_groups=8,
            sampling_layout=self.pact_sampling_layout,
        )
        self.update_block = MultiPromptUpdateBlock(args, hidden_dim=args.hidden_dims[0], feat_dim=16,
                                                   volume_dim=1, warp_feat_dim=64,
                                                   harddim=self.corr_fn.output_channels,
                                                   adaptive_search=True, use_base_selection=False)

        # Anchor-free PACT consumes the 1/4, 1/8 and 1/16 feature outputs but
        # only needs context features at 1/4. Keep compatibility branches in
        # the state dict, but exclude them from the optimizer and DDP.
        self._freeze_unused_compatibility_branches()

    def _freeze_unused_compatibility_branches(self) -> None:
        unused_modules = (
            self.fnet.layer6,
            self.fnet.out64,
            self.cnet.outputs08[1],
            self.cnet.outputs16[1],
            self.cnet.outputs32[1],
        )
        for module in unused_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def _check_finite(self, stage: str, **tensors: torch.Tensor) -> None:
        """Fail at the first non-finite PACT stage when diagnostics are enabled."""
        if not self.debug_finite:
            return
        for name, tensor in tensors.items():
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            finite = torch.isfinite(tensor)
            if bool(finite.all()):
                continue
            finite_values = tensor.detach()[finite]
            value_range = (
                "no finite values"
                if finite_values.numel() == 0
                else f"finite_range=[{finite_values.min().item():.6g}, "
                f"{finite_values.max().item():.6g}]"
            )
            raise FloatingPointError(
                f"PACT non-finite tensor at {stage}: {name}, "
                f"shape={tuple(tensor.shape)}, nan={torch.isnan(tensor).sum().item()}, "
                f"inf={torch.isinf(tensor).sum().item()}, {value_range}"
            )

    def upsample_field(self, field, mask, value_scale=1.0):
        """Convexly upsample a field, optionally scaling its values."""
        batch, channels, height, width = field.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(batch, 1, 9, factor, factor, height, width)
        mask = torch.softmax(mask.float(), dim=2)
        up_field = F.unfold(
            float(value_scale) * field.float(), [3, 3], padding=1
        )
        up_field = up_field.view(
            batch, channels, 9, 1, 1, height, width
        )
        up_field = torch.sum(mask * up_field, dim=2)
        up_field = up_field.permute(0, 1, 4, 2, 5, 3)
        return up_field.reshape(
            batch, channels, factor * height, factor * width
        )

    def upsample_flow(self, flow, mask):
        """Convexly upsample quarter-resolution disparity to full resolution."""
        factor = 2 ** self.args.n_downsample
        return self.upsample_field(flow, mask, value_scale=factor)

    def _dynamic_corr(self, fmap1, fmap2, disp, radius, coarse_std,
                      coarse_entropy, coarse_margin, return_aux=False):
        inputs = (fmap1, fmap2, disp, radius, coarse_std, coarse_entropy, coarse_margin)
        if return_aux:
            # Checkpointing only accepts the regular tensor output contract.
            # Diagnostic evaluation runs without gradients, so bypassing it
            # here does not alter the training path.
            return self.corr_fn(*inputs, return_aux=True)
        if (
            self.training
            and self.checkpoint_dynamic_corr
            and torch.is_grad_enabled()
        ):
            return checkpoint(self.corr_fn, *inputs, use_reentrant=False)
        return self.corr_fn(*inputs)

    def _mid_refine(self, fmap1, fmap2, disp, radius, coarse_std,
                    coarse_entropy, coarse_margin, return_aux=False):
        inputs = (fmap1, fmap2, disp, radius, coarse_std,
                  coarse_entropy, coarse_margin)
        if return_aux:
            return self.mid_refiner(*inputs, return_aux=True)
        if self.training and self.checkpoint_dynamic_corr and torch.is_grad_enabled():
            return checkpoint(self.mid_refiner, *inputs, use_reentrant=False)
        return self.mid_refiner(*inputs)

    @staticmethod
    def _make_auxiliary(coarse: Dict[str, torch.Tensor],
                        initializer: Dict[str, torch.Tensor],
                        refined_disp: torch.Tensor,
                        mid_confidence: torch.Tensor,
                        include_diagnostics: bool = False) -> Dict[str, object]:
        batch = coarse["logits"].shape[0]
        coarse_bins = coarse["logits"].shape[1]
        auxiliary: Dict[str, object] = {
            "coarse_logits": coarse["logits"],
            "coarse_posterior": coarse["posterior"].float(),
            # Keep a batch dimension so nn.DataParallel gathers candidates
            # along the same axis as logits instead of concatenating D.
            "coarse_candidates": (
                torch.arange(
                    coarse_bins,
                    device=coarse["logits"].device,
                    dtype=torch.float32,
                )
                .view(1, coarse_bins, 1, 1)
                .expand(batch, -1, -1, -1)
                * 16.0
            ),
            "coarse_disp": coarse["coarse_disp"].float() * 4.0,
            "coarse_valid": coarse["valid"],
            "pre_refine_disp": initializer["disp0"].float() * 4.0,
            "init_disp": refined_disp.float() * 4.0,
            "mono_calibrated": initializer["mono_calibrated"].float() * 4.0,
            "mono_confidence": initializer["mono_confidence"].float(),
            "mid_confidence": mid_confidence.float(),
            "delta_info_preds": [],
        }
        if include_diagnostics:
            auxiliary.update(
                {
                    "coarse_peak1": coarse["refined_peak_bins"][:, 0:1].float() * 4.0,
                    "coarse_peak2": coarse["refined_peak_bins"][:, 1:2].float() * 4.0,
                    "coarse_peak_scores": coarse["peak_scores"].float(),
                    "coarse_peak_valid": coarse["peak_valid"],
                    "search_radius_preds": [],
                    "update_confidence_preds": [],
                    "candidate_valid_ratio_preds": [],
                }
            )
        return auxiliary

    def forward(self, image1, image2, iters=12, scale_iters=3, test_mode=False,
                return_init_disp=False, return_mono_disp=False, return_aux=False,
                pact_ablation="full", return_diagnostics=False):
        _, _, height, width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()
        danv2_io_sizes = get_danv2_io_size(height, width, self.args.n_downsample)

        # Depth Anything is frozen and runs without autograd.  It can remain
        # under AMP without exposing trainable feature gradients to FP16.
        with autocast(enabled=self.args.mixed_precision):
            d_features, dfeat1, dfeat2, mono_disp1, _ = self.defomencoder(
                [image1, image2], danv2_io_sizes, return_idepth=True)

        # Run all trainable stereo feature/context/matching preparation in
        # FP32 by default.  In particular, fmap1/fmap2 must be created in
        # FP32 rather than merely cast afterwards: casting an FP16 activation
        # to FP32 would still cast its backward gradient back to FP16 and can
        # overflow before it reaches the feature encoder.
        with autocast(
            enabled=self.args.mixed_precision and not self.fp32_stereo
        ):
            fmap1, fmap2 = self.fnet([image1, image2], [dfeat1.float(), dfeat2.float()], num_layers=3)
            cnet_list = self.cnet(
                image1, [feature.float() for feature in d_features],
                output_counts=(1, 1, 1),
            )
            net_list = [torch.tanh(item[0]).float() for item in cnet_list]
            context = self.context_adapter(d_features[0]).float()

            coarse = self.coarse_volume(fmap1[2], fmap2[2])
        # self._check_finite(
        #     "encoder",
        #     mono_disp=mono_disp1,
        #     dfeat1=dfeat1,
        #     dfeat2=dfeat2,
        #     fmap1_4=fmap1[0],
        #     fmap2_4=fmap2[0],
        #     context=context,
        #     net_4=net_list[0],
        #     net_8=net_list[1],
        #     net_16=net_list[2],
        # )
        # self._check_finite(
        #     "coarse",
        #     logits=coarse["logits"],
        #     posterior=coarse["posterior"],
        #     coarse_disp=coarse["coarse_disp"],
        #     std=coarse["std"],
        #     entropy=coarse["entropy"],
        #     margin=coarse["margin"],
        # )

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
            coarse_disp_8, mono_disp_8, coarse_std_8,
            coarse_entropy_8, coarse_margin_8,
        )
        disp_8 = initialization["disp0"].float()
        radius_8 = initialization["initial_radius"].float().clamp(
            self.min_radius, self.max_radius
        )
        # self._check_finite(
        #     "initialization",
        #     disp0=disp_8,
        #     mono_calibrated=initialization["mono_calibrated"],
        #     mono_confidence=initialization["mono_confidence"],
        #     radius=radius_8,
        # )

        mid_confidence = torch.zeros_like(disp_8)
        mid_match_peak = torch.zeros_like(disp_8)
        refine_rounds = 0 if pact_ablation == "no_mid_refine" else self.mid_refine_iters
        for _ in range(refine_rounds):
            mid_result = self._mid_refine(
                fmap1[1], fmap2[1], disp_8, radius_8, coarse_std_8,
                coarse_entropy_8, coarse_margin_8,
                return_aux=return_diagnostics,
            )
            if return_diagnostics:
                disp_8, radius_8, mid_confidence, mid_aux = mid_result
                mid_match_peak = mid_aux["match_weights"].max(
                    dim=1, keepdim=True
                ).values
            else:
                disp_8, radius_8, mid_confidence = mid_result
        disp = F.interpolate(disp_8, **interp_args).clamp(
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
        self._check_finite(
            "mid-scale refinement", disp=disp, radius=search_radius,
            confidence=mid_confidence,
        )
        mono_disp_up = upflow(mono_calibrated, factor=2 ** self.args.n_downsample, sacle=True)

        aux = self._make_auxiliary(
            coarse, initialization, disp, mid_confidence,
            include_diagnostics=return_diagnostics,
        ) if return_aux else None
        if aux is not None and return_diagnostics:
            aux["mid_match_peak"] = mid_match_peak.float()
        init_disp_up = upflow(disp, factor=2 ** self.args.n_downsample, sacle=True)
        disp_predictions: List[torch.Tensor] = [init_disp_up]

        for iteration in range(iters):
            disp = disp.detach()
            radius_base = search_radius.detach()
            corr_result = self._dynamic_corr(
                fmap1[0], fmap2[0], disp, search_radius,
                coarse_std, coarse_entropy, coarse_margin,
                return_aux=return_diagnostics,
            )
            if not return_diagnostics:
                corr, galerkin_feat = corr_result
                corr_aux = None
            else:
                corr, galerkin_feat, corr_aux = corr_result
                # Radius is stored in full-resolution pixel units; confidence
                # and valid ratio are dimensionless.  Keep these at 1/4
                # spatial resolution and let the evaluator resize them.
                aux["search_radius_preds"].append(search_radius.float() * 4.0)
                aux["candidate_valid_ratio_preds"].append(
                    corr_aux["valid"].float().mean(dim=1, keepdim=True)
                )
            # self._check_finite(f"iteration {iteration} matching", corr=corr,
            #                    galerkin_feat=galerkin_feat)
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
            with autocast(
                enabled=self.args.mixed_precision and not self.fp32_update
            ):
                net_list, raw_delta, up_mask, delta_info = self.update_block(
                    net_list, corr, update_disp, update_context,
                    update_galerkin, update_mono_prompt,
                )
            # self._check_finite(
            #     f"iteration {iteration} update",
            #     raw_delta=raw_delta,
            #     up_mask=up_mask,
            #     delta_info=delta_info,
            #     net_4=net_list[0],
            #     net_8=net_list[1],
            #     net_16=net_list[2],
            # )
            local_delta = search_radius * torch.tanh(raw_delta.float())
            disp = (disp + local_delta).clamp(0.0, float(self.max_disp) - 1.0e-3)
            confidence = torch.softmax(
                delta_info.float()[:, :2], dim=1
            )[:, 1:2]
            if return_diagnostics:
                aux["update_confidence_preds"].append(confidence)
            next_radius = confidence_to_radius(
                confidence, self.min_radius, self.max_radius
            )
            if pact_ablation == "fixed_radius":
                next_radius = radius_base

            # self._check_finite(
            #     f"iteration {iteration} state",
            #     disp=disp,
            #     radius=next_radius,
            # )

            search_radius = next_radius

            if test_mode and iteration < iters - 1:
                continue
            if up_mask is None:
                disp_up = upflow(disp, factor=2 ** self.args.n_downsample, sacle=True)
                delta_info_up = F.interpolate(
                    delta_info.float(), size=(height, width), mode="bilinear",
                    align_corners=True,
                )
            else:
                disp_up = self.upsample_flow(disp, up_mask)
                delta_info_up = self.upsample_field(
                    delta_info, up_mask, value_scale=1.0
                )
            if aux is not None:
                aux["delta_info_preds"].append(delta_info_up)
            self._check_finite(
                f"iteration {iteration} output",
                disp_up=disp_up,
                delta_info_up=delta_info_up,
            )
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

        if return_aux:
            return result, aux
        return result


__all__ = ["DEFOMStereo", "DINOContextAdapter"]
