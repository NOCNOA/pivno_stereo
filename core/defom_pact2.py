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

from core.adaptive_tube_pact2 import (
    PACT2DualScaleGwcBlock,
    PACT2FixedRadiusMidScaleRefiner,
    PACT2GlobalCoarseGwcAggregator,
)
from core.extractor import DefomEncoder
from core.extractor_pact2 import PACT2ContextEncoder, PACT2FeatureEncoder
from core.update_pru_pact2 import PACT2FixedRadiusUpdateBlock
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
        return {
            "disp0": disp0,
            "mono_calibrated": mono_calibrated,
            "mono_confidence": mono_confidence,
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

    MODEL_VARIANT = "pact2_mobilenetv2_dual_gwc_fixed_r4"
    FEATURE_BACKBONE = "mobilenetv2_100"
    FIXED_RADIUS_QUARTER = 4.0
    SAMPLING_LAYOUT = "legacy9"

    def __init__(self, args):
        super().__init__()
        self.args = args
        if int(args.n_downsample) != 2:
            raise ValueError("PACT2 requires n_downsample=2")
        if int(args.n_gru_layers) != 3:
            raise ValueError("PACT2 requires exactly three GRU levels")
        hidden_dims = tuple(int(dim) for dim in args.hidden_dims)
        if len(hidden_dims) != 3 or len(set(hidden_dims)) != 1:
            raise ValueError(
                "PACT2 requires three equal hidden dimensions, got "
                f"{list(hidden_dims)}"
            )
        if int(args.max_disp) <= 0 or int(args.max_disp) % 4 != 0:
            raise ValueError("PACT2 max_disp must be positive and divisible by 4")
        self.register_buffer("mean", torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer("std", torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)

        self.max_disp_full = int(args.max_disp)
        self.max_disp = self.max_disp_full // 4
        # All model states use quarter-resolution disparity units. PACT2 has a
        # fixed integer candidate grid and does not accept adaptive/wide search.
        self.pact_sampling_layout = self.SAMPLING_LAYOUT
        requested_layout = str(
            getattr(args, "pact_sampling_layout", self.SAMPLING_LAYOUT)
        )
        if requested_layout != self.SAMPLING_LAYOUT:
            raise ValueError(
                "PACT2 fixed-radius matching requires pact_sampling_layout="
                f"{self.SAMPLING_LAYOUT!r}, got {requested_layout!r}"
            )
        self.checkpoint_corr = bool(getattr(args, "pact_checkpoint_corr", True))
        # Candidate correlation normalizes sampled features with a very small
        # epsilon.  Keeping the trainable stereo path in FP32 prevents its
        # backward derivative from overflowing at zero/OOB samples under AMP.
        # The frozen Depth-Anything encoder remains under autocast, which
        # retains most of AMP's memory and throughput benefit.
        self.fp32_stereo = bool(getattr(args, "pact_fp32_stereo", True))
        self.fp32_update = bool(getattr(args, "pact_fp32_update", True))
        self.mid_refine_iters = int(getattr(args, "pact_mid_refine_iters", 1))
        if self.mid_refine_iters < 0:
            raise ValueError("pact_mid_refine_iters must be non-negative")

        self.defomencoder = DefomEncoder(args.dinov2_encoder, idepth_scale=args.idepth_scale)
        self.fnet = PACT2FeatureEncoder(
            self.defomencoder.out_dim,
            output_dim=[64, 128, 192],
            norm_fn="instance",
            downsample=args.n_downsample,
        )
        self.cnet = PACT2ContextEncoder(
            self.defomencoder.out_dim,
            output_dim=args.hidden_dims,
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
        )
        self.context_adapter = DINOContextAdapter(
            self.defomencoder.out_dim, args.hidden_dims[0]
        )

        self.coarse_volume = PACT2GlobalCoarseGwcAggregator(
            max_disp=self.max_disp_full,
            num_groups=8,
            hidden_channels=16,
        )
        self.initializer = MonoCoarseInitializer(self.max_disp_full)
        self.mid_refiner = PACT2FixedRadiusMidScaleRefiner(
            max_disp=self.max_disp_full,
        )
        self.corr_fn = PACT2DualScaleGwcBlock(
            max_disp=self.max_disp_full,
            num_groups=8,
        )
        self.update_block = PACT2FixedRadiusUpdateBlock(
            args,
            hidden_dim=args.hidden_dims[0],
            warp_feat_dim=64,
            harddim=self.corr_fn.output_channels,
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

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

    def _fixed_corr(self, fmap1, fmap2, disp, coarse_gwc, coarse_valid):
        inputs = (fmap1, fmap2, disp, coarse_gwc, coarse_valid)
        if (
            self.training
            and self.checkpoint_corr
            and torch.is_grad_enabled()
        ):
            return checkpoint(self.corr_fn, *inputs, use_reentrant=False)
        return self.corr_fn(*inputs)

    def _mid_refine(self, fmap1, fmap2, disp, coarse_std,
                    coarse_entropy, coarse_margin):
        inputs = (fmap1, fmap2, disp, coarse_std,
                  coarse_entropy, coarse_margin)
        if self.training and self.checkpoint_corr and torch.is_grad_enabled():
            return checkpoint(self.mid_refiner, *inputs, use_reentrant=False)
        return self.mid_refiner(*inputs)

    @staticmethod
    def _make_auxiliary(coarse: Dict[str, torch.Tensor],
                        initializer: Dict[str, torch.Tensor],
                        refined_disp: torch.Tensor) -> Dict[str, object]:
        return {
            "coarse_logits": coarse["logits"],
            "coarse_disp": coarse["coarse_disp"].float() * 4.0,
            "coarse_valid": coarse["valid"],
            "init_disp": refined_disp.float() * 4.0,
            "mono_calibrated": initializer["mono_calibrated"].float() * 4.0,
            "mono_confidence": initializer["mono_confidence"].float(),
        }

    def forward(self, image1, image2, iters=12, scale_iters=3, test_mode=False,
                return_aux=False):
        if test_mode and iters < 1:
            raise ValueError("PACT2 test_mode requires iters >= 1")
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
            fmap1, fmap2 = self.fnet(
                [image1, image2], [dfeat1.float(), dfeat2.float()]
            )
            cnet_list = self.cnet(
                image1, [feature.float() for feature in d_features]
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
            coarse_disp_8, mono_disp_8, coarse_std_8,
            coarse_entropy_8, coarse_margin_8,
        )
        disp_8 = initialization["disp0"].float()
        for _ in range(self.mid_refine_iters):
            disp_8 = self._mid_refine(
                fmap1[1], fmap2[1], disp_8, coarse_std_8,
                coarse_entropy_8, coarse_margin_8,
            )
        disp = F.interpolate(disp_8, **interp_args).clamp(
            0.0, float(self.max_disp) - 1.0e-3
        )
        mono_calibrated = F.interpolate(
            initialization["mono_calibrated"].float(), **interp_args
        )
        aux = self._make_auxiliary(
            coarse, initialization, disp,
        ) if return_aux else None
        disp_predictions: List[torch.Tensor] = []
        if not test_mode:
            disp_predictions.append(
                upflow(disp, factor=2 ** self.args.n_downsample, sacle=True)
            )

        for iteration in range(iters):
            disp = disp.detach()
            corr, galerkin_feat = self._fixed_corr(
                fmap1[0], fmap2[0], disp,
                coarse["gwc_volume"], coarse["valid"],
            )
            if self.fp32_update:
                net_list = [state.float() for state in net_list]
                corr = corr.float()
                update_disp = disp.float()
                update_context = context.float()
                update_galerkin = galerkin_feat.float()
                update_mono_prompt = mono_calibrated.float()
            else:
                update_disp = disp
                update_context = context
                update_galerkin = galerkin_feat
                update_mono_prompt = mono_calibrated
            with autocast(
                enabled=self.args.mixed_precision and not self.fp32_update
            ):
                net_list, raw_delta, up_mask = self.update_block(
                    net_list, corr, update_disp, update_context,
                    update_galerkin, update_mono_prompt,
                )
            local_delta = self.FIXED_RADIUS_QUARTER * torch.tanh(
                raw_delta.float()
            )
            disp = (disp + local_delta).clamp(0.0, float(self.max_disp) - 1.0e-3)
            if test_mode and iteration < iters - 1:
                continue
            disp_up = self.upsample_flow(disp, up_mask)
            if not test_mode:
                disp_predictions.append(disp_up)

        if test_mode:
            result = disp_up
        else:
            result = disp_predictions

        if return_aux:
            return result, aux
        return result


__all__ = ["DEFOMStereo", "DINOContextAdapter"]
