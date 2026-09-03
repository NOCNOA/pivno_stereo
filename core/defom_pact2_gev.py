"""PACT2 with scale-matched coarse and dynamic geometry aggregation."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.adaptive_tube_pact2_gev import (
    PACT2GevDualScaleGwcBlock,
    PACT2GevGlobalCoarseAggregator,
)
from core.defom_pact2 import DEFOMStereo as PACT2DEFOMStereo
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


class MonoCoarseGevInitializer(nn.Module):
    """Fuse scalar coarse statistics, mono disparity and 1/16 GEV geometry."""

    def __init__(
        self,
        max_disp: int,
        geometry_channels: int = 16,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.max_disp_quarter = float(max_disp) / 4.0
        self.geometry_channels = int(geometry_channels)
        self.net = nn.Sequential(
            nn.Conv2d(6 + self.geometry_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            # log scale, shift, mono fusion confidence and residual
            nn.Conv2d(hidden_channels, 4, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        coarse_disp: torch.Tensor,
        mono_disp: torch.Tensor,
        coarse_std: torch.Tensor,
        entropy: torch.Tensor,
        margin: torch.Tensor,
        coarse_geometry: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if coarse_geometry.shape[:2] != (
            coarse_disp.shape[0], self.geometry_channels
        ) or coarse_geometry.shape[-2:] != coarse_disp.shape[-2:]:
            raise ValueError(
                "coarse geometry must be [B,16,H,W] and align with coarse disparity"
            )
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
                coarse_geometry.float(),
            ),
            dim=1,
        )
        raw_scale, raw_shift, raw_confidence, raw_residual = self.net(
            inputs
        ).chunk(4, dim=1)
        mono_scale = torch.exp(0.25 * torch.tanh(raw_scale.float()))
        mono_shift = 0.25 * scale_norm * torch.tanh(raw_shift.float())
        mono_calibrated = (
            mono_disp.float() * mono_scale + mono_shift
        ).clamp(0.0, self.max_disp_quarter - 1.0e-3)

        disagreement = (mono_calibrated - coarse_disp.float()).abs()
        correction_radius = (disagreement + coarse_std.float()).clamp(
            1.0, min(32.0, self.max_disp_quarter)
        )
        mono_confidence = torch.sigmoid(raw_confidence.float())
        correction = (
            mono_confidence
            * correction_radius
            * torch.tanh(raw_residual.float())
        )
        disp0 = (coarse_disp.float() + correction).clamp(
            0.0, self.max_disp_quarter - 1.0e-3
        )
        return {
            "disp0": disp0,
            "mono_calibrated": mono_calibrated,
            "mono_confidence": mono_confidence,
        }


class DEFOMStereo(PACT2DEFOMStereo):
    """Isolated PACT2-GEV variant; the existing PACT2 path is unchanged."""

    MODEL_VARIANT = "pact2_gev_mobilenetv2_fixed_r4_v1"
    GEV_MODES = PACT2GevDualScaleGwcBlock.MODES

    def __init__(self, args) -> None:
        gev_mode = str(getattr(args, "pact_gev_mode", "dual"))
        if gev_mode not in self.GEV_MODES:
            raise ValueError(
                f"pact_gev_mode must be one of {self.GEV_MODES}, got {gev_mode!r}"
            )
        super().__init__(args)
        self.pact_gev_mode = gev_mode
        self.coarse_volume = PACT2GevGlobalCoarseAggregator(
            max_disp=self.max_disp_full,
            num_groups=8,
            feature_channels=192,
        )
        self.initializer = MonoCoarseGevInitializer(self.max_disp_full)
        self.corr_fn = PACT2GevDualScaleGwcBlock(
            max_disp=self.max_disp_full,
            gev_mode=self.pact_gev_mode,
            num_groups=8,
            feature_channels=64,
        )

    def forward(
        self,
        image1,
        image2,
        iters=12,
        scale_iters=3,
        test_mode=False,
        return_aux=False,
    ):
        if test_mode and iters < 1:
            raise ValueError("PACT2-GEV test_mode requires iters >= 1")
        _, _, height, width = image1.shape
        image1 = ((image1 - self.mean) / self.std).contiguous().float()
        image2 = ((image2 - self.mean) / self.std).contiguous().float()
        danv2_io_sizes = get_danv2_io_size(
            height, width, self.args.n_downsample
        )

        with autocast(enabled=self.args.mixed_precision):
            d_features, dfeat1, dfeat2, mono_disp1, _ = self.defomencoder(
                [image1, image2], danv2_io_sizes, return_idepth=True
            )

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
        interp_args = {
            "size": state_size,
            "mode": "bilinear",
            "align_corners": True,
        }
        mid_interp = {
            "size": mid_size,
            "mode": "bilinear",
            "align_corners": True,
        }
        mono_disp_8 = F.interpolate(mono_disp1.float(), **mid_interp)
        coarse_disp_8 = F.interpolate(
            coarse["coarse_disp"].float(), **mid_interp
        )
        coarse_std_8 = F.interpolate(coarse["std"].float(), **mid_interp)
        coarse_entropy_8 = F.interpolate(
            coarse["entropy"].float(), **mid_interp
        )
        coarse_margin_8 = F.interpolate(
            coarse["margin"].float(), **mid_interp
        )
        coarse_geometry_8 = F.interpolate(
            coarse["coarse_geometry_2d"].float(), **mid_interp
        )
        initialization = self.initializer(
            coarse_disp_8,
            mono_disp_8,
            coarse_std_8,
            coarse_entropy_8,
            coarse_margin_8,
            coarse_geometry_8,
        )
        disp_8 = initialization["disp0"].float()
        for _ in range(self.mid_refine_iters):
            disp_8 = self._mid_refine(
                fmap1[1],
                fmap2[1],
                disp_8,
                coarse_std_8,
                coarse_entropy_8,
                coarse_margin_8,
            )
        disp = F.interpolate(disp_8, **interp_args).clamp(
            0.0, float(self.max_disp) - 1.0e-3
        )
        mono_calibrated = F.interpolate(
            initialization["mono_calibrated"].float(), **interp_args
        )
        aux = self._make_auxiliary(
            coarse, initialization, disp
        ) if return_aux else None
        disp_predictions: List[torch.Tensor] = []
        if not test_mode:
            disp_predictions.append(
                upflow(
                    disp,
                    factor=2 ** self.args.n_downsample,
                    sacle=True,
                )
            )

        for iteration in range(iters):
            disp = disp.detach()
            corr, galerkin_feat = self._fixed_corr(
                fmap1[0],
                fmap2[0],
                disp,
                coarse["gwc_volume"],
                coarse["valid"],
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
                    net_list,
                    corr,
                    update_disp,
                    update_context,
                    update_galerkin,
                    update_mono_prompt,
                )
            local_delta = self.FIXED_RADIUS_QUARTER * torch.tanh(
                raw_delta.float()
            )
            disp = (disp + local_delta).clamp(
                0.0, float(self.max_disp) - 1.0e-3
            )
            if test_mode and iteration < iters - 1:
                continue
            disp_up = self.upsample_flow(disp, up_mask)
            if not test_mode:
                disp_predictions.append(disp_up)

        result = disp_up if test_mode else disp_predictions
        if return_aux:
            return result, aux
        return result


__all__ = ["DEFOMStereo", "MonoCoarseGevInitializer"]
