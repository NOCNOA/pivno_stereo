"""Frozen-Full PACT-SMD with a bimodal Laplace head after the last GRU.

Only ``final_smd_head`` is trainable.  The head consumes the last quarter-grid
hidden state, disparity, calibrated mono disparity and left stereo feature.
Its selected quarter-grid mode is upsampled with the *frozen* final GRU mask.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defom_pact_smd import DEFOMStereo as PACTSMDDEFOMStereo
from core.utils.utils import upflow


class FinalBimodalLaplaceHead(nn.Module):
    """Predict a two-Laplace residual distribution on the 1/4 grid."""

    NET_CHANNELS = 128
    FMAP_CHANNELS = 64
    STAT_CHANNELS = 3

    def __init__(
        self,
        max_disp: int,
        net_channels: int = NET_CHANNELS,
        local_radius: float = 1.0,
        broad_radius_min: float = 2.0,
        broad_radius_max: float = 32.0,
        scale_min: float = 0.125,
        scale_max: float = 16.0,
        mode_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.net_channels = int(net_channels)
        self.max_disp_quarter = float(max_disp) / 4.0
        self.local_radius = float(local_radius)
        self.broad_radius_min = float(broad_radius_min)
        self.broad_radius_max = min(
            float(broad_radius_max), self.max_disp_quarter
        )
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.mode_threshold = float(mode_threshold)
        if self.local_radius <= 0.0:
            raise ValueError("post-GRU local radius must be positive")
        if not 0.0 < self.broad_radius_min <= self.broad_radius_max:
            raise ValueError("invalid post-GRU broad-radius interval")

        self.net_branch = nn.Sequential(
            nn.Conv2d(self.net_channels, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.fmap_branch = nn.Sequential(
            nn.Conv2d(self.FMAP_CHANNELS, 16, 1, bias=False),
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
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.mean_head = nn.Conv2d(32, 2, 1)
        self.scale_head = nn.Conv2d(32, 2, 1)
        self.weight_head = nn.Conv2d(32, 1, 1)
        self._initialize_output_heads()

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return math.log(math.expm1(value))

    def _initialize_output_heads(self) -> None:
        # Both means initially equal the frozen GRU disparity.  The prior gives
        # mode one a 90% mixture weight, so the hard MAP output is an exact
        # identity before this head receives any training.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        with torch.no_grad():
            self.scale_head.bias.copy_(torch.tensor([
                self._inverse_softplus(0.5),
                self._inverse_softplus(2.0),
            ]))
        nn.init.zeros_(self.weight_head.weight)
        nn.init.constant_(self.weight_head.bias, math.log(0.9 / 0.1))

    @staticmethod
    def _check_scalar_field(
        name: str, value: torch.Tensor, reference: torch.Tensor
    ) -> None:
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"{name} must be [B,1,H,W], got {tuple(value.shape)}")
        if value.shape != reference.shape:
            raise ValueError(
                f"post-GRU field mismatch: disp={tuple(reference.shape)}, "
                f"{name}={tuple(value.shape)}"
            )

    def forward(
        self,
        net_4: torch.Tensor,
        disp_4: torch.Tensor,
        mono_4: torch.Tensor,
        fmap1_4: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._check_scalar_field("mono_4", mono_4, disp_4)
        expected_spatial = disp_4.shape[-2:]
        if (
            net_4.ndim != 4
            or net_4.shape[1] != self.net_channels
            or net_4.shape[0] != disp_4.shape[0]
            or net_4.shape[-2:] != expected_spatial
        ):
            raise ValueError(
                f"net_4 must be [B,{self.net_channels},H,W] aligned with disp, "
                f"got {tuple(net_4.shape)}"
            )
        if (
            fmap1_4.ndim != 4
            or fmap1_4.shape[1] != self.FMAP_CHANNELS
            or fmap1_4.shape[0] != disp_4.shape[0]
            or fmap1_4.shape[-2:] != expected_spatial
        ):
            raise ValueError(
                "fmap1_4 must be [B,64,H,W] aligned with disp, got "
                f"{tuple(fmap1_4.shape)}"
            )

        normalizer = max(self.max_disp_quarter, 1.0)
        statistics = torch.cat((
            disp_4.float() / normalizer,
            mono_4.float() / normalizer,
            (mono_4.float() - disp_4.float()) / normalizer,
        ), dim=1)
        shared = self.shared_trunk(torch.cat((
            self.net_branch(net_4.float()),
            self.fmap_branch(fmap1_4.float()),
            self.statistics_branch(statistics),
        ), dim=1))

        raw_means = self.mean_head(shared).float()
        raw_scales = self.scale_head(shared).float()
        alpha = self.weight_head(shared).float()
        broad_radius = (
            self.broad_radius_min + (mono_4.float() - disp_4.float()).abs()
        ).clamp(self.broad_radius_min, self.broad_radius_max)
        mu1 = disp_4.float() + self.local_radius * torch.tanh(raw_means[:, 0:1])
        mu2 = disp_4.float() + broad_radius * torch.tanh(raw_means[:, 1:2])
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
        disp_new = torch.where(selected_mode, means[:, 1:2], means[:, 0:1])
        return {
            "disp": disp_new,
            "means": means,
            "scales": scales,
            "mixture_logits": mixture_logits,
            "weights": weights,
            "peak_scores": peak_scores,
            "selected_mode": selected_mode,
            "broad_radius": broad_radius,
        }


class DEFOMStereo(PACTSMDDEFOMStereo):
    """Train only a final bimodal head on top of a frozen Full checkpoint."""

    MODEL_VARIANT = "pact_smd_post_gru_v1"

    def __init__(self, args) -> None:
        parent_args = copy.copy(args)
        parent_args.pact_smd_stage = "full"
        super().__init__(parent_args)
        self.args = args
        self.pact_smd_stage = "post"
        self.final_smd_head = FinalBimodalLaplaceHead(
            max_disp=self.max_disp_full,
            net_channels=int(args.hidden_dims[0]),
            local_radius=float(getattr(args, "pact_smd_post_local_radius", 1.0)),
            broad_radius_min=float(
                getattr(args, "pact_smd_post_broad_radius_min", 2.0)
            ),
            broad_radius_max=float(
                getattr(args, "pact_smd_post_broad_radius_max", 32.0)
            ),
            mode_threshold=float(
                getattr(args, "pact_smd_post_mode_threshold", 0.5)
            ),
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.final_smd_head.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        # Frozen modules stay in inference mode (including every BN/dropout),
        # while the root flag remains true for the training-loop contract.
        nn.Module.train(self, False)
        self.training = bool(mode)
        self.final_smd_head.train(bool(mode))
        return self

    def optimizer_parameter_groups(self, args):
        parameters = [
            parameter for parameter in self.final_smd_head.parameters()
            if parameter.requires_grad
        ]
        return [{
            "params": parameters,
            "lr": float(args.pact_smd_post_lr),
            "name": "final_smd_head",
        }]

    def _postprocess_final_prediction(
        self,
        net_4: torch.Tensor,
        disp_4: torch.Tensor,
        mono_4: torch.Tensor,
        fmap1_4: torch.Tensor,
        up_mask: Optional[torch.Tensor],
        raw_disp_up: torch.Tensor,
    ):
        mixture = self.final_smd_head(
            net_4.detach(),
            disp_4.detach(),
            mono_4.detach(),
            fmap1_4.detach(),
        )
        if up_mask is None:
            post_disp_up = upflow(
                mixture["disp"], factor=2 ** self.args.n_downsample, sacle=True
            )
        else:
            post_disp_up = self.upsample_flow(
                mixture["disp"], up_mask.detach()
            )
        auxiliary = {
            "post_raw_final": raw_disp_up.detach(),
            "post_final": post_disp_up,
            "post_smd_means": mixture["means"] * 4.0,
            "post_smd_scales": mixture["scales"] * 4.0,
            "post_smd_mixture_logits": mixture["mixture_logits"],
            "post_smd_weights": mixture["weights"],
            "post_smd_peak_scores": mixture["peak_scores"],
            "post_smd_selected_mode": mixture["selected_mode"].float(),
            "post_smd_broad_radius": mixture["broad_radius"] * 4.0,
        }
        return post_disp_up, auxiliary

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if not self.training:
            return output
        return_aux = bool(kwargs.get("return_aux", False))
        if return_aux:
            predictions, auxiliary = output
            return [predictions[-1]], auxiliary
        return [output[-1]]


__all__ = ["FinalBimodalLaplaceHead", "DEFOMStereo"]
