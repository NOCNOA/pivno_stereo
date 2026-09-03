"""Direct SR prediction of the final outer-GRU disparity increment.

This isolated variant keeps every completed C32/GWC4/enc16 gated-GRU3
parameter unchanged.  Ordinary recurrent refinement is used until the last
iteration.  At the last iteration a small low-resolution head consumes the
updated quarter-resolution hidden state together with the disparity, context,
and motion inputs, and PixelShuffle directly produces a full-resolution
``delta_d``.  That increment is added to the already-upsampled prediction from
the preceding iteration.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import (
    CheckpointCompatibleGWC4GatedGRU3,
)


class DirectLastDeltaSRHead(nn.Module):
    """Predict one full-resolution final ``delta_d`` with PixelShuffle."""

    HIDDEN_CHANNELS = 128
    CONTEXT_CHANNELS = 3 * 128
    MOTION_CHANNELS = 128
    INPUT_CHANNELS = (
        HIDDEN_CHANNELS + 1 + CONTEXT_CHANNELS + MOTION_CHANNELS
    )
    FEATURE_CHANNELS = 128
    FEATURE_SOURCE = "final_outer_hidden_disp_context_zqr_motion"
    OUTPUT_MODE = "direct_pixelshuffle_final_outer_delta_d"
    UPSAMPLE_MODE = "pixel_shuffle"

    def __init__(
        self,
        *,
        factor: int,
        max_disp_low: float,
        max_delta_disp_low: float,
    ) -> None:
        super().__init__()
        if int(factor) <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        if float(max_disp_low) <= 0:
            raise ValueError(
                f"max_disp_low must be positive, got {max_disp_low}"
            )
        if float(max_delta_disp_low) <= 0:
            raise ValueError(
                "max_delta_disp_low must be positive, got "
                f"{max_delta_disp_low}"
            )

        self.factor = int(factor)
        self.max_disp_low = float(max_disp_low)
        self.max_delta_disp_low = float(max_delta_disp_low)
        self.max_delta_disp_hr = self.factor * self.max_delta_disp_low
        # Shared SR plumbing records this bound under its historical metadata
        # name even though this head predicts a complete increment.
        self.residual_max = self.max_delta_disp_hr

        self.input_projection = nn.Conv2d(
            self.INPUT_CHANNELS,
            self.FEATURE_CHANNELS,
            kernel_size=1,
        )
        self.refine = nn.Conv2d(
            self.FEATURE_CHANNELS,
            self.FEATURE_CHANNELS,
            kernel_size=3,
            padding=1,
        )
        self.delta_head = nn.Conv2d(
            self.FEATURE_CHANNELS,
            self.factor * self.factor,
            kernel_size=3,
            padding=1,
        )

        # Start from the preceding recurrent prediction. The new branch learns
        # only the final high-resolution increment during head fine-tuning.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        disparity_low: torch.Tensor,
        context_zqr: Sequence[torch.Tensor],
        motion_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hidden.ndim != 4 or hidden.shape[1] != self.HIDDEN_CHANNELS:
            raise ValueError(
                "hidden must be [B,128,h,w], got "
                f"{tuple(hidden.shape)}"
            )
        if disparity_low.ndim != 4 or disparity_low.shape[1] != 1:
            raise ValueError(
                "disparity_low must be [B,1,h,w], got "
                f"{tuple(disparity_low.shape)}"
            )
        if len(context_zqr) != 3:
            raise ValueError(
                f"context_zqr must contain three tensors, got {len(context_zqr)}"
            )

        batch, _, height, width = hidden.shape
        expected_128 = (batch, 128, height, width)
        if disparity_low.shape != (batch, 1, height, width):
            raise ValueError("hidden and disparity_low shapes must align")
        if motion_features.shape != expected_128:
            raise ValueError(
                f"motion_features must be {expected_128}, got "
                f"{tuple(motion_features.shape)}"
            )
        for index, context in enumerate(context_zqr):
            if context.shape != expected_128:
                raise ValueError(
                    f"context_zqr[{index}] must be {expected_128}, got "
                    f"{tuple(context.shape)}"
                )

        sr_input = torch.cat(
            [
                hidden,
                disparity_low / self.max_disp_low,
                *context_zqr,
                motion_features,
            ],
            dim=1,
        )
        if sr_input.shape[1] != self.INPUT_CHANNELS:
            raise RuntimeError(
                f"SR input channel mismatch: expected {self.INPUT_CHANNELS}, "
                f"got {sr_input.shape[1]}"
            )

        feature = F.gelu(self.input_projection(sr_input))
        feature = F.gelu(self.refine(feature))
        delta_subpixels = self.delta_head(feature)
        delta_raw_hr = F.pixel_shuffle(delta_subpixels, self.factor)
        delta_sr_hr = self.max_delta_disp_hr * torch.tanh(delta_raw_hr)
        return delta_sr_hr, {
            "delta_sr_hr": delta_sr_hr,
            "delta_sr_raw_hr": delta_raw_hr,
        }


class DEFOMStereo(CheckpointCompatibleGWC4GatedGRU3):
    """Completed gated-GRU3 with a direct final-iteration delta SR head."""

    MODEL_VARIANT = "defom_pivno_gated_gru3_gwc4_last_delta_direct_sr"
    BASE_MODEL_VARIANT = CheckpointCompatibleGWC4GatedGRU3.MODEL_VARIANT
    TRAIN_STAGES = ("head", "joint")

    def __init__(self, args):
        super().__init__(args)
        self.sr_stage = str(getattr(args, "pivno_mask_sr_stage", "head"))
        if self.sr_stage not in self.TRAIN_STAGES:
            raise ValueError(
                f"pivno_mask_sr_stage must be one of {self.TRAIN_STAGES}, "
                f"got {self.sr_stage!r}"
            )

        factor = 2 ** int(args.n_downsample)
        self.sr_head = DirectLastDeltaSRHead(
            factor=factor,
            max_disp_low=float(args.max_disp) / factor,
            max_delta_disp_low=float(self.max_delta_disp),
        )
        self.set_train_stage(self.sr_stage)

    def set_train_stage(self, stage: str) -> None:
        if stage not in self.TRAIN_STAGES:
            raise ValueError(
                f"stage must be one of {self.TRAIN_STAGES}, got {stage!r}"
            )
        self.sr_stage = stage
        if stage == "joint":
            for parameter in self.parameters():
                parameter.requires_grad_(True)
            return
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.sr_head.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.sr_stage == "head":
            for child in self.children():
                child.eval()
            self.sr_head.train(True)
            self.training = True
        return self

    def optimizer_parameter_groups(self, args):
        parameters = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("direct last-delta SR has no trainable parameters")
        return [{"params": parameters, "lr": float(args.lr)}]

    def forward(
        self,
        image1,
        image2,
        iters=12,
        scale_iters=None,
        test_mode=False,
        return_sr_aux=False,
    ):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            return self._forward_impl(
                image1,
                image2,
                iters=iters,
                scale_iters=scale_iters,
                test_mode=test_mode,
                return_sr_aux=return_sr_aux,
            )

    def _forward_impl(
        self,
        image1,
        image2,
        iters=12,
        scale_iters=None,
        test_mode=False,
        return_sr_aux=False,
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
        context_image = ((image1 - self.mean) / self.std).contiguous().float()

        init_predictions, fmap1_4, fmap2_4, disparity = self.pivno(
            pivno_image1,
            pivno_image2,
            return_imgfeature=True,
        )
        cnet_list = self.cnet(
            context_image,
            num_layers=self.args.n_gru_layers,
        )
        net_list = [torch.tanh(value[0]) for value in cnet_list]
        inp_list = [torch.relu(value[1]) for value in cnet_list]
        inp_list = [
            list(conv(context).chunk(3, dim=1))
            for context, conv in zip(inp_list, self.context_zqr_convs)
        ]

        fmap1_low = self.low_channel(fmap1_4)
        fmap2_low = self.low_channel(fmap2_4)
        fmap2_half, fmap2_quarter = self.right_width_compressor(fmap2_low)
        right_feature_pyramid = (fmap2_low, fmap2_half, fmap2_quarter)

        predictions = []
        previous_disparity_up = init_predictions[-1]
        sr_aux = None
        for iteration in range(iters):
            disparity = disparity.detach()
            disparity_before_update = disparity
            warp_feature = self._refine_warp_feature(
                fmap1_low,
                right_feature_pyramid,
                disparity_before_update,
            )
            net_list, up_mask, delta_disp = self.update_block(
                net_list,
                inp_list,
                warp_feature,
                disparity_before_update,
                iter32=self.args.n_gru_layers == 3,
                iter16=self.args.n_gru_layers >= 2,
            )
            delta_disp = self._clamp_delta_disp(delta_disp)

            if iteration == iters - 1:
                # Recreate the exact motion tensor consumed by gru08. Keeping
                # this variant isolated avoids changing the shared update block
                # return contract for all existing models and checkpoints.
                motion_features = self.update_block.encoder(
                    disparity_before_update,
                    warp_feature,
                )
                delta_sr_hr, sr_aux = self.sr_head(
                    net_list[0],
                    disparity_before_update,
                    inp_list[0],
                    motion_features,
                )
                disparity_up = previous_disparity_up + delta_sr_hr
                sr_aux.update({
                    "disp_previous_hr": previous_disparity_up,
                    "delta_gru_low": delta_disp,
                })
            else:
                disparity = disparity_before_update + delta_disp
                disparity_up = self.upsample_flow(disparity, up_mask)
                previous_disparity_up = disparity_up
            predictions.append(disparity_up)

        if sr_aux is None:
            raise RuntimeError("direct final-delta SR head was not executed")
        if test_mode:
            if return_sr_aux:
                return disparity_up, sr_aux
            return disparity_up
        if return_sr_aux:
            return init_predictions, predictions, sr_aux
        return init_predictions, predictions
