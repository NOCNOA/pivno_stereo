"""Full-resolution left-RGB-guided SR for the completed gated-GRU3 model.

This is an isolated alternative to ``defom_pivno_gated_gru3_gwc4_mask_sr``.
The completed C32/GWC4/enc16 recurrent base and its checkpoint layout are
preserved, while the final SR residual head uses shallow full-resolution left
image features instead of the quarter-resolution outer-GRU hidden state.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import (
    DEFOMStereo as HiddenGuidedMaskSR,
)


class FullResolutionRGBGuidedSRHead(nn.Module):
    """Predict bounded ``delta_d`` from mask-upsampled disparity and left RGB."""

    FEATURE_CHANNELS = 32
    FEATURE_SOURCE = "full_resolution_normalized_left_rgb"
    INPUT_CHANNELS = 9 + 9 + 1 + FEATURE_CHANNELS + 2

    def __init__(self, *, max_disp: float, residual_max: float = 4.0) -> None:
        super().__init__()
        if float(max_disp) <= 0:
            raise ValueError(f"max_disp must be positive, got {max_disp}")
        if float(residual_max) <= 0:
            raise ValueError(
                f"residual_max must be positive, got {residual_max}"
            )
        self.max_disp = float(max_disp)
        self.residual_max = float(residual_max)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, self.FEATURE_CHANNELS, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                self.FEATURE_CHANNELS,
                self.FEATURE_CHANNELS,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )
        self.conv1 = nn.Conv2d(
            self.INPUT_CHANNELS, 64, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # A base checkpoint plus this new head must initially reproduce the
        # completed model's convex-mask prediction exactly.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    @staticmethod
    def _subpixels_to_image(value: torch.Tensor) -> torch.Tensor:
        """Rearrange ``[B,C,F,F,h,w]`` into ``[B,C,Fh,Fw]``."""
        if value.ndim != 6:
            raise ValueError(
                f"subpixel tensor must be 6D, got shape {tuple(value.shape)}"
            )
        batch, channels, factor_y, factor_x, height, width = value.shape
        if factor_y != factor_x:
            raise ValueError(
                f"upsampling factors must match, got {factor_y} and {factor_x}"
            )
        return value.permute(0, 1, 4, 2, 5, 3).reshape(
            batch, channels, height * factor_y, width * factor_x
        )

    @staticmethod
    def _relative_subpixel_coordinates(
        *,
        batch: int,
        factor: int,
        height: int,
        width: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        axis = (
            2.0
            * (
                torch.arange(
                    factor,
                    device=reference.device,
                    dtype=reference.dtype,
                )
                + 0.5
            )
            / factor
            - 1.0
        )
        coord = torch.stack(
            torch.meshgrid(axis, axis, indexing="ij"), dim=0
        )
        coord = coord.view(1, 2, factor, factor, 1, 1).expand(
            batch, -1, -1, -1, height, width
        )
        return FullResolutionRGBGuidedSRHead._subpixels_to_image(coord)

    def forward(
        self,
        disparity_low: torch.Tensor,
        mask_logits: torch.Tensor,
        left_image: torch.Tensor,
        *,
        factor: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if disparity_low.ndim != 4 or disparity_low.shape[1] != 1:
            raise ValueError(
                "disparity_low must be [B,1,h,w], got "
                f"{tuple(disparity_low.shape)}"
            )
        if left_image.ndim != 4 or left_image.shape[1] != 3:
            raise ValueError(
                "left_image must be [B,3,Fh,Fw], got "
                f"{tuple(left_image.shape)}"
            )
        if int(factor) <= 0:
            raise ValueError(f"factor must be positive, got {factor}")

        batch, _, height, width = disparity_low.shape
        factor = int(factor)
        expected_image_shape = (batch, 3, factor * height, factor * width)
        if left_image.shape != expected_image_shape:
            raise ValueError(
                f"left_image must be {expected_image_shape}, got "
                f"{tuple(left_image.shape)}"
            )
        expected_mask_channels = 9 * factor * factor
        if mask_logits is None or mask_logits.shape != (
            batch,
            expected_mask_channels,
            height,
            width,
        ):
            actual = None if mask_logits is None else tuple(mask_logits.shape)
            raise ValueError(
                "mask_logits must be "
                f"[B,{expected_mask_channels},h,w], got {actual}"
            )

        weights = mask_logits.view(
            batch, 1, 9, factor, factor, height, width
        )
        weights = torch.softmax(weights, dim=2)

        # Preserve the completed base model's disparity upsampling exactly.
        disparity_neighbours = F.unfold(
            factor * disparity_low, kernel_size=3, padding=1
        ).view(batch, 1, 9, height, width)
        disparity_neighbours_sub = disparity_neighbours[:, :, :, None, None]
        disparity_neighbours_sub = disparity_neighbours_sub.expand(
            -1, -1, -1, factor, factor, -1, -1
        )
        disparity_base_sub = (
            weights * disparity_neighbours_sub
        ).sum(dim=2)

        disparity_base = self._subpixels_to_image(disparity_base_sub)
        disparity_neighbours_hr = self._subpixels_to_image(
            disparity_neighbours_sub[:, 0].reshape(
                batch, 9, factor, factor, height, width
            )
        )
        weights_hr = self._subpixels_to_image(
            weights[:, 0].reshape(
                batch, 9, factor, factor, height, width
            )
        )
        image_feature = self.image_encoder(left_image)
        relative_coord = self._relative_subpixel_coordinates(
            batch=batch,
            factor=factor,
            height=height,
            width=width,
            reference=left_image,
        )

        sr_input = torch.cat(
            [
                disparity_neighbours_hr / self.max_disp,
                weights_hr,
                disparity_base / self.max_disp,
                image_feature,
                relative_coord,
            ],
            dim=1,
        )
        if sr_input.shape[1] != self.INPUT_CHANNELS:
            raise RuntimeError(
                f"SR input channel mismatch: expected {self.INPUT_CHANNELS}, "
                f"got {sr_input.shape[1]}"
            )
        residual = self.residual_max * torch.tanh(
            self.conv2(F.gelu(self.conv1(sr_input)))
        )
        prediction = disparity_base + residual
        return prediction, {
            "disp_mask_base": disparity_base,
            "sr_delta_d": residual,
        }


class DEFOMStereo(HiddenGuidedMaskSR):
    """Completed C32/GWC4 gated-GRU3 plus left-RGB-guided final SR."""

    MODEL_VARIANT = "defom_pivno_gated_gru3_gwc4_mask_rgb_sr"
    BASE_MODEL_VARIANT = "defom_pivno_gated_gru3"

    def __init__(self, args):
        super().__init__(args)
        self.sr_head = FullResolutionRGBGuidedSRHead(
            max_disp=float(args.max_disp),
            residual_max=float(
                getattr(args, "pivno_mask_sr_residual_max", 4.0)
            ),
        )
        self.set_train_stage(self.sr_stage)

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
        factor = 2 ** self.args.n_downsample
        predictions = []
        sr_aux = None
        for iteration in range(iters):
            disparity = disparity.detach()
            warp_feature = self._refine_warp_feature(
                fmap1_low,
                right_feature_pyramid,
                disparity,
            )
            net_list, up_mask, delta_disp = self.update_block(
                net_list,
                inp_list,
                warp_feature,
                disparity,
                iter32=self.args.n_gru_layers == 3,
                iter16=self.args.n_gru_layers >= 2,
            )
            disparity = disparity + self._clamp_delta_disp(delta_disp)
            if iteration == iters - 1:
                disparity_up, sr_aux = self.sr_head(
                    disparity,
                    up_mask,
                    context_image,
                    factor=factor,
                )
            else:
                disparity_up = self.upsample_flow(disparity, up_mask)
            predictions.append(disparity_up)

        if sr_aux is None:
            raise RuntimeError("final-only RGB-guided SR head was not executed")
        if test_mode:
            if return_sr_aux:
                return disparity_up, sr_aux
            return disparity_up
        if return_sr_aux:
            return init_predictions, predictions, sr_aux
        return init_predictions, predictions
