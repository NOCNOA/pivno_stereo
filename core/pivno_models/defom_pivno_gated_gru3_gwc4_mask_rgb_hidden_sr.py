"""Full-resolution RGB plus recurrent-hidden SR for gated-GRU3.

This isolated performance-oriented variant preserves the completed
C32/GWC4/enc16 base.  It retains the existing mask-upsampled outer-GRU hidden
feature, adds a shallow full-resolution left-RGB feature, and fuses both back
to 32 channels before the existing disparity-residual decoder.

The fusion is initialized as ``[identity(hidden), zero(rgb)]``.  Consequently,
loading a trained hidden-only mask-SR checkpoint reproduces its prediction
before any RGB-fusion optimization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import (
    DEFOMStereo as HiddenGuidedMaskSR,
)
from core.pivno_models.defom_pivno_gwc4_enc16_concat_gru3_mask_sr import (
    SimpleMaskGuidedSRHead,
)


class RGBHiddenFusionSRHead(SimpleMaskGuidedSRHead):
    """Fuse mask-upsampled hidden and full-resolution left-RGB features."""

    FEATURE_SOURCE = "mask_upsampled_hidden_plus_full_resolution_left_rgb"

    def __init__(self, *, max_disp: float, residual_max: float = 4.0) -> None:
        super().__init__(max_disp=max_disp, residual_max=residual_max)
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
        self.feature_fusion = nn.Conv2d(
            2 * self.FEATURE_CHANNELS,
            self.FEATURE_CHANNELS,
            kernel_size=1,
        )
        self._initialize_hidden_identity_fusion()

    def _initialize_hidden_identity_fusion(self) -> None:
        nn.init.zeros_(self.feature_fusion.weight)
        nn.init.zeros_(self.feature_fusion.bias)
        with torch.no_grad():
            channels = torch.arange(self.FEATURE_CHANNELS)
            self.feature_fusion.weight[channels, channels, 0, 0] = 1.0

    def forward(
        self,
        disparity_low: torch.Tensor,
        mask_logits: torch.Tensor,
        hidden: torch.Tensor,
        left_image: torch.Tensor,
        *,
        factor: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if disparity_low.ndim != 4 or disparity_low.shape[1] != 1:
            raise ValueError(
                "disparity_low must be [B,1,h,w], got "
                f"{tuple(disparity_low.shape)}"
            )
        if hidden.ndim != 4 or hidden.shape[1] != 128:
            raise ValueError(
                f"hidden must be [B,128,h,w], got {tuple(hidden.shape)}"
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
        if hidden.shape[0] != batch or hidden.shape[-2:] != (height, width):
            raise ValueError("hidden and disparity_low spatial shapes must match")
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

        hidden_low = self.feature_projection(hidden)
        hidden_neighbours = F.unfold(
            hidden_low, kernel_size=3, padding=1
        ).view(batch, self.FEATURE_CHANNELS, 9, height, width)
        hidden_sub = torch.einsum(
            "bcnhw,bnuvhw->bcuvhw",
            hidden_neighbours,
            weights[:, 0],
        )
        hidden_hr = self._subpixels_to_image(hidden_sub)
        image_hr = self.image_encoder(left_image)
        fused_feature = self.feature_fusion(
            torch.cat([hidden_hr, image_hr], dim=1)
        )
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
                fused_feature,
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
    """Completed gated-GRU3 plus identity-initialized RGB/hidden SR fusion."""

    MODEL_VARIANT = "defom_pivno_gated_gru3_gwc4_mask_rgb_hidden_sr"
    BASE_MODEL_VARIANT = "defom_pivno_gated_gru3"
    SR_SOURCE_MODEL_VARIANT = "defom_pivno_gated_gru3_gwc4_mask_sr"

    def __init__(self, args):
        super().__init__(args)
        self.sr_head = RGBHiddenFusionSRHead(
            max_disp=float(args.max_disp),
            residual_max=float(
                getattr(args, "pivno_mask_sr_residual_max", 4.0)
            ),
        )
        self.set_train_stage(self.sr_stage)

    def optimizer_parameter_groups(self, args):
        new_parameters = [
            *self.sr_head.image_encoder.parameters(),
            *self.sr_head.feature_fusion.parameters(),
        ]
        new_ids = {id(parameter) for parameter in new_parameters}
        pretrained_parameters = [
            parameter
            for parameter in self.sr_head.parameters()
            if parameter.requires_grad and id(parameter) not in new_ids
        ]
        groups = []
        if pretrained_parameters:
            pretrained_lr = getattr(
                args, "pivno_mask_sr_pretrained_lr", None
            )
            if pretrained_lr is None:
                pretrained_lr = 0.1 * float(args.lr)
            groups.append({
                "params": pretrained_parameters,
                "lr": float(pretrained_lr),
            })
        active_new_parameters = [
            parameter for parameter in new_parameters if parameter.requires_grad
        ]
        if active_new_parameters:
            groups.append({"params": active_new_parameters, "lr": float(args.lr)})

        sr_ids = {id(parameter) for parameter in self.sr_head.parameters()}
        base_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in sr_ids
        ]
        if base_parameters:
            groups.append({"params": base_parameters, "lr": float(args.lr)})
        if not groups:
            raise ValueError("RGB/hidden mask-SR has no trainable parameters")
        return groups

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
                    net_list[0],
                    context_image,
                    factor=factor,
                )
            else:
                disparity_up = self.upsample_flow(disparity, up_mask)
            predictions.append(disparity_up)

        if sr_aux is None:
            raise RuntimeError("final-only RGB/hidden SR head was not executed")
        if test_mode:
            if return_sr_aux:
                return disparity_up, sr_aux
            return disparity_up
        if return_sr_aux:
            return init_predictions, predictions, sr_aux
        return init_predictions, predictions
