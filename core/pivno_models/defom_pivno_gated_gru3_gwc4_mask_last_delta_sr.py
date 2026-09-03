"""SR reconstruction of the final outer-GRU disparity update.

This isolated variant leaves the completed C32/GWC4/enc16 gated-GRU3 model
unchanged.  At the last outer recurrent iteration it reconstructs the
low-resolution ``delta_disp`` at full resolution from the same convex mask and
the final outer hidden state, then learns a spatial weight for that update.

Both output branches are zero initialized: the reconstructed delta initially
equals the ordinary mask-upsampled delta and its weight is exactly one.  The
model therefore reproduces the base checkpoint before head optimization.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import (
    CheckpointCompatibleGWC4GatedGRU3,
)


class LastDeltaWeightedSRHead(nn.Module):
    """Refine and spatially reweight only the final outer-GRU ``delta_disp``."""

    FEATURE_CHANNELS = 32
    INPUT_CHANNELS = 9 + 9 + 1 + FEATURE_CHANNELS + 2
    FEATURE_SOURCE = "mask_upsampled_final_outer_gru_hidden"
    OUTPUT_MODE = "weighted_sr_reconstruction_of_final_outer_delta_d"
    WEIGHT_MODE = "two_sigmoid_identity_one"

    def __init__(
        self,
        *,
        max_delta_disp_low: float,
        residual_max: float = 4.0,
    ) -> None:
        super().__init__()
        if float(max_delta_disp_low) <= 0:
            raise ValueError(
                "max_delta_disp_low must be positive, got "
                f"{max_delta_disp_low}"
            )
        if float(residual_max) <= 0:
            raise ValueError(
                f"residual_max must be positive, got {residual_max}"
            )
        self.max_delta_disp_low = float(max_delta_disp_low)
        self.residual_max = float(residual_max)
        self.feature_projection = nn.Conv2d(
            128, self.FEATURE_CHANNELS, kernel_size=1
        )
        self.conv1 = nn.Conv2d(
            self.INPUT_CHANNELS, 64, kernel_size=3, padding=1
        )
        self.delta_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.weight_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # residual=0 and 2*sigmoid(0)=1 preserve the base model exactly.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    @staticmethod
    def _subpixels_to_image(value: torch.Tensor) -> torch.Tensor:
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

    @classmethod
    def _relative_subpixel_coordinates(
        cls,
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
        return cls._subpixels_to_image(coord)

    def forward(
        self,
        disparity_base_hr: torch.Tensor,
        delta_disp_low: torch.Tensor,
        mask_logits: torch.Tensor,
        hidden: torch.Tensor,
        *,
        factor: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:


        batch, _, height, width = delta_disp_low.shape
        factor = int(factor)
        expected_hr_shape = (batch, 1, factor * height, factor * width)
        expected_mask_channels = 9 * factor * factor

        weights = mask_logits.view(
            batch, 1, 9, factor, factor, height, width
        )
        weights = torch.softmax(weights, dim=2)

        # ``delta_disp_low`` is in feature-grid pixels. Match the base
        # upsampler by converting it to full-image pixel units before unfold.
        delta_neighbours = F.unfold(
            factor * delta_disp_low, kernel_size=3, padding=1
        ).view(batch, 1, 9, height, width)
        delta_neighbours_sub = delta_neighbours[:, :, :, None, None]
        delta_neighbours_sub = delta_neighbours_sub.expand(
            -1, -1, -1, factor, factor, -1, -1
        )#到这里已经相当于一个factor*factor大小的patch，看到这里感觉这个mask才是最需要SR的
        delta_base_sub = (weights * delta_neighbours_sub).sum(dim=2)
        #转为原图大小
        delta_base_hr = self._subpixels_to_image(delta_base_sub)
        delta_neighbours_hr = self._subpixels_to_image(
            delta_neighbours_sub[:, 0].reshape(
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
        relative_coord = self._relative_subpixel_coordinates(
            batch=batch,
            factor=factor,
            height=height,
            width=width,
            reference=hidden,
        )

        max_delta_disp_hr = factor * self.max_delta_disp_low
        sr_input = torch.cat(
            [
                delta_neighbours_hr / max_delta_disp_hr,
                weights_hr,
                delta_base_hr / max_delta_disp_hr,
                hidden_hr,
                relative_coord,
            ],
            dim=1,
        )
        if sr_input.shape[1] != self.INPUT_CHANNELS:
            raise RuntimeError(
                f"SR input channel mismatch: expected {self.INPUT_CHANNELS}, "
                f"got {sr_input.shape[1]}"
            )

        decoded = F.gelu(self.conv1(sr_input))
        delta_residual = self.residual_max * torch.tanh(
            self.delta_head(decoded)
        )
        delta_refined_hr = delta_base_hr + delta_residual
        delta_weight = 2.0 * torch.sigmoid(self.weight_head(decoded))
        delta_weighted_hr = delta_weight * delta_refined_hr

        # Replace only the contribution of the last recurrent update. At
        # initialization the parenthesized correction is identically zero.
        final_correction = delta_weighted_hr - delta_base_hr
        prediction = disparity_base_hr + final_correction
        return prediction, {
            "disp_mask_base": disparity_base_hr,
            "delta_mask_base": delta_base_hr,
            "delta_sr_residual": delta_residual,
            "delta_sr_refined": delta_refined_hr,
            "delta_sr_weight": delta_weight,
            "sr_delta_d": final_correction,
        }


class DEFOMStereo(CheckpointCompatibleGWC4GatedGRU3):
    """Completed gated-GRU3 plus weighted SR of its final update."""

    MODEL_VARIANT = "defom_pivno_gated_gru3_gwc4_mask_last_delta_sr"
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
        self.sr_head = LastDeltaWeightedSRHead(
            max_delta_disp_low=float(self.max_delta_disp),
            residual_max=float(
                getattr(args, "pivno_mask_sr_residual_max", 4.0)
            ),
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
            raise ValueError("last-delta SR has no trainable parameters")
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
            delta_disp = self._clamp_delta_disp(delta_disp)
            disparity = disparity + delta_disp
            disparity_base_hr = self.upsample_flow(disparity, up_mask)
            if iteration == iters - 1:
                disparity_up, sr_aux = self.sr_head(
                    disparity_base_hr,
                    delta_disp,
                    up_mask,
                    net_list[0],
                    factor=factor,
                )
            else:
                disparity_up = disparity_base_hr
            predictions.append(disparity_up)

        if sr_aux is None:
            raise RuntimeError("final-only last-delta SR head was not executed")
        if test_mode:
            if return_sr_aux:
                return disparity_up, sr_aux
            return disparity_up
        if return_sr_aux:
            return init_predictions, predictions, sr_aux
        return init_predictions, predictions
