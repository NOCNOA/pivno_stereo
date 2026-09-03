"""Final-only mask-guided disparity SR for the GWC4/enc16/GRU3 model.

The base model and its checkpoint layout stay unchanged.  This isolated
variant reuses the base model's final convex upsampling mask and predicts a
small bounded full-resolution disparity residual.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .defom_pivno_gwc4_enc16_concat_gru3 import (
    DEFOMStereo as BaseGWC4Enc16ConcatGRU3,
)


class SimpleMaskGuidedSRHead(nn.Module):
    """Reuse convex weights for disparity/features, then predict ``delta_d``."""

    FEATURE_CHANNELS = 32
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
        self.feature_projection = nn.Conv2d(
            128, self.FEATURE_CHANNELS, kernel_size=1
        )
        self.conv1 = nn.Conv2d(
            self.INPUT_CHANNELS, 64, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # Before SR training, the new model must exactly reproduce the base
        # model's convex-mask prediction.
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
        return SimpleMaskGuidedSRHead._subpixels_to_image(coord)

    def forward(
        self,
        disparity_low: torch.Tensor,
        mask_logits: torch.Tensor,
        hidden: torch.Tensor,
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
        if int(factor) <= 0:
            raise ValueError(f"factor must be positive, got {factor}")

        batch, _, height, width = disparity_low.shape
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
        if hidden.shape[0] != batch or hidden.shape[-2:] != (height, width):
            raise ValueError("hidden and disparity_low spatial shapes must match")

        weights = mask_logits.view(
            batch, 1, 9, factor, factor, height, width
        )
        weights = torch.softmax(weights, dim=2)

        # This is exactly the base GWC4/enc16/GRU3 convex upsampling formula.
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

        feature_low = self.feature_projection(hidden)
        feature_neighbours = F.unfold(
            feature_low, kernel_size=3, padding=1
        ).view(batch, self.FEATURE_CHANNELS, 9, height, width)
        feature_sub = torch.einsum(
            "bcnhw,bnuvhw->bcuvhw",
            feature_neighbours,
            weights[:, 0],
        )
        feature_hr = self._subpixels_to_image(feature_sub)
        relative_coord = self._relative_subpixel_coordinates(
            batch=batch,
            factor=factor,
            height=height,
            width=width,
            reference=hidden,
        )

        sr_input = torch.cat(
            [
                disparity_neighbours_hr / self.max_disp,
                weights_hr,
                disparity_base / self.max_disp,
                feature_hr,
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


class DEFOMStereo(BaseGWC4Enc16ConcatGRU3):
    """GWC4/enc16/direct-concat/GRU3 plus final mask-guided SR."""

    MODEL_VARIANT = "defom_pivno_gwc4_enc16_concat_gru3_mask_sr"
    BASE_MODEL_VARIANT = BaseGWC4Enc16ConcatGRU3.MODEL_VARIANT
    TRAIN_STAGES = ("head", "joint")

    def __init__(self, args):
        super().__init__(args)
        self.sr_stage = str(getattr(args, "pivno_mask_sr_stage", "head"))
        if self.sr_stage not in self.TRAIN_STAGES:
            raise ValueError(
                f"pivno_mask_sr_stage must be one of {self.TRAIN_STAGES}, "
                f"got {self.sr_stage!r}"
            )
        self.sr_head = SimpleMaskGuidedSRHead(
            max_disp=float(args.max_disp),
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
        """Keep the frozen base, including BN statistics, in evaluation mode."""
        super().train(mode)
        if mode and self.sr_stage == "head":
            for child in self.children():
                child.eval()
            self.sr_head.train(True)
            self.training = True
        return self

    def optimizer_parameter_groups(self, args):
        parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("GWC4/enc16/GRU3-mask-SR has no trainable parameters")
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
            context_image, num_layers=self.args.n_gru_layers
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
                fmap1_low, right_feature_pyramid, disparity
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
                    factor=factor,
                )
            else:
                disparity_up = self.upsample_prediction(disparity, up_mask)
            predictions.append(disparity_up)

        if sr_aux is None:
            raise RuntimeError("final-only SR head was not executed")
        if test_mode:
            if return_sr_aux:
                return disparity_up, sr_aux
            return disparity_up
        if return_sr_aux:
            return init_predictions, predictions, sr_aux
        return init_predictions, predictions
