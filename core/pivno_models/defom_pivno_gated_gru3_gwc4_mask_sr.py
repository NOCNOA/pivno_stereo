"""Final-only mask-guided SR for the completed C32/GWC4 gated-GRU3 model.

The live ``defom_pivno_gated_gru3.py`` has drifted to C48/GWC8, while the
completed 200k checkpoint is C32/GWC4/enc16.  This isolated module restores
the checkpoint-compatible modules without modifying the live base class, then
adds the same small final disparity-residual head used by the direct-concat SR
ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.pivno_models.defom_pivno_gated_gru3 import (
    DEFOMStereo as LiveGatedGRU3,
)
from core.pivno_models.defom_pivno_gwc4_enc16_concat_gru3_mask_sr import (
    SimpleMaskGuidedSRHead,
)
from core.submodules import ParallelRightWidthCompressor


class CheckpointCompatibleGWC4GatedGRU3(LiveGatedGRU3):
    """C32/GWC4 form encoded by the completed gated-GRU3 200k checkpoint."""

    MODEL_VARIANT = "defom_pivno_gated_gru3"
    LOW_FEATURE_DIM = 32
    MATCH_NUM_GROUPS = 4
    MATCH_ENCODED_CHANNELS = 16

    def __init__(self, args):
        super().__init__(args)

        # ``super`` currently constructs the drifted C48 modules. Replace only
        # that shape-dependent group so the state dict exactly matches the
        # completed C32/GWC4 checkpoint. Shared PIVNO/context/GRU modules stay
        # the implementation from defom_pivno_gated_gru3.py.
        low_dim = self.LOW_FEATURE_DIM
        self.low_channel = nn.Conv2d(64, low_dim, kernel_size=1)
        self.right_width_compressor = ParallelRightWidthCompressor(
            low_dim,
            low_dim,
            mode="conv",
        )
        self.sample_match_encoder = nn.Sequential(
            nn.Conv2d(
                low_dim + self.MATCH_NUM_GROUPS,
                self.MATCH_ENCODED_CHANNELS,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(4, self.MATCH_ENCODED_CHANNELS),
            nn.GELU(),
        )

        num_scales = len(self.compression_ratios)
        num_offsets = int(self.warp_offsets.numel())
        self.scale_gate = nn.Sequential(
            nn.Conv2d(
                low_dim + num_scales * num_offsets,
                low_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(low_dim, num_scales, kernel_size=1),
        )
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)
        self.refine_right_fuse = self._make_right_fuser(
            num_scales * num_offsets * self.MATCH_ENCODED_CHANNELS
        )


class DEFOMStereo(CheckpointCompatibleGWC4GatedGRU3):
    """Completed GWC4 gated-GRU3 plus final-only mask-guided ``delta_d``."""

    MODEL_VARIANT = "defom_pivno_gated_gru3_gwc4_mask_sr"
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
            raise ValueError("GWC4 gated-GRU3-mask-SR has no trainable parameters")
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
            disparity = disparity + self._clamp_delta_disp(delta_disp)
            if iteration == iters - 1:
                disparity_up, sr_aux = self.sr_head(
                    disparity,
                    up_mask,
                    net_list[0],
                    factor=factor,
                )
            else:
                disparity_up = self.upsample_flow(disparity, up_mask)
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
