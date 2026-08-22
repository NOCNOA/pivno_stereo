"""PIVNO-initialized recurrent DEFOM-Stereo model.

PIVNO supplies a quarter-resolution disparity and stereo feature pair. The
right feature is compressed at width ratios 1/2/4, locally sampled around the
current disparity, and concatenated for recurrent quarter-resolution updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIVNO.models.sronet import PIVNO

from core.extractor import MultiBasicEncoder2
from core.update import BasicMultiUpdateBlock
from core.utils.utils import upflow
from core.submodules import ParallelRightWidthCompressor, lookup_right_features
from core.galerkin import simple_attn_rope_2d


class DEFOMStereo(nn.Module):
    """
    特征提取头以及视差初始化换成了pivno
    """
    def __init__(self, args):
        super().__init__()
        if int(args.n_downsample) != 2:
            raise ValueError(
                "PIVNO produces quarter-resolution features, so "
                f"n_downsample must be 2, got {args.n_downsample}"
            )
        if len(args.hidden_dims) != 3:
            raise ValueError(
                "BasicMultiUpdateBlock requires three hidden dimensions, "
                f"got {args.hidden_dims}"
            )

        self.args = args
        self.register_buffer('mean', torch.tensor([[0.485, 0.456, 0.406]])[..., None, None] * 255)
        self.register_buffer('std', torch.tensor([[0.229, 0.224, 0.225]])[..., None, None] * 255)
        self.pivno_input_channels = int(
            getattr(args, 'pivno_input_channels', 3)
        )
        if self.pivno_input_channels not in (1, 3):
            raise ValueError(
                "PIVNO input_channels must be 1 or 3, got "
                f"{self.pivno_input_channels}"
            )
        self.pivno = PIVNO(input_channels=self.pivno_input_channels)
        context_dims = args.hidden_dims
        self.cnet = MultiBasicEncoder2(
            output_dim=[args.hidden_dims, context_dims],
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
        )
        level_context_dims = list(reversed(context_dims))
        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(dim, dim * 3, 3, padding=1)
            for dim in level_context_dims[:args.n_gru_layers]
        ])

        low_feature_dim = 32
        self.corr_radius = int(args.corr_radius)
        self.compression_ratios = (1, 2, 4)
        self.max_delta_disp = float(
            max(self.compression_ratios) * self.corr_radius
        )
        self.register_buffer(
            'warp_offsets',
            torch.arange(
                -self.corr_radius,
                self.corr_radius + 1,
                dtype=torch.float32,
            ),
        )
        self.low_channel = nn.Conv2d(64, low_feature_dim, kernel_size=1)
        self.right_width_compressor = ParallelRightWidthCompressor(
            low_feature_dim,
            low_feature_dim,
            mode="conv",
        )

        num_scales = len(self.compression_ratios)
        num_offsets = self.warp_offsets.numel()
        refine_fuse_channels = low_feature_dim * (1 + num_scales * num_offsets)
        self.refine_right_fuse = self._make_right_fuser(refine_fuse_channels)
        self.refine_rope = simple_attn_rope_2d(128, 4)
        self.update_block = BasicMultiUpdateBlock(
            self.args,
            hidden_dims=args.hidden_dims,
        )

    @staticmethod
    def _make_right_fuser(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def _refine_warp_feature(self, fmap1_low, right_feature_pyramid, disp):
        fused_feature = lookup_right_features(
            fmap1_low,
            right_feature_pyramid,
            disp,
            warp_offsets=self.warp_offsets,
            feature_fuse=self.refine_right_fuse,
            rope_galerkin=None,
            scale_weights=None,
            compression_ratios=self.compression_ratios,
        )
        # The untrained three-scale fusion can have a large activation range.
        # At the SceneFlow crop size the Galerkin projection overflows FP16 on
        # the first batch, turning the first recurrent prediction into NaNs.
        # Keep the large fusion convolutions under the caller's autocast, but
        # perform only this numerically sensitive attention block in FP32.
        with torch.cuda.amp.autocast(enabled=False):
            return self.refine_rope(fused_feature.float())

    def _clamp_delta_disp(self, delta_disp):
        """Limit one update to the widest sampled search support."""
        return delta_disp.clamp(
            min=-self.max_delta_disp,
            max=self.max_delta_disp,
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def upsample_flow(self, flow, mask):
        """Upsample a quarter-resolution disparity with a convex mask."""
        batch, channels, height, width = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(batch, 1, 9, factor, factor, height, width)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(batch, channels, 9, 1, 1, height, width)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(batch, channels, factor * height, factor * width)

    def upsample_prediction(self, disp, mask):
        if mask is None:
            return upflow(disp, factor=2 ** self.args.n_downsample)
        return self.upsample_flow(disp, mask)

    @staticmethod
    def _to_pivno_rgb(image):
        """Scale DEFOM's RGB [0,255] input to PIVNO's RGB [0,1]."""
        return DEFOMStereo._prepare_pivno_input(image, input_channels=3)

    @staticmethod
    def _prepare_pivno_input(image, input_channels):
        """Reproduce RGB or legacy grayscale PIVNO preprocessing."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected RGB input [B,3,H,W], got {tuple(image.shape)}")
        if input_channels == 3:
            return image / 255.0
        if input_channels == 1:
            coefficients = image.new_tensor(
                [0.299, 0.587, 0.114]
            ).view(1, 3, 1, 1)
            return (image * coefficients).sum(dim=1, keepdim=True) / 255.0
        raise ValueError(f"Unsupported PIVNO input channel count: {input_channels}")

    def forward(self, image1, image2, iters=12, scale_iters=None, test_mode=False):
        # Kept for compatibility with the repository-wide train/eval API.
        del scale_iters
        if image1.shape != image2.shape:
            raise ValueError(
                f"Stereo images must have equal shapes, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
        if iters < 1:
            raise ValueError(f"iters must be at least 1, got {iters}")

        pivno_image1 = self._prepare_pivno_input(
            image1, self.pivno_input_channels
        ).contiguous().float()
        pivno_image2 = self._prepare_pivno_input(
            image2, self.pivno_input_channels
        ).contiguous().float()
        image1 = ((image1 - self.mean) / self.std).contiguous().float()

        init_disp_predictions, fmap1_4, fmap2_4, d0 = self.pivno(
            pivno_image1,
            pivno_image2,
            return_imgfeature=True,
        )
        disp = d0
        cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [
            list(conv(context).chunk(3, dim=1))
            for context, conv in zip(inp_list, self.context_zqr_convs)
        ]

        fmap1_low = self.low_channel(fmap1_4)
        fmap2_low = self.low_channel(fmap2_4)
        fmap2_half, fmap2_quarter = self.right_width_compressor(fmap2_low)
        right_feature_pyramid = (fmap2_low, fmap2_half, fmap2_quarter)
        disp_predictions = []
        for _ in range(iters):
            disp = disp.detach()
            warp_feature = self._refine_warp_feature(fmap1_low, right_feature_pyramid, disp)
            net_list, up_mask, delta_disp = self.update_block(
                net_list,
                inp_list,
                warp_feature,
                disp,
                iter32=self.args.n_gru_layers == 3,
                iter16=self.args.n_gru_layers >= 2,
            )
            delta_disp = self._clamp_delta_disp(delta_disp)
            disp = disp + delta_disp
            disp_up = self.upsample_prediction(disp, up_mask)
            disp_predictions.append(disp_up)
        if test_mode:
            return disp_up

        return init_disp_predictions , disp_predictions
