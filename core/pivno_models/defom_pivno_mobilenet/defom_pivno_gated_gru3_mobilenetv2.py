"""Gated GRU3 with IGEV++ image features and recurrent disparity upsampling.

Reference: gangweix/IGEV-plusplus, core/extractor.py::Feature.
The FPN and extra stem each produce 48 channels at stride 4. Their 96-channel
concatenation is projected to PIVNO's 64-channel contract.
"""

from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
import torch.nn.functional as F

from core.pivno_models.defom_pivno_gated_gru3 import DEFOMStereo as BaseStereo


class MobileNetV2PIVNOEncoder(nn.Module):
    """Accept PIVNO RGB [0,1] tensors or a stereo list; return stride-4 C64."""

    def __init__(self, pretrained=True):
        super().__init__()
        # Keep this dependency local so importing the legacy model is unaffected.
        import timm
        from core.pivno_models.igevpp_match_encoder import BasicConv, MobileNetV2FPN

        self.feature = MobileNetV2FPN()
        if pretrained:
            # Prefer an existing official timm cache over a Hugging Face request.
            # A missing cache uses timm's normal download; failures are not hidden.
            config = timm.models.get_pretrained_cfg('mobilenetv2_100')
            cached = Path(torch.hub.get_dir()) / 'checkpoints' / Path(
                urlparse(config.url).path
            ).name
            overlay = {'file': str(cached)} if cached.is_file() else None
            backbone = timm.create_model(
                'mobilenetv2_100', pretrained=True, features_only=True,
                pretrained_cfg_overlay=overlay,
            )
            self.feature.conv_stem = backbone.conv_stem
            self.feature.bn1 = backbone.bn1
            # Recent timm integrates the activation into bn1 (BatchNormAct2d).
            self.feature.act1 = getattr(backbone, 'act1', nn.Identity())
            boundaries = (0, 1, 2, 3, 5, 6)
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                setattr(self.feature, f'block{index}', nn.Sequential(
                    *backbone.blocks[start:end]
                ))
        # IGEV++'s extra RGB stem, parallel to its MobileNetV2/FPN.
        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
        self.projection = nn.Conv2d(96, 64, kernel_size=1)

    def forward(self, images, return_stem=False):
        is_pair = isinstance(images, (tuple, list))
        if is_pair:
            if len(images) != 2 or images[0].shape != images[1].shape:
                raise ValueError('Expected two equally shaped stereo tensors')
            batch = images[0].shape[0]
            images = torch.cat(images, dim=0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f'Expected RGB [B,3,H,W], got {tuple(images.shape)}')
        # IGEV++ feeds 2*(RGB/255)-1 to Feature. PIVNO already divides by 255.
        images = (2.0 * images - 1.0).contiguous()
        feature4 = self.feature(images)
        stem2 = self.stem_2(images)
        stem4 = self.stem_4(stem2)
        features = self.projection(torch.cat((feature4, stem4), dim=1))
        if is_pair:
            features = features.split(batch, dim=0)
            stem2 = stem2.split(batch, dim=0)
        return (features, stem2) if return_stem else features


class IGEVContextUpsampler(nn.Module):
    """IGEV++ mask_feat_4 + spx_2_gru + spx_gru + context_upsample."""

    def __init__(self, hidden_channels):
        super().__init__()
        from core.pivno_models.igevpp_match_encoder import Conv2x

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spx_2_gru = Conv2x(64, 32, deconv=True, concat=True)
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(64, 9, kernel_size=4, stride=2, padding=1),
        )

    @staticmethod
    def context_upsample(disparity, weights):
        """Apply full-resolution weights; disparity uses stride-4 pixel units."""
        batch, _, height, width = disparity.shape
        candidates = F.unfold(4.0 * disparity, kernel_size=3, padding=1)
        candidates = candidates.reshape(batch, 9, height, width)
        candidates = F.interpolate(
            candidates, size=(height * 4, width * 4), mode='nearest'
        )
        return (candidates * weights).sum(dim=1, keepdim=True)

    def forward(self, disparity, hidden, left_stem2):
        feature4 = self.mask_feat_4(hidden)
        feature2 = self.spx_2_gru(feature4, left_stem2)
        weights = torch.softmax(self.spx_gru(feature2), dim=1)
        return self.context_upsample(disparity, weights)


class DEFOMStereo(BaseStereo):
    """Keep PIVNO initialization/lookup/GRUs; replace features and upsampling."""

    MODEL_VARIANT = 'defom_pivno_gated_gru3_mobilenetv2'
    ENCODER_CONFIG = {
        'feature_backbone': 'timm_mobilenetv2_100_igevpp_fpn',
        'pivno_feature_encoder': 'igevpp_feature_stem_96_project_64',
        'pivno_encoder_normalization': 'rgb_0_1_to_minus1_1',
        'pivno_encoder_output_channels': 64,
        'pivno_upsampling': 'igevpp_context_stem2_all_refinement_iters',
    }

    def __init__(self, args):
        super().__init__(args)
        self.imagenet_pretrained = bool(getattr(args, 'pivno_mobilenet_pretrained', True))
        restoring = bool(
            getattr(args, 'resume_ckpt', None) or getattr(args, 'restore_ckpt', None)
        )
        self.pivno.snet = MobileNetV2PIVNOEncoder(
            pretrained=self.imagenet_pretrained and not restoring
        )
        # No unused legacy 144-channel mask parameters in the optimizer/DDP.
        del self.update_block.mask
        self.igev_upsampler = IGEVContextUpsampler(int(args.hidden_dims[2]))

    def _initialize_with_stem(self, image1, image2):
        """Original PIVNO initialization, reusing the encoder's half-res stem.

        This mirrors PIVNO.forward after snet without changing its attention,
        internal GRU, predictions or disparity units. snet runs only once.
        """
        (left, right), (left_stem2, _) = self.pivno.snet(
            [image1, image2], return_stem=True
        )
        smap = self.pivno.query_rgb(torch.cat((left, right), dim=1))
        hidden = smap.clone()
        predictions = []
        output_size = image1.shape[-2:]
        for _ in range(self.pivno.iters):
            hidden = self.pivno.gru(hidden, smap)
            disparity = self.pivno.disp_head(hidden)
            scale_x = output_size[1] / disparity.shape[-1]
            predictions.append(F.interpolate(
                disparity, size=output_size, mode='bilinear', align_corners=False
            ) * scale_x)
        return predictions, left, right, disparity, left_stem2

    def _forward_impl(self, image1, image2, iters=12, scale_iters=None,
                      test_mode=False):
        del scale_iters
        pivno_image1 = self._to_pivno_rgb(image1).contiguous().float()
        pivno_image2 = self._to_pivno_rgb(image2).contiguous().float()
        init_predictions, left4, right4, disparity, left_stem2 = (
            self._initialize_with_stem(pivno_image1, pivno_image2)
        )
        context_image = ((image1 - self.mean) / self.std).contiguous().float()
        contexts = self.cnet(context_image, num_layers=self.args.n_gru_layers)
        net = [torch.tanh(item[0]) for item in contexts]
        inp = [torch.relu(item[1]) for item in contexts]
        inp = [list(conv(item).chunk(3, dim=1))
               for item, conv in zip(inp, self.context_zqr_convs)]
        left_low = self.low_channel(left4)
        right_low = self.low_channel(right4)
        right_half, right_quarter = self.right_width_compressor(right_low)
        right_pyramid = (right_low, right_half, right_quarter)
        predictions = []
        for _ in range(iters):
            disparity = disparity.detach()
            warp_feature = self._refine_warp_feature(left_low, right_pyramid, disparity)
            # Identical GRU update, omitting only the old convex-mask head.
            net = self.update_block(
                net, inp, warp_feature, disparity,
                iter32=self.args.n_gru_layers == 3,
                iter16=self.args.n_gru_layers >= 2, update=False,
            )
            delta = self.update_block.disp_head(net[0])
            disparity = disparity + self._clamp_delta_disp(delta)
            predictions.append(self.igev_upsampler(disparity, net[0], left_stem2))
        return predictions[-1] if test_mode else (init_predictions, predictions)
