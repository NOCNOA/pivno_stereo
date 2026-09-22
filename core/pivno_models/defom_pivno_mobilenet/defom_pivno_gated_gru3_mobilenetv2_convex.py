"""MobileNetV2 PIVNO encoder with the original gated-GRU3 upsampler.

This controlled variant changes only ``pivno.snet``. The PIVNO initializer,
right-feature sampling, scale gate, recurrent updates, 144-channel mask head,
and original convex ``upsample_flow`` path are inherited without modification.
"""

from core.pivno_models.defom_pivno_gated_gru3 import DEFOMStereo as BaseStereo
from core.pivno_models.defom_pivno_mobilenet.defom_pivno_gated_gru3_mobilenetv2 import (
    MobileNetV2PIVNOEncoder,
)


class DEFOMStereo(BaseStereo):
    """Replace only PIVNO's RGB feature encoder with MobileNetV2/FPN/stem."""

    MODEL_VARIANT = 'defom_pivno_gated_gru3_mobilenetv2_convex'
    ENCODER_CONFIG = {
        'feature_backbone': 'timm_mobilenetv2_100_igevpp_fpn',
        'pivno_feature_encoder': 'igevpp_feature_stem_96_project_64',
        'pivno_encoder_normalization': 'rgb_0_1_to_minus1_1',
        'pivno_encoder_output_channels': 64,
        'pivno_upsampling': 'original_gated_gru3_convex_mask_144',
    }

    def __init__(self, args):
        super().__init__(args)
        self.imagenet_pretrained = bool(
            getattr(args, 'pivno_mobilenet_pretrained', True)
        )
        restoring = bool(
            getattr(args, 'resume_ckpt', None)
            or getattr(args, 'restore_ckpt', None)
        )
        self.pivno.snet = MobileNetV2PIVNOEncoder(
            pretrained=self.imagenet_pretrained and not restoring
        )
