"""IGEV++-style MobileNetV2 encoder with frozen BN and convex upsampling.

The tensor architecture is identical to the MobileNetV2 convex variant.  Its
separate model identity selects the IGEV++ training policy: preserve timm's
BatchNormAct2d activations, skip blanket SyncBatchNorm conversion, and keep BN
running statistics frozen while the remaining parameters train.
"""

from core.pivno_models.defom_pivno_mobilenet.defom_pivno_gated_gru3_mobilenetv2_convex import (
    DEFOMStereo as MobileNetV2ConvexStereo,
)


class DEFOMStereo(MobileNetV2ConvexStereo):
    MODEL_VARIANT = 'defom_pivno_gated_gru3_mobilenetv2_convex_bnfreeze'
    ENCODER_CONFIG = dict(
        MobileNetV2ConvexStereo.ENCODER_CONFIG,
        pivno_bn_training_policy='igevpp_frozen_batchnorm_preserve_activation',
    )
