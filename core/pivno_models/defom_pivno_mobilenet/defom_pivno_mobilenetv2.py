"""Isolated MobileNetV2 feature-backbone variant of RGB DEFOM-PIVNO."""

from __future__ import annotations

from PIVNO.models.sronet_mobilenetv2 import PIVNOMobileNetV2

from core.pivno_models.defom_pivno import DEFOMStereo as BaseDEFOMStereo


class DEFOMStereo(BaseDEFOMStereo):
    """DEFOM-PIVNO with a MobileNetV2 multi-scale stereo image encoder."""

    MODEL_VARIANT = "defom_pivno_mobilenetv2"
    FEATURE_BACKBONE = PIVNOMobileNetV2.FEATURE_BACKBONE

    def __init__(self, args):
        # Construct the unchanged RGB DEFOM-PIVNO modules, then replace only
        # this new subclass's PIVNO instance. Existing model classes and their
        # state-dict layouts are therefore untouched.
        super().__init__(args)
        self.pivno = PIVNOMobileNetV2(input_channels=3)
