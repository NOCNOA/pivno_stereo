import unittest
from types import SimpleNamespace

import torch

from PIVNO.models.sronet_mobilenetv2 import MobileNetV2FeatureEncoder
from core.pivno_models.defom_pivno import DEFOMStereo as BaseDEFOMStereo
from core.pivno_models.defom_pivno_mobilenetv2 import DEFOMStereo


def make_args():
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        context_norm="instance",
        corr_radius=4,
    )


class DefomPivnoMobileNetV2Tests(unittest.TestCase):
    def test_encoder_fuses_complete_backbone_to_quarter_resolution(self):
        encoder = MobileNetV2FeatureEncoder().eval()
        left = torch.rand(1, 3, 64, 96)
        right = torch.rand(1, 3, 64, 96)

        with torch.no_grad():
            left_feature, right_feature = encoder((left, right))

        self.assertEqual(tuple(left_feature.shape), (1, 64, 16, 24))
        self.assertEqual(tuple(right_feature.shape), (1, 64, 16, 24))

    def test_variant_does_not_change_base_state_dict_layout(self):
        base = BaseDEFOMStereo(make_args())
        mobile = DEFOMStereo(make_args())
        base_keys = set(base.state_dict())
        mobile_keys = set(mobile.state_dict())

        self.assertIn("pivno.snet.conv1.weight", base_keys)
        self.assertNotIn("pivno.snet.conv1.weight", mobile_keys)
        self.assertIn("pivno.snet.backbone.0.0.weight", mobile_keys)
        self.assertNotIn("pivno.snet.backbone.0.0.weight", base_keys)
        self.assertEqual(mobile.MODEL_VARIANT, "defom_pivno_mobilenetv2")

    def test_cpu_forward_preserves_existing_prediction_contract(self):
        model = DEFOMStereo(make_args()).eval()
        left = torch.rand(1, 3, 64, 96) * 255.0
        right = torch.rand(1, 3, 64, 96) * 255.0

        with torch.no_grad():
            init_predictions, recurrent_predictions = model(
                left,
                right,
                iters=1,
            )

        self.assertEqual(len(init_predictions), model.pivno.iters)
        self.assertEqual(len(recurrent_predictions), 1)
        self.assertTrue(all(
            tuple(prediction.shape) == (1, 1, 64, 96)
            for prediction in init_predictions
        ))
        self.assertEqual(
            tuple(recurrent_predictions[-1].shape),
            (1, 1, 64, 96),
        )


if __name__ == "__main__":
    unittest.main()
