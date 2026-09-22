import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru3 import (
    DEFOMStereo as BaseDEFOMStereo,
)
from core.pivno_models.defom_pivno_gated_gru3_bins import (
    DEFOMStereo,
    MultiRangeBinInitializer,
    PIVNOFeatureExtractor,
)


def make_args(stage="init"):
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        context_norm="instance",
        corr_radius=4,
        mixed_precision=False,
        max_disp=768,
        pivno_bins_stage=stage,
        pivno_num_init_bins=8,
        pivno_bins_max_offset=8.0,
        pivno_init_corr_scale=10.0,
    )


class DefomPivnoGatedGRU3BinsTests(unittest.TestCase):
    def test_registers_only_pivno_snet(self):
        model = DEFOMStereo(make_args())

        self.assertIsInstance(model.pivno, PIVNOFeatureExtractor)
        self.assertFalse(hasattr(model.pivno, "gru"))
        self.assertFalse(hasattr(model.pivno, "disp_head"))
        self.assertEqual(model.pivno.snet.conv1.in_channels, 3)

    def test_base_checkpoint_key_contract(self):
        base = BaseDEFOMStereo(make_args())
        bins = DEFOMStereo(make_args())
        incompatible = bins.load_state_dict(base.state_dict(), strict=False)

        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(name.startswith((
            "bin_initializer.", "initial_weight_head.",
        )) for name in incompatible.missing_keys))
        self.assertTrue(incompatible.unexpected_keys)
        self.assertTrue(all(name.startswith((
            "pivno.conv0.", "pivno.conv1.",
            "pivno.disp_head.", "pivno.gru.",
        )) for name in incompatible.unexpected_keys))

    def test_initializer_has_no_mono_channel(self):
        initializer = MultiRangeBinInitializer(
            feature_dim=8,
            num_bins=4,
            max_offset=4.0,
        )
        self.assertEqual(initializer.logit_refiners[0][0].in_channels, 4)

        left = torch.randn(1, 8, 2, 8)
        pyramid = (
            torch.randn(1, 8, 2, 8),
            torch.randn(1, 8, 2, 4),
            torch.randn(1, 8, 2, 2),
        )
        disps, logits, confidence, valid = initializer(left, pyramid)
        self.assertEqual(len(disps), 3)
        self.assertEqual(tuple(logits.shape), (1, 3, 4, 2, 8))
        self.assertEqual(tuple(confidence.shape), (1, 3, 2, 8))
        self.assertEqual(tuple(valid.shape), (1, 3, 4, 2, 8))
        self.assertTrue(all(torch.isfinite(disp).all() for disp in disps))

    def test_finetune_stages(self):
        init_model = DEFOMStereo(make_args("init"))
        init_trainable = {
            name for name, parameter in init_model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(init_trainable)
        self.assertTrue(all(name.startswith((
            "bin_initializer.", "initial_weight_head.",
        )) for name in init_trainable))

        joint_model = DEFOMStereo(make_args("joint"))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in joint_model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
