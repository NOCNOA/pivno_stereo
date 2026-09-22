import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_mobilenet.defom_pivno_gated_gru3_mobilenetv2 import (
    MobileNetV2PIVNOEncoder,
)
from core.pivno_models.defom_pivno_mobilenet.defom_pivno_gated_gru3_mobilenetv2_bins import (
    DEFOMStereo,
)


def args(stage):
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        context_norm='instance',
        corr_radius=4,
        mixed_precision=False,
        max_disp=768,
        lr=0.0002,
        pivno_bins_stage=stage,
        pivno_num_init_bins=8,
        pivno_bins_max_offset=8.0,
        pivno_init_corr_scale=10.0,
        pivno_mobilenet_pretrained=False,
    )


class MobileNetV2BinsTests(unittest.TestCase):
    def test_init_stage_reapplies_freeze_after_encoder_replacement(self):
        model = DEFOMStereo(args('init'))
        self.assertIsInstance(model.pivno.snet, MobileNetV2PIVNOEncoder)
        trainable = [name for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith(('bin_initializer.',
                                             'initial_weight_head.'))
                            for name in trainable))
        self.assertEqual(len(model.optimizer_parameter_groups(args('init'))), 1)

    def test_joint_stage_produces_full_resolution_initial_and_refined_disp(self):
        model = DEFOMStereo(args('joint')).eval()
        self.assertTrue(all(parameter.requires_grad
                            for parameter in model.parameters()))
        image1 = torch.rand(1, 3, 64, 128) * 255
        image2 = torch.rand(1, 3, 64, 128) * 255
        with torch.no_grad():
            initial, recurrent = model(image1, image2, iters=1)
        self.assertEqual(tuple(initial[-1].shape), (1, 1, 64, 128))
        self.assertEqual(tuple(recurrent[-1].shape), (1, 1, 64, 128))
        self.assertTrue(torch.isfinite(initial[-1]).all())
        self.assertTrue(torch.isfinite(recurrent[-1]).all())
        self.assertEqual(len(model.optimizer_parameter_groups(args('joint'))), 2)


if __name__ == '__main__':
    unittest.main()
