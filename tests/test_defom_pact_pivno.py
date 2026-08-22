import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pact_pivno import DEFOMStereo


def make_args(n_gru_layers=3, pivno_input_channels=3):
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=n_gru_layers,
        context_norm="instance",
        corr_levels=4,
        corr_radius=4,
        pivno_input_channels=pivno_input_channels,
    )


class DefomPactPivnoTests(unittest.TestCase):
    def test_scheme3_fuser_channels_match_update_block(self):
        model = DEFOMStereo(make_args())

        self.assertEqual(model.pivno.input_channels, 3)
        self.assertEqual(model.pivno.snet.conv1.in_channels, 3)
        self.assertEqual(model.refine_right_fuse[0].in_channels, 896)
        self.assertEqual(model.refine_right_fuse[-2].out_channels, 128)
        self.assertEqual(model.update_block.encoder.convc1.in_channels, 128)
        self.assertEqual(model.max_delta_disp, 16.0)

    def test_pivno_rgb_preprocessing_preserves_three_channels(self):
        image = torch.tensor(
            [[[[0.0]], [[127.5]], [[255.0]]]], dtype=torch.float32
        )

        prepared = DEFOMStereo._to_pivno_rgb(image)

        self.assertEqual(tuple(prepared.shape), (1, 3, 1, 1))
        self.assertTrue(torch.equal(
            prepared,
            torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]]),
        ))

    def test_legacy_pivno_preprocessing_converts_rgb_to_luminance(self):
        image = torch.tensor(
            [[[[255.0]], [[0.0]], [[0.0]]]], dtype=torch.float32
        )

        prepared = DEFOMStereo._prepare_pivno_input(image, input_channels=1)

        self.assertEqual(tuple(prepared.shape), (1, 1, 1, 1))
        self.assertTrue(torch.allclose(prepared, torch.tensor([[[[0.299]]]])))

    def test_legacy_checkpoint_architecture_uses_one_input_channel(self):
        model = DEFOMStereo(make_args(pivno_input_channels=1))

        self.assertEqual(model.pivno.input_channels, 1)
        self.assertEqual(model.pivno.snet.conv1.in_channels, 1)

    def test_delta_disp_is_clamped_to_widest_scale_support(self):
        model = DEFOMStereo(make_args())
        delta = torch.tensor([-20.0, -16.0, 0.0, 16.0, 20.0])

        clamped = model._clamp_delta_disp(delta)

        self.assertTrue(torch.equal(
            clamped,
            torch.tensor([-16.0, -16.0, 0.0, 16.0, 16.0]),
        ))

    def test_cpu_forward_returns_full_resolution_predictions(self):
        model = DEFOMStereo(make_args()).eval()
        image1 = torch.rand(1, 3, 64, 64) * 255.0
        image2 = torch.rand(1, 3, 64, 64) * 255.0

        with torch.no_grad():
            init_predictions, predictions = model(image1, image2, iters=1)
            final_prediction = model(image1, image2, iters=1, test_mode=True)

        self.assertEqual(len(init_predictions), model.pivno.iters)
        self.assertEqual(len(predictions), 1)
        self.assertTrue(all(tuple(item.shape) == (1, 1, 64, 64) for item in init_predictions))
        self.assertTrue(all(tuple(item.shape) == (1, 1, 64, 64) for item in predictions))
        self.assertEqual(tuple(final_prediction.shape), (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
