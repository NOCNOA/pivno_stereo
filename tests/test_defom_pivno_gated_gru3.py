import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated import (
    DEFOMStereo as GatedDEFOMStereo,
)
from core.pivno_models.defom_pivno_gated_gru3 import (
    DEFOMStereo as GatedGRU3DEFOMStereo,
)
from core.submodules import encode_sampled_right_features


def make_args():
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        context_norm="instance",
        corr_radius=4,
        mixed_precision=False,
    )


class DefomPivnoGatedGRU3Tests(unittest.TestCase):
    def test_new_model_is_standalone(self):
        self.assertEqual(GatedGRU3DEFOMStereo.__bases__, (torch.nn.Module,))

    def test_existing_gated_model_keeps_1x1_grus(self):
        model = GatedDEFOMStereo(make_args())

        for gru in (
            model.update_block.gru08,
            model.update_block.gru16,
            model.update_block.gru32,
        ):
            self.assertEqual(gru.convz.kernel_size, (1, 1))

    def test_new_model_uses_3x3_grus_for_all_gates(self):
        model = GatedGRU3DEFOMStereo(make_args())

        self.assertEqual(model.MODEL_VARIANT, "defom_pivno_gated_gru3")
        for gru in (
            model.update_block.gru08,
            model.update_block.gru16,
            model.update_block.gru32,
        ):
            self.assertEqual(gru.convz.kernel_size, (3, 3))
            self.assertEqual(gru.convr.kernel_size, (3, 3))
            self.assertEqual(gru.convq.kernel_size, (3, 3))

    def test_new_model_uses_rgb_pivno(self):
        model = GatedGRU3DEFOMStereo(make_args())

        self.assertEqual(model.pivno.input_channels, 3)
        self.assertEqual(model.pivno.snet.conv1.in_channels, 3)
        self.assertFalse(model.mixed_precision)
        self.assertEqual(
            model.AMP_POLICY,
            "fp16_compute_fp32_corr_attention_softmax",
        )

    def test_sample_encoder_concatenates_residual_and_group_correlation(self):
        left = torch.ones(1, 4, 1, 1)
        sampled = torch.stack(
            [left, -left, torch.zeros_like(left)],
            dim=1,
        ).reshape(1, 1, 3, 4, 1, 1)

        encoded = encode_sampled_right_features(
            left,
            sampled,
            num_groups=2,
        )

        self.assertEqual(tuple(encoded.shape), (1, 1, 3, 6, 1, 1))
        residual = encoded[:, :, :, :4]
        correlation = encoded[:, :, :, 4:]
        self.assertTrue(torch.equal(residual[0, 0, 0], torch.zeros_like(left[0])))
        self.assertTrue(torch.equal(residual[0, 0, 1], -2.0 * torch.ones_like(left[0])))
        self.assertTrue(torch.equal(residual[0, 0, 2], -torch.ones_like(left[0])))
        self.assertTrue(torch.allclose(correlation[0, 0, 0], torch.ones(2, 1, 1)))
        self.assertTrue(torch.allclose(correlation[0, 0, 1], -torch.ones(2, 1, 1)))
        self.assertTrue(torch.equal(correlation[0, 0, 2], torch.zeros(2, 1, 1)))

    def test_encoded_fuser_channel_count(self):
        model = GatedGRU3DEFOMStereo(make_args())

        self.assertEqual(model.MATCH_NUM_GROUPS, 4)
        self.assertEqual(
            model.RIGHT_SAMPLE_ENCODING,
            "residual_gwc4_conv16_no_left_concat",
        )
        # Each (32 residual + 4 GWC) sample is compressed from 36 to 16.
        self.assertEqual(model.sample_match_encoder[0].in_channels, 36)
        self.assertEqual(model.sample_match_encoder[0].out_channels, 16)
        # No direct left-feature concat: 3 scales * 9 samples * 16 channels.
        self.assertEqual(model.refine_right_fuse[0].in_channels, 432)

    def test_sample_encoder_has_finite_gradients_for_zero_samples(self):
        left = torch.randn(1, 8, 2, 3, requires_grad=True)
        sampled = torch.randn(1, 2, 3, 8, 2, 3, requires_grad=True)
        with torch.no_grad():
            sampled[:, :, -1].zero_()

        encoded = encode_sampled_right_features(
            left,
            sampled,
            num_groups=2,
        )
        encoded.square().mean().backward()

        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(sampled.grad).all())

    def test_scale_gate_reuses_four_group_correlations(self):
        model = GatedGRU3DEFOMStereo(make_args())
        left = torch.randn(1, 32, 2, 3)
        group_correlation = torch.randn(1, 3, 9, 4, 2, 3)

        weights = model._apply_scale_gate(left, group_correlation)

        self.assertEqual(
            model.SCALE_GATE_MODE,
            "gwc4_mean_softmax_weighted_encoded_concat",
        )
        self.assertEqual(tuple(weights.shape), (1, 3, 2, 3))
        self.assertTrue(torch.allclose(
            weights,
            torch.full_like(weights, 1.0 / 3.0),
        ))


if __name__ == "__main__":
    unittest.main()
