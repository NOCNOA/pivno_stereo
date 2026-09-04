import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru3 import (
    DEFOMStereo as ResidualGWCDEFOMStereo,
)
from core.pivno_models.defom_pivno_gated_gru3_gwc_only import (
    DEFOMStereo as GWCOnlyDEFOMStereo,
    encode_group_correlations,
)


def make_args():
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        context_norm="instance",
        corr_radius=4,
        mixed_precision=False,
    )


class DefomPivnoGatedGRU3GWCOnlyTests(unittest.TestCase):
    def test_model_is_standalone(self):
        self.assertEqual(GWCOnlyDEFOMStereo.__bases__, (torch.nn.Module,))
        self.assertFalse(issubclass(
            GWCOnlyDEFOMStereo,
            ResidualGWCDEFOMStereo,
        ))

    def test_group_correlation_only_encoding(self):
        left = torch.ones(1, 48, 1, 1)
        sampled = torch.stack(
            [left, -left, torch.zeros_like(left)],
            dim=1,
        ).reshape(1, 1, 3, 48, 1, 1)

        encoded = encode_group_correlations(
            left,
            sampled,
            num_groups=8,
        )

        self.assertEqual(tuple(encoded.shape), (1, 1, 3, 8, 1, 1))
        self.assertTrue(torch.allclose(
            encoded[0, 0, 0],
            torch.ones(8, 1, 1),
        ))
        self.assertTrue(torch.allclose(
            encoded[0, 0, 1],
            -torch.ones(8, 1, 1),
        ))
        self.assertTrue(torch.equal(
            encoded[0, 0, 2],
            torch.zeros(8, 1, 1),
        ))

    def test_gwc_only_channel_contract(self):
        model = GWCOnlyDEFOMStereo(make_args())

        self.assertEqual(
            model.MODEL_VARIANT,
            "defom_pivno_gated_gru3_gwc_only",
        )
        self.assertEqual(model.low_channel.out_channels, 48)
        self.assertEqual(model.MATCH_NUM_GROUPS, 8)
        self.assertEqual(
            model.RIGHT_SAMPLE_ENCODING,
            "gwc8_only_conv16_no_left_concat",
        )
        self.assertEqual(model.sample_match_encoder[0].in_channels, 8)
        self.assertEqual(model.sample_match_encoder[0].out_channels, 16)
        self.assertEqual(model.refine_right_fuse[0].in_channels, 432)

    def test_only_parameter_shape_change_is_match_encoder_input(self):
        original = ResidualGWCDEFOMStereo(make_args())
        gwc_only = GWCOnlyDEFOMStereo(make_args())
        original_state = original.state_dict()
        gwc_only_state = gwc_only.state_dict()

        self.assertEqual(set(original_state), set(gwc_only_state))
        different_shapes = {
            name: (original_state[name].shape, gwc_only_state[name].shape)
            for name in original_state
            if original_state[name].shape != gwc_only_state[name].shape
        }
        self.assertEqual(
            different_shapes,
            {
                "sample_match_encoder.0.weight": (
                    torch.Size([16, 56, 1, 1]),
                    torch.Size([16, 8, 1, 1]),
                ),
            },
        )

    def test_zero_samples_have_finite_gradients(self):
        left = torch.randn(1, 48, 2, 3, requires_grad=True)
        sampled = torch.randn(1, 3, 2, 48, 2, 3, requires_grad=True)
        with torch.no_grad():
            sampled[:, :, -1].zero_()

        encoded = encode_group_correlations(
            left,
            sampled,
            num_groups=8,
        )
        encoded.square().mean().backward()

        self.assertTrue(torch.isfinite(left.grad).all())
        self.assertTrue(torch.isfinite(sampled.grad).all())


if __name__ == "__main__":
    unittest.main()
