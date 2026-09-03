import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru3 import (
    DEFOMStereo as ExistingGatedGRU3DEFOMStereo,
)
from core.pivno_models.defom_pivno_gwc4_enc16_concat_gru3 import (
    DEFOMStereo as DirectConcatGRU3DEFOMStereo,
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


class DefomPivnoGWC4Enc16ConcatGRU3Tests(unittest.TestCase):
    def test_model_is_isolated_and_has_no_scale_gate(self):
        model = DirectConcatGRU3DEFOMStereo(make_args())
        existing = ExistingGatedGRU3DEFOMStereo(make_args())

        self.assertIsNot(type(model), type(existing))
        self.assertFalse(hasattr(model, "scale_gate"))
        self.assertTrue(hasattr(existing, "scale_gate"))
        self.assertFalse(any(
            name.startswith("scale_gate.")
            for name, _ in model.named_parameters()
        ))

    def test_fixed_gwc4_enc16_no_left_contract(self):
        model = DirectConcatGRU3DEFOMStereo(make_args())

        self.assertEqual(
            model.MODEL_VARIANT,
            "defom_pivno_gwc4_enc16_concat_gru3",
        )
        self.assertEqual(
            model.FUSION_MODE,
            "gwc4_enc16_direct_concat_no_gate_no_left",
        )
        self.assertEqual(model.low_channel.out_channels, 32)
        self.assertEqual(model.MATCH_NUM_GROUPS, 4)
        self.assertEqual(model.sample_match_encoder[0].in_channels, 36)
        self.assertEqual(model.sample_match_encoder[0].out_channels, 16)
        self.assertEqual(model.refine_right_fuse[0].in_channels, 432)

    def test_all_grus_use_3x3_kernels(self):
        model = DirectConcatGRU3DEFOMStereo(make_args())

        for gru in (
            model.update_block.gru08,
            model.update_block.gru16,
            model.update_block.gru32,
        ):
            self.assertEqual(gru.convz.kernel_size, (3, 3))
            self.assertEqual(gru.convr.kernel_size, (3, 3))
            self.assertEqual(gru.convq.kernel_size, (3, 3))

    def test_encoded_candidates_are_directly_concatenated(self):
        torch.manual_seed(5)
        model = DirectConcatGRU3DEFOMStereo(make_args()).eval()
        left = torch.randn(1, 32, 2, 3)
        sampled = torch.randn(1, 3, 9, 32, 2, 3)

        with torch.no_grad():
            direct = model._encode_direct_concat(left, sampled)
            raw = encode_sampled_right_features(
                left,
                sampled,
                num_groups=4,
            )
            manual = model.sample_match_encoder(
                raw.reshape(27, 36, 2, 3)
            ).reshape(1, 3, 9, 16, 2, 3).flatten(1, 3)

        self.assertEqual(tuple(direct.shape), (1, 432, 2, 3))
        self.assertTrue(torch.equal(direct, manual))

    def test_gated_gwc4_checkpoint_diff_is_only_gate_keys(self):
        model = DirectConcatGRU3DEFOMStereo(make_args())
        simulated_gated_state = dict(model.state_dict())
        simulated_gated_state.update({
            "scale_gate.0.weight": torch.empty(32, 59, 3, 3),
            "scale_gate.0.bias": torch.empty(32),
            "scale_gate.2.weight": torch.empty(3, 32, 1, 1),
            "scale_gate.2.bias": torch.empty(3),
        })

        incompatible = model.load_state_dict(
            simulated_gated_state,
            strict=False,
        )

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(
            set(incompatible.unexpected_keys),
            {
                "scale_gate.0.weight",
                "scale_gate.0.bias",
                "scale_gate.2.weight",
                "scale_gate.2.bias",
            },
        )


if __name__ == "__main__":
    unittest.main()
