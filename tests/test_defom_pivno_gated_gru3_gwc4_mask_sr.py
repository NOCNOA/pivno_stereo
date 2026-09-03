import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import (
    CheckpointCompatibleGWC4GatedGRU3,
    DEFOMStereo as SRModel,
)


def make_args(stage="head"):
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        context_norm="instance",
        corr_levels=4,
        corr_radius=4,
        max_disp=768,
        mixed_precision=False,
        pivno_mask_sr_stage=stage,
        pivno_mask_sr_residual_max=4.0,
        lr=1e-4,
    )


class GatedGRU3GWC4MaskSRTests(unittest.TestCase):
    def test_restored_base_shapes_match_completed_gwc4_checkpoint(self):
        model = CheckpointCompatibleGWC4GatedGRU3(make_args())

        self.assertEqual(tuple(model.low_channel.weight.shape), (32, 64, 1, 1))
        self.assertEqual(
            tuple(model.sample_match_encoder[0].weight.shape),
            (16, 36, 1, 1),
        )
        self.assertEqual(
            tuple(model.scale_gate[0].weight.shape),
            (32, 59, 3, 3),
        )
        self.assertEqual(model.MATCH_NUM_GROUPS, 4)

    def test_base_state_load_allows_only_sr_head_keys(self):
        base = CheckpointCompatibleGWC4GatedGRU3(make_args())
        model = SRModel(make_args())

        incompatible = model.load_state_dict(base.state_dict(), strict=False)
        expected_missing = {
            key for key in model.state_dict() if key.startswith("sr_head.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_zero_head_matches_gwc4_gated_base_and_runs_once(self):
        torch.manual_seed(11)
        base = CheckpointCompatibleGWC4GatedGRU3(make_args()).eval()
        model = SRModel(make_args()).eval()
        model.load_state_dict(base.state_dict(), strict=False)
        calls = []
        handle = model.sr_head.register_forward_hook(
            lambda _module, _inputs, _output: calls.append(1)
        )
        left = torch.rand(1, 3, 64, 64) * 255.0
        right = torch.rand(1, 3, 64, 64) * 255.0
        try:
            with torch.no_grad():
                base_init, base_recurrent = base(left, right, iters=2)
                sr_init, sr_recurrent, aux = model(
                    left,
                    right,
                    iters=2,
                    return_sr_aux=True,
                )
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        for expected, actual in zip(base_init, sr_init):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(base_recurrent, sr_recurrent):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(
            torch.equal(sr_recurrent[-1], aux["disp_mask_base"])
        )
        self.assertEqual(int(torch.count_nonzero(aux["sr_delta_d"])), 0)

    def test_head_stage_freezes_base_parameters_and_bn_state(self):
        model = SRModel(make_args()).train()
        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(model.training)
        self.assertTrue(model.sr_head.training)
        self.assertFalse(model.pivno.training)
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("sr_head.") for name in trainable))

    def test_full_backward_reaches_only_sr_head(self):
        torch.manual_seed(12)
        model = SRModel(make_args()).train()
        left = torch.rand(1, 3, 64, 64) * 255.0
        right = torch.rand(1, 3, 64, 64) * 255.0

        _, predictions = model(left, right, iters=1)
        predictions[-1].square().mean().backward()

        for name, parameter in model.named_parameters():
            if name.startswith("sr_head."):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)


if __name__ == "__main__":
    unittest.main()
