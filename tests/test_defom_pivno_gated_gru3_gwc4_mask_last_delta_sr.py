import math
import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_last_delta_sr import (
    DEFOMStereo as LastDeltaSRModel,
    LastDeltaWeightedSRHead,
)
from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import (
    CheckpointCompatibleGWC4GatedGRU3,
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


class LastDeltaWeightedSRHeadTests(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(41)
        disparity_base = torch.randn(2, 1, 16, 20)
        delta = torch.randn(2, 1, 4, 5)
        mask = torch.randn(2, 9 * 4 * 4, 4, 5)
        hidden = torch.randn(2, 128, 4, 5)
        return disparity_base, delta, mask, hidden

    def test_identity_initialization_preserves_final_disparity_exactly(self):
        head = LastDeltaWeightedSRHead(max_delta_disp_low=16.0)
        disparity_base, delta, mask, hidden = self._inputs()

        prediction, aux = head(
            disparity_base, delta, mask, hidden, factor=4
        )

        self.assertTrue(torch.equal(prediction, disparity_base))
        self.assertEqual(int(torch.count_nonzero(aux["delta_sr_residual"])), 0)
        self.assertEqual(int(torch.count_nonzero(aux["sr_delta_d"])), 0)
        self.assertTrue(torch.equal(
            aux["delta_sr_weight"],
            torch.ones_like(aux["delta_sr_weight"]),
        ))
        self.assertTrue(torch.equal(
            aux["delta_sr_refined"], aux["delta_mask_base"]
        ))

    def test_weight_replaces_only_last_delta_contribution(self):
        head = LastDeltaWeightedSRHead(max_delta_disp_low=16.0)
        with torch.no_grad():
            # 2*sigmoid(log(3)) = 1.5.
            head.weight_head.bias.fill_(math.log(3.0))
        disparity_base, delta, mask, hidden = self._inputs()

        prediction, aux = head(
            disparity_base, delta, mask, hidden, factor=4
        )

        self.assertTrue(torch.allclose(
            aux["delta_sr_weight"],
            torch.full_like(aux["delta_sr_weight"], 1.5),
        ))
        self.assertTrue(torch.allclose(
            prediction - disparity_base,
            0.5 * aux["delta_mask_base"],
            atol=1e-6,
            rtol=1e-6,
        ))

    def test_reconstructed_delta_matches_base_convex_upsampling(self):
        head = LastDeltaWeightedSRHead(max_delta_disp_low=16.0)
        base = CheckpointCompatibleGWC4GatedGRU3(make_args())
        disparity_base, delta, mask, hidden = self._inputs()

        _, aux = head(disparity_base, delta, mask, hidden, factor=4)
        expected = base.upsample_flow(delta, mask)

        self.assertTrue(torch.equal(aux["delta_mask_base"], expected))

    def test_both_output_branches_receive_initial_gradient(self):
        head = LastDeltaWeightedSRHead(max_delta_disp_low=16.0)
        disparity_base, delta, mask, hidden = self._inputs()
        prediction, _ = head(
            disparity_base, delta, mask, hidden, factor=4
        )

        prediction.square().mean().backward()

        self.assertGreater(float(head.delta_head.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(head.weight_head.weight.grad.abs().sum()), 0.0)


class GatedGRU3GWC4MaskLastDeltaSRTests(unittest.TestCase):
    def test_base_state_load_allows_only_new_sr_head(self):
        base = CheckpointCompatibleGWC4GatedGRU3(make_args())
        model = LastDeltaSRModel(make_args())

        incompatible = model.load_state_dict(base.state_dict(), strict=False)
        expected_missing = {
            key for key in model.state_dict() if key.startswith("sr_head.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_identity_full_model_matches_base_and_runs_only_at_final(self):
        torch.manual_seed(42)
        base = CheckpointCompatibleGWC4GatedGRU3(make_args()).eval()
        model = LastDeltaSRModel(make_args()).eval()
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
        self.assertTrue(torch.equal(
            sr_recurrent[-1], aux["disp_mask_base"]
        ))

    def test_head_stage_freezes_everything_except_last_delta_sr(self):
        model = LastDeltaSRModel(make_args()).train()
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(model.training)
        self.assertTrue(model.sr_head.training)
        self.assertFalse(model.pivno.training)
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("sr_head.") for name in trainable))

    def test_full_backward_reaches_only_last_delta_sr_head(self):
        torch.manual_seed(43)
        model = LastDeltaSRModel(make_args()).train()
        with torch.no_grad():
            torch.nn.init.normal_(model.sr_head.delta_head.weight, std=0.01)
            torch.nn.init.normal_(model.sr_head.weight_head.weight, std=0.01)
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
