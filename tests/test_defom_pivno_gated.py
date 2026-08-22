import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno import DEFOMStereo as BaseDEFOMStereo
from core.pivno_models.defom_pivno_gated import DEFOMStereo as GatedDEFOMStereo


def make_args():
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        context_norm="instance",
        corr_radius=4,
    )


class DefomPivnoGatedTests(unittest.TestCase):
    def test_uniform_initial_gate_preserves_every_scale_branch(self):
        model = GatedDEFOMStereo(make_args())
        left = torch.randn(2, 32, 3, 5)
        sampled = torch.randn(2, 3, 9, 32, 3, 5)

        weighted, weights = model._apply_scale_gate(left, sampled)

        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1.0 / 3.0)))
        self.assertTrue(torch.allclose(weighted, sampled, atol=1e-6, rtol=1e-6))
        self.assertEqual(model.refine_right_fuse[0].in_channels, 896)

    def test_base_checkpoint_is_missing_only_gate_parameters(self):
        base = BaseDEFOMStereo(make_args())
        gated = GatedDEFOMStereo(make_args())

        incompatible = gated.load_state_dict(base.state_dict(), strict=False)

        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            key.startswith("scale_gate.") for key in incompatible.missing_keys
        ))
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_gate_receives_gradient_from_weighted_branches(self):
        model = GatedDEFOMStereo(make_args())
        left = torch.randn(1, 32, 3, 5)
        sampled = torch.randn(1, 3, 9, 32, 3, 5)
        sampled[:, 1] += 1.0
        sampled[:, 2] -= 1.0

        weighted, _ = model._apply_scale_gate(left, sampled)
        weighted.sum().backward()

        gradient = model.scale_gate[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_identity_gate_matches_base_forward_after_warm_start(self):
        torch.manual_seed(7)
        base = BaseDEFOMStereo(make_args()).eval()
        gated = GatedDEFOMStereo(make_args()).eval()
        gated.load_state_dict(base.state_dict(), strict=False)
        image1 = torch.rand(1, 3, 64, 64) * 255.0
        image2 = torch.rand(1, 3, 64, 64) * 255.0

        with torch.no_grad():
            base_prediction = base(image1, image2, iters=1, test_mode=True)
            gated_prediction = gated(image1, image2, iters=1, test_mode=True)

        self.assertTrue(torch.allclose(
            gated_prediction,
            base_prediction,
            atol=1e-5,
            rtol=1e-5,
        ))

    def test_optimizer_groups_use_separate_base_and_gate_rates(self):
        model = GatedDEFOMStereo(make_args())
        optimizer_args = SimpleNamespace(lr=2e-5, pivno_gate_lr=2e-4)

        groups = model.optimizer_parameter_groups(optimizer_args)

        self.assertEqual([group["lr"] for group in groups], [2e-5, 2e-4])
        base_ids = {id(parameter) for parameter in groups[0]["params"]}
        gate_ids = {id(parameter) for parameter in groups[1]["params"]}
        self.assertFalse(base_ids & gate_ids)
        self.assertEqual(
            base_ids | gate_ids,
            {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
        )


if __name__ == "__main__":
    unittest.main()
