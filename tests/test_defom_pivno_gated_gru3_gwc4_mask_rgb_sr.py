import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_rgb_sr import (
    DEFOMStereo as RGBSRModel,
    FullResolutionRGBGuidedSRHead,
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


class FullResolutionRGBGuidedSRHeadTests(unittest.TestCase):
    def test_zero_head_reproduces_convex_base(self):
        torch.manual_seed(21)
        head = FullResolutionRGBGuidedSRHead(
            max_disp=768,
            residual_max=4.0,
        )
        disparity = torch.randn(2, 1, 4, 5)
        mask = torch.randn(2, 9 * 4 * 4, 4, 5)
        left = torch.randn(2, 3, 16, 20)

        prediction, aux = head(disparity, mask, left, factor=4)

        self.assertTrue(torch.equal(prediction, aux["disp_mask_base"]))
        self.assertEqual(int(torch.count_nonzero(aux["sr_delta_d"])), 0)
        self.assertEqual(tuple(prediction.shape), (2, 1, 16, 20))

    def test_residual_path_uses_full_resolution_left_image(self):
        torch.manual_seed(22)
        head = FullResolutionRGBGuidedSRHead(
            max_disp=768,
            residual_max=4.0,
        )
        torch.nn.init.normal_(head.conv2.weight, std=0.01)
        disparity = torch.randn(1, 1, 4, 5)
        mask = torch.randn(1, 9 * 4 * 4, 4, 5)
        left = torch.randn(1, 3, 16, 20, requires_grad=True)

        prediction, _ = head(disparity, mask, left, factor=4)
        prediction.square().mean().backward()

        self.assertIsNotNone(left.grad)
        self.assertGreater(float(left.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.image_encoder[0].weight.grad)
        self.assertGreater(
            float(head.image_encoder[0].weight.grad.abs().sum()),
            0.0,
        )

    def test_rejects_misaligned_full_resolution_image(self):
        head = FullResolutionRGBGuidedSRHead(max_disp=768)
        disparity = torch.randn(1, 1, 4, 5)
        mask = torch.randn(1, 9 * 4 * 4, 4, 5)
        left = torch.randn(1, 3, 15, 20)

        with self.assertRaisesRegex(ValueError, "left_image must be"):
            head(disparity, mask, left, factor=4)


class GatedGRU3GWC4MaskRGBSRTests(unittest.TestCase):
    def test_base_state_load_allows_only_new_rgb_sr_head(self):
        base = CheckpointCompatibleGWC4GatedGRU3(make_args())
        model = RGBSRModel(make_args())

        incompatible = model.load_state_dict(base.state_dict(), strict=False)
        expected_missing = {
            key for key in model.state_dict() if key.startswith("sr_head.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(
            model.sr_head.FEATURE_SOURCE,
            "full_resolution_normalized_left_rgb",
        )

    def test_zero_head_matches_completed_base_and_runs_once(self):
        torch.manual_seed(23)
        base = CheckpointCompatibleGWC4GatedGRU3(make_args()).eval()
        model = RGBSRModel(make_args()).eval()
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
                rgb_init, rgb_recurrent, aux = model(
                    left,
                    right,
                    iters=2,
                    return_sr_aux=True,
                )
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        for expected, actual in zip(base_init, rgb_init):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(base_recurrent, rgb_recurrent):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(
            torch.equal(rgb_recurrent[-1], aux["disp_mask_base"])
        )
        self.assertEqual(int(torch.count_nonzero(aux["sr_delta_d"])), 0)

    def test_head_stage_freezes_everything_except_rgb_sr_head(self):
        model = RGBSRModel(make_args()).train()
        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("sr_head.") for name in trainable))
        self.assertTrue(model.sr_head.training)
        self.assertFalse(model.pivno.training)

    def test_full_backward_reaches_only_rgb_sr_head(self):
        torch.manual_seed(24)
        model = RGBSRModel(make_args()).train()
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
