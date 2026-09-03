import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_rgb_hidden_sr import (
    DEFOMStereo as FusionSRModel,
    RGBHiddenFusionSRHead,
)
from core.pivno_models.defom_pivno_gated_gru3_gwc4_mask_sr import (
    DEFOMStereo as HiddenSRModel,
)
from core.pivno_models.defom_pivno_gwc4_enc16_concat_gru3_mask_sr import (
    SimpleMaskGuidedSRHead,
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
        pivno_mask_sr_pretrained_lr=1e-5,
        lr=1e-4,
    )


class RGBHiddenFusionSRHeadTests(unittest.TestCase):
    def test_hidden_identity_rgb_zero_matches_hidden_head_exactly(self):
        torch.manual_seed(31)
        hidden_head = SimpleMaskGuidedSRHead(max_disp=768)
        torch.nn.init.normal_(hidden_head.conv2.weight, std=0.01)
        fusion_head = RGBHiddenFusionSRHead(max_disp=768)

        incompatible = fusion_head.load_state_dict(
            hidden_head.state_dict(), strict=False
        )
        expected_missing = {
            key
            for key in fusion_head.state_dict()
            if key.startswith("image_encoder.")
            or key.startswith("feature_fusion.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])

        disparity = torch.randn(2, 1, 4, 5)
        mask = torch.randn(2, 9 * 4 * 4, 4, 5)
        hidden = torch.randn(2, 128, 4, 5)
        left = torch.randn(2, 3, 16, 20)
        expected, expected_aux = hidden_head(
            disparity, mask, hidden, factor=4
        )
        actual, actual_aux = fusion_head(
            disparity, mask, hidden, left, factor=4
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(
            torch.equal(actual_aux["disp_mask_base"], expected_aux["disp_mask_base"])
        )
        self.assertTrue(
            torch.equal(actual_aux["sr_delta_d"], expected_aux["sr_delta_d"])
        )

    def test_initial_backward_reaches_rgb_fusion_weights(self):
        torch.manual_seed(32)
        head = RGBHiddenFusionSRHead(max_disp=768)
        torch.nn.init.normal_(head.conv2.weight, std=0.01)
        disparity = torch.randn(1, 1, 4, 5)
        mask = torch.randn(1, 9 * 4 * 4, 4, 5)
        hidden = torch.randn(1, 128, 4, 5)
        left = torch.randn(1, 3, 16, 20)

        prediction, _ = head(disparity, mask, hidden, left, factor=4)
        prediction.square().mean().backward()

        rgb_weight_grad = head.feature_fusion.weight.grad[
            :, head.FEATURE_CHANNELS:
        ]
        self.assertGreater(float(rgb_weight_grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.image_encoder[0].weight.grad)
        self.assertTrue(torch.isfinite(head.image_encoder[0].weight.grad).all())


class GatedGRU3GWC4MaskRGBHiddenSRTests(unittest.TestCase):
    def test_hidden_sr_state_load_adds_only_rgb_fusion_tensors(self):
        hidden_model = HiddenSRModel(make_args())
        fusion_model = FusionSRModel(make_args())

        incompatible = fusion_model.load_state_dict(
            hidden_model.state_dict(), strict=False
        )
        expected_missing = {
            key
            for key in fusion_model.state_dict()
            if key.startswith("sr_head.image_encoder.")
            or key.startswith("sr_head.feature_fusion.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_identity_initialized_full_model_matches_hidden_sr(self):
        torch.manual_seed(33)
        hidden_model = HiddenSRModel(make_args()).eval()
        torch.nn.init.normal_(hidden_model.sr_head.conv2.weight, std=0.01)
        fusion_model = FusionSRModel(make_args()).eval()
        fusion_model.load_state_dict(hidden_model.state_dict(), strict=False)
        left = torch.rand(1, 3, 64, 64) * 255.0
        right = torch.rand(1, 3, 64, 64) * 255.0

        with torch.no_grad():
            hidden_output, hidden_aux = hidden_model(
                left,
                right,
                iters=2,
                test_mode=True,
                return_sr_aux=True,
            )
            fusion_output, fusion_aux = fusion_model(
                left,
                right,
                iters=2,
                test_mode=True,
                return_sr_aux=True,
            )

        self.assertTrue(torch.equal(fusion_output, hidden_output))
        self.assertTrue(
            torch.equal(
                fusion_aux["disp_mask_base"], hidden_aux["disp_mask_base"]
            )
        )
        self.assertTrue(
            torch.equal(fusion_aux["sr_delta_d"], hidden_aux["sr_delta_d"])
        )

    def test_head_stage_optimizer_uses_two_learning_rates(self):
        model = FusionSRModel(make_args()).train()
        groups = model.optimizer_parameter_groups(make_args())

        self.assertEqual(len(groups), 2)
        self.assertEqual([group["lr"] for group in groups], [1e-5, 1e-4])
        grouped_ids = {
            id(parameter)
            for group in groups
            for parameter in group["params"]
        }
        trainable_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(grouped_ids, trainable_ids)
        self.assertTrue(
            all(
                name.startswith("sr_head.")
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        )

    def test_full_backward_reaches_only_fusion_sr_head(self):
        torch.manual_seed(34)
        model = FusionSRModel(make_args()).train()
        torch.nn.init.normal_(model.sr_head.conv2.weight, std=0.01)
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
