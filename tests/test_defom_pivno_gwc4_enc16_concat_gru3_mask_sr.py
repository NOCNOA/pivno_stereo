import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gwc4_enc16_concat_gru3 import (
    DEFOMStereo as BaseModel,
)
from core.pivno_models.defom_pivno_gwc4_enc16_concat_gru3_mask_sr import (
    DEFOMStereo as SRModel,
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
        pivno_mask_sr_stage=stage,
        pivno_mask_sr_residual_max=4.0,
    )


class SimpleMaskGuidedSRHeadTests(unittest.TestCase):
    def test_zero_initialization_matches_base_convex_upsampling(self):
        torch.manual_seed(0)
        head = SimpleMaskGuidedSRHead(max_disp=768, residual_max=4.0)
        disparity = torch.randn(1, 1, 3, 4)
        mask = torch.randn(1, 9 * 4 * 4, 3, 4)
        hidden = torch.randn(1, 128, 3, 4)

        prediction, aux = head(disparity, mask, hidden, factor=4)

        base = BaseModel(make_args())
        expected = base.upsample_prediction(disparity, mask)
        self.assertEqual(tuple(prediction.shape), (1, 1, 12, 16))
        self.assertTrue(torch.equal(prediction, expected))
        self.assertTrue(torch.equal(aux["disp_mask_base"], expected))
        self.assertEqual(int(torch.count_nonzero(aux["sr_delta_d"])), 0)

    def test_sr_input_contract_and_backward_are_finite(self):
        torch.manual_seed(1)
        head = SimpleMaskGuidedSRHead(max_disp=768, residual_max=4.0)
        disparity = torch.randn(1, 1, 3, 4)
        mask = torch.randn(1, 9 * 4 * 4, 3, 4)
        hidden = torch.randn(1, 128, 3, 4)

        prediction, _ = head(disparity, mask, hidden, factor=4)
        prediction.square().mean().backward()

        self.assertEqual(head.conv1.in_channels, 53)
        self.assertEqual(head.conv2.out_channels, 1)
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(head.conv2.weight.grad.abs().sum()), 0.0)


class GWC4Enc16ConcatGRU3MaskSRTests(unittest.TestCase):
    def test_head_stage_freezes_base_parameters_and_state(self):
        model = SRModel(make_args(stage="head"))
        model.train()
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

    def test_base_checkpoint_load_allows_only_new_sr_keys(self):
        base = BaseModel(make_args())
        model = SRModel(make_args(stage="head"))

        incompatible = model.load_state_dict(base.state_dict(), strict=False)
        expected_missing = {
            key for key in model.state_dict() if key.startswith("sr_head.")
        }

        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_forward_calls_sr_once_and_preserves_output_contract(self):
        torch.manual_seed(2)
        model = SRModel(make_args(stage="head")).eval()
        calls = []
        handle = model.sr_head.register_forward_hook(
            lambda _module, _inputs, _output: calls.append(1)
        )
        image1 = torch.rand(1, 3, 64, 64) * 255.0
        image2 = torch.rand(1, 3, 64, 64) * 255.0
        try:
            with torch.no_grad():
                init_predictions, predictions, aux = model(
                    image1,
                    image2,
                    iters=2,
                    return_sr_aux=True,
                )
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(init_predictions), model.pivno.iters)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(tuple(predictions[-1].shape), (1, 1, 64, 64))
        self.assertTrue(torch.equal(predictions[-1], aux["disp_mask_base"]))
        self.assertEqual(int(torch.count_nonzero(aux["sr_delta_d"])), 0)

    def test_full_model_backward_reaches_only_sr_head(self):
        torch.manual_seed(3)
        model = SRModel(make_args(stage="head")).train()
        image1 = torch.rand(1, 3, 64, 64) * 255.0
        image2 = torch.rand(1, 3, 64, 64) * 255.0

        _, predictions = model(image1, image2, iters=1)
        predictions[-1].square().mean().backward()

        for name, parameter in model.named_parameters():
            if name.startswith("sr_head."):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertGreater(
            float(model.sr_head.conv2.weight.grad.abs().sum()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
