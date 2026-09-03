import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru3_gwc4_last_delta_direct_sr import (
    DEFOMStereo as DirectLastDeltaSRModel,
    DirectLastDeltaSRHead,
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
        lr=1e-4,
    )


def make_head_inputs(batch=2, height=4, width=5):
    torch.manual_seed(51)
    hidden = torch.randn(batch, 128, height, width)
    disparity = torch.rand(batch, 1, height, width) * 192.0
    context = [
        torch.randn(batch, 128, height, width) for _ in range(3)
    ]
    motion = torch.randn(batch, 128, height, width)
    return hidden, disparity, context, motion


class DirectLastDeltaSRHeadTests(unittest.TestCase):
    def test_zero_initialization_outputs_no_final_increment(self):
        head = DirectLastDeltaSRHead(
            factor=4,
            max_disp_low=192.0,
            max_delta_disp_low=16.0,
        )
        delta, aux = head(*make_head_inputs())

        self.assertEqual(delta.shape, (2, 1, 16, 20))
        self.assertEqual(int(torch.count_nonzero(delta)), 0)
        self.assertTrue(torch.equal(delta, aux["delta_sr_hr"]))

    def test_pixelshuffle_channels_are_distinct_subpixel_predictions(self):
        head = DirectLastDeltaSRHead(
            factor=4,
            max_disp_low=192.0,
            max_delta_disp_low=16.0,
        )
        with torch.no_grad():
            head.delta_head.bias.copy_(
                torch.arange(1, 17, dtype=torch.float32) / 100.0
            )
        inputs = make_head_inputs(batch=1, height=2, width=3)

        delta, _ = head(*inputs)

        for sub_y in range(4):
            for sub_x in range(4):
                channel = 4 * sub_y + sub_x
                expected = 64.0 * torch.tanh(
                    torch.tensor((channel + 1) / 100.0)
                )
                actual = delta[0, 0, sub_y::4, sub_x::4]
                self.assertTrue(torch.allclose(
                    actual, torch.full_like(actual, expected)
                ))


class DirectLastDeltaSRModelTests(unittest.TestCase):
    def test_base_state_load_allows_only_new_sr_head(self):
        base = CheckpointCompatibleGWC4GatedGRU3(make_args())
        model = DirectLastDeltaSRModel(make_args())

        incompatible = model.load_state_dict(base.state_dict(), strict=False)
        expected_missing = {
            key for key in model.state_dict() if key.startswith("sr_head.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_final_is_previous_upsampled_prediction_plus_direct_delta(self):
        torch.manual_seed(52)
        base = CheckpointCompatibleGWC4GatedGRU3(make_args()).eval()
        model = DirectLastDeltaSRModel(make_args()).eval()
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
                direct_init, direct_recurrent, aux = model(
                    left,
                    right,
                    iters=2,
                    return_sr_aux=True,
                )
        finally:
            handle.remove()

        self.assertEqual(len(calls), 1)
        for expected, actual in zip(base_init, direct_init):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(
            direct_recurrent[0], base_recurrent[0]
        ))
        self.assertTrue(torch.equal(
            direct_recurrent[-1], direct_recurrent[-2]
        ))
        self.assertTrue(torch.equal(
            direct_recurrent[-1], aux["disp_previous_hr"]
        ))
        self.assertEqual(int(torch.count_nonzero(aux["delta_sr_hr"])), 0)

    def test_one_iteration_uses_last_pivno_prediction_as_previous_disp(self):
        torch.manual_seed(53)
        model = DirectLastDeltaSRModel(make_args()).eval()
        left = torch.rand(1, 3, 64, 64) * 255.0
        right = torch.rand(1, 3, 64, 64) * 255.0

        with torch.no_grad():
            init_predictions, recurrent, aux = model(
                left,
                right,
                iters=1,
                return_sr_aux=True,
            )

        self.assertTrue(torch.equal(recurrent[-1], init_predictions[-1]))
        self.assertTrue(torch.equal(
            recurrent[-1], aux["disp_previous_hr"]
        ))

    def test_head_stage_freezes_everything_except_direct_sr(self):
        model = DirectLastDeltaSRModel(make_args()).train()
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(model.training)
        self.assertTrue(model.sr_head.training)
        self.assertFalse(model.pivno.training)
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("sr_head.") for name in trainable))

    def test_full_backward_reaches_only_direct_sr_head(self):
        torch.manual_seed(54)
        model = DirectLastDeltaSRModel(make_args()).train()
        with torch.no_grad():
            torch.nn.init.normal_(model.sr_head.delta_head.weight, std=1e-4)
        left = torch.rand(1, 3, 64, 64) * 255.0
        right = torch.rand(1, 3, 64, 64) * 255.0

        _, predictions = model(left, right, iters=2)
        predictions[-1].square().mean().backward()

        for name, parameter in model.named_parameters():
            if name.startswith("sr_head."):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)


if __name__ == "__main__":
    unittest.main()
