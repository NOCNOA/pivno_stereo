import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru_kernel_ablation import (
    DEFOMStereo as GRUKernelAblationDEFOMStereo,
)


def make_args(kernel_size):
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=3,
        context_norm="instance",
        corr_radius=4,
        mixed_precision=False,
        pivno_gru_kernel_size=kernel_size,
    )


class DefomPivnoGatedGRUKernelAblationTests(unittest.TestCase):
    def test_requires_kernel_one_or_three(self):
        with self.assertRaisesRegex(ValueError, "required"):
            GRUKernelAblationDEFOMStereo(make_args(None))
        with self.assertRaisesRegex(ValueError, "must be 1 or 3"):
            GRUKernelAblationDEFOMStereo(make_args(5))

    def test_feature_contract_is_frozen_to_completed_baseline(self):
        model = GRUKernelAblationDEFOMStereo(make_args(3))

        self.assertEqual(model.LOW_FEATURE_DIM, 32)
        self.assertEqual(model.MATCH_NUM_GROUPS, 4)
        self.assertEqual(model.MATCH_ENCODED_CHANNELS, 16)
        self.assertEqual(model.low_channel.out_channels, 32)
        self.assertEqual(model.sample_match_encoder[0].in_channels, 36)
        self.assertEqual(model.refine_right_fuse[0].in_channels, 432)

    def test_only_gru_gate_weight_shapes_change(self):
        torch.manual_seed(1234565)
        gru1 = GRUKernelAblationDEFOMStereo(make_args(1))
        torch.manual_seed(1234565)
        gru3 = GRUKernelAblationDEFOMStereo(make_args(3))

        state1 = gru1.state_dict()
        state3 = gru3.state_dict()
        self.assertEqual(set(state1), set(state3))
        differences = {
            key for key in state1
            if state1[key].shape != state3[key].shape
        }
        expected = {
            f"update_block.{level}.{gate}.weight"
            for level in ("gru08", "gru16", "gru32")
            for gate in ("convz", "convr", "convq")
        }
        self.assertEqual(differences, expected)

        for key in state1:
            if key not in expected:
                self.assertTrue(torch.equal(state1[key], state3[key]), key)

    def test_all_outer_gru_gates_use_requested_kernel(self):
        for kernel_size in (1, 3):
            model = GRUKernelAblationDEFOMStereo(make_args(kernel_size))
            expected = (kernel_size, kernel_size)
            for gru in (
                model.update_block.gru08,
                model.update_block.gru16,
                model.update_block.gru32,
            ):
                self.assertEqual(gru.convz.kernel_size, expected)
                self.assertEqual(gru.convr.kernel_size, expected)
                self.assertEqual(gru.convq.kernel_size, expected)

    def test_update_blocks_forward_and_backward_are_finite(self):
        for kernel_size in (1, 3):
            torch.manual_seed(7)
            block = GRUKernelAblationDEFOMStereo(
                make_args(kernel_size)
            ).update_block
            net = [
                torch.randn(1, 128, 4, 8, requires_grad=True),
                torch.randn(1, 128, 2, 4, requires_grad=True),
                torch.randn(1, 128, 1, 2, requires_grad=True),
            ]
            inp = [
                [torch.randn_like(level) for _ in range(3)]
                for level in net
            ]
            corr = torch.randn(1, 128, 4, 8)
            disp = torch.randn(1, 1, 4, 8)

            updated, mask, delta = block(
                net, inp, corr=corr, disp=disp
            )
            loss = sum(value.square().mean() for value in updated)
            loss = loss + mask.square().mean() + delta.square().mean()
            loss.backward()

            self.assertEqual(tuple(mask.shape), (1, 144, 4, 8))
            self.assertEqual(tuple(delta.shape), (1, 1, 4, 8))
            for name, parameter in block.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(), name
                )


if __name__ == "__main__":
    unittest.main()
