import unittest
from types import SimpleNamespace

import torch

from core.pivno_models.defom_pivno_gated_gru1 import (
    DEFOMStereo as GatedGRU1DEFOMStereo,
)
from core.pivno_models.defom_pivno_gated_gru3 import (
    DEFOMStereo as GatedGRU3DEFOMStereo,
)
from core.update import BasicMultiUpdateBlock


def make_args(n_gru_layers=3):
    return SimpleNamespace(
        n_downsample=2,
        hidden_dims=[128, 128, 128],
        n_gru_layers=n_gru_layers,
        context_norm="instance",
        corr_radius=4,
        mixed_precision=False,
    )


class DefomPivnoGatedGRU1Tests(unittest.TestCase):
    def test_new_variant_changes_only_gru_gate_weight_shapes(self):
        args = make_args()
        gru1 = GatedGRU1DEFOMStereo(args)
        gru3 = GatedGRU3DEFOMStereo(args)

        self.assertEqual(gru1.MODEL_VARIANT, "defom_pivno_gated_gru1")
        self.assertEqual(gru1.GRU_KERNEL_SIZE, 1)
        self.assertIs(type(gru1.update_block), BasicMultiUpdateBlock)

        state1 = gru1.state_dict()
        state3 = gru3.state_dict()
        self.assertEqual(set(state1), set(state3))
        shape_differences = {
            key for key in state1
            if state1[key].shape != state3[key].shape
        }
        expected = {
            f"update_block.{level}.{gate}.weight"
            for level in ("gru08", "gru16", "gru32")
            for gate in ("convz", "convr", "convq")
        }
        self.assertEqual(shape_differences, expected)

    def test_gru1_does_not_mutate_existing_gru3_model(self):
        args = make_args()
        gru1 = GatedGRU1DEFOMStereo(args)
        gru3 = GatedGRU3DEFOMStereo(args)

        for model, expected_kernel in ((gru1, (1, 1)), (gru3, (3, 3))):
            for gru in (
                model.update_block.gru08,
                model.update_block.gru16,
                model.update_block.gru32,
            ):
                self.assertEqual(gru.convz.kernel_size, expected_kernel)
                self.assertEqual(gru.convr.kernel_size, expected_kernel)
                self.assertEqual(gru.convq.kernel_size, expected_kernel)

    def test_gru3_checkpoint_is_rejected_by_gru1_strict_load(self):
        args = make_args()
        gru1 = GatedGRU1DEFOMStereo(args)
        gru3 = GatedGRU3DEFOMStereo(args)

        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            gru1.load_state_dict(gru3.state_dict(), strict=True)

    def test_update_block_forward_and_backward_are_finite(self):
        torch.manual_seed(7)
        model = GatedGRU1DEFOMStereo(make_args())
        block = model.update_block

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

        updated, mask, delta = block(net, inp, corr=corr, disp=disp)
        loss = sum(item.square().mean() for item in updated)
        loss = loss + mask.square().mean() + delta.square().mean()
        loss.backward()

        self.assertEqual(tuple(mask.shape), (1, 144, 4, 8))
        self.assertEqual(tuple(delta.shape), (1, 1, 4, 8))
        for name, parameter in block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
