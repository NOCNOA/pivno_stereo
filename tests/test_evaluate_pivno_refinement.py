import unittest
from types import SimpleNamespace

import torch

from tools.evaluate_pivno_refinement import (
    _validate_checkpoint,
    build_d0_error_masks,
    build_gt_disparity_masks,
    search_support_counts,
)


class EvaluatePivnoRefinementTests(unittest.TestCase):
    def test_d0_error_bins_are_mutually_exclusive_and_cover_valid_pixels(self):
        error = torch.tensor([[[0.0, 16.0, 16.1, 32.0, 32.1, 64.0, 64.1]]])
        valid = torch.ones_like(error, dtype=torch.bool)

        masks = build_d0_error_masks(error, valid)

        self.assertEqual(
            {name: mask.flatten().nonzero().flatten().tolist() for name, mask in masks.items()},
            {
                "0_16px": [0, 1],
                "16_32px": [2, 3],
                "32_64px": [4, 5],
                "over_64px": [6],
            },
        )
        coverage = torch.stack(list(masks.values())).sum(dim=0)
        self.assertTrue(torch.equal(coverage, valid.to(coverage.dtype)))

    def test_invalid_pixels_are_excluded_from_every_bin(self):
        error = torch.tensor([[[1.0, 20.0, 40.0, 80.0]]])
        valid = torch.zeros_like(error, dtype=torch.bool)

        masks = build_d0_error_masks(error, valid)

        self.assertFalse(any(bool(mask.any()) for mask in masks.values()))

    def test_gt_disparity_masks_have_disjoint_and_cumulative_semantics(self):
        gt = torch.tensor([[[0.0, 191.9, 192.0, 383.9, 384.0, 767.9]]])
        valid = torch.ones_like(gt, dtype=torch.bool)

        bands, cumulative = build_gt_disparity_masks(
            gt,
            valid,
            thresholds=(192.0, 384.0, 768.0),
        )

        self.assertEqual(
            {
                name: mask.flatten().nonzero().flatten().tolist()
                for name, mask in bands.items()
            },
            {
                "0_192px": [0, 1],
                "192_384px": [2, 3],
                "384_768px": [4, 5],
            },
        )
        self.assertEqual(
            {
                name: mask.flatten().nonzero().flatten().tolist()
                for name, mask in cumulative.items()
            },
            {
                "lt_192px": [0, 1],
                "lt_384px": [0, 1, 2, 3],
                "lt_768px": [0, 1, 2, 3, 4, 5],
            },
        )

    def test_search_support_counts_are_nested(self):
        error = torch.tensor([[[0.0, 4.0, 4.1, 8.0, 8.1, 16.0, 16.1]]])
        valid = torch.ones_like(error, dtype=torch.bool)

        counts = search_support_counts(error, valid, (4.0, 8.0, 16.0))

        self.assertEqual(
            counts,
            {
                "le_4px": 2,
                "le_8px": 4,
                "le_16px": 6,
                "gt_16px": 1,
            },
        )

    def test_completed_gated_gru3_checkpoint_contract_is_accepted(self):
        config = {
            "model": "defom_pivno_gated_gru3",
            "pivno_input_channels": 3,
            "corr_radius": 4,
            "pivno_match_num_groups": 4,
            "pivno_match_encoded_channels": 16,
            "pivno_gru_kernel_size": 3,
            "pivno_right_sample_encoding": (
                "residual_gwc4_conv16_no_left_concat"
            ),
            "pivno_scale_gate": (
                "gwc4_mean_softmax_weighted_encoded_concat"
            ),
        }
        state = {
            "low_channel.weight": torch.empty(32, 64, 1, 1),
            "sample_match_encoder.0.weight": torch.empty(16, 36, 1, 1),
            "refine_right_fuse.0.weight": torch.empty(128, 432, 3, 3),
            "scale_gate.0.weight": torch.empty(32, 59, 3, 3),
            "update_block.gru08.convz.weight": torch.empty(128, 384, 3, 3),
        }

        validated_config, checkpoint_model = _validate_checkpoint(
            {"model_config": config},
            state,
            SimpleNamespace(corr_radius=4),
        )

        self.assertIs(validated_config, config)
        self.assertEqual(checkpoint_model, "defom_pivno_gated_gru3")


if __name__ == "__main__":
    unittest.main()
