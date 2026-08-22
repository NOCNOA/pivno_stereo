import unittest

import torch

from tools.evaluate_pivno_refinement import build_d0_error_masks


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


if __name__ == "__main__":
    unittest.main()
