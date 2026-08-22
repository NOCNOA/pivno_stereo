import unittest

import torch

from tools.evaluate_pivno_d0 import (
    extract_pivno_state,
    infer_pivno_input_channels,
    prepare_pivno_input,
)


class EvaluatePivnoD0Tests(unittest.TestCase):
    def test_infers_legacy_grayscale_checkpoint(self):
        state = {"pivno.snet.conv1.weight": torch.zeros(64, 1, 3, 3)}

        self.assertEqual(infer_pivno_input_channels(state), 1)

    def test_grayscale_preprocessing_matches_training_transform(self):
        image = torch.tensor([[[[255.0]], [[0.0]], [[0.0]]]])

        prepared = prepare_pivno_input(image, input_channels=1)

        self.assertEqual(tuple(prepared.shape), (1, 1, 1, 1))
        self.assertTrue(torch.allclose(prepared, torch.tensor([[[[0.299]]]])))

    def test_rgb_preprocessing_only_rescales(self):
        image = torch.tensor([[[[0.0]], [[127.5]], [[255.0]]]])

        prepared = prepare_pivno_input(image, input_channels=3)

        self.assertTrue(torch.equal(
            prepared,
            torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]]),
        ))

    def test_extracts_only_pivno_parameters(self):
        state = {
            "pivno.snet.conv1.weight": torch.ones(1),
            "cnet.conv1.weight": torch.zeros(1),
        }

        extracted = extract_pivno_state(state)

        self.assertEqual(list(extracted), ["snet.conv1.weight"])


if __name__ == "__main__":
    unittest.main()
