import unittest

import torch

from utils.pact_pivno_loss import _sequence_weight, pact_pivno_sequence_loss


class PactPivnoLossTests(unittest.TestCase):
    def test_sequence_weight_keeps_latest_prediction_at_unit_weight(self):
        count = 5
        weights = [_sequence_weight(index, count, 0.9) for index in range(count)]

        self.assertAlmostEqual(weights[-1], 1.0)
        self.assertEqual(weights, sorted(weights))

    def test_every_init_and_recurrent_prediction_receives_gradient(self):
        target = torch.zeros(1, 1, 8, 8)
        target[:, :, :, 4:] = 4.0
        valid = torch.ones_like(target)
        init_predictions = [
            torch.full_like(target, value, requires_grad=True)
            for value in (3.0, 2.0, 1.0)
        ]
        recurrent_predictions = [
            torch.full_like(target, value, requires_grad=True)
            for value in (1.5, 0.5)
        ]

        loss, metrics = pact_pivno_sequence_loss(
            init_predictions,
            recurrent_predictions,
            target,
            valid,
            max_disp=768,
        )
        loss.backward()

        for prediction in [*init_predictions, *recurrent_predictions]:
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        self.assertGreater(metrics["pivno_init_smooth_l1"], 0.0)
        self.assertGreater(metrics["pivno_recurrent_edge_l1"], 0.0)
        self.assertGreater(metrics["edge_pixel_ratio"], 0.0)

    def test_rejects_empty_stage_predictions(self):
        target = torch.zeros(1, 1, 4, 4)
        valid = torch.ones_like(target)

        with self.assertRaisesRegex(ValueError, "init_predictions"):
            pact_pivno_sequence_loss([], [target], target, valid)
        with self.assertRaisesRegex(ValueError, "recurrent_predictions"):
            pact_pivno_sequence_loss([target], [], target, valid)


if __name__ == "__main__":
    unittest.main()
