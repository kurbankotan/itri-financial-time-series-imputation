"""Stress-test methods at the longest, densest prespecified block setting."""

import unittest

import numpy as np

from masking import strict_block_mask
from run_classical_benchmark import (
    METHODS, corrupt, run_methods, score_features, stable_mask_seed,
    train_observed_scaler,
)


class MethodSmokeTests(unittest.TestCase):
    def test_long_block_thirty_percent(self):
        rng = np.random.default_rng(20260923)
        train = rng.standard_normal((1006, 5)).cumsum(axis=0) + 100
        test = rng.standard_normal((250, 5)).cumsum(axis=0) + 100
        train_hidden = strict_block_mask(train.shape, 0.30, 20, 113,
                                         edge_guard=20)
        test_hidden = strict_block_mask(test.shape, 0.30, 20, 337,
                                        edge_guard=20)
        mean, scale = train_observed_scaler(train, train_hidden)
        train_scaled = (train - mean) / scale
        test_scaled = (test - mean) / scale
        input_train = corrupt(train_scaled, train_hidden)
        input_test = corrupt(test_scaled, test_hidden)
        estimates = run_methods(input_train, input_test, 11)
        self.assertEqual(tuple(estimates), METHODS)
        for method, (estimate, seconds, warning_count) in estimates.items():
            self.assertGreaterEqual(seconds, 0)
            self.assertGreaterEqual(warning_count, 0)
            self.assertTrue(np.isfinite(estimate).all())
            self.assertTrue(np.array_equal(estimate[~test_hidden],
                                           input_test[~test_hidden]))
            scored = score_features(test, estimate, test_hidden, mean, scale,
                                    {"method": method}, seconds, warning_count)
            self.assertEqual(len(scored), 5)
            self.assertTrue(all(row["rmse_trainstd"] >= 0 for row in scored))

    def test_seed_is_stable_and_partition_specific(self):
        first = stable_mask_seed("abc", "EWC", "A", "test", 20, .30, 11)
        self.assertEqual(first, stable_mask_seed("abc", "EWC", "A", "test",
                                                 20, .30, 11))
        self.assertNotEqual(first, stable_mask_seed("abc", "EWC", "A", "train",
                                                    20, .30, 11))


if __name__ == "__main__":
    unittest.main()
