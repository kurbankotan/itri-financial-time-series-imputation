"""Invariant tests for the proposed strict-run mask generator."""

import unittest

import numpy as np

from masking import run_lengths, strict_block_mask


class StrictBlockMaskTests(unittest.TestCase):
    def test_original_grid_shapes_and_rates(self) -> None:
        for rows in (409, 412, 413, 1907, 1922):
            for block in (1, 2, 3, 5, 10, 20):
                for rate in (0.10, 0.20, 0.30):
                    for seed in (11, 23, 37):
                        with self.subTest(rows=rows, block=block, rate=rate, seed=seed):
                            mask = strict_block_mask((rows, 14), rate, block, seed)
                            target = max(1, round(rate * (rows - 20) * 14))
                            lengths = run_lengths(mask)
                            self.assertEqual(int(mask.sum()), target)
                            self.assertLessEqual(int(lengths.max()), block)
                            self.assertEqual(int((lengths < block).sum()),
                                             int(target % block != 0))
                            self.assertFalse(mask[:10].any())
                            self.assertFalse(mask[-10:].any())

    def test_seed_reproducibility(self) -> None:
        one = strict_block_mask((409, 14), 0.2, 5, 11)
        two = strict_block_mask((409, 14), 0.2, 5, 11)
        self.assertTrue(np.array_equal(one, two))

    def test_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            strict_block_mask((20, 14), 0.2, 5, 11)


if __name__ == "__main__":
    unittest.main()
