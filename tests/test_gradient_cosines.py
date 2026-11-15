"""CPU numerical regressions for long diagnostic gradient vectors; no datasets."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

# These are standalone investigation CLIs, not another installed training package.
with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'scripts'), *sys.path]):
    from scripts.film_scaling_gradients import _gradient_vector, cosine, matrix_summary


class TestGradientCosines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_long_gradient_cosine_stays_normalized(self):
        # Million-dimensional, heterogeneous gradients exposed a float32 reduction
        # bias in the real diagnostic (self-cosines up to 1.000657).
        raw = torch.linspace(1e-6, 1., 2_000_000).pow(8)
        vector = _gradient_vector([raw[:700_000], raw[700_000:]])
        self.assertAlmostEqual(cosine(vector, vector), 1., places=12)
        self.assertAlmostEqual(cosine(vector, -vector), -1., places=12)

    def test_zero_gradients_have_undefined_cosines(self):
        report = matrix_summary({'a': torch.zeros(3), 'b': torch.zeros(3)}, ['a', 'b'])
        self.assertEqual(report['valid_off_diagonal_pairs'], 0)
        self.assertIsNone(report['off_diagonal_mean'])
        self.assertIsNone(report['off_diagonal_negative_fraction'])
        self.assertEqual(report['cosine'], [[None, None], [None, None]])


if __name__ == '__main__':
    unittest.main()
