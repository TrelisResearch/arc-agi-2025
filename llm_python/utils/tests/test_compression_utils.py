#!/usr/bin/env python3

import unittest
import numpy as np
from llm_python.utils.compression_utils import (
    calculate_gzip_ratio,
    calculate_gzip_size,
    calculate_combined_gzip_ratio
)


class TestCompressionUtils(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Simple 2x2 grid
        self.simple_grid = [[0, 1], [1, 0]]

        # Repetitive grid (should compress well)
        self.repetitive_grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        # Random-ish grid (should compress poorly)
        self.random_grid = [[0, 3, 7], [2, 5, 1], [8, 4, 6]]

        # Numpy array version
        self.numpy_grid = np.array([[0, 1], [1, 0]])

        # Empty and None cases
        self.empty_grid = []
        self.none_grid = None

    def test_calculate_gzip_ratio_simple(self):
        """Test gzip ratio calculation for simple grid."""
        ratio = calculate_gzip_ratio(self.simple_grid)

        # Should be a valid ratio > 0 (can be > 1 due to gzip overhead on small data)
        self.assertGreater(ratio, 0.0)

    def test_calculate_gzip_ratio_repetitive(self):
        """Test that repetitive grids compress better."""
        repetitive_ratio = calculate_gzip_ratio(self.repetitive_grid)
        random_ratio = calculate_gzip_ratio(self.random_grid)

        # Repetitive grid should compress better (lower ratio)
        self.assertLess(repetitive_ratio, random_ratio)

    def test_calculate_gzip_ratio_numpy(self):
        """Test gzip ratio with numpy arrays."""
        ratio = calculate_gzip_ratio(self.numpy_grid)

        # Should work with numpy arrays
        self.assertGreater(ratio, 0.0)

    def test_calculate_gzip_ratio_methods(self):
        """Test different serialization methods."""
        json_ratio = calculate_gzip_ratio(self.simple_grid, method='json')
        pickle_ratio = calculate_gzip_ratio(self.simple_grid, method='pickle')
        string_ratio = calculate_gzip_ratio(self.simple_grid, method='string')

        # All should be valid ratios > 0
        for ratio in [json_ratio, pickle_ratio, string_ratio]:
            self.assertGreater(ratio, 0.0)

    def test_calculate_gzip_ratio_edge_cases(self):
        """Test edge cases."""
        # None grid
        none_ratio = calculate_gzip_ratio(self.none_grid)
        self.assertEqual(none_ratio, 1.0)

        # Empty grid
        empty_ratio = calculate_gzip_ratio(self.empty_grid)
        self.assertGreaterEqual(empty_ratio, 0.0)

    def test_calculate_gzip_size(self):
        """Test gzip size calculation."""
        size = calculate_gzip_size(self.simple_grid)

        # Should return a positive integer
        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)

    def test_calculate_gzip_size_none(self):
        """Test gzip size with None input."""
        size = calculate_gzip_size(self.none_grid)
        self.assertEqual(size, 0)

    def test_calculate_combined_gzip_ratio(self):
        """Test combined grid compression."""
        ratio = calculate_combined_gzip_ratio(self.simple_grid, self.repetitive_grid)

        # Should be a valid ratio > 0
        self.assertGreater(ratio, 0.0)

    def test_invalid_method(self):
        """Test invalid serialization method."""
        # Should return 1.0 for invalid method
        ratio = calculate_gzip_ratio(self.simple_grid, method='invalid')
        self.assertEqual(ratio, 1.0)


if __name__ == '__main__':
    unittest.main()