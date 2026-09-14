# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import unittest
import numpy as np
from gazefollower.filter import HeuristicFilter


class TestHeuristicFilter(unittest.TestCase):

    def setUp(self):
        # Initialize the filter with a look-ahead of 2
        self.filter = HeuristicFilter(look_ahead=2)

    def test_initial_state(self):
        # Test that the initial state of the filter has NaN values for dummy_x and dummy_y
        self.assertTrue(np.isnan(self.filter.dummy_x))
        self.assertTrue(np.isnan(self.filter.dummy_y))

    def test_filter_with_insufficient_data(self):
        # Test that the filter returns the original values if there's not enough history
        result = self.filter.filter_values([1.0, 2.0])
        self.assertEqual(result, [1.0, 2.0])

        result = self.filter.filter_values([2.0, 3.0])
        self.assertEqual(result, [2.0, 3.0])

    def test_filter_with_sufficient_data(self):
        # Feed enough values to fill the look-ahead window
        points = [
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]
        ]
        result = None
        for p in points:
            result = self.filter.filter_values(p)

        # Checking if dummy_x and dummy_y are updated
        self.assertFalse(np.isnan(self.filter.dummy_x))
        self.assertFalse(np.isnan(self.filter.dummy_y))
        self.assertEqual(result, [self.filter.dummy_x, self.filter.dummy_y])

    def test_filter_raises_value_error(self):
        # Test that a ValueError is raised if input does not contain exactly two values
        with self.assertRaises(ValueError):
            self.filter.filter_values([1.0])

    def test_smoothing_logic(self):
        # Test that the smoothing logic correctly updates dummy values
        values = [
            [1.0, 2.0],
            [5.0, 4.0],
            [2.0, 6.0],
            [3.0, 8.0],
            [4.0, 7.0]
        ]

        for v in values:
            self.filter.filter_values(v)

        self.assertFalse(np.isnan(self.filter.dummy_x))
        self.assertFalse(np.isnan(self.filter.dummy_y))


if __name__ == '__main__':
    unittest.main()
