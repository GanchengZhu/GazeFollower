# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import unittest
from gazefollower.filter.OneEuroFilter import OneEuroFilter, LowPassFilter


class TestOneEuroFilter(unittest.TestCase):

    def test_low_pass_filter(self):
        lpf = LowPassFilter(alpha=0.5, initval=0.0)
        self.assertFalse(lpf.has_last_raw_value())
        
        # First sample initializes filter
        val1 = lpf.filter(10.0)
        self.assertEqual(val1, 10.0)
        self.assertTrue(lpf.has_last_raw_value())
        self.assertEqual(lpf.last_raw_value(), 10.0)

        # Second sample smooths: 0.5 * 20.0 + 0.5 * 10.0 = 15.0
        val2 = lpf.filter(20.0)
        self.assertEqual(val2, 15.0)

    def test_one_euro_filter_initialization(self):
        oef = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta_=0.007, d_cutoff=1.0)
        self.assertEqual(oef.freq, 30.0)
        self.assertEqual(oef.min_cutoff, 1.0)

    def test_one_euro_filter_single_value(self):
        oef = OneEuroFilter(freq=30.0)
        res = oef.filter(10.0, timestamp=1000)
        self.assertEqual(res, 10.0)
        
        # Filter with subsequent timestamps
        res2 = oef.filter(20.0, timestamp=1033)
        self.assertIsInstance(res2, float)
        self.assertGreater(res2, 10.0)
        self.assertLess(res2, 20.0)

    def test_one_euro_filter_values_list(self):
        oef = OneEuroFilter(freq=30.0)
        filtered_coords = oef.filter_values([100.0, 200.0], timestamp=1000)
        self.assertEqual(len(filtered_coords), 2)
        self.assertEqual(filtered_coords[0], 100.0)
        self.assertGreater(filtered_coords[1], 100.0)
        self.assertLess(filtered_coords[1], 200.0)

        filtered_coords2 = oef.filter_values([110.0, 210.0], timestamp=1033)
        self.assertEqual(len(filtered_coords2), 2)
        self.assertIsInstance(filtered_coords2[0], float)
        self.assertIsInstance(filtered_coords2[1], float)


if __name__ == '__main__':
    unittest.main()
