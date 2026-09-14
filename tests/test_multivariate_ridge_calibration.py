# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import unittest
import numpy as np

from gazefollower.calibration import MultivariateRidgeCalibration


class TestMultivariateRidgeCalibration(unittest.TestCase):

    def setUp(self):
        """
        Set up a MultivariateRidgeCalibration instance and prepare test data.
        """
        self.calibration = MultivariateRidgeCalibration(alpha=0.1)
        # Create synthetic data with a known linear relation: Y = X @ W + bias
        # For instance: y0 = 2*x0 + 3*x1 + 5, y1 = -x0 + 4*x1 - 2
        np.random.seed(42)
        self.features = np.random.randn(20, 2).astype(np.float32)
        true_w = np.array([[2.0, -1.0], [3.0, 4.0]], dtype=np.float32)
        true_b = np.array([5.0, -2.0], dtype=np.float32)
        self.labels = (self.features @ true_w + true_b).astype(np.float32)
        self.estimated_coordinate = [100.0, 200.0]

    def test_predict_before_calibration(self):
        """
        Test the predict method before calibration.
        """
        calibrated, result = self.calibration.predict(self.features[0], self.estimated_coordinate)
        self.assertFalse(calibrated)
        self.assertEqual(result, self.estimated_coordinate)

    def test_calibrate_and_predict(self):
        """
        Test calibration and prediction methods.
        """
        has_calibrated, mean_error, predictions = self.calibration.calibrate(self.features, self.labels)
        self.assertTrue(has_calibrated)
        self.assertLess(mean_error, 1.0)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.labels))

        # Predict on a test point
        test_feat = np.array([1.0, 2.0], dtype=np.float32)
        calibrated, pred = self.calibration.predict(test_feat, self.estimated_coordinate)
        self.assertTrue(calibrated)
        self.assertEqual(len(pred), 2)

        # Expected values close to: 2*1 + 3*2 + 5 = 13, -1*1 + 4*2 - 2 = 5
        self.assertAlmostEqual(pred[0], 13.0, delta=0.5)
        self.assertAlmostEqual(pred[1], 5.0, delta=0.5)

    def test_alpha_parameter(self):
        """
        Test initializing with different alpha values.
        """
        calib_large_alpha = MultivariateRidgeCalibration(alpha=100.0)
        has_calibrated, mean_error, _ = calib_large_alpha.calibrate(self.features, self.labels)
        self.assertTrue(has_calibrated)
        self.assertNotEqual(mean_error, float('inf'))

    def test_release(self):
        """
        Test that release does not raise exceptions.
        """
        self.calibration.release()


if __name__ == '__main__':
    unittest.main()
