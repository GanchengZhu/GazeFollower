# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import unittest
import numpy as np

from gazefollower.calibration import SVRCalibration


class TestSVRCalibration(unittest.TestCase):

    def setUp(self):
        """
        Set up a SVRCalibration instance and prepare test data.
        """
        self.calibration = SVRCalibration()
        # Create dummy data for testing
        self.features = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32)
        self.labels = np.array([[10.0, 20.0], [20.0, 30.0], [30.0, 40.0], [40.0, 50.0]], dtype=np.float32)
        self.estimated_coordinate = [0.0, 0.0]

    def test_calibrate_and_predict(self):
        """
        Test calibration and prediction methods.
        """
        has_calibrated, mean_error, predictions = self.calibration.calibrate(self.features, self.labels)
        self.assertTrue(has_calibrated)
        self.assertNotEqual(mean_error, float('inf'))
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.labels))

        # Predict using the calibrated model
        calibrated, result = self.calibration.predict(self.features[0], self.estimated_coordinate)
        self.assertTrue(calibrated)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], (float, np.floating))
        self.assertIsInstance(result[1], (float, np.floating))

    def test_predict_before_calibration(self):
        """
        Test the predict method before calibration.
        """
        calibrated, result = self.calibration.predict(self.features[0], self.estimated_coordinate)
        self.assertFalse(calibrated)
        self.assertEqual(list(result), self.estimated_coordinate)

    def test_save_and_load_model(self):
        """
        Test the model saving and loading functionality.
        """
        self.calibration.calibrate(self.features, self.labels)
        save_success = self.calibration.save_model()
        self.assertTrue(save_success, "Model saving failed")

        new_calibration = SVRCalibration()
        self.assertTrue(new_calibration.has_calibrated, "Model loading failed")

    def tearDown(self):
        """
        Clean up model files generated during tests.
        """
        self.calibration.release()
        svr_x_path = self.calibration.svr_x_path
        svr_y_path = self.calibration.svr_y_path
        if svr_x_path.exists():
            svr_x_path.unlink()
        if svr_y_path.exists():
            svr_y_path.unlink()


if __name__ == '__main__':
    unittest.main()
