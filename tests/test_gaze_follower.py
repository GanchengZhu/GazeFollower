# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import unittest
from unittest.mock import MagicMock
import numpy as np

from gazefollower import GazeFollower
from gazefollower.calibration import MultivariateRidgeCalibration, SVRCalibration
from gazefollower.face_alignment import MediaPipeFaceAlignment
from gazefollower.gaze_estimator import MGazeNetGazeEstimator
from gazefollower.filter import HeuristicFilter, OneEuroFilter
from gazefollower.camera import Camera


class DummyCamera(Camera):
    def __init__(self):
        super().__init__()
        self.started = False
        self.stopped = False

    def capture(self):
        pass

    def start_sampling(self):
        self.started = True

    def stop_sampling(self):
        self.stopped = True

    def close(self):
        pass

    def release(self):
        pass


class TestGazeFollower(unittest.TestCase):

    def setUp(self):
        self.dummy_cam = DummyCamera()
        self.gf = GazeFollower(camera=self.dummy_cam)

    def test_default_components(self):
        """
        Verify that default components are properly initialized,
        especially MultivariateRidgeCalibration as default calibration.
        """
        self.assertIsInstance(self.gf.calibration, MultivariateRidgeCalibration)
        self.assertIsInstance(self.gf.face_alignment, MediaPipeFaceAlignment)
        self.assertIsInstance(self.gf.gaze_estimator, MGazeNetGazeEstimator)
        self.assertIsInstance(self.gf.gaze_filter, HeuristicFilter)

    def test_custom_components(self):
        """
        Verify that custom components can be injected.
        """
        custom_calib = SVRCalibration()
        custom_filter = OneEuroFilter(freq=30.0)
        custom_gf = GazeFollower(
            camera=DummyCamera(),
            calibration=custom_calib,
            gaze_filter=custom_filter
        )
        self.assertIs(custom_gf.calibration, custom_calib)
        self.assertIs(custom_gf.gaze_filter, custom_filter)
        custom_gf.release()

    def test_subscribers(self):
        """
        Test adding and removing subscribers.
        """
        callback_called = []

        def test_callback(gaze_info):
            callback_called.append(gaze_info)

        self.gf.add_subscriber(test_callback)
        self.assertTrue(any(sub[0] == test_callback for sub in self.gf.subscribers))

        self.gf.remove_subscriber(test_callback)
        self.assertFalse(any(sub[0] == test_callback for sub in self.gf.subscribers))

    def test_sampling_control(self):
        """
        Test starting and stopping sampling.
        """
        self.gf.start_sampling()
        self.assertTrue(self.dummy_cam.started)

        self.gf.stop_sampling()
        self.assertTrue(self.dummy_cam.stopped)

    def test_get_gaze_info_initial(self):
        """
        get_gaze_info should return None before any frame is processed.
        """
        self.assertIsNone(self.gf.get_gaze_info())

    def tearDown(self):
        self.gf.release()


if __name__ == '__main__':
    unittest.main()
