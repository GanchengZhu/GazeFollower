# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import os
from pathlib import Path
import unittest
import cv2

from gazefollower.face_alignment import BlazeFaceAlignment, MediaPipeFaceAlignment
from gazefollower.gaze_estimator import MGazeNetGazeEstimator


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = Path(__file__).parent
        cls.image_path = test_dir / "asset" / "example.jpg"
        if not cls.image_path.exists():
            raise unittest.SkipTest(f"Test image not found at {cls.image_path}")
        cls.frame = cv2.imread(str(cls.image_path))
        if cls.frame is None:
            raise unittest.SkipTest(f"Failed to read image at {cls.image_path}")

    def test_blazeface_alignment(self):
        aligner = BlazeFaceAlignment()
        face_info = aligner.detect(0, self.frame)
        self.assertIsNotNone(face_info)
        self.assertTrue(face_info.status)
        self.assertTrue(face_info.can_gaze_estimation)
        self.assertEqual(len(face_info.face_rect), 4)
        aligner.release()

    def test_mediapipe_alignment(self):
        aligner = MediaPipeFaceAlignment()
        face_info = aligner.detect(0, self.frame)
        self.assertIsNotNone(face_info)
        self.assertTrue(face_info.status)
        self.assertTrue(face_info.can_gaze_estimation)
        aligner.release()

    def test_gaze_estimator_with_blazeface(self):
        aligner = BlazeFaceAlignment()
        face_info = aligner.detect(0, self.frame)
        estimator = MGazeNetGazeEstimator()
        gaze_info = estimator.detect(self.frame, face_info)
        self.assertIsNotNone(gaze_info)
        self.assertTrue(gaze_info.status)
        self.assertIsNotNone(gaze_info.raw_gaze_coordinates)
        self.assertEqual(len(gaze_info.raw_gaze_coordinates), 2)
        estimator.release()


if __name__ == '__main__':
    unittest.main()
