# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import time
import unittest
import numpy as np

from gazefollower.calibration import CalibrationController
from gazefollower.misc import CalibrationMode, GazeInfo, FaceInfo


class TestCalibrationController(unittest.TestCase):

    def _create_mock_face_gaze(self, feature_dim=128):
        face_info = FaceInfo(
            status=True,
            can_gaze_estimation=True,
            left_eye_openness=20.0,
            right_eye_openness=20.0,
            face_rect=(100, 100, 200, 200),
            left_rect=(120, 140, 50, 30),
            right_rect=(220, 140, 50, 30)
        )
        gaze_info = GazeInfo(
            status=True,
            features=np.random.randn(feature_dim).astype(np.float32)
        )
        return face_info, gaze_info

    def test_click_calibration_advancement(self):
        """
        Scheme A (Point-and-Click):
        Verify that user click confirms dot and advances to the next point,
        and that feature shapes are properly aligned.
        """
        controller = CalibrationController(
            cali_mode=CalibrationMode.FIVE_POINT,
            camera_pos=(17.15, -0.68),
            screen_size=np.array([1920, 1080]),
            cali_click_mode=True
        )
        controller.new_session()
        self.assertTrue(controller.calibrating)
        self.assertEqual(controller._current_index, 0)

        # Click warmup point (index 0) -> advances to point 1
        clicked = controller.on_target_clicked()
        self.assertTrue(clicked)
        self.assertEqual(controller._current_index, 1)

        # Feed 15 frames for point 1
        face_info, gaze_info = self._create_mock_face_gaze()
        for _ in range(15):
            controller.add_cali_feature(gaze_info, face_info)

        # Click point 1 -> should commit data (resampled/padded to 45 frames) and advance to point 2
        clicked = controller.on_target_clicked()
        self.assertTrue(clicked)
        self.assertEqual(controller._current_index, 2)
        self.assertEqual(len(controller.feature_vectors[0]), 45)
        self.assertEqual(len(controller.label_vectors[0]), 45)

        # Complete points 2, 3, 4, 5
        for pt in range(2, 6):
            self.assertEqual(controller._current_index, pt)
            for _ in range(10):
                controller.add_cali_feature(gaze_info, face_info)
            clicked = controller.on_target_clicked()
            self.assertTrue(clicked)

        # After point 5 is clicked, calibration should complete
        self.assertFalse(controller.calibrating)
        features_arr = np.array(controller.feature_vectors)
        self.assertEqual(features_arr.shape, (5, 45, 128))
        labels_arr = np.array(controller.label_vectors)
        self.assertEqual(labels_arr.shape, (5, 45, 2))

    def test_lissajous_no_click(self):
        """
        Lissajous calibration pattern does NOT support click.
        on_target_clicked() must return False and not alter state.
        """
        controller = CalibrationController(
            cali_mode=CalibrationMode.LISSAJOUS,
            camera_pos=(17.15, -0.68),
            screen_size=np.array([1920, 1080]),
            lissajous_frame_latency=4
        )
        controller.new_session()
        self.assertTrue(controller.calibrating)

        # Attempt to click during Lissajous
        clicked = controller.on_target_clicked()
        self.assertFalse(clicked)
        self.assertTrue(controller.calibrating)

    def test_lissajous_frame_latency_compensation(self):
        """
        Lissajous pursuit calibration:
        Verify that frame latency (default: 4 frames) pairs the current
        captured gaze frame with the target position from 4 frames ago.
        """
        latency = 4
        controller = CalibrationController(
            cali_mode=CalibrationMode.LISSAJOUS,
            camera_pos=(17.15, -0.68),
            screen_size=np.array([1920, 1080]),
            lissajous_frame_latency=latency,
            lissajous_duration=24.0
        )
        controller.new_session()

        # Fast forward past prepare time into movement phase
        controller._lissajous_start_time = time.time() - (controller._prepare_time + 0.1)

        face_info, gaze_info = self._create_mock_face_gaze()

        # Record 10 frames
        for step in range(10):
            # Advance time slightly to move the dot
            controller._lissajous_start_time -= 0.05
            controller.add_cali_feature(gaze_info, face_info)

        # Verify 10 frames collected
        self.assertEqual(len(controller.feature_vectors[0]), 10)
        self.assertEqual(len(controller._target_history), 10)

        # Verify latency compensation for frame 6:
        # Frame index 6 should be paired with target from index 6 - 4 = 2!
        expected_x, expected_y = controller._target_history[6 - latency]
        actual_label = controller.label_vectors[0][6]
        self.assertAlmostEqual(actual_label[0], expected_x, places=5)
        self.assertAlmostEqual(actual_label[1], expected_y, places=5)

    def test_lissajous_custom_latency(self):
        """
        Test configurable frame latency (e.g. 2 frames).
        """
        latency = 2
        controller = CalibrationController(
            cali_mode=CalibrationMode.LISSAJOUS,
            camera_pos=(17.15, -0.68),
            screen_size=np.array([1920, 1080]),
            lissajous_frame_latency=latency,
            lissajous_duration=24.0
        )
        controller.new_session()
        controller._lissajous_start_time = time.time() - (controller._prepare_time + 0.1)

        face_info, gaze_info = self._create_mock_face_gaze()
        for step in range(8):
            controller._lissajous_start_time -= 0.05
            controller.add_cali_feature(gaze_info, face_info)

        # Frame 5 should be paired with target 5 - 2 = 3
        expected_x, expected_y = controller._target_history[5 - latency]
        actual_label = controller.label_vectors[0][5]
        self.assertAlmostEqual(actual_label[0], expected_x, places=5)
        self.assertAlmostEqual(actual_label[1], expected_y, places=5)


if __name__ == '__main__':
    unittest.main()
