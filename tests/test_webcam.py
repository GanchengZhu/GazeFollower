# tests/test_webcam.py
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from gazefollower.camera import WebCamCamera
from gazefollower.misc import CameraRunningState


class TestWebCamCamera(unittest.TestCase):
    @patch.object(cv2, 'VideoCapture')
    def setUp(self, mock_video_capture):
        """
        Set up the test environment before each test.
        Mock the VideoCapture object to avoid using a real camera.
        """
        self.mock_cap = MagicMock()
        mock_video_capture.return_value = self.mock_cap
        self.mock_cap.isOpened.return_value = True
        self.camera = WebCamCamera()
        self.received_frames = []

    def frame_callback(self, running_state, timestamp, frame, *args, **kwargs):
        """
        Callback function to record the captured frame without opening GUI windows.
        """
        self.received_frames.append((running_state, timestamp, frame))

    def test_camera_functionality(self):
        """
        Test the camera frame capture and callback execution in headless mode.
        """
        self.camera.set_on_image_callback(self.frame_callback)
        self.camera.camera_running_state = CameraRunningState.SAMPLING

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera._camera_thread_running = True

        def mock_read():
            self.camera._camera_thread_running = False
            return True, dummy_frame

        self.mock_cap.read.side_effect = mock_read

        # Run the capture method with mock time to simulate frame capture
        with patch('time.time_ns', return_value=123456789):
            self.camera.capture()

        self.assertEqual(len(self.received_frames), 1)
        state, ts, frame = self.received_frames[0]
        self.assertEqual(state, CameraRunningState.SAMPLING)
        self.assertEqual(ts, 123456789)
        self.assertEqual(frame.shape, (480, 640, 3))

    def tearDown(self):
        """
        Clean up after each test case.
        """
        self.camera.close()


if __name__ == '__main__':
    unittest.main()
