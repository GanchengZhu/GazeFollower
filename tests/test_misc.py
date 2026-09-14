# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import unittest
import numpy as np
from gazefollower.misc import (
    FaceInfo, GazeInfo, DefaultConfig, CameraRunningState,
    clip_patch, px2cm, cm2px, generate_points
)


class TestMisc(unittest.TestCase):

    def test_clip_patch_valid(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:30, 10:30] = 255
        patch = clip_patch(frame, (10, 10, 20, 20))
        self.assertIsNotNone(patch)
        self.assertEqual(patch.shape, (20, 20, 3))
        self.assertEqual(patch[0, 0, 0], 255)

    def test_clip_patch_invalid(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Out of bounds or negative dimensions
        self.assertIsNone(clip_patch(frame, (-1, 0, 10, 10)))
        self.assertIsNone(clip_patch(frame, (0, -1, 10, 10)))
        self.assertIsNone(clip_patch(frame, (0, 0, 0, 10)))
        self.assertIsNone(clip_patch(frame, (100, 50, 10, 10)))

    def test_px2cm_and_cm2px_roundtrip(self):
        cam_pos = (15.0, 10.0)
        phys_size = (30.0, 20.0)
        px_size = (1920, 1080)

        original_px = (960.0, 540.0)
        cm_pos = px2cm(original_px, cam_pos, phys_size, px_size)
        recovered_px = cm2px(cm_pos, cam_pos, phys_size, px_size)

        self.assertAlmostEqual(original_px[0], recovered_px[0], places=4)
        self.assertAlmostEqual(original_px[1], recovered_px[1], places=4)

    def test_generate_points(self):
        pts = generate_points()
        self.assertIsInstance(pts, np.ndarray)
        self.assertEqual(pts.shape, (45, 2))
        # Ensure all points are within [0, 1] normalized space
        self.assertTrue(np.all(pts >= 0.0) and np.all(pts <= 1.0))

    def test_default_config(self):
        cfg = DefaultConfig()
        self.assertIsNotNone(cfg.cali_instruction)
        self.assertIsNotNone(cfg.camera_position)
        self.assertEqual(cfg.lissajous_frame_latency, 4)
        self.assertFalse(cfg.cali_click_mode)

        # Test Lissajous mode
        cfg.cali_mode = 0
        from gazefollower.misc import CalibrationMode
        self.assertEqual(cfg.cali_mode, CalibrationMode.LISSAJOUS)
        self.assertIn("Lissajous", cfg.cali_instruction)

        # Test click mode
        cfg.cali_mode = 9
        cfg.cali_click_mode = True
        self.assertIn("click", cfg.cali_instruction.lower())

    def test_ui_backend_texture(self):
        import pygame
        from pathlib import Path
        import gazefollower
        from gazefollower.ui.UIBackend import PyGameUIBackend
        pygame.init()
        surface = pygame.Surface((640, 480))
        backend = PyGameUIBackend(surface)

        # 1. Standard 3D RGB array
        test_img = np.zeros((64, 64, 3), dtype=np.uint8)
        backend.draw_texture(test_img, (10, 10, 50, 50))
        backend.draw_image(test_img, (10, 10, 50, 50))

        # 2. String file path (regression test for ValueError: axes don't match array)
        img_path = str(Path(gazefollower.__file__).parent / "res" / "image" / "frame.jpg")
        backend.draw_texture(img_path, (10, 10, 50, 50))
        backend.draw_image(img_path, (10, 10, 50, 50))

        # 3. 2D grayscale array
        gray_img = np.zeros((64, 64), dtype=np.uint8)
        backend.draw_texture(gray_img, (10, 10, 50, 50))
        backend.draw_image(gray_img, (10, 10, 50, 50))

        # 4. None and empty
        backend.draw_texture(None, (10, 10, 50, 50))
        backend.draw_image(None, (10, 10, 50, 50))
        backend.draw_texture(np.array([]), (10, 10, 50, 50))


if __name__ == '__main__':
    unittest.main()
