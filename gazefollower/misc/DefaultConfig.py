# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from enum import IntEnum
from pathlib import Path
from typing import Tuple

import numpy as np
try:
    from screeninfo import get_monitors
except Exception:
    get_monitors = None


class CalibrationMode(IntEnum):
    """Enum representing calibration modes"""
    LISSAJOUS = 0
    THIRTEEN_POINT = 13
    NINE_POINT = 9
    FIVE_POINT = 5


class DefaultConfig:
    def __init__(self):
        """System default configuration class containing all parameters required for program execution

        Consolidates paths, hyperparameters, screen settings, calibration resources, and hardware configurations.
            model_fit_instruction (str): Display message during model fitting process
            eye_blink_threshold (int): Frame count threshold for blink detection (unit: frames)
            screen_size (np.ndarray): Screen resolution in pixels [width, height]
            cali_target_sound (str): File path for calibration target beep sound
            cali_target_img (str): File path for calibration target dot image
            cali_target_size (tuple): Display dimensions of calibration target (width, height) in pixels
            camera_position (Tuple[float, float]): Physical camera coordinates (x, y) in centimeters
            screen_physical_size (None|tuple): Physical screen dimensions (width, height) in centimeters
            cali_instruction (str): Instruction text displayed during calibration
            cali_click_mode (bool): Whether point calibration requires clicking the target (Scheme A)
            lissajous_frame_latency (int): Camera and pipeline latency in frames for Lissajous pursuit calibration (default: 4)
            lissajous_duration (float): Total duration in seconds for Lissajous pursuit calibration
            lissajous_freq_x (float): Horizontal frequency for Lissajous curve
            lissajous_freq_y (float): Vertical frequency for Lissajous curve
        """

        self.model_fit_instruction = "Calibration model is fitting.\nPlease wait."
        self.eye_blink_threshold = 10
        self.cali_mode = 13

        # Click calibration mode (Scheme A: point-and-click)
        self.cali_click_mode = False

        # Lissajous calibration pattern settings
        self.lissajous_frame_latency = 4
        self.lissajous_duration = 24.0
        self.lissajous_freq_x = 1.0 / 12.0
        self.lissajous_freq_y = 1.0 / 8.0

        # Processing mode: multiprocessing vs multithreading
        self.use_multiprocessing = True

        # 1-Euro Filter settings for face and eye smoothing
        self.enable_face_filter = True
        self.filter_min_cutoff = 1.0
        self.filter_beta = 0.01

        self._monitors = []
        self.screen_size = np.array([1920, 1080])
        if get_monitors is not None:
            try:
                monitors = get_monitors()
                if monitors and len(monitors) > 0:
                    self._monitors = monitors
                    self.screen_size = np.array([monitors[0].width, monitors[0].height])
            except Exception:
                self._monitors = []
                self.screen_size = np.array([1920, 1080])

        self._current_dir = Path(__file__).parent.parent.absolute()
        # Calibration resource file paths
        # Sound file for target beep during calibration
        self.cali_target_sound = str(self._current_dir / 'res' / 'audio' / 'beep.wav')
        self.cali_target_img = str(self._current_dir / 'res' / 'image' / 'dot.png')
        self.cali_target_size = (70, 70)

        self.camera_position: Tuple = (17.15, -0.68)
        self.screen_physical_size = None
        self._custom_cali_instruction = None

    @property
    def cali_instruction(self):
        if self._custom_cali_instruction is not None:
            return self._custom_cali_instruction
        if self.cali_mode == CalibrationMode.LISSAJOUS:
            return "Lissajous Smooth Pursuit Calibration\nPlease follow the moving dot with your eyes.\nPress `SPACE` to continue."
        elif self.cali_click_mode:
            return "Point-and-Click Calibration\nPlease look directly at each dot and click it with the mouse.\nPress `SPACE` to continue."
        else:
            return "Please look at the dot.\nPress `SPACE` to continue."

    @cali_instruction.setter
    def cali_instruction(self, text):
        self._custom_cali_instruction = text

    @property
    def cali_mode(self):
        return self._cali_mode

    @cali_mode.setter
    def cali_mode(self, mode):
        if isinstance(mode, CalibrationMode):
            self._cali_mode = mode
        elif mode == 0 or str(mode).lower() in ('0', 'lissajous'):
            self._cali_mode = CalibrationMode.LISSAJOUS
        elif mode == 5:
            self._cali_mode = CalibrationMode.FIVE_POINT
        elif mode == 9:
            self._cali_mode = CalibrationMode.NINE_POINT
        elif mode == 13:
            self._cali_mode = CalibrationMode.THIRTEEN_POINT
        else:
            raise ValueError("Invalid calibration mode. Must be 5, 9, 13, 0 (Lissajous), or a CalibrationMode instance.")
