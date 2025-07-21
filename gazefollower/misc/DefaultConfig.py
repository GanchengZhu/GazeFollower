# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from enum import IntEnum
from pathlib import Path
from typing import Tuple

import numpy as np
from screeninfo import get_monitors


class CalibrationMode(IntEnum):
    """Enum representing calibration modes"""
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
        """

        self.model_fit_instruction = "Calibration model is fitting.\nPlease wait."
        self.eye_blink_threshold = 10
        self.cali_mode = 13

        self._monitors = get_monitors()
        self.screen_size = np.array([self._monitors[0].width, self._monitors[0].height])

        self._current_dir = Path(__file__).parent.parent.absolute()
        # Calibration resource file paths
        # Sound file for target beep during calibration
        self.cali_target_sound = str(self._current_dir / 'res' / 'audio' / 'beep.wav')
        self.cali_target_img = str(self._current_dir / 'res' / 'image' / 'dot.png')
        self.cali_target_size = (70, 70)

        self.camera_position: Tuple = (17.15, -0.68)
        self.screen_physical_size = None
        self.cali_instruction = "Please look at the dot.\nPress `SPACE` to continue."

    @property
    def cali_mode(self):
        return self._cali_mode

    @cali_mode.setter
    def cali_mode(self, mode):
        if isinstance(mode, CalibrationMode):
            self._cali_mode = mode
        elif mode == 5:
            self._cali_mode = CalibrationMode.FIVE_POINT
        elif mode == 9:
            self._cali_mode = CalibrationMode.NINE_POINT
        elif mode == 13:
            self._cali_mode = CalibrationMode.THIRTEEN_POINT
        else:
            raise ValueError("Invalid calibration mode. Must be 5, 9 13, or a CalibrationMode instance.")
