# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import time

import numpy as np

from ..logger import Log
from ..misc import FaceInfo, GazeInfo, px2cm, generate_points, CalibrationMode, cm2px


class CalibrationController:
    def __init__(self, cali_mode, camera_pos, screen_size, physical_screen_size=None, eye_blink_threshold=10):
        self.mean_euclidean_error = None
        self.cali_available = False
        self.labels = None
        self.predictions = None
        self.normalized_point = generate_points()
        self._nine_cali_idx = [23, 1, 5, 9, 19, 27, 37, 41, 45, 23]
        self._five_cali_idx = [23, 1, 9, 37, 45, 23]
        self._thirteen_cali_idx = [23, 1, 5, 9, 12, 16, 19, 27, 30, 34, 37, 41, 45, 23]

        self._six_vali_idx = [2, 8, 22, 24, 38, 44]
        self._eight_vali_idx = [2, 8, 13, 15, 31, 33, 38, 44]
        self._twenty_vali_idx = [2, 4, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35, 38, 40, 42, 44]

        self.cam_pos = camera_pos
        self.screen_size = screen_size
        self.eye_blink_threshold = eye_blink_threshold
        self.physical_screen_size = physical_screen_size
        self.cali_mode: CalibrationMode = cali_mode

        self.x = self.screen_size[0] // 2
        self.y = self.screen_size[1] // 2
        self.progress = 0

        self._prepare_time = 1.5  # time for waiting subject look at the dot
        self._wait_time = 0.5
        self._n_frame_need_collect = 45

        self.feature_ids = []
        self.feature_vectors = []
        self.label_vectors = []
        self._n_frame_added = 0
        self._current_index = 0
        self._feature_full_time = 0
        self._each_point_onset_time = 0
        self.cali_model_fitted = False
        self.calibrating = False

    def update_position(self):
        if self.cali_mode == CalibrationMode.NINE_POINT:
            position_idx = self._nine_cali_idx[self._current_index]
        elif self.cali_mode == CalibrationMode.FIVE_POINT:
            position_idx = self._five_cali_idx[self._current_index]
        else:
            position_idx = self._thirteen_cali_idx[self._current_index]

        percent_point = self.normalized_point[position_idx - 1]
        self.x = percent_point[0]
        self.y = percent_point[1]
        self.progress = int(np.round(self._n_frame_added * 100 / self._n_frame_need_collect))

    def new_session(self):
        self.feature_ids.clear()
        self.feature_vectors.clear()
        self.label_vectors.clear()
        self._n_frame_added = 0
        self._current_index = 0
        self.cali_model_fitted = False
        self.calibrating = True
        self.update_position()
        self._each_point_onset_time = time.time()

        for _ in range(self.cali_mode.value):
            self.feature_ids.append([])
            self.feature_vectors.append([])
            self.label_vectors.append([])

    def add_cali_feature(self, gaze_info: GazeInfo, face_info: FaceInfo):
        if self._current_index == self.cali_mode.value + 1:
            Log.i("calibrating shutdowns")
            self.calibrating = False
            return
        self.update_position()
        if (time.time() - self._each_point_onset_time) >= self._prepare_time:
            if gaze_info.status and (self._n_frame_added < self._n_frame_need_collect) and (
                    face_info.left_eye_openness > self.eye_blink_threshold) and (
                    face_info.right_eye_openness > self.eye_blink_threshold):
                if self._current_index != 0 and self._n_frame_added < self._n_frame_need_collect:
                    self.feature_vectors[self._current_index - 1].append(gaze_info.features)
                    # self.label_vectors[self._current_index - 1].append(gaze_info.)
                    self.feature_ids[self._current_index - 1].append([self._current_index - 1])

                    if self.physical_screen_size:
                        # has physical_screen_size
                        added_pos = px2cm((self.x * self.screen_size[0], self.y * self.screen_size[1]),
                                          self.cam_pos, self.physical_screen_size, self.screen_size)
                    else:
                        added_pos = [self.x, self.y]
                    self.label_vectors[self._current_index - 1].append(added_pos)

                self._n_frame_added += 1
                if self._n_frame_added == self._n_frame_need_collect:
                    self._feature_full_time = time.time()

            if self._n_frame_added == self._n_frame_need_collect:
                if time.time() - self._feature_full_time >= self._wait_time:
                    self._current_index += 1
                    self._n_frame_added = 0
                    self._each_point_onset_time = time.time()

    def set_calibration_results(self, has_calibrated, mean_euclidean_error, labels, predictions):
        self.cali_available = has_calibrated
        self.mean_euclidean_error = mean_euclidean_error
        self.labels = labels
        self.predictions = predictions

    def convert_to_pixel(self, raw_pos):
        """
        Convert raw position values to pixel coordinates.

        The raw position could be either percentage-based (relative to screen dimensions)
        or in physical centimeters, depending on whether physical screen size is specified.

        Args:
            raw_pos (tuple): Input position coordinates, could be either
                (percentage_x, percentage_y) if using relative units, or
                (cm_x, cm_y) if physical screen size is available.

        Returns:
            tuple: Pixel coordinates (x, y) converted based on input format.
                Uses cm2px conversion if physical screen size exists, otherwise
                interprets input as screen percentages.

        Note:
            Requires either physical_screen_size for centimeter conversion or
            screen_size for percentage conversion to be properly configured.
        """
        if self.physical_screen_size:
            return cm2px(raw_pos, self.cam_pos, self.physical_screen_size, self.screen_size)
        else:
            return raw_pos[0] * self.screen_size[0], raw_pos[1] * self.screen_size[1]
