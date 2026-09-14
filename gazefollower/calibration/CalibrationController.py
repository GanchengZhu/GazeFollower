# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import math
import time
from typing import List, Tuple

import numpy as np

from ..logger import Log
from ..misc import FaceInfo, GazeInfo, px2cm, generate_points, CalibrationMode, cm2px


class CalibrationController:
    def __init__(self, cali_mode, camera_pos, screen_size, physical_screen_size=None,
                 eye_blink_threshold=10, cali_click_mode: bool = False,
                 lissajous_frame_latency: int = 4, lissajous_duration: float = 24.0,
                 lissajous_freq_x: float = 1.0 / 12.0, lissajous_freq_y: float = 1.0 / 8.0):
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
        self.cali_click_mode: bool = cali_click_mode

        # Lissajous pattern parameters
        self.lissajous_frame_latency: int = lissajous_frame_latency
        self.lissajous_duration: float = lissajous_duration
        self.lissajous_freq_x: float = lissajous_freq_x
        self.lissajous_freq_y: float = lissajous_freq_y
        self._visible_margin = 0.08
        self._amp_x = (1.0 - 2 * self._visible_margin) / 2
        self._amp_y = (1.0 - 2 * self._visible_margin) / 2
        self._center_x = 0.5
        self._center_y = 0.5
        self._target_history: List[Tuple[float, float]] = []
        self._lissajous_start_time: float = 0

        self.x = 0.5
        self.y = 0.5
        self.progress = 0

        self._prepare_time = 1.5  # time for waiting subject look at the dot
        self._wait_time = 0.5
        self._n_frame_need_collect = 45

        self.feature_ids = []
        self.feature_vectors = []
        self.label_vectors = []
        self._current_point_features = []
        self._current_point_labels = []
        self._n_frame_added = 0
        self._current_index = 0
        self._feature_full_time = 0
        self._each_point_onset_time = 0
        self.cali_model_fitted = False
        self.calibrating = False
        self.is_point_collecting = False

    def update_position(self):
        if self.cali_mode == CalibrationMode.LISSAJOUS:
            now = time.time()
            t = now - self._lissajous_start_time
            if t < self._prepare_time:
                self.x = self._center_x
                self.y = self._center_y
                self.progress = 0
            else:
                t_move = t - self._prepare_time
                if t_move >= self.lissajous_duration:
                    self.calibrating = False
                    self.progress = 100
                    return
                self.x = self._center_x + self._amp_x * math.sin(2 * math.pi * self.lissajous_freq_x * t_move)
                self.y = self._center_y + self._amp_y * math.sin(2 * math.pi * self.lissajous_freq_y * t_move)
                self.progress = int(np.round(t_move * 100 / self.lissajous_duration))
            return

        if self.cali_mode == CalibrationMode.NINE_POINT:
            position_idx = self._nine_cali_idx[self._current_index]
        elif self.cali_mode == CalibrationMode.FIVE_POINT:
            position_idx = self._five_cali_idx[self._current_index]
        else:
            position_idx = self._thirteen_cali_idx[self._current_index]

        percent_point = self.normalized_point[position_idx - 1]
        self.x = percent_point[0]
        self.y = percent_point[1]
        if self.cali_click_mode:
            if not self.is_point_collecting:
                self.progress = 0
            else:
                self.progress = min(100, int(np.round(self._n_frame_added * 100 / self._n_frame_need_collect)))
        else:
            self.progress = int(np.round(self._n_frame_added * 100 / self._n_frame_need_collect))

    def new_session(self):
        self.feature_ids.clear()
        self.feature_vectors.clear()
        self.label_vectors.clear()
        self._current_point_features.clear()
        self._current_point_labels.clear()
        self._target_history.clear()
        self._n_frame_added = 0
        self._current_index = 0
        self.cali_model_fitted = False
        self.calibrating = True
        self.is_point_collecting = False
        self._each_point_onset_time = time.time()
        self._lissajous_start_time = time.time()

        if self.cali_mode == CalibrationMode.LISSAJOUS:
            self.feature_ids.append([])
            self.feature_vectors.append([])
            self.label_vectors.append([])
        else:
            for _ in range(self.cali_mode.value):
                self.feature_ids.append([])
                self.feature_vectors.append([])
                self.label_vectors.append([])

        self.update_position()

    def on_target_clicked(self) -> bool:
        """
        Scheme A (Point-and-Click):
        Called when the user clicks on the target dot.
        Triggers frame collection for the current point (progress counts from 0 to 100).
        Lissajous calibration pattern does NOT support click (viewing-only).
        """
        if self.cali_mode == CalibrationMode.LISSAJOUS:
            # Lissajous calibration pattern does not support click
            return False

        if not self.calibrating:
            return False

        if not self.is_point_collecting:
            self.is_point_collecting = True
            self._n_frame_added = 0
            self.progress = 0
            self._current_point_features.clear()
            self._current_point_labels.clear()
            self._each_point_onset_time = time.time()
            Log.i(f"Target point {self._current_index} clicked. Collecting data (0 to 100)...")
            return True

        return False

    def _add_lissajous_feature(self, gaze_info: GazeInfo, face_info: FaceInfo):
        """
        Record gaze features along the Lissajous smooth pursuit trajectory
        with frame latency compensation.
        """
        now = time.time()
        t = now - self._lissajous_start_time
        if t < self._prepare_time:
            self.x = self._center_x
            self.y = self._center_y
            self.progress = 0
            return

        t_move = t - self._prepare_time
        if t_move >= self.lissajous_duration:
            Log.i("Lissajous calibration complete")
            self.calibrating = False
            return

        self.update_position()

        if (gaze_info.status and gaze_info.features is not None and
                face_info.left_eye_openness > self.eye_blink_threshold and
                face_info.right_eye_openness > self.eye_blink_threshold):
            self._target_history.append((self.x, self.y))

            # Apply frame latency compensation (default 4 frames delay)
            latency_idx = max(0, len(self._target_history) - 1 - self.lissajous_frame_latency)
            comp_x, comp_y = self._target_history[latency_idx]

            if self.physical_screen_size:
                added_pos = px2cm((comp_x * self.screen_size[0], comp_y * self.screen_size[1]),
                                  self.cam_pos, self.physical_screen_size, self.screen_size)
            else:
                added_pos = [comp_x, comp_y]

            # Assign 24 benchmark segments across the duration for evaluation
            segment_id = min(23, int((t_move / self.lissajous_duration) * 24))
            self.feature_vectors[0].append(gaze_info.features)
            self.label_vectors[0].append(added_pos)
            self.feature_ids[0].append([segment_id])
            self._n_frame_added += 1

    def add_cali_feature(self, gaze_info: GazeInfo, face_info: FaceInfo):
        if not self.calibrating:
            return

        if self.cali_mode == CalibrationMode.LISSAJOUS:
            self._add_lissajous_feature(gaze_info, face_info)
            return

        if self._current_index == self.cali_mode.value + 1:
            Log.i("calibrating shutdowns")
            self.calibrating = False
            return

        self.update_position()

        if self.cali_click_mode:
            if not self.is_point_collecting:
                return

            if gaze_info.status and gaze_info.features is not None and (
                    face_info.left_eye_openness > self.eye_blink_threshold) and (
                    face_info.right_eye_openness > self.eye_blink_threshold):
                if self._current_index != 0 and self._n_frame_added < self._n_frame_need_collect:
                    self.feature_vectors[self._current_index - 1].append(gaze_info.features)
                    self.feature_ids[self._current_index - 1].append([self._current_index - 1])

                    if self.physical_screen_size:
                        added_pos = px2cm((self.x * self.screen_size[0], self.y * self.screen_size[1]),
                                          self.cam_pos, self.physical_screen_size, self.screen_size)
                    else:
                        added_pos = [self.x, self.y]
                    self.label_vectors[self._current_index - 1].append(added_pos)

                self._n_frame_added += 1
                self.progress = min(100, int(np.round(self._n_frame_added * 100 / self._n_frame_need_collect)))

                if self._n_frame_added >= self._n_frame_need_collect:
                    Log.i(f"Point {self._current_index}/{self.cali_mode.value} collected 100%. Advancing.")
                    self._current_index += 1
                    self.is_point_collecting = False
                    self._n_frame_added = 0
                    self.progress = 0
                    self._each_point_onset_time = time.time()
                    if self._current_index == self.cali_mode.value + 1:
                        Log.i("All calibration points completed")
                        self.calibrating = False
                    else:
                        self.update_position()
            return

        # Automatic timer / frame-count based mode (default passive mode)
        if (time.time() - self._each_point_onset_time) >= self._prepare_time:
            if gaze_info.status and (self._n_frame_added < self._n_frame_need_collect) and (
                    face_info.left_eye_openness > self.eye_blink_threshold) and (
                    face_info.right_eye_openness > self.eye_blink_threshold):
                if self._current_index != 0 and self._n_frame_added < self._n_frame_need_collect:
                    self.feature_vectors[self._current_index - 1].append(gaze_info.features)
                    self.feature_ids[self._current_index - 1].append([self._current_index - 1])

                    if self.physical_screen_size:
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

