# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pygame

from .mp_worker import mp_worker_entry
from ..calibration import CalibrationController, MultivariateRidgeCalibration, SVRCalibration
from ..face_alignment import BlazeFaceAlignment, MediaPipeFaceAlignment
from ..filter import HeuristicFilter, OneEuroFilter
from ..gaze_estimator import MGazeNetGazeEstimator
from ..logger import Log
from ..misc import DefaultConfig, FaceInfo, GazeInfo
from ..ui import CalibrationUI, CameraPreviewerUI


class MultiprocessGazeFollower:
    """
    MultiprocessGazeFollower provides the exact same API as GazeFollower, but runs
    camera capture, image preprocessing, face alignment, and neural network gaze
    estimation inside a dedicated child process (multiprocessing).

    This completely decouples the Python GIL from the main process, ensuring that
    PsychoPy stimulus rendering and OpenGL win.flip() never drop frames due to
    eye-tracking model inference.
    """

    def __init__(self,
                 camera=None,
                 face_alignment=None,
                 gaze_estimator=None,
                 gaze_filter=None,
                 calibration=None,
                 config=None):
        self.config = config if config is not None else DefaultConfig()
        self.screen_size = self.config.screen_size
        self.calibration = calibration if calibration is not None else MultivariateRidgeCalibration()
        self.gaze_filter = gaze_filter if gaze_filter is not None else HeuristicFilter()

        self._calibration_controller = CalibrationController(
            cali_mode=self.config.cali_mode,
            camera_pos=self.config.camera_position,
            screen_size=self.screen_size,
            physical_screen_size=self.config.screen_physical_size,
            eye_blink_threshold=self.config.eye_blink_threshold,
            cali_click_mode=self.config.cali_click_mode,
            lissajous_frame_latency=self.config.lissajous_frame_latency,
            lissajous_duration=self.config.lissajous_duration,
            lissajous_freq_x=self.config.lissajous_freq_x,
            lissajous_freq_y=self.config.lissajous_freq_y,
        )

        self._gaze_info = None
        self._subscribers = []
        self._subscriber_lock = threading.Lock()

        # Temporary logging files
        self._tmp_dir = Path(tempfile.gettempdir()) / "gazefollower"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_sample_path = self._tmp_dir / f"sample_{int(time.time() * 1000)}.csv"

        # Config dictionary to pass across process boundary
        config_dict = {
            'webcam_id': getattr(camera, 'webcam_id', 0) if camera is not None else 0,
            'screen_size': self.screen_size,
            'screen_physical_size': self.config.screen_physical_size,
            'camera_position': self.config.camera_position,
            'eye_blink_threshold': self.config.eye_blink_threshold,
            'alpha': getattr(self.calibration, 'alpha', 0.1),
            'face_alignment_type': 'blazeface' if isinstance(face_alignment, BlazeFaceAlignment) else 'mediapipe',
            'enable_face_filter': getattr(face_alignment, 'enable_filter', getattr(self.config, 'enable_face_filter', True)),
            'filter_min_cutoff': getattr(face_alignment, 'filter_min_cutoff', getattr(self.config, 'filter_min_cutoff', 1.0)),
            'filter_beta': getattr(face_alignment, 'filter_beta', getattr(self.config, 'filter_beta', 0.01)),
            'gaze_estimator_type': 'mgazenet',
            'calibration_type': 'svr' if isinstance(self.calibration, SVRCalibration) else 'ridge',
            'filter_type': 'one_euro' if isinstance(self.gaze_filter, OneEuroFilter) else 'heuristic',
        }
        self.config_dict = config_dict

        # IPC Primitives
        ctx = multiprocessing.get_context("spawn")
        self._cmd_queue = ctx.Queue()
        self._preview_queue = ctx.Queue(maxsize=1)
        self._cali_feature_queue = ctx.Queue(maxsize=200)
        self._gaze_queue = ctx.Queue(maxsize=2)

        self._worker_process = ctx.Process(
            target=mp_worker_entry,
            args=(self._cmd_queue, self._preview_queue, self._cali_feature_queue,
                  self._gaze_queue, config_dict)
        )
        self._worker_process.daemon = True
        self._worker_process.start()

        self.calibration_ui = None
        self.camera_previewer_ui = None

    @staticmethod
    def backend_name(screen):
        from pygame import Surface
        if isinstance(screen, Surface):
            return 'pygame'
        else:
            from psychopy.visual import Window
            if isinstance(screen, Window):
                return 'psychopy'
        raise Exception("Screen cannot be None. Please pass pygame window or psychopy window instance")

    def preview(self, win=None):
        if win is None:
            pygame.init()
            win = pygame.display.set_mode(self.screen_size.tolist(), pygame.FULLSCREEN)
            pygame.display.set_caption("Camera Preview")
            backend_name = "pygame"
        else:
            backend_name = self.backend_name(win)

        self.camera_previewer_ui = CameraPreviewerUI(win=win, backend_name=backend_name)

        import cv2
        from ..camera import WebCamCamera
        from ..misc import clip_patch
        webcam_id = self.config_dict.get('webcam_id', 0)
        local_cam = WebCamCamera(webcam_id=webcam_id)
        enable_filter = self.config_dict.get('enable_face_filter', True)
        min_cutoff = self.config_dict.get('filter_min_cutoff', 1.0)
        beta = self.config_dict.get('filter_beta', 0.01)
        if self.config_dict.get('face_alignment_type') == 'blazeface':
            local_fa = BlazeFaceAlignment(enable_filter=enable_filter, filter_min_cutoff=min_cutoff, filter_beta=beta)
        else:
            local_fa = MediaPipeFaceAlignment(enable_filter=enable_filter, filter_min_cutoff=min_cutoff, filter_beta=beta)

        def on_preview_frame(state, timestamp, frame):
            face_info = local_fa.detect(timestamp, frame)
            face_patch = None
            left_eye_patch = None
            right_eye_patch = None
            if face_info.status and face_info.can_gaze_estimation:
                face_patch = clip_patch(frame, face_info.face_rect)
                left_eye_patch = clip_patch(frame, face_info.left_rect)
                right_eye_patch = clip_patch(frame, face_info.right_rect)

                x, y, w, h = face_info.left_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                x, y, w, h = face_info.right_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

                if face_patch is not None:
                    fx, fy, fw, fh = face_info.face_rect
                    lx, ly, lw, lh = face_info.left_rect
                    relative_left_x = lx - fx
                    relative_left_y = ly - fy
                    cv2.rectangle(face_patch, (relative_left_x, relative_left_y),
                                  (relative_left_x + lw, relative_left_y + lh), color=(255, 0, 0), thickness=2)
                    rx, ry, rw, rh = face_info.right_rect
                    relative_right_x = rx - fx
                    relative_right_y = ry - fy
                    cv2.rectangle(face_patch, (relative_right_x, relative_right_y),
                                  (relative_right_x + rw, relative_right_y + rh), color=(0, 0, 255), thickness=2)

            self.camera_previewer_ui.update_images(frame, face_patch, left_eye_patch, right_eye_patch)
            self.camera_previewer_ui.face_info_dict = face_info.to_dict()

        local_cam.set_on_image_callback(on_preview_frame)
        local_cam.start_previewing()
        self.camera_previewer_ui.draw()
        local_cam.stop_previewing()
        try:
            local_fa.release()
        except Exception:
            pass

    def calibrate(self, win=None):
        if win is None:
            pygame.init()
            win = pygame.display.set_mode(self.screen_size.tolist(), pygame.FULLSCREEN)
            pygame.display.set_caption("Calibration UI")
            backend_name = "pygame"
        else:
            backend_name = self.backend_name(win)

        self.calibration_ui = CalibrationUI(win=win, backend_name=backend_name, config=self.config)

        while True:
            self._calibration_controller.new_session()
            self.calibration_ui.new_session()
            self.calibration_ui.draw_guidance(self.config.cali_instruction)

            # Start calibration on worker
            self._cmd_queue.put(('START_CALIBRATING',))

            # Consumer thread in main process to safely feed extracted features to controller
            draining = True

            def feature_consumer():
                while draining:
                    try:
                        item = self._cali_feature_queue.get(timeout=0.02)
                        ts, gaze_info, face_info = item
                        self._calibration_controller.add_cali_feature(gaze_info, face_info)
                    except queue.Empty:
                        continue
                    except Exception:
                        break

            consumer_thread = threading.Thread(target=feature_consumer)
            consumer_thread.daemon = True
            consumer_thread.start()

            # Draw calibration animation on UI
            self.calibration_ui.draw(self._calibration_controller)

            draining = False
            consumer_thread.join(timeout=0.5)

            # Fit calibration model
            features = np.array(self._calibration_controller.feature_vectors)
            n_point, n_frame, feature_dim = features.shape
            features = np.reshape(features, (n_point * n_frame, feature_dim))

            labels = np.array(self._calibration_controller.label_vectors)
            n_point, n_frame, label_dim = labels.shape
            labels = np.reshape(labels, (n_point * n_frame, label_dim))

            ids = np.array(self._calibration_controller.feature_ids)
            n_point, n_frame, ids_dim = ids.shape
            point_ids = np.reshape(ids, (n_point * n_frame, ids_dim))

            has_calibrated, mean_euclidean_error, predictions = self.calibration.calibrate(
                features, labels, point_ids
            )
            self._calibration_controller.set_calibration_results(
                has_calibrated, mean_euclidean_error, labels, predictions
            )
            self._calibration_controller.cali_model_fitted = True

            # Synchronize fitted calibration model to worker process
            self._cmd_queue.put(('SET_CALIBRATION_MODEL', {
                'weights': getattr(self.calibration, 'weights', None),
                'has_calibrated': has_calibrated
            }))

            user_response = self.calibration_ui.draw_cali_result(
                self._calibration_controller, self.config.model_fit_instruction
            )
            self._cmd_queue.put(('STOP_CALIBRATING',))

            if user_response:
                break

    def start_sampling(self):
        self._cmd_queue.put(('START_SAMPLING', str(self._tmp_sample_path)))

    def stop_sampling(self):
        self._cmd_queue.put(('STOP_SAMPLING',))

    def get_gaze_info(self) -> GazeInfo:
        try:
            while not self._gaze_queue.empty():
                self._gaze_info = self._gaze_queue.get_nowait()
                with self._subscriber_lock:
                    for func, args, kwargs in self._subscribers:
                        func(FaceInfo(), self._gaze_info, *args, **kwargs)
        except queue.Empty:
            pass
        return self._gaze_info

    def add_subscriber(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with self._subscriber_lock:
            self._subscribers.append((func, args, kwargs))

    def remove_subscriber(self, func):
        with self._subscriber_lock:
            for sub in list(self._subscribers):
                if func == sub[0]:
                    self._subscribers.remove(sub)

    def send_trigger(self, trigger_num: int):
        self._cmd_queue.put(('SEND_TRIGGER', int(trigger_num)))

    def save_data(self, path: str):
        time.sleep(0.05)
        if self._tmp_sample_path.exists():
            shutil.copyfile(str(self._tmp_sample_path), path)
            Log.i(f"Saved gaze sample data to {path}")

    def release(self):
        try:
            self._cmd_queue.put(('TERMINATE',))
            self._worker_process.join(timeout=1.5)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
                self._worker_process.join(timeout=1.0)
        except Exception:
            pass
        if hasattr(self, '_tmp_sample_path') and self._tmp_sample_path.exists():
            try:
                self._tmp_sample_path.unlink()
            except Exception:
                pass
