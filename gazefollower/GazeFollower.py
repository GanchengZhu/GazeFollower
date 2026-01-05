# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import datetime
import pathlib
import re
import shutil
import threading

import cv2
import numpy as np
import pygame

from .calibration import Calibration, CalibrationController
from .calibration import SVRCalibration
from .camera import Camera
from .camera import WebCamCamera
from .face_alignment import FaceAlignment
from .face_alignment import MediaPipeFaceAlignment
from .filter import HeuristicFilter
from .filter.Filter import Filter
from .gaze_estimator import GazeEstimator, MGazeNetGazeEstimator
from .logger import Log
from .misc import CameraRunningState, clip_patch, GazeInfo, DefaultConfig
from .ui import CameraPreviewerUI, CalibrationUI

# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

class GazeFollower:

    def __init__(self, camera: Camera = WebCamCamera(),
                 face_alignment: FaceAlignment = MediaPipeFaceAlignment(),
                 gaze_estimator: GazeEstimator = MGazeNetGazeEstimator(),
                 gaze_filter: Filter = HeuristicFilter(),
                 calibration: Calibration = SVRCalibration(),
                 config: DefaultConfig = DefaultConfig()):
        """
        Initializes the main components of the eye-tracking system.

        Parameters:
        camera (Camera): The camera object, default is WebCamCamera.
        face_alignment (FaceAlignment): The face alignment module, default is MediaPipeFaceAlignment.
        gaze_estimator (GazeEstimator): The gaze estimation module, default is GazeEstimator.
        gaze_filter (Filter): The gaze filter for smoothing estimation results, default is HeuristicFilter.
        calibration (Calibration): The gaze calibration module, default is SVRCalibration.
        """
        self._create_session("my_session")

        # eye tracking components
        self.camera: Camera = camera
        self.face_alignment: FaceAlignment = face_alignment
        self.gaze_estimator: GazeEstimator = gaze_estimator
        self.gaze_filter: Filter = gaze_filter
        self.calibration: Calibration = calibration

        # default config
        self.config: DefaultConfig = config

        # set the camera to call process_frame method when a new image is captured

        self.camera.set_on_image_callback(self.process_frame)

        # lock for synchronizing access to shared resources among threads
        self.subscriber_lock = threading.Lock()

        # list to hold subscribers for the sampling events
        self.subscribers = []

        # trigger variable
        self._trigger = 0

        # screen size
        self.screen_size = config.screen_size

        # ui instance
        self.calibration_ui = None
        self.camera_previewer_ui = None
        self._calibration_controller: CalibrationController = CalibrationController(config.cali_mode,
                                                                                    config.camera_position,
                                                                                    self.screen_size,
                                                                                    config.screen_physical_size,
                                                                                    self.config.eye_blink_threshold)

        self._gaze_info = None

    def get_gaze_info(self):
        """
        Returns the gaze information of the eye-tracking system or none.
        """
        return self._gaze_info

    def add_subscriber(self, subscriber_fuc, args=(), kwargs=None):
        """
        Adds a subscriber function to the list of subscribers.

        :param subscriber_fuc: callable
            The function that will be called when an event occurs.

        :param args: tuple (optional)
            A tuple of positional arguments to be passed to the subscriber function
            when it is called.

        :param kwargs: dict (optional)
            A dictionary of keyword arguments to be passed to the subscriber function
            when it is called. If None, an empty dictionary will be used.

        :return: None
            This method does not return any value. It modifies the internal state
            by adding the subscriber to the list.
        """
        if kwargs is None:
            kwargs = {}
        with self.subscriber_lock:
            self.subscribers.append((subscriber_fuc, args, kwargs))

    def remove_subscriber(self, subscriber_fuc):
        """
        Removes a subscriber function from the list of subscribers.

        :param subscriber_fuc: callable
            The function to be removed from the subscriber list.

        :return: None
            This method does not return any value. It modifies the internal state
            by removing the specified subscriber from the list.
        """
        with self.subscriber_lock:
            for subscribe in self.subscribers:
                if subscriber_fuc in subscribe:
                    self.subscribers.remove(subscribe)

    def start_sampling(self):
        self.camera.start_sampling()
        self.add_subscriber(self._write_sample)

    def stop_sampling(self):
        self.camera.stop_sampling()
        self.remove_subscriber(self._write_sample)

    def save_data(self, path):
        self._tmpSampleDataSteam.close()
        # Open the file and write the sampled data
        shutil.copyfile(str(self._tmpSampleDataPath), path)
        Log.i("save data to {}".format(path))

    def send_trigger(self, trigger_num: int):
        self._trigger = trigger_num

    @staticmethod
    def backend_name(screen):
        """
        Determines the backend type based on the given screen instance.

        Parameters:
            screen: The screen instance, which can be either a pygame Surface or a PsychoPy Window.

        Returns:
            str: The backend name, either 'pygame' or 'psychopy'.

        Raises:
            Exception: If the screen is None or not a valid pygame or PsychoPy window instance.
        """
        from pygame import Surface
        screen_type = ""
        if isinstance(screen, Surface):
            screen_type = 'pygame'
        else:
            from psychopy.visual import Window
            if isinstance(screen, Window):
                screen_type = 'psychopy'

        if screen_type == "":
            raise Exception("Screen cannot be None. Please pass pygame window or psychopy window instance")
        return screen_type

    def preview(self, win=None):
        """
        Starts the camera preview and displays images.

        Parameters:
            win (None|pygame.Surface|psychopy.visual.Window): The pygame window or psychopy window.
                If you pass None, the default pygame window will be used.
        """
        if win is None:
            pygame.init()
            win = pygame.display.set_mode(self.screen_size.tolist(), pygame.FULLSCREEN)
            pygame.display.set_caption("Calibration UI")
            backend_name = "pygame"
        else:
            backend_name = self.backend_name(win)

        self.camera_previewer_ui = CameraPreviewerUI(win=win, backend_name=backend_name)
        self.camera.start_previewing()
        self.camera_previewer_ui.draw()
        self.camera.stop_previewing()

    def calibrate(self, win=None):
        """
        Initiates a calibration session for gaze estimation and optionally validates
        the calibration with provided validation points.

        Parameters:
            win (None|pygame.Surface|psychopy.visual.Window): The pygame window or psychopy window.
                If you pass None, the default pygame window will be used.
        """

        if win is None:
            pygame.init()
            win = pygame.display.set_mode(self.screen_size.tolist(), pygame.FULLSCREEN)
            pygame.display.set_caption("Calibration UI")
            backend_name = "pygame"
        else:
            backend_name = self.backend_name(win)

        self.calibration_ui = CalibrationUI(win=win, backend_name=backend_name)
        while 1:
            # new session
            self._new_calibration_session()
            self._calibration_controller.new_session()
            self.calibration_ui.new_session()
            # draw guidance
            self.calibration_ui.draw_guidance(self.config.cali_instruction)
            # start calibration
            self.camera.start_calibrating()
            # draw calibration points
            self.calibration_ui.draw(self._calibration_controller)
            user_response = self.calibration_ui.draw_cali_result(self._calibration_controller,
                                                                 self.config.model_fit_instruction)
            self.camera.stop_calibrating()

            if user_response:
                break

    def _new_calibration_session(self):
        """
        Initializes a new calibration session by resetting necessary data collections.

        This method clears the ground truth points, gaze feature collection,
        and point ID collection to prepare for a fresh calibration session.

        :return: None
            This method does not return any value. It modifies the internal state
            of the object by resetting the calibration data collections.
        """
        self.ground_truth_points = []
        self.gaze_feature_collection = []
        self.point_id_collection = []

    def _drop_last_three_frames(self):
        """
       Drops the last three occurrences of each unique point ID from the collections
       of gaze features, ground truth points, and point IDs.

       This method is useful for ensuring that only the most relevant data points
       are retained for calibration, particularly in cases where certain points
       may be over-represented.

       :return: None
           This method does not return any value. It modifies the internal state
           of the object by filtering the gaze feature collection, ground truth points,
           and point ID collection.
       """
        # Convert point_id_collection to a NumPy array
        point_ids = np.array(self.point_id_collection)

        # Initialize a mask with True values
        mask = np.ones(len(point_ids), dtype=bool)

        # Get unique point indices and their corresponding indices in the array
        unique_ids, counts = np.unique(point_ids, return_counts=True)

        # Iterate over each unique point index
        for point_id in unique_ids:
            # Get all indices of the current point_id
            indices = np.where(point_ids == point_id)[0]

            # If there are more than three occurrences, mark the last three indices as False
            if len(indices) > 3:
                mask[indices[-3:]] = False

        # Apply the mask to filter collections
        self.gaze_feature_collection = np.array(self.gaze_feature_collection)[mask]
        self.ground_truth_points = np.array(self.ground_truth_points)[mask]
        self.point_id_collection = point_ids[mask]

    def process_frame(self, state, timestamp, frame):
        """
        Processes the received frame.

        :param state: camera state
        :param timestamp: long, the timestamp when the frame was captured.
        :param frame: The captured image frame (np.ndarray).
        :return: None
        """
        if state == CameraRunningState.PREVIEWING:
            # face detection
            face_info = self.face_alignment.detect(timestamp, frame)
            face_patch, left_eye_patch, right_eye_patch = None, None, None

            if face_info.status and face_info.can_gaze_estimation:
                # clip face and eye patches
                face_patch = clip_patch(frame, face_info.face_rect)
                left_eye_patch = clip_patch(frame, face_info.left_rect)
                right_eye_patch = clip_patch(frame, face_info.right_rect)

                # draw left eye rectangle on the image frame
                x, y, w, h = face_info.left_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              color=(255, 0, 0), thickness=2)
                # draw right eye rectangle on the image frame
                x, y, w, h = face_info.right_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              color=(0, 0, 255), thickness=2)
                # draw eye rectangles on the face patch
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
            # send the image frame to the screen
            self.camera_previewer_ui.update_images(frame, face_patch, left_eye_patch, right_eye_patch)
            self.camera_previewer_ui.face_info_dict = face_info.to_dict()

        elif state == CameraRunningState.SAMPLING:
            # detect face
            face_info = self.face_alignment.detect(timestamp, frame)
            # detect gaze
            gaze_info = self.gaze_estimator.detect(frame, face_info)

            if gaze_info.status and gaze_info.features is not None:
                calibrated, calibrated_gaze_coordinates = self.calibration.predict(gaze_info.features,
                                                                                   gaze_info.raw_gaze_coordinates)

                if not calibrated:
                    Log.e("No calibration model is available, please perform calibration")
                    raise Exception("No calibration model is available, please perform calibration")
                else:
                    # scale to pixel
                    calibrated_gaze_coordinates = self._calibration_controller.convert_to_pixel(
                        calibrated_gaze_coordinates)
                    gaze_info.calibrated_gaze_coordinates = calibrated_gaze_coordinates

                # do filter
                filtered_gaze_coordinates = self.gaze_filter.filter_values(calibrated_gaze_coordinates)
                gaze_info.filtered_gaze_coordinates = filtered_gaze_coordinates
            self.dispatch_face_gaze_info(face_info, gaze_info)

        elif state == CameraRunningState.CALIBRATING:
            if self._calibration_controller.calibrating:
                face_info = self.face_alignment.detect(timestamp, frame)
                gaze_info = self.gaze_estimator.detect(frame, face_info)
                self._calibration_controller.add_cali_feature(gaze_info=gaze_info, face_info=face_info)
            elif not self._calibration_controller.cali_model_fitted:
                features = np.array(self._calibration_controller.feature_vectors)
                n_point, n_frame, feature_dim = features.shape
                print("feature shape: ", features.shape)

                features = np.reshape(features, (n_point * n_frame, feature_dim))

                labels = np.array(self._calibration_controller.label_vectors)
                n_point, n_frame, label_dim = labels.shape
                labels = np.reshape(labels, (n_point * n_frame, label_dim))
                ids = np.array(self._calibration_controller.feature_ids)
                n_point, n_frame, ids_dim = ids.shape
                point_ids = np.reshape(ids, (n_point * n_frame, ids_dim))

                # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # features_path = f"features_{timestamp}.npz"
                # labels_path = f"labels_{timestamp}.npz"
                # point_ids_path = f"point_ids_{timestamp}.npz"
                #
                # np.savez_compressed(features_path, data=features)
                # np.savez_compressed(labels_path, data=labels)
                # np.savez_compressed(point_ids_path, data=point_ids)
                #
                # print(f"Features saved to: {features_path}")
                # print(f"Labels saved to: {labels_path}")
                # print(f"Point IDs saved to: {point_ids_path}")

                has_calibrated, mean_euclidean_error, predictions \
                    = self.calibration.calibrate(features, labels, point_ids)

                self._calibration_controller.set_calibration_results(has_calibrated, mean_euclidean_error, labels,
                                                                     predictions)
                self._calibration_controller.cali_model_fitted = True

        elif state == CameraRunningState.CLOSING:
            # Do nothing
            pass

    # def pixel_to_cm(self, camera_position, coordination_pixel):
    def dispatch_face_gaze_info(self, face_info, gaze_info):
        """
        Dispatches face and gaze information to all subscribed functions.

        This method iterates through the list of subscriber functions and calls
        each one with the provided face and gaze information, along with any
        additional positional and keyword arguments specified during subscription.

        :param face_info: any
            The information related to the detected face that will be passed to
            the subscriber functions.

        :param gaze_info: any
            The gaze information associated with the detected face that will be
            passed to the subscriber functions.

        :return: None
            This method does not return any value. It modifies the state of
            subscriber functions by invoking them with the given parameters.
        """
        with self.subscriber_lock:
            for (subscriber_func, args, kwargs) in self.subscribers:
                subscriber_func(face_info, gaze_info, *args, **kwargs)

    def release(self):
        """

        :return: None
        """
        self.camera.close()
        self.camera.set_on_image_callback(None)
        self.camera.release()
        self.gaze_filter.release()
        self.gaze_estimator.release()
        self.face_alignment.release()
        self.calibration.release()

    @staticmethod
    def _gaze_info_2_string(gaze_info: GazeInfo, trigger):
        """
        timestamp,gaze_position_x,gaze_position_y,left_eye_openness,
        right_eye_openness,tracking_status,status,event,trigger\n
        """

        ret_str = (f"{gaze_info.timestamp},{gaze_info.raw_gaze_coordinates[0]},{gaze_info.raw_gaze_coordinates[1]},"
                   f"{gaze_info.calibrated_gaze_coordinates[0]},{gaze_info.calibrated_gaze_coordinates[1]},"
                   f"{gaze_info.filtered_gaze_coordinates[0]},{gaze_info.filtered_gaze_coordinates[1]},"
                   f"{gaze_info.left_openness},{gaze_info.right_openness},{gaze_info.tracking_state.value},"
                   f"{int(gaze_info.status)},{int(gaze_info.event.value)},{trigger}\n")
        return ret_str

    def _write_sample(self, face_info, gaze_info):
        """
        Write the sample data.
        """
        _ = face_info
        self._gaze_info = gaze_info
        tmp_trigger = 0
        if isinstance(self._trigger, int):
            if self._trigger != 0:
                tmp_trigger = self._trigger
                self._trigger = 0
        else:
            Log.e("Trigger must be an integer, but you gave {}".format(type(self._trigger)))
            raise Exception("Trigger must be an integer, but you gave {}".format(type(self._trigger)))

        self._tmpSampleDataSteam.write(self._gaze_info_2_string(gaze_info, tmp_trigger))
        self._tmpSampleDataSteam.flush()

    def _create_session(self, session_name: str):
        """
        Create a new session with the given name.
        Sets up directories, log files, and logger for the session.

        Args:
            session_name: Name of the session. For defining logging files and temporary files.
        """
        # logging.info(f"Creating session: {session_name}")

        available_session = bool(re.fullmatch(r'^[a-zA-Z0-9_]+$', session_name))
        if not available_session:
            raise Exception(
                f"Session name '{session_name}' is invalid. Ensure it includes only letters, digits, "
                f"or underscores without any special characters.")

        self._session_name = session_name
        self._workSpace = pathlib.Path.home().joinpath("GazeFollower")

        if not self._workSpace.exists():
            self._workSpace.mkdir()

        # Set up the log directory
        _logDir = self._workSpace.joinpath("log")
        if not _logDir.exists():
            _logDir.mkdir()

        # Set up the temporary directory
        _tmpDir = self._workSpace.joinpath("tmp")
        if not _tmpDir.exists():
            _tmpDir.mkdir()

        _currentTime = datetime.datetime.now()
        _timeString = _currentTime.strftime("%Y_%m_%d_%H_%M_%S")

        # Set up the log file
        _logFile = _logDir.joinpath(f"log_{session_name}_{_timeString}.txt")
        Log.init(_logFile)

        self._tmpSampleDataPath = _tmpDir.joinpath(f"em_{session_name}_{_timeString}.csv")
        self._tmpSampleDataSteam = self._tmpSampleDataPath.open("w", encoding="utf-8")
        self._tmpSampleDataSteam.write(
            "timestamp,raw_gaze_position_x,raw_gaze_position_y,"
            "calibrated_gaze_position_x,calibrated_gaze_position_y,"
            "filtered_gaze_position_x,filtered_gaze_position_y,left_eye_openness," +
            "right_eye_openness,tracking_status,status,event,trigger\n"
        )

    def fine_tuning(self):
        Log.w("Finetuning is awaiting author's updating. Coming soon.")
        raise NotImplementedError("Finetuning feature is under development.")
