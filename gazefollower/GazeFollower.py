#!/usr/bin/bash
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import copy
import threading

import numpy as np
import pygame
from screeninfo import get_monitors

from .calibration import Calibration
from .calibration import SVRCalibration
from .camera import Camera
from .camera import WebCamCamera
from .face_alignment import FaceAlignment
from .face_alignment import MediaPipeFaceAlignment
from .filter import HeuristicFilter
from .filter.Filter import Filter
from .gaze_estimator import GazeEstimator
from .gaze_estimator import MGazeNetGazeEstimator
from .misc import CameraRunningState, clip_patch
from .ui import CameraPreviewerUI, CalibrationUI


class GazeFollower:

    def __init__(self, pipeline=None):
        """
        Initializes the UniTracker instance with a specified pipeline for gaze tracking.

        :param pipeline: dict (optional)
            A dictionary that specifies components for the gaze tracking pipeline.
            If None, a default pipeline will be created with the following components:
                - WebCamCamera for capturing images.
                - MediaPipeFaceAlignment for detecting and aligning faces.
                - MGazeNetGazeEstimator for estimating gaze direction.
                - HeuristicFilter for filtering gaze estimates.
                - SVRCalibration for calibrating the gaze estimation.
        """

        if pipeline is None:
            # Create a default pipeline with specified components
            default_pipeline = {
                "camera": WebCamCamera(),
                "face_alignment": MediaPipeFaceAlignment(),
                "gaze_estimator": MGazeNetGazeEstimator(),
                "filter": HeuristicFilter(look_ahead=2),
                "calibration": SVRCalibration(),
            }
            self.pipeline = default_pipeline
        else:
            self.pipeline = pipeline

        # Assign the pipeline components to instance variables
        self.camera: Camera = self.pipeline["camera"]
        self.face_alignment: FaceAlignment = self.pipeline["face_alignment"]
        self.gaze_estimator: GazeEstimator = self.pipeline["gaze_estimator"]
        self.filter: Filter = self.pipeline["filter"]
        self.calibration: Calibration = self.pipeline["calibration"]

        # Set the camera to call process_frame method when a new image is captured
        self.camera.set_on_image_callback(self.process_frame)

        # Define the camera position relative to the screen
        self.camera_position = (17.15, -0.68)

        # Get monitor details and set the screen size
        self.monitors = get_monitors()
        self.screen_size = np.array([self.monitors[0].width, self.monitors[0].height])

        # Define calibration points as percentage coordinates on the screen
        self.calibration_percentage_points = (
            (0.5, 0.5),  # Center
            (0.5, 0.08),  # Top center
            (0.08, 0.5),  # Left center
            (0.92, 0.5),  # Right center
            (0.5, 0.92),  # Bottom center
            (0.08, 0.08),  # Top left corner
            (0.92, 0.08),  # Top right corner
            (0.08, 0.92),  # Bottom left corner
            (0.92, 0.92),  # Bottom right corner
            (0.25, 0.25),  # Inner top left
            (0.75, 0.25),  # Inner top right
            (0.25, 0.75),  # Inner bottom left
            (0.75, 0.75),  # Inner bottom right
            (0.5, 0.5)  # Center
        )

        # Create a deep copy of calibration points for validation purposes
        self.validation_percentage_points = copy.deepcopy(self.calibration_percentage_points)

        # Initialize pygame for UI rendering
        pygame.init()

        # Create UI components for camera preview and calibration
        self.camera_previewer_ui = CameraPreviewerUI()
        self.calibration_ui = CalibrationUI(camera_position=self.camera_position,
                                            screen_size=self.screen_size,
                                            calibration_percentage_points=self.calibration_percentage_points,
                                            validation_percentage_points=self.validation_percentage_points)
        # Lock for synchronizing access to shared resources among threads
        self.subscriber_lock = threading.Lock()
        # List to hold subscribers for the sampling events
        self.subscribers = []

        self._trigger = 0

    def set_calibration_percentage_points(self, calibration_percentage_points):
        """
        Sets the calibration percentage points for gaze estimation calibration
        and updates the calibration UI with the new points.

        :param calibration_percentage_points: tuple
            A sequence of tuples representing percentage coordinates for calibration
            points on the screen. Each tuple should contain two float values
            representing the x and y coordinates, where (0, 0) is the top-left
            corner and (1, 1) is the bottom-right corner of the screen.

        :return: None
            This method does not return any value. It updates the internal state
            of the object and the UI.
        """
        self.calibration_percentage_points = calibration_percentage_points
        self.calibration_ui.set_calibration_points(calibration_percentage_points)

    def set_validation_percentage_points(self, validation_percentage_points):
        """
        Sets the validation percentage points for gaze estimation validation
        and updates the calibration UI with the new points.

        :param validation_percentage_points: tuple
            A sequence of tuples representing percentage coordinates for validation
            points on the screen. Each tuple should contain two float values
            representing the x and y coordinates, where (0, 0) is the top-left
            corner and (1, 1) is the bottom-right corner of the screen.

        :return: None
            This method does not return any value. It updates the internal state
            of the object and the UI.
        """
        self.validation_percentage_points = validation_percentage_points
        self.calibration_ui.set_validation_points(validation_percentage_points)

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
        # TODO
        pass

    def stop_sampling(self):
        # TODO
        pass

    def save_data(self, path):
        # TODO
        pass

    def send_trigger(self, trigger_num: int):
        self._trigger = trigger_num
        pass

    def preview(self):
        """
        Starts the camera preview and displays it in a Pygame window.

        This method initializes the Pygame display, sets the window title,
        and draws the camera preview UI on the screen. After displaying the
        preview, it stops the camera preview.

        :return: None
            This method does not return any value.
        """
        self.camera.start_previewing()
        screen = pygame.display.set_mode(
            (self.camera_previewer_ui.screen_width, self.camera_previewer_ui.screen_height))
        self.camera_previewer_ui.draw(screen)
        self.camera.stop_previewing()

    def calibrate(self, validation=True, validation_percentage_points=None):
        """
        Initiates a calibration session for gaze estimation and optionally validates
        the calibration with provided validation points.

        :param validation: bool (default=True)
            Indicates whether to perform validation after calibration. If True,
            the method will start a validation sampling session.

        :param validation_percentage_points: tuple (optional)
            A sequence of tuples representing percentage coordinates for validation
            points on the screen. If provided, these points will be used for
            validation after calibration.

        :return: None
            This method does not return any value. It modifies the internal state
            of the object based on calibration and validation results.
        """
        self._new_calibration_session()
        screen = pygame.display.set_mode(self.screen_size.tolist(), pygame.FULLSCREEN)
        pygame.display.set_caption("Calibration UI")
        self.camera.start_calibrating()
        cali_user_response = self.calibration_ui.draw(screen)
        self.camera.stop_calibrating()
        self._drop_last_three_frames()
        fitness = self.calibration.calibrate(self.gaze_feature_collection, self.ground_truth_points)

        if validation_percentage_points is not None:
            self.set_validation_percentage_points(validation_percentage_points)

        if validation:
            self.camera.start_sampling()
            self.add_subscriber(self.calibration_ui.validation_sample_subscriber)
            vali_user_response = self.calibration_ui.draw(screen, draw_type="validation")
            self.camera.stop_sampling()
            self.remove_subscriber(self.calibration_ui.validation_sample_subscriber)

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
        Processes the received frame by ensuring it is in RGB format and resizing it to 640x480.

        :param state: camera state
        :param timestamp: long, the timestamp when the frame was captured.
        :param frame: The captured image frame (np.ndarray).
        :return: None
        """
        if state == CameraRunningState.PREVIEWING:
            # face detection
            # face patch, eye patches
            face_info = self.face_alignment.detect(timestamp, frame)
            face_patch, left_eye_patch, right_eye_patch = None, None, None
            if face_info.status:
                face_patch = clip_patch(frame, face_info.face_rect)
                left_eye_patch = clip_patch(frame, face_info.left_rect)
                right_eye_patch = clip_patch(frame, face_info.right_rect)

            self.camera_previewer_ui.update_images(frame, face_patch, left_eye_patch, right_eye_patch)
            self.camera_previewer_ui.face_info_dict = face_info.to_dict()

        elif state == CameraRunningState.SAMPLING:
            face_info = self.face_alignment.detect(timestamp, frame)
            gaze_info = self.gaze_estimator.detect(frame, face_info)
            if gaze_info.status and gaze_info.features is not None:
                gaze_coordinates = self.calibration.predict(gaze_info.features, gaze_info.gaze_coordinates)
                # scale to pixel
                gaze_coordinates *= self.screen_size
                # do filter
                gaze_coordinates = self.filter.filter_values(gaze_coordinates)
                gaze_info.gaze_coordinates = gaze_coordinates

            self.dispatch_face_gaze_info(face_info, gaze_info)

        elif state == CameraRunningState.CALIBRATING:
            if self.calibration_ui.point_showing:
                face_info = self.face_alignment.detect(timestamp, frame)
                gaze_info = self.gaze_estimator.detect(frame, face_info)
                current_point_index = self.calibration_ui.current_point_index
                if current_point_index >= len(self.calibration_ui.calibration_points):
                    return
                # Collect data
                ground_truth_point = self.calibration_ui.calibration_points[current_point_index]
                # Normalized the ground truth point
                ground_truth_point = np.array(ground_truth_point) / self.screen_size
                # ground_truth_point_cm = self.pixel_to_cm(self.camera_position, ground_truth_point_pixel)
                if (self.calibration_ui.point_elapsed_time > 0.5 and gaze_info.features is not None
                        and gaze_info.left_openness > 20 and gaze_info.right_openness > 20):
                    self.gaze_feature_collection.append(gaze_info.features)
                    # self.estimated_gaze_points.append(gaze_info.gaze_coordinates)
                    self.ground_truth_points.append(ground_truth_point)
                    self.point_id_collection.append(current_point_index)

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
        for component in self.pipeline.items():
            component.release()