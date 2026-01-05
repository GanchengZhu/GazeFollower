# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np

from .Enumeration import EyeMovementEvent, TrackingState


class GazeInfo:
    """
    Class to hold gaze information obtained from gaze estimation.

    Attributes:
        features (np.array): The features extracted from the gaze estimation model.
        raw_gaze_coordinates (np.array): The x, y coordinates of the raw gaze point.
        calibrated_gaze_coordinates (np.array): The x, y coordinates of the calibrated gaze point.
        filtered_gaze_coordinates (np.array): The x, y coordinates of the filtered gaze point.
        left_openness (float): The openness of the left eye.
        right_openness (float): The openness of the right eye.
        event (EyeMovementEvent): The type of eye movement event.
        status (bool): Indicates whether the gaze estimation was successful.
        tracking_state (TrackingState): The current state of gaze tracking.
        timestamp (int): The timestamp when the gaze information was recorded.
    """

    def __init__(self,
                 features: np.array = None,
                 raw_gaze_coordinates: np.array = None,
                 calibrated_gaze_coordinates: np.array = None,
                 filtered_gaze_coordinates: np.array = None,
                 left_openness: float = 0.0,
                 right_openness: float = 0.0,
                 event: EyeMovementEvent = EyeMovementEvent.UNKNOWN,
                 status: bool = False,
                 tracking_state: TrackingState = TrackingState.FACE_MISSING,
                 timestamp: int = 0):
        """
        Initializes a GazeInfo instance.
        :param features: The features extracted from the gaze estimation model.
        :param raw_gaze_coordinates: The x, y coordinates of the gaze point from the gaze estimation model.
        :param calibrated_gaze_coordinates: The x, y coordinates of the gaze point from the calibration model.
        :param filtered_gaze_coordinates: The x, y coordinates of the gaze point from the filter.
        :param left_openness: The openness of the left eye.
        :param right_openness: The openness of the right eye.
        :param event: The type of eye movement event (default is EyeMovementEvent.UNKNOWN).
        :param status: Indicates whether the gaze estimation was successful (default is False).
        :param tracking_state: The current state of gaze tracking (default is TrackingState.FACE_MISSING).
        :param timestamp: The timestamp when the gaze information was recorded (default is 0).
        """
        self.features = features
        self.left_openness = left_openness
        self.right_openness = right_openness
        self.event = event
        self.status = status
        self.tracking_state = tracking_state
        self.timestamp = timestamp
        self.raw_gaze_coordinates = raw_gaze_coordinates
        self.calibrated_gaze_coordinates = calibrated_gaze_coordinates
        self.filtered_gaze_coordinates = filtered_gaze_coordinates

    def __str__(self):
        """
        Returns a string representation of the GazeInfo instance.

        :return: A string describing the GazeInfo instance.
        """
        return (
            f"GazeInfo(timestamp={self.timestamp}, status={self.status}, "
            f"raw_gaze_coordinates={self.raw_gaze_coordinates[:2]}, "
            f"cali_gaze_coordinates={self.calibrated_gaze_coordinates}, "
            f"filtered_gaze_coordinates={self.filtered_gaze_coordinates}, "
            f"left_openness={self.left_openness}, right_openness={self.right_openness}, "
            f"event={self.event}, tracking_state={self.tracking_state}, features={self.features})")
