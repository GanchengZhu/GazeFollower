# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from enum import Enum


class TrackingState(Enum):
    """
    Enumeration for the tracking state of the gaze estimation process.

    Attributes:
        FACE_MISSING: Indicates that the face is not detected.
        SUCCESS: Indicates successful gaze tracking.
        OUT_OF_BOUNDARIES: Indicates that gaze tracking is out of valid boundaries.
        FAILURE: Indicates a failure in gaze tracking.
    """
    FACE_MISSING = 0
    SUCCESS = 1
    OUT_OF_BOUNDARIES = 2
    FAILURE = 3


class EyeMovementEvent(Enum):
    """
    Enumeration for different types of eye movement events.

    Attributes:
        SACCADE: Represents a rapid movement of the eye between fixation points.
        FIXATION: Represents a stable point of gaze.
        UNKNOWN: Indicates an unrecognized type of eye movement.
    """
    SACCADE = 2
    FIXATION = 1
    UNKNOWN = 0


class CameraRunningState(Enum):
    """
    Enumeration for the various states of the camera during operation.

    Attributes:
        PREVIEWING: Indicates that the camera is in preview mode.
        SAMPLING: Indicates that the camera is sampling data.
        CALIBRATING: Indicates that the camera is in calibration mode.
        CLOSING: Indicates that the camera is closing.
    """
    PREVIEWING = 0
    SAMPLING = 1
    CALIBRATING = 2
    CLOSING = 3
