# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np


def _format_array(arr: np.array) -> str:
    """
    Format a NumPy array into a string representation.

    :param arr: A NumPy array containing numeric values.
    :return: A string representation of the array with each value formatted as a three-digit number.
    """
    return ', '.join('%03d' % num for num in arr)


class FaceInfo:
    """
    Class to encapsulate information related to facial detection and attributes.

    Attributes:
        timestamp (int): Timestamp in milliseconds.
        status (bool): Detection status (True if face detected, False otherwise).
        face_rect (np.array): Bounding box coordinates for the face.
        right_rect (np.array): Bounding box coordinates for the right eye.
        left_rect (np.array): Bounding box coordinates for the left eye.
        can_gaze_estimation (bool): Whether gaze estimation is enabled.
        face_landmarks (np.array): Array of facial landmarks (e.g., eyes, nose).
        img_w (int): Width of the image.
        img_h (int): Height of the image.
        left_eye_openness (float): Openness of the left eye.
        right_eye_openness (float): Openness of the right eye.
        left_eye_distance (float): Distance between the eyes.
        right_eye_distance (float): Distance from the center of the image to the eyes.
    """

    def __init__(self,
                 timestamp: int = 0,
                 status: bool = False,
                 face_rect: np.array = np.array([0.0, 0.0, 0.0, 0.0]),
                 right_rect: np.array = np.array([0.0, 0.0, 0.0, 0.0]),
                 left_rect: np.array = np.array([0.0, 0.0, 0.0, 0.0]),
                 can_gaze_estimation: bool = False,
                 face_landmarks: np.array = np.zeros(shape=(478, 3)),
                 img_w: int = 0, img_h: int = 0,
                 left_eye_openness: float = 0.0,
                 right_eye_openness: float = 0.0,
                 left_eye_distance: float = 0.0,
                 right_eye_distance: float = 0.0):
        """
        Initialize a FaceInfo object with provided attributes.

        :param timestamp: Timestamp of the frame in milliseconds.
        :param status: Detection status (True if face is detected).
        :param face_rect: Bounding box for the face.
        :param right_rect: Bounding box for the right eye.
        :param left_rect: Bounding box for the left eye.
        :param can_gaze_estimation: Whether gaze estimation is enabled.
        :param face_landmarks: Array containing the landmarks of the face.
        :param img_w: Width of the image.
        :param img_h: Height of the image.
        :param left_eye_openness: Openness of the left eye.
        :param right_eye_openness: Openness of the right eye.
        :param left_eye_distance: Distance between the eyes.
        :param right_eye_distance: Distance from the center of the image to the eyes.
        """
        self.timestamp = timestamp
        self.status = status
        self.face_rect = face_rect
        self.right_rect = right_rect
        self.left_rect = left_rect
        self.can_gaze_estimation = can_gaze_estimation
        self.face_landmarks = face_landmarks
        self.img_w = img_w
        self.img_h = img_h
        self.left_eye_openness = left_eye_openness
        self.right_eye_openness = right_eye_openness
        self.left_eye_distance = left_eye_distance
        self.right_eye_distance = right_eye_distance

    def __str__(self):
        """
        Return a string representation of the FaceInfo object.

        :return: A string representing the FaceInfo object.
        """
        return str(self.to_dict())

    def to_dict(self):
        """
        Convert the FaceInfo object to a dictionary format.

        :return: A dictionary containing relevant face detection information.
        """

        # Determine the status message based on detection results
        if not self.can_gaze_estimation and self.status:
            status_value = "Face Out of Boundary"
        elif self.status and self.can_gaze_estimation:
            status_value = "Face Detected"
        else:
            status_value = "Face Not Detected"

        # Return a dictionary containing all relevant information
        return {
            'Timestamp': self.timestamp // 1000,  # Convert timestamp to seconds
            'Status': status_value,
            'Face Rect': _format_array(self.face_rect),  # Format bounding box
            'Right Rect': _format_array(self.right_rect),  # Format right eye bounding box
            'Left Rect': _format_array(self.left_rect),  # Format left eye bounding box
            'Face Landmarks': str(self.face_landmarks.shape),  # Shape of landmarks array
            'Image Width': self.img_w,
            'Image Height': self.img_h,
            'Left Eye Openness': f'{self.left_eye_openness:.2f}',  # Format eye openness
            'Right Eye Openness': f'{self.right_eye_openness:.2f}',  # Format eye openness
            'Left Eye Distance': f'{self.left_eye_distance:.2f}',  # Format eye distance
            'Right Eye Distance': f'{self.right_eye_distance:.2f}',  # Format eye distance
            'Estimate Gaze': 'Enable' if self.can_gaze_estimation else 'Disable'  # Gaze estimation status
        }
