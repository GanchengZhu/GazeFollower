# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from ..misc import GazeInfo, FaceInfo


class GazeEstimator:
    """
    A class to estimate gaze direction based on facial information and an input image.

    This class provides methods to detect gaze direction from a given image using
    provided face information.
    """

    def __init__(self):
        """
        Initialize the GazeEstimator.

        This constructor can be expanded to initialize any required parameters or models
        for gaze estimation.
        """
        pass

    def detect(self, image, face_info: FaceInfo) -> GazeInfo:
        """
        Detect gaze direction from the given image and facial information.

        :param image: The input image from which to estimate gaze direction.
        :param face_info: An instance of FaceInfo containing facial landmarks and status.
        :return: An instance of GazeInfo representing the estimated gaze direction.
        """
        raise NotImplementedError("Subclasses must implement this method.")  # Implement gaze detection logic here

    def release(self):
        raise NotImplementedError("Subclasses must implement this method.")
