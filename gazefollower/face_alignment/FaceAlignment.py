# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from ..misc import FaceInfo


class FaceAlignment:
    def __init__(self):
        """
        Initializes the FaceAlignment object.

        This constructor currently does not initialize any attributes
        or perform any actions, but it can be extended in the future
        to set up resources or parameters needed for face alignment.
        """
        pass

    def detect(self, timestamp, image) -> FaceInfo:
        """
        Detects face landmarks in the given image.

        This method processes the input image to detect facial features
        based on the provided timestamp. The detected features are
        encapsulated in a FaceInfo object.

        :param timestamp: A timestamp indicating when the image was captured.
                          This can be used for logging or processing purposes.
        :param image: The image in which to detect faces. This should be
                      a valid image format that the detection algorithm can process.

        :return: An instance of FaceInfo containing details about the
                 detected faces, such as their positions and landmarks.
                 If no faces are detected, the return value may indicate
                 that no information is available.
        """

        # Implementation of the detection logic will go here.
        raise NotImplementedError("Subclasses must implement this method.")

    def release(self):
        raise NotImplementedError("Subclasses must implement this method.")