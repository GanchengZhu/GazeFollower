# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import threading

from ..misc import CameraRunningState  # Importing the CameraRunningState enum for managing camera states


class Camera:
    """
    A base class representing a camera with different operational states.
    This class provides methods to start and stop different modes of operation
    such as sampling, previewing, and calibrating. It also manages the state
    of the camera and allows for setting a callback function to handle captured images.
    """

    def __init__(self):
        """
        Initialize the camera and the constant.
        """
        # Initializes the camera state to CLOSING and prepares a lock for callback management
        self.camera_running_state = CameraRunningState.CLOSING
        self.callback_func = None  # Stores the callback function and its parameters
        self.callback_args = ()
        self.callback_kwargs = {}
        self.callback_and_param_lock = threading.Lock()  # A lock to manage access to callback settings

    def start_sampling(self):
        """
        Starts the sampling mode if the camera is in the CLOSING state.
        Changes the state to SAMPLING and opens the camera. 
        Raises an error if the camera is already in an incompatible state.
        """
        if self.camera_running_state == CameraRunningState.CLOSING:
            self.camera_running_state = CameraRunningState.SAMPLING
            self.open()
        elif self.camera_running_state in {CameraRunningState.PREVIEWING, CameraRunningState.CALIBRATING}:
            raise RuntimeError("It is under previewing or calibration and cannot start sampling")
        else:
            print("Please do not call start_sampling repeatedly")

    def stop_sampling(self):
        """
        Stops the sampling mode if the camera is in the SAMPLING state.
        Changes the state to CLOSING and closes the camera.
        Raises an error if the camera is in an incompatible state.
        """
        if self.camera_running_state == CameraRunningState.SAMPLING:
            self.close()
            self.camera_running_state = CameraRunningState.CLOSING
        elif self.camera_running_state in {CameraRunningState.PREVIEWING, CameraRunningState.CALIBRATING}:
            raise RuntimeError("It is under previewing or calibration and cannot stop sampling")
        else:
            print("Please do not call stop_sampling repeatedly")

    def start_previewing(self):
        """
        Starts the previewing mode if the camera is in the CLOSING state.
        Changes the state to PREVIEWING and opens the camera.
        Raises an error if the camera is in an incompatible state.
        """
        if self.camera_running_state == CameraRunningState.CLOSING:
            self.camera_running_state = CameraRunningState.PREVIEWING
            self.open()
        elif self.camera_running_state in {CameraRunningState.SAMPLING, CameraRunningState.CALIBRATING}:
            raise RuntimeError("It is under sampling or calibrating and cannot start previewing")
        else:
            print("Please do not call start_previewing repeatedly")

    def stop_previewing(self):
        """
        Stops the previewing mode if the camera is in the PREVIEWING state.
        Changes the state to CLOSING and closes the camera.
        Raises an error if the camera is in an incompatible state.
        """
        if self.camera_running_state == CameraRunningState.PREVIEWING:
            self.camera_running_state = CameraRunningState.CLOSING
            self.close()
        elif self.camera_running_state in {CameraRunningState.SAMPLING, CameraRunningState.CALIBRATING}:
            raise RuntimeError("It is under sampling or calibrating and cannot stop previewing")
        else:
            print("It has already stopped previewing")

    def start_calibrating(self):
        """
        Starts the calibrating mode if the camera is in the CLOSING state.
        Changes the state to CALIBRATING and opens the camera.
        Raises an error if the camera is in an incompatible state.
        """
        if self.camera_running_state == CameraRunningState.CLOSING:
            self.camera_running_state = CameraRunningState.CALIBRATING
            self.open()
        elif self.camera_running_state in {CameraRunningState.SAMPLING, CameraRunningState.PREVIEWING}:
            raise RuntimeError("It is under sampling or previewing and cannot start calibrating")
        else:
            print("Please do not call start_calibrating repeatedly")

    def stop_calibrating(self):
        """
        Stops the calibrating mode if the camera is in the CALIBRATING state.
        Changes the state to CLOSING and closes the camera.
        Raises an error if the camera is in an incompatible state.
        """
        if self.camera_running_state == CameraRunningState.CALIBRATING:
            self.camera_running_state = CameraRunningState.CLOSING
            self.close()
        elif self.camera_running_state in {CameraRunningState.SAMPLING, CameraRunningState.PREVIEWING}:
            raise RuntimeError("It is under sampling or previewing and cannot stop calibrating")
        else:
            print("It has already stopped calibrating")

    def open(self):
        """
        Opens the camera. This method should be implemented in subclasses to define 
        specific behavior for opening the camera hardware.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def close(self):
        """
        Closes the camera. This method should be implemented in subclasses to define 
        specific behavior for closing the camera hardware.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def release(self):
        """
        Releases resources associated with the camera. This method should be implemented 
        in subclasses to define specific behavior for resource management.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def set_on_image_callback(self, func, args=None, kwargs=None):
        """
        Sets the callback function that will be triggered on capturing an image.
        The callback function must have the following args,
            timestamp and frame, which are the timestamp when the image was
            captured and the captured image frame (np.ndarray).

        Args:
            func: The callback function to be executed.
            args: Positional arguments for the callback function.
            kwargs: Keyword arguments for the callback function.
        The function and its parameters are stored and protected by a threading lock.
        """
        with self.callback_and_param_lock:
            if args is None:
                self.callback_args = ()
            else:
                self.callback_args = args

            if kwargs is None:
                self.callback_kwargs = {}
            else:
                self.callback_kwargs = kwargs

            if func is not None and not callable(func):
                raise TypeError("func must be callable or None")
            self.callback_func = func
