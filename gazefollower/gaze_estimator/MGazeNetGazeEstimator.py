# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import pathlib

import MNN
import cv2
import numpy as np

from .GazeEstimator import GazeEstimator
from ..misc import FaceInfo, GazeInfo, TrackingState, clip_patch


class MGazeNetGazeEstimator(GazeEstimator):
    """
    A gaze estimation class that utilizes a pre-trained MNN model.

    This class is responsible for loading the model, preparing input data,
    and running inference to estimate gaze direction based on facial information.
    """

    def __init__(self, model_path: str = ""):
        """
        Initialize the MGazeNetGazeEstimator.

        Loads the model weights and sets up the interpreter and session for inference.
        """
        super().__init__()

        # Load model weights from the specified path
        if model_path == "":
            self.model_path = pathlib.Path(__file__).parent.parent / "res/model_weights/base.mnn"
        else:
            self.model_path = pathlib.Path(model_path).resolve()
        self.interpreter = MNN.Interpreter(str(self.model_path))

        # Create a session for running the model
        self.session = self.interpreter.createSession({'precision': 'normal',
                                                       'numThread': 6,
                                                       'backend': 0})

        # Get input tensors for face and eyes
        self.face_input_tensor = self.interpreter.getSessionInput(self.session, "face")
        self.left_input_tensor = self.interpreter.getSessionInput(self.session, "left")
        self.right_input_tensor = self.interpreter.getSessionInput(self.session, "right")
        self.rect_input_tensor = self.interpreter.getSessionInput(self.session, "rect")

        # Temporary output tensor for storing inference results
        self.tmp_output = MNN.Tensor((1, 258),
                                     MNN.Halide_Type_Float,
                                     np.zeros([1, 258]).astype(np.float32),
                                     MNN.Tensor_DimensionType_Tensorflow)

        # Define input dimensions for model
        self.face_input_format = (1, 224, 224, 3)
        self.eye_input_format = (1, 112, 112, 3)
        self.rect_input_format = (1, 12)

        # Size definitions for face and eye patches
        self.face_size = (224, 224)
        self.eye_size = (112, 112)
        self.rect_size = (1, 12)

    def detect(self, image, face_info: FaceInfo) -> GazeInfo:
        """
        Detect gaze direction from the input image and face information.

        :param image: The input image from which to estimate gaze direction.
        :param face_info: An instance of FaceInfo containing information about the detected face.
        :return: An instance of GazeInfo containing estimated gaze direction and related features.
        """
        gaze_info = GazeInfo()
        gaze_info.timestamp = face_info.timestamp

        # Check if face is detected
        if not face_info.status:
            gaze_info.tracking_state = TrackingState.FACE_MISSING
            return gaze_info

        # Check if gaze estimation is possible
        if not face_info.can_gaze_estimation:
            gaze_info.tracking_state = TrackingState.OUT_OF_BOUNDARIES
            return gaze_info

        # Extract face and eye rectangles
        f_x, f_y, f_w, f_h = face_info.face_rect
        le_x, le_y, le_w, le_h = face_info.left_rect
        re_x, re_y, re_w, re_h = face_info.right_rect

        # Clip patches for face and eyes
        face_patch = clip_patch(image, face_info.face_rect)
        left_eye_patch = clip_patch(image, face_info.left_rect)
        right_eye_patch = clip_patch(image, face_info.right_rect)

        # Check if any patch is None
        if face_patch is None or left_eye_patch is None or right_eye_patch is None:
            gaze_info.tracking_state = TrackingState.OUT_OF_BOUNDARIES
            return gaze_info

        # Prepare the rectangle input tensor with normalized values
        rect = (np.array([f_w, f_h, f_x, f_y,
                          le_w, le_h, le_x, le_y,
                          re_w, re_h, re_x, re_y], dtype=np.float32) /
                np.array(([face_info.img_w, face_info.img_h] * 6), dtype=np.float32))

        # Resize and normalize the patches for input
        face_patch = cv2.resize(face_patch, self.face_size).astype(np.float32) / 255.0
        left_patch = cv2.resize(left_eye_patch, self.eye_size).astype(np.float32) / 255.0
        right_patch = cv2.resize(right_eye_patch, self.eye_size)

        # Flip the right eye patch horizontally for correct gaze estimation
        right_patch = cv2.flip(right_patch, 1).astype(np.float32) / 255.0

        # Create input tensors for the model
        face_tmp = MNN.Tensor(self.face_input_format,
                              MNN.Halide_Type_Float,
                              face_patch,
                              MNN.Tensor_DimensionType_Tensorflow)

        left_tmp = MNN.Tensor(self.eye_input_format,
                              MNN.Halide_Type_Float,
                              left_patch,
                              MNN.Tensor_DimensionType_Tensorflow)

        right_tmp = MNN.Tensor(self.eye_input_format,
                               MNN.Halide_Type_Float,
                               right_patch,
                               MNN.Tensor_DimensionType_Tensorflow)

        rect_tmp = MNN.Tensor(self.rect_input_format,
                              MNN.Halide_Type_Float,
                              rect,
                              MNN.Tensor_DimensionType_Tensorflow)

        # Copy input tensors to the session
        self.face_input_tensor.copyFromHostTensor(face_tmp)
        self.left_input_tensor.copyFromHostTensor(left_tmp)
        self.right_input_tensor.copyFromHostTensor(right_tmp)
        self.rect_input_tensor.copyFromHostTensor(rect_tmp)

        # Run the inference
        self.interpreter.runSession(self.session)

        # Get the output tensor and copy the results
        output_tensor = self.interpreter.getSessionOutput(self.session, "output_0")
        output_tensor.copyToHostTensor(self.tmp_output)

        # Retrieve and process the results
        res = self.tmp_output.getNumpyData().copy()[0]
        gaze_info.features = res
        gaze_info.raw_gaze_coordinates = res[:2]  # Extract gaze coordinates
        gaze_info.status = True
        gaze_info.left_openness = face_info.left_eye_openness
        gaze_info.right_openness = face_info.right_eye_openness
        gaze_info.tracking_state = TrackingState.SUCCESS
        return gaze_info

    def release(self):
        pass
