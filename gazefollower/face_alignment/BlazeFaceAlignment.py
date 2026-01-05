import math
import os.path
import pathlib

import MNN
import cv2
import numpy as np

from gazefollower.face_alignment import FaceAlignment
from gazefollower.misc import FaceInfo


class BlazeFaceAlignment(FaceAlignment):
    def __init__(self, model_path="",
                 max_num_faces=1, min_confidence=0.5, min_iou_thresh=0.3):
        """
        Initializes the BlazeFaceAlignment object with MNN model.

        Args:
            model_path: Path to the MNN model file
            max_num_faces: Maximum number of faces to detect
            min_confidence: Minimum confidence threshold for detection
            min_iou_thresh: Minimum IoU threshold for NMS
        """
        super().__init__()
        self.max_num_faces = max_num_faces
        self.min_confidence = min_confidence
        self.min_iou_thresh = min_iou_thresh

        if model_path == "":
            self.model_path = pathlib.Path(__file__).parent.parent / "res/model_weights/blaze_face.mnn"
        else:
            self.model_path = pathlib.Path(model_path).resolve()
        # print(model_path)
        # Initialize MNN model
        try:
            config = {'precision': 'low', 'backend': 0, 'numThread': 4}
            rt = MNN.nn.create_runtime_manager((config,))
            self.face_detector = MNN.nn.load_module_from_file(
                str(self.model_path),
                ["image", "conf_threshold", "max_detections", "iou_threshold"],
                ["selectedBoxes"],
                runtime_manager=rt
            )
        except Exception as e:
            raise e

        # Define vertex indices for lip and eye regions (kept for compatibility)
        self.lip_vertices_index = [61, 91, 14, 178, 402, 324, 95]
        self.left_vertices_index = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
        self.right_vertices_index = [362, 388, 384, 385, 386, 387, 388, 466, 263, 249, 380, 373, 374, 380, 381, 382,
                                     362]

    @staticmethod
    def calculate_polygon_area(vertices) -> float:
        """
        Calculates the area of a polygon defined by its vertices.

        Args:
            vertices: A numpy array of shape (n, 2) where n is the number of vertices.
        Returns:
            The area of the polygon.
        """
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def _crop_img(img, X, Y, W, H):
        """
        Safely crops an image region.

        Args:
            img: Input image
            X, Y: Top-left coordinates
            W, H: Width and height
        Returns:
            Cropped image region
        """
        try:
            Y_lim, X_lim, _ = img.shape
            H = min(H, Y_lim)
            W = min(W, X_lim)
            X, Y, W, H = list(map(int, [X, Y, W, H]))
            X = max(X, 0)
            Y = max(Y, 0)
            if X + W > X_lim:
                X = X_lim - W
            if Y + H > Y_lim:
                Y = Y_lim - H
            return img[Y:(Y + H), X:(X + W)]
        except Exception as e:
            return None

    def _detect_landmarks(self, image):
        """
        Detects face landmarks using the MNN model.

        Args:
            image: Input image in RGB format
        Returns:
            Tuple of (image_shape, landmarks_array)
        """
        try:
            image_rows, image_cols, _ = image.shape

            # Preprocess image
            img128 = cv2.resize(image, (128, 128))
            img_np = np.asarray(img128, dtype=np.float32) / 255.0

            # Prepare MNN inputs
            image_var = MNN.expr.placeholder([1, 128, 128, 3], MNN.expr.NHWC)
            image_var.write(img_np)

            conf_var = MNN.expr.const([self.min_confidence], [1], MNN.expr.NCHW, MNN.expr.dtype.float)
            max_det_var = MNN.expr.const([float(self.max_num_faces)], [1], MNN.expr.NCHW, MNN.expr.dtype.float)
            iou_var = MNN.expr.const([self.min_iou_thresh], [1], MNN.expr.NCHW, MNN.expr.dtype.float)

            # Run inference
            inputs = [image_var, conf_var, max_det_var, iou_var]
            outputs = self.face_detector.onForward(inputs)

            # Extract results
            if len(outputs) > 0 and len(outputs[0]) > 0:
                boxes = outputs[0][0]
                if boxes.ndim != 1:
                    boxes = boxes[0]
                landmarks = boxes.read()
                return (image_cols, image_rows), landmarks
            else:
                return (image_cols, image_rows), None

        except Exception as e:
            return None, None

    def detect(self, timestamp, image) -> FaceInfo:
        """
        Detects face landmarks and returns relevant face information.

        Args:
            timestamp: A timestamp indicating when the image was captured.
            image: The image in which to detect faces, expected in BGR format.
        Returns:
            An instance of FaceInfo containing details about the detected face.
        """
        face_info = FaceInfo()
        face_info.timestamp = timestamp

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        face_info.img_w = image_width
        face_info.img_h = image_height

        # Detect landmarks
        (width, height), landmarks = self._detect_landmarks(image_rgb)

        if landmarks is None or len(landmarks) < 8:
            face_info.status = False
            face_info.can_gaze_estimation = False
            return face_info

        # Extract key points (normalized coordinates)
        top_y, top_x, bot_y, bot_x = landmarks[0], landmarks[1], landmarks[2], landmarks[3]
        ley_x, ley_y = landmarks[4], landmarks[5]
        rey_x, rey_y = landmarks[6], landmarks[7]

        # Convert to pixel coordinates
        x1, y1 = int(top_x * width), int(top_y * height)
        x2, y2 = int(bot_x * width), int(bot_y * height)
        ley_cx, ley_cy = int(ley_x * width), int(ley_y * height)
        rey_cx, rey_cy = int(rey_x * width), int(rey_y * height)

        # Check if bbox is valid
        if x2 - x1 < 5 or y2 - y1 < 5:
            face_info.status = False
            face_info.can_gaze_estimation = False
            return face_info

        face_width = x2 - x1
        face_height = y2 - y1

        # Calculate eye boxes (based on original algorithm)
        base_eye_size = max(face_width * 0.2, 25)
        eye_width = int(base_eye_size * 1.7)
        eye_height = int(base_eye_size * 1.1)

        left_eye_x1 = ley_cx - eye_width // 2
        left_eye_y1 = ley_cy - eye_height // 2 - 5
        right_eye_x1 = rey_cx - eye_width // 2
        right_eye_y1 = rey_cy - eye_height // 2 - 5

        # Create face landmarks array (for compatibility with existing code)
        # We'll create a simplified landmarks array with 12 points
        face_landmarks_simple = np.zeros((12, 2), dtype=np.float32)
        face_landmarks_simple[0] = [top_x * width, top_y * height]  # Top-left
        face_landmarks_simple[1] = [bot_x * width, bot_y * height]  # Bottom-right
        face_landmarks_simple[2] = [ley_x * width, ley_y * height]  # Left eye center
        face_landmarks_simple[3] = [rey_x * width, rey_y * height]  # Right eye center

        # Fill the rest with interpolated points for compatibility
        for i in range(4, 12):
            t = (i - 4) / 8.0
            face_landmarks_simple[i] = [
                (1 - t) * ley_x * width + t * rey_x * width,
                (1 - t) * ley_y * height + t * rey_y * height
            ]

        # Set face rectangles
        face_info.face_rect = [x1, y1, face_width, face_height]
        face_info.left_rect = [left_eye_x1, left_eye_y1, eye_width, eye_height]
        face_info.right_rect = [right_eye_x1, right_eye_y1, eye_width, eye_height]

        # # Create full face mesh (468 points) for compatibility
        # # We'll interpolate from the detected points
        # full_face_mesh = np.zeros((468, 3), dtype=np.int16)

        # Place detected points at appropriate indices
        # Note: This is a simplified approximation
        # Left eye region (indices 33, 133, etc.)
        # for idx in self.left_vertices_index:
        #     if idx < 468:
        #         # Interpolate between left eye center and face boundary
        #         full_face_mesh[idx] = [ley_cx, ley_cy, 0]
        #
        # # Right eye region
        # for idx in self.right_vertices_index:
        #     if idx < 468:
        #         full_face_mesh[idx] = [rey_cx, rey_cy, 0]
        #
        # # Fill other points with interpolated values
        # for i in range(468):
        #     if np.all(full_face_mesh[i] == 0):
        #         # Simple interpolation based on face position
        #         t = i / 468.0
        #         full_face_mesh[i] = [
        #             int(x1 + t * face_width),
        #             int(y1 + t * face_height),
        #             0
        #         ]
        #
        # face_info.face_landmarks = full_face_mesh

        # Calculate eye openness (simplified)
        # Using the distance between eye centers as a proxy
        eye_distance = math.sqrt((rey_cx - ley_cx) ** 2 + (rey_cy - ley_cy) ** 2)
        face_info.left_eye_openness = 100 # Simplified
        face_info.right_eye_openness = 100  # Simplified

        face_info.status = True
        face_info.can_gaze_estimation = True

        return face_info

    def release(self):
        """
        Release resources. MNN model doesn't need explicit release in this case.
        """
        pass
