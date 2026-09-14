# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import math
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .FaceAlignment import FaceAlignment
from ..misc import FaceInfo


class _LandmarkPoint:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class MediaPipeFaceAlignment(FaceAlignment):
    def __init__(self, model_path: str = None):
        """
        Initializes the MediaPipeFaceAlignment object.
        Supports both modern MediaPipe Tasks API (FaceLandmarker) and legacy solutions API.
        """
        super().__init__()
        self.static_image_mode = False
        self.max_num_faces = 1
        self.min_detection_confidence = 0.1
        self.min_tracking_confidence = 0.1

        self.use_tasks_api = False
        self.landmarker = None
        self.face_mesh = None

        # 1. Try modern MediaPipe Tasks API
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            resolved_model_path = model_path
            if resolved_model_path is None:
                default_path = Path(__file__).parent.parent / "res" / "model_weights" / "face_landmarker.task"
                if not default_path.exists():
                    default_path.parent.mkdir(parents=True, exist_ok=True)
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    try:
                        urllib.request.urlretrieve(url, str(default_path))
                    except Exception:
                        pass
                if default_path.exists():
                    resolved_model_path = str(default_path)

            if resolved_model_path and os.path.exists(resolved_model_path):
                base_options = python.BaseOptions(model_asset_path=resolved_model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=self.max_num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_face_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.use_tasks_api = True
        except Exception:
            self.use_tasks_api = False

        # 2. Fallback to legacy solutions API if Tasks API is not available
        if not self.use_tasks_api:
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    self.static_image_mode,
                    self.max_num_faces,
                    True,
                    self.min_detection_confidence,
                    self.min_tracking_confidence
                )
            else:
                raise RuntimeError(
                    "Neither MediaPipe Tasks FaceLandmarker (with face_landmarker.task) "
                    "nor legacy mediapipe.solutions.face_mesh could be initialized."
                )

        # Define vertex indices for lip and eye regions
        self.lip_vertices_index = [61, 91, 14, 178, 402, 324, 95]
        self.left_vertices_index = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
        self.right_vertices_index = [362, 388, 384, 385, 386, 387, 388, 466, 263, 249, 380, 373, 374, 380, 381, 382,
                                     362]

    @staticmethod
    def calculate_polygon_area(vertices) -> float:
        """
        Calculates the area of a polygon defined by its vertices.

        :param vertices: A numpy array of shape (n, 2) where n is the number of vertices.
                         Each vertex is defined by its (x, y) coordinates.
        :return: The area of the polygon.

        Example usage:
        polygon_vertices = [(0, 0), (0, 5), (5, 5), (5, 0)]
        area = calculate_polygon_area(polygon_vertices)
        """
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def detect(self, timestamp, image) -> FaceInfo:
        """
        Detects face landmarks and returns relevant face information.

        This method processes the input image to detect facial landmarks,
        computes the face bounding box, and extracts eye openness metrics.

        :param timestamp: A timestamp indicating when the image was captured.
        :param image: The image in which to detect faces, expected in BGR format.
        :return: An instance of FaceInfo containing details about the detected face,
                 including status, bounding box, eye openness, and landmarks.
        """
        # s_time = time.time()
        face_info = FaceInfo()
        face_info.timestamp = timestamp
        image_height, image_width, _ = image.shape
        face_info.img_w = image_width
        face_info.img_h = image_height

        if self.use_tasks_api:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = self.landmarker.detect(mp_image)
            if not result.face_landmarks or len(result.face_landmarks) == 0:
                face_info.status = False
                face_info.can_gaze_estimation = False
                return face_info

            raw_landmarks = result.face_landmarks[0]
            scaled_landmarks = [
                _LandmarkPoint(
                    float(np.round(lm.x * image_width)),
                    float(np.round(lm.y * image_height)),
                    float(np.round(lm.z * image_width))
                )
                for lm in raw_landmarks
            ]
        else:
            outputs = self.face_mesh.process(image)
            _multi_face_landmarks = outputs.multi_face_landmarks
            if not _multi_face_landmarks:
                face_info.status = False
                face_info.can_gaze_estimation = False
                return face_info

            raw_landmarks = _multi_face_landmarks[0].landmark
            scaled_landmarks = [
                _LandmarkPoint(
                    float(np.round(lm.x * image_width)),
                    float(np.round(lm.y * image_height)),
                    float(np.round(lm.z * image_width))
                )
                for lm in raw_landmarks
            ]

        face_info.status = True
        _face_mesh = np.array([[p.x, p.y, p.z] for p in scaled_landmarks], dtype=np.int16)

        # Computing face box from the face mesh
        max_x = np.max(_face_mesh[:, 0])
        min_x = np.min(_face_mesh[:, 0])
        max_y = np.max(_face_mesh[:, 1])
        min_y = np.min(_face_mesh[:, 1])

        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > image_width:
            max_x = image_width
        if max_y > image_height:
            max_y = image_height

        # Filtering the instances near the screen edge
        lip_position_y = np.mean(_face_mesh[:, 1][self.lip_vertices_index])

        if lip_position_y >= image_height:
            face_info.status = True
            face_info.can_gaze_estimation = False
            return face_info

        face_height = math.fabs(min_y - max_y)
        face_width = math.fabs(min_x - max_x)

        delta = (face_width - face_height) / 4

        left_top_face_point = [min_x + delta, min_y - delta]
        right_bottom_face_point = [max_x - delta, max_y + delta]

        if right_bottom_face_point[1] > image_height:
            right_bottom_face_point[1] = image_height - 1
        if left_top_face_point[1] < 0:
            left_top_face_point[1] = 0
        if left_top_face_point[0] < 0:
            left_top_face_point[0] = 0
        if right_bottom_face_point[0] > image_width:
            right_bottom_face_point[0] = image_width - 1

        # face box to integer format
        left_top_face_point[0] = int(left_top_face_point[0])
        left_top_face_point[1] = int(left_top_face_point[1])
        right_bottom_face_point[0] = int(right_bottom_face_point[0])
        right_bottom_face_point[1] = int(right_bottom_face_point[1])

        # Get the mirrored eye box,
        # which means the left eye box is the right eye box in non-mirror.

        # split the distance between the inner eye corners into 100 units
        scale = math.fabs(scaled_landmarks[362].x - scaled_landmarks[133].x) / 100
        x_padding = 20  # padding the eye corners
        y_0 = 0.6  # less area for upper portion of the eye area
        y_1 = 0.4  # more area for the lower portion of the eye area

        # get the X-coords for the left eye
        eye_left_xs = [scaled_landmarks[33].x - x_padding * scale, scaled_landmarks[133].x + x_padding * scale]
        # height of the eyebox is 0.75 of the width of the eyebox
        eye_left_height = math.fabs(eye_left_xs[0] - eye_left_xs[1]) * 0.75
        # get the Y-coords for the left eye
        eye_left_y = (scaled_landmarks[33].y + scaled_landmarks[133].y) / 2.0
        eye_left_ys = [eye_left_y - eye_left_height * y_0, eye_left_y + eye_left_height * y_1]

        # get the X-coords for the right eye
        eye_right_xs = [scaled_landmarks[362].x - x_padding * scale, scaled_landmarks[263].x + x_padding * scale]
        # height of the eyebox is 0.75 of the width of the eyebox
        eye_right_height = math.fabs(eye_right_xs[0] - eye_right_xs[1]) * 0.75
        # get the Y-coords for the right eye
        eye_right_y = (scaled_landmarks[362].y + scaled_landmarks[263].y) / 2.0
        eye_right_ys = [eye_right_y - eye_right_height * y_0, eye_right_y + eye_right_height * y_1]

        # get the eye coords for json files
        left_top_eye_left_point = [int(eye_left_xs[0]), int(eye_left_ys[0])]
        right_bottom_eye_left_point = [int(eye_left_xs[1]), int(eye_left_ys[1])]
        left_top_eye_right_point = [int(eye_right_xs[0]), int(eye_right_ys[0])]
        right_bottom_eye_right_point = [int(eye_right_xs[1]), int(eye_right_ys[1])]

        # Filter eyes out of image
        if (left_top_eye_left_point[0] <= 0) or (left_top_eye_left_point[1] <= 0) \
                or (right_bottom_eye_left_point[0] >= image_width) or (right_bottom_eye_left_point[1] >= image_height):
            face_info.can_gaze_estimation = False
            return face_info

        if (left_top_eye_right_point[0] <= 0) or (left_top_eye_right_point[1] <= 0) \
                or (right_bottom_eye_right_point[0] >= image_width) or (
                right_bottom_eye_right_point[1] >= image_height):
            face_info.can_gaze_estimation = False
            return face_info

        # left-eye AREA
        left_vertices = _face_mesh[:, :2][self.left_vertices_index]
        left_eye_area = self.calculate_polygon_area(left_vertices)
        left_eye_ear = np.abs(scaled_landmarks[33].y - scaled_landmarks[133].y) / max(
            1e-6, np.abs(scaled_landmarks[33].x - scaled_landmarks[133].x)
        )

        # right-eye AREA
        right_vertices = _face_mesh[:, :2][self.right_vertices_index]
        right_eye_area = self.calculate_polygon_area(right_vertices)
        right_eye_ear = np.abs(scaled_landmarks[362].y - scaled_landmarks[263].y) / max(
            1e-6, np.abs(scaled_landmarks[362].x - scaled_landmarks[263].x)
        )

        face_info.face_rect = [  # x, y, w, h
            left_top_face_point[0],
            left_top_face_point[1],
            right_bottom_face_point[0] - left_top_face_point[0],
            right_bottom_face_point[1] - left_top_face_point[1],
        ]
        # left box (non-mirror)
        face_info.left_rect = [  # x, y, w, h
            left_top_eye_left_point[0],
            left_top_eye_left_point[1],
            right_bottom_eye_left_point[0] - left_top_eye_left_point[0],
            right_bottom_eye_left_point[1] - left_top_eye_left_point[1],
        ]
        # right box (non-mirror)
        face_info.right_rect = [  # x, y, w, h
            left_top_eye_right_point[0],
            left_top_eye_right_point[1],
            right_bottom_eye_right_point[0] - left_top_eye_right_point[0],
            right_bottom_eye_right_point[1] - left_top_eye_right_point[1],
        ]

        face_info.face_landmarks = _face_mesh
        face_info.left_eye_openness = left_eye_area
        face_info.right_eye_openness = right_eye_area
        face_info.can_gaze_estimation = True
        # print(f"Time cost in {time.time() - s_time} s")
        return face_info

    def release(self):
        if self.landmarker is not None and hasattr(self.landmarker, 'close'):
            self.landmarker.close()
        if self.face_mesh is not None and hasattr(self.face_mesh, 'close'):
            self.face_mesh.close()
