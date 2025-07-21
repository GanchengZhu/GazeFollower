# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import math

import mediapipe as mp
import numpy as np

from .FaceAlignment import FaceAlignment
from ..misc import FaceInfo


class MediaPipeFaceAlignment(FaceAlignment):
    def __init__(self):
        """
        Initializes the MediaPipeFaceAlignment object.

        This constructor sets up the MediaPipe face mesh parameters, including
        detection and tracking confidence levels, and initializes the facial
        landmarks' configuration.
        """
        super().__init__()
        self.static_image_mode = False
        self.max_num_faces = 1
        self.min_detection_confidence = 0.1
        self.min_tracking_confidence = 0.1

        # Initialize MediaPipe FaceMesh solution
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence
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

        outputs = self.face_mesh.process(image)
        _multi_face_landmarks = outputs.multi_face_landmarks
        if not _multi_face_landmarks:
            face_info.status = False
            face_info.can_gaze_estimation = False
            return face_info
        else:
            # Normalised landmarks to pixel format
            face_landmarks = _multi_face_landmarks[0].landmark
            face_info.status = True
            _face_mesh = []

            for i in range(len(face_landmarks)):
                face_landmarks[i].x = np.round(face_landmarks[i].x * image_width)
                face_landmarks[i].y = np.round(face_landmarks[i].y * image_height)
                face_landmarks[i].z = np.round(face_landmarks[i].z * image_width)
                _face_mesh.append([face_landmarks[i].x, face_landmarks[i].y, face_landmarks[i].z])
            _face_mesh = np.array(_face_mesh, dtype=np.int16)
            # print(_face_mesh.shape)
            # face_landmarks to numpy array
            # np.savez_compressed(face_mesh_npz_path, face_mesh=_face_mesh)

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
        scale = math.fabs(face_landmarks[362].x - face_landmarks[133].x) / 100
        x_padding = 20  # padding the eye corners
        y_0 = 0.6  # less area for upper portion of the eye area
        y_1 = 0.4  # more area for the lower portion of the eye area

        # get the X-coords for the left eye
        eye_left_xs = [face_landmarks[33].x - x_padding * scale, face_landmarks[133].x + x_padding * scale]
        # height of the eyebox is 0.75 of the width of the eyebox
        eye_left_height = math.fabs(eye_left_xs[0] - eye_left_xs[1]) * 0.75
        # get the Y-coords for the left eye
        eye_left_y = (face_landmarks[33].y + face_landmarks[133].y) / 2.0
        eye_left_ys = [eye_left_y - eye_left_height * y_0, eye_left_y + eye_left_height * y_1]

        # get the X-coords for the right eye
        eye_right_xs = [face_landmarks[362].x - x_padding * scale, face_landmarks[263].x + x_padding * scale]
        # height of the eyebox is 0.75 of the width of the eyebox
        eye_right_height = math.fabs(eye_right_xs[0] - eye_right_xs[1]) * 0.75
        # get the Y-coords for the right eye
        eye_right_y = (face_landmarks[362].y + face_landmarks[263].y) / 2.0
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
        left_eye_ear = np.abs(face_landmarks[33].y - face_landmarks[133].y) / np.abs(
            face_landmarks[33].x - face_landmarks[133].x)

        # right-eye AREA
        right_vertices = _face_mesh[:, :2][self.right_vertices_index]
        right_eye_area = self.calculate_polygon_area(right_vertices)
        right_eye_ear = np.abs(face_landmarks[362].y - face_landmarks[263].y) / np.abs(
            face_landmarks[362].x - face_landmarks[263].x)

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
        pass
