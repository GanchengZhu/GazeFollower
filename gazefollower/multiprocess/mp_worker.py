# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import os
import queue
import time
import cv2
import numpy as np

from ..camera import WebCamCamera
from ..face_alignment import MediaPipeFaceAlignment, BlazeFaceAlignment
from ..gaze_estimator import MGazeNetGazeEstimator
from ..calibration import MultivariateRidgeCalibration, SVRCalibration
from ..filter import HeuristicFilter, OneEuroFilter
from ..misc import clip_patch, CameraRunningState, FaceInfo, GazeInfo, cm2px
from ..logger import Log


def create_component(comp_type, comp_kind, kwargs=None):
    if kwargs is None:
        kwargs = {}
    if comp_kind == 'face_alignment':
        if comp_type == 'blazeface':
            return BlazeFaceAlignment(**kwargs)
        return MediaPipeFaceAlignment(**kwargs)
    elif comp_kind == 'gaze_estimator':
        return MGazeNetGazeEstimator(**kwargs)
    elif comp_kind == 'calibration':
        if comp_type == 'svr':
            return SVRCalibration(**kwargs)
        alpha = kwargs.get('alpha', 0.1)
        return MultivariateRidgeCalibration(alpha=alpha)
    elif comp_kind == 'filter':
        if comp_type == 'one_euro':
            return OneEuroFilter(**kwargs)
        return HeuristicFilter(**kwargs)
    elif comp_kind == 'camera':
        webcam_id = kwargs.get('webcam_id', 0)
        return WebCamCamera(webcam_id=webcam_id)
    return None


def mp_worker_entry(cmd_queue, preview_queue, cali_feature_queue, gaze_queue, config_dict):
    """
    Worker process entrypoint for eye tracking.
    Runs camera capture, face detection, gaze estimation, calibration, and filtering
    in a dedicated process, completely isolating heavy ML inference from the main UI GIL.
    """
    camera = create_component(config_dict.get('camera_type', 'webcam'), 'camera',
                              {'webcam_id': config_dict.get('webcam_id', 0)})
    face_alignment = create_component(config_dict.get('face_alignment_type', 'mediapipe'), 'face_alignment')
    gaze_estimator = create_component(config_dict.get('gaze_estimator_type', 'mgazenet'), 'gaze_estimator')
    calibration = create_component(config_dict.get('calibration_type', 'ridge'), 'calibration',
                                   {'alpha': config_dict.get('alpha', 0.1)})
    gaze_filter = create_component(config_dict.get('filter_type', 'heuristic'), 'filter')

    screen_size = config_dict.get('screen_size', np.array([1920, 1080]))
    physical_screen_size = config_dict.get('screen_physical_size', None)
    camera_position = config_dict.get('camera_position', (15.0, 10.0))

    sample_stream = None
    trigger = 0

    def convert_to_pixel(raw_pos):
        if physical_screen_size is not None:
            return cm2px(raw_pos, camera_position, physical_screen_size, screen_size)
        return raw_pos[0] * screen_size[0], raw_pos[1] * screen_size[1]

    def on_frame(state, timestamp, frame):
        nonlocal trigger, sample_stream
        if state == CameraRunningState.CALIBRATING:
            face_info = face_alignment.detect(timestamp, frame)
            gaze_info = gaze_estimator.detect(frame, face_info)
            try:
                cali_feature_queue.put_nowait((timestamp, gaze_info, face_info))
            except Exception:
                pass

        elif state == CameraRunningState.SAMPLING:
            face_info = face_alignment.detect(timestamp, frame)
            gaze_info = gaze_estimator.detect(frame, face_info)

            if gaze_info.status and gaze_info.features is not None:
                calibrated, calibrated_coords = calibration.predict(gaze_info.features,
                                                                    gaze_info.raw_gaze_coordinates)
                if calibrated:
                    gaze_info.calibrated_gaze_coordinates = convert_to_pixel(calibrated_coords)
                else:
                    gaze_info.calibrated_gaze_coordinates = convert_to_pixel(gaze_info.raw_gaze_coordinates)

                filtered_coords = gaze_filter.filter_values(gaze_info.calibrated_gaze_coordinates)
                gaze_info.filtered_gaze_coordinates = filtered_coords

            if gaze_queue.full():
                try:
                    gaze_queue.get_nowait()
                except Exception:
                    pass
            try:
                gaze_queue.put_nowait(gaze_info)
            except Exception:
                pass

            if sample_stream is not None:
                tmp_trig = 0
                if trigger != 0:
                    tmp_trig = trigger
                    trigger = 0

                raw_x, raw_y = gaze_info.raw_gaze_coordinates if gaze_info.raw_gaze_coordinates is not None else (0, 0)
                cal_x, cal_y = gaze_info.calibrated_gaze_coordinates if gaze_info.calibrated_gaze_coordinates is not None else (0, 0)
                flt_x, flt_y = gaze_info.filtered_gaze_coordinates if gaze_info.filtered_gaze_coordinates is not None else (0, 0)

                row = (f"{gaze_info.timestamp},{raw_x},{raw_y},"
                       f"{cal_x},{cal_y},"
                       f"{flt_x},{flt_y},"
                       f"{gaze_info.left_openness},{gaze_info.right_openness},{gaze_info.tracking_state.value},"
                       f"{int(gaze_info.status)},{int(gaze_info.event.value)},{tmp_trig}\n")
                sample_stream.write(row)
                sample_stream.flush()

    camera.set_on_image_callback(on_frame)
    running = True

    while running:
        try:
            msg = cmd_queue.get(timeout=0.02)
        except queue.Empty:
            continue

        cmd = msg[0]
        if cmd == 'START_PREVIEW':
            camera.start_previewing()
        elif cmd == 'STOP_PREVIEW':
            camera.stop_previewing()
        elif cmd == 'START_CALIBRATING':
            camera.start_calibrating()
        elif cmd == 'STOP_CALIBRATING':
            camera.stop_calibrating()
        elif cmd == 'SET_CALIBRATION_MODEL':
            cali_data = msg[1]
            calibration.weights = cali_data.get('weights')
            calibration.has_calibrated = cali_data.get('has_calibrated', True)
        elif cmd == 'START_SAMPLING':
            sample_path = msg[1] if len(msg) > 1 else None
            if sample_path:
                sample_stream = open(sample_path, 'a', encoding='utf-8')
            camera.start_sampling()
        elif cmd == 'STOP_SAMPLING':
            camera.stop_sampling()
            if sample_stream is not None:
                try:
                    sample_stream.close()
                except Exception:
                    pass
                sample_stream = None
        elif cmd == 'SEND_TRIGGER':
            trigger = msg[1]
        elif cmd == 'TERMINATE':
            running = False

    try:
        camera.close()
        camera.release()
    except Exception:
        pass
    try:
        face_alignment.release()
        gaze_estimator.release()
        gaze_filter.release()
        calibration.release()
    except Exception:
        pass
