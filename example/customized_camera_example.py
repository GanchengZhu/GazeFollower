
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes

import cv2
import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_RETURN

from gazefollower import GazeFollower
from gazefollower.camera import Camera
from gazefollower.logger import Log

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]


class PupilioCamera(Camera):
    """
    Pupil.IO (https://github.com/GanchengZhu/Pupilio) is an eye-tracking device based on a dual-camera,
    single-light-source system. This script utilizes Pupil.IO’s built-in cameras to capture near-infrared images
    and performs appearance-based eye tracking.
    """
    def __init__(self):
        """
        Initialize the camera and the constant.
        """
        # Initializes the camera state to CLOSING and prepares a lock for callback management
        super().__init__()
        self.camera_utils = None
        # image size
        self.source_img_width = 1280
        self.source_img_height = 1024
        self.target_img_width = 640
        self.target_img_height = 480
        self.dll_handle = None
        self.camera_utils = ctypes.CDLL('customized_camera_native_library/lib_pupil_io_camera.dll', winmode=0)
        self.dll_handle = self.camera_utils._handle
        # camera util
        self.camera_utils.init.restype = ctypes.c_int
        self.camera_utils.deinit.restype = ctypes.c_int
        self.camera_utils.retrieve_images.restype = ctypes.c_int
        # initialize the camera
        result = self.camera_utils.init()
        self._create_capture_thread()

    def open(self):
        # Load the DLL
        pass

    def _create_capture_thread(self):
        """
        Creates and starts a daemon thread for continuously capturing frames from the camera.
        """
        self._camera_thread_running = True
        self._camera_thread = threading.Thread(target=self._processing_frame)
        self._camera_thread.daemon = True
        self._camera_thread.start()

    def _processing_frame(self):
        while self._camera_thread_running:
            timestamp = time.time_ns()
            img_1 = np.zeros((self.source_img_height, self.source_img_width), dtype=np.uint8)
            img_2 = np.zeros((self.source_img_height, self.source_img_width), dtype=np.uint8)
            img_1_ptr = img_1.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            img_2_ptr = img_2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

            result = self.camera_utils.retrieve_images(img_1_ptr, img_2_ptr)

            if result == 1:
                # only use left image
                frame = cv2.cvtColor(img_1, cv2.COLOR_GRAY2RGB)
                # Resize the frame to 640x480 if necessary
                frame0 = cv2.resize(frame, (self.target_img_width, self.target_img_height))
                try:
                    with self.callback_and_param_lock:
                        if self.callback_func is not None:
                            self.callback_func(self.camera_running_state, timestamp, frame, *self.callback_args,
                                               **self.callback_kwargs)
                except Exception as e:
                    Log.e(str(e))
                    # sys.exit(1)
            else:
                Log.d("Failed to retrieve images.")

    def close(self):
        Log.i("WebCam closed")

    def release(self):
        try:
            super().release()

            if self.camera_utils is not None:
                self.camera_utils.deinit()

                if self.dll_handle is not None:
                    kernel32.FreeLibrary(self.dll_handle)
                    self.dll_handle = None

                self.camera_utils = None

        except Exception as e:
            Log.e(f"Release failed: {str(e)}")


if __name__ == '__main__':
    # init pygame
    pygame.init()
    win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

    # init GazeFollower
    gf = GazeFollower(camera=PupilioCamera())
    # previewing
    gf.preview(win=win)
    # calibrating
    gf.calibrate(win=win)
    # sampling
    gf.start_sampling()
    pygame.time.wait(100)  # sleep for 100 ms so the tracker caches some samples

    # A free viewing task, in which we show a picture and overlay the gaze cursor
    img_folder = 'images'
    images = ['grid.jpg']

    # show the images one by one in a loop, press the ENTER key to exit the program
    for _img in images:
        # show the image on screen
        win.fill((128, 128, 128))
        im = pygame.image.load(os.path.join(img_folder, _img))
        win.blit(im, (0, 0))
        pygame.display.flip()
        # send a trigger to record in the eye movement data to mark picture onset
        gf.send_trigger(202)

        # now let's show the gaze point, press any key to close the window
        got_key = False
        max_duration = 20000
        t_start = pygame.time.get_ticks()
        pygame.event.clear()  # clear all cached events if there were any
        while not (got_key or (pygame.time.get_ticks() - t_start) >= max_duration):
            # check key presses
            for ev in pygame.event.get():
                if ev.type == KEYDOWN:
                    if ev.key == K_RETURN:
                        got_key = True

            # update the visual (image and cursor)
            win.blit(im, (0, 0))
            # show gaze cursor; when in a formal experiment you can remove this code
            gaze_info = gf.get_gaze_info()
            if gaze_info and gaze_info.status:
                gx = int(gaze_info.filtered_gaze_coordinates[0])
                gy = int(gaze_info.filtered_gaze_coordinates[1])
                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 50, 5)  # cursor for the left eye

            # flip the frame
            pygame.display.flip()

    # stop sampling
    pygame.time.wait(100)  # sleep for 100 ms to capture ending samples
    gf.stop_sampling()

    # save the sample data to file
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    file_name = "free_viewing_pygame_demo.csv"
    gf.save_data(os.path.join(data_dir, file_name))
    # release gaze follower
    gf.release()
