# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import os

import pygame
from pygame.locals import KEYDOWN, K_RETURN

from gazefollower import GazeFollower
from gazefollower.camera import WebCamCamera
from gazefollower.gaze_estimator import MGazeNetGazeEstimator

if __name__ == '__main__':
    # init pygame
    pygame.init()
    win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

    # init GazeFollower
    gf = GazeFollower(camera=WebCamCamera(webcam_id=0, cam_fps=30))
    # previewing
    gf.preview(win=win)
    # calibrating
    gf.calibrate(win=win)
    # sampling
    gf.start_sampling()
    pygame.time.wait(100)  # sleep for 100 ms so the tracker cache some sample

    # A free viewing task, in which we show a picture and overlay the gaze cursor
    img_folder = 'images'
    images = ['grid.jpg']

    # show the images one by one in a loop, press a ENTER key to exit the program
    for _img in images:
        # show the image on screen
        win.fill((128, 128, 128))
        im = pygame.image.load(os.path.join(img_folder, _img))
        win.blit(im, (0, 0))
        pygame.display.flip()
        # send a trigger to record in the eye movement data to mark picture onset
        gf.send_trigger(202)

        # now lets show the gaze point, press any key to close the window
        got_key = False
        max_duration = 20 * 1000 # 20 seconds
        t_start = pygame.time.get_ticks()
        pygame.event.clear()  # clear all cached events if there were any
        gx, gy = -65536, -65536
        while not (got_key or (pygame.time.get_ticks() - t_start) >= max_duration):
            # check key presses
            for ev in pygame.event.get():
                if ev.type == KEYDOWN:
                    if ev.key == K_RETURN:
                        got_key = True

            # update the visual (image and cursor)
            win.blit(im, (0, 0))
            # show gaze cursor, when formal experiment
            # you can remove this code
            # ++++++++++++++++++++++++
            gaze_info = gf.get_gaze_info()
            if gaze_info and gaze_info.status:
                gx = int(gaze_info.filtered_gaze_coordinates[0])
                gy = int(gaze_info.filtered_gaze_coordinates[1])
                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 50, 5)  # cursor for the left eye
            # ++++++++++++++++++++++++

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

