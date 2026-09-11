import os
import pygame
from pygame.locals import *
from gazefollower import GazeFollower
from gazefollower.face_alignment import BlazeFaceAlignment
from gazefollower.gaze_estimator import MGazeNetGazeEstimator
from gazefollower.calibration import MultivariateRidgeCalibration
from pathlib import Path
import gazefollower

if __name__ == '__main__':
    # init pygame
    pygame.init()
    win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

    # get path to the mobilenet_v4.mnn model inside the res folder
    res_dir = Path(gazefollower.__file__).parent / "res"
    # model_path = res_dir / "model_weights" /"base.mnn"
    model_path = res_dir / "model_weights" / "mobilenet_v4.mnn"

    # init components
    face_alignment = BlazeFaceAlignment()
    gaze_estimator = MGazeNetGazeEstimator(model_path=str(model_path))
    calibration = MultivariateRidgeCalibration(alpha=0.1)

    # init GazeFollower
    gf = GazeFollower(
        face_alignment=face_alignment,
        gaze_estimator=gaze_estimator,
        calibration=calibration
    )
    
    # previewing
    gf.preview(win=win)
    # calibrating
    gf.calibrate(win=win)
    # sampling
    gf.start_sampling()
    pygame.time.wait(100)

    img_folder = 'images'
    images = ['grid.jpg']

    for _img in images:
        win.fill((128, 128, 128))
        im = pygame.image.load(os.path.join(img_folder, _img))
        win.blit(im, (0, 0))
        pygame.display.flip()
        
        gf.send_trigger(202)

        got_key = False
        max_duration = 20000
        t_start = pygame.time.get_ticks()
        pygame.event.clear()
        
        while not (got_key or (pygame.time.get_ticks() - t_start) >= max_duration):
            for ev in pygame.event.get():
                if ev.type == KEYDOWN:
                    if ev.key == K_RETURN or ev.key == K_SPACE:
                        got_key = True

            win.blit(im, (0, 0))
            
            gaze_info = gf.get_gaze_info()
            if gaze_info and gaze_info.status:
                gx = int(gaze_info.filtered_gaze_coordinates[0])
                gy = int(gaze_info.filtered_gaze_coordinates[1])
                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 50, 5)

            pygame.display.flip()

    pygame.time.wait(100)
    gf.stop_sampling()

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    file_name = "fastest_mnn_ridge_demo.csv"
    gf.save_data(os.path.join(data_dir, file_name))
    gf.release()
