# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import json
import math
import os.path
import time
import tkinter as tk
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import pygame
from screeninfo import get_monitors

from ..logger import Log
from ..ui.ParticipantInfoDialog import ParticipantInfoDialog


class Recorder:
    def __init__(self, camera=None,
                 dataset_dir: str = None, subject_dir_format: str = "{subject_id}_{age}_{gender}_{wears_glasses}",
                 frame_name_format: str = "{frame_id:06d}_{point_index:03d}_{ground_truth_x:.6f}_"
                                          "{ground_truth_y:.6f}.jpg"):
        """
        Initializes the Recorder instance, setting up the camera, directories,
        and GUI for participant information.

        :param camera: An instance of a Camera subclass; if None, a WebCamCamera will be used.
        :param dataset_dir: Directory to save the dataset; defaults to the current working directory.
        :param subject_dir_format: Format for the subject directory name.
        :param frame_name_format: Format for the names of saved image frames.
        """
        Log.init("tmp.log")
        self.point_showing = False
        self.formal_exp = False

        if camera is None:
            from ..camera import WebCamCamera
            self.camera = WebCamCamera()
        else:
            from ..camera import Camera
            if not isinstance(camera, Camera):
                raise TypeError(f"Expected an instance of Camera, but got {type(camera).__name__}.")
            else:
                self.camera = camera

        self.camera.set_on_image_callback(self._on_image_available)
        self.camera.start_sampling()

        if dataset_dir is None:
            self.dataset_dir = Path.cwd() / "gaze_dataset"
        else:
            self.dataset_dir = dataset_dir

        root = tk.Tk()
        dialog = ParticipantInfoDialog(root)
        root.mainloop()  # Start the Tkinter main loop

        # Get the results after the dialog is closed
        self.participant_info = dialog.get_info()
        self.subject_dir = os.path.join(str(self.dataset_dir), subject_dir_format.format(**self.participant_info))

        self.image_save_dir = os.path.join(self.subject_dir, "images")
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

        # root.destroy()
        self.frame_name_format = frame_name_format

        pygame.init()

        self._color_white = (255, 255, 255)
        self._color_red = (255, 0, 0)
        self._color_black = (0, 0, 0)
        self._color_blue = (0, 0, 255)
        self._color_green = (0, 255, 0)
        self._color_gray = (128, 128, 128)

        # hidden the mouse cursor
        pygame.mouse.set_visible(False)

        # Initialize the font
        self.guidance_font = pygame.font.SysFont('Microsoft YaHei', 28)

        self.monitors = get_monitors()
        self.screen_size = np.array([self.monitors[0].width, self.monitors[0].height])
        self.screen_width = self.monitors[0].width
        self.screen_height = self.monitors[0].height
        self.screen = pygame.display.set_mode(self.screen_size, pygame.FULLSCREEN)

        # Initialize the mixer and load sound
        pygame.mixer.init()
        _audio_path = Path(__file__).parent.parent.absolute() / 'res/audio/beep.wav'
        self.feedback_sound = pygame.mixer.Sound(_audio_path)  # Replace with the path to your sound file

        # Load the arrow images
        _left_arrow_path = Path(__file__).parent.parent.absolute() / 'res/image/left_arrow.png'

        self._arrow_image_size = 36
        # Load and resize the left arrow image
        self.left_red_arrow_image = pygame.transform.smoothscale(pygame.image.load(_left_arrow_path),
                                                                 (self._arrow_image_size, self._arrow_image_size))

        # Create and resize the right arrow image by rotating the left arrow image
        self.right_red_arrow_image = pygame.transform.rotate(self.left_red_arrow_image, 180)

        # Function to change the color of the arrow image

        # Create left and right green arrow images
        self.left_green_arrow_image = self._change_arrow_color(self.left_red_arrow_image)  # Green color
        self.right_green_arrow_image = self._change_arrow_color(self.right_red_arrow_image)  # Green color

    @staticmethod
    def _change_arrow_color(image):
        """
        Changes the color of the given arrow image by swapping the red and green channels.

        This method processes each pixel in the image and modifies its color if it is not transparent.

        :param image: The image to be color-modified. This should be a Pygame surface.
        :return: A new image with the red and green color channels swapped.
        """
        colored_image = image.copy()
        for x in range(colored_image.get_width()):
            for y in range(colored_image.get_height()):
                pixel = colored_image.get_at((x, y))
                if pixel[3] != 0:  # Check if the pixel is not transparent
                    # Swap R and G values
                    new_pixel = (pixel[1], pixel[0], pixel[2], pixel[3])  # (G, R, B, A)
                    colored_image.set_at((x, y), new_pixel)
        return colored_image

    def draw_guidance_text(self, screen, instruction_text):
        """
        Draws the guidance text on the specified screen.

        This method takes a list of instruction text lines and renders them
        on the screen in the center, adjusting for vertical spacing.

        :param screen: The Pygame surface where the text will be drawn.
        :param instruction_text: A list of strings representing the lines of
                                 text to be displayed as guidance for the user.
        """
        text_surfaces = [self.guidance_font.render(line, True, self._color_white) for line in instruction_text]

        total_text_height = sum(text_surface.get_height() for text_surface in text_surfaces) + (
                len(instruction_text) - 1) * 10
        start_y = (self.screen_height - total_text_height) // 2

        for i, text_surface in enumerate(text_surfaces):
            text_rect = text_surface.get_rect(
                center=(self.screen_width // 2, start_y + i * (text_surface.get_height() + 10)))
            screen.blit(text_surface, text_rect)

    def draw_arrows(self, screen, center: Tuple, direction: str, color: str):
        """
        Draws an arrow on the specified screen at the given center position.

        The arrow's direction (left or right) and color (green or red) determine
        which image to render.

        :param screen: The Pygame surface where the arrow will be drawn.
        :param center: A tuple representing the (x, y) coordinates where the
                       arrow should be centered.
        :param direction: A string indicating the direction of the arrow ('left' or 'right').
        :param color: A string indicating the color of the arrow ('green' or 'red').
        """
        # Select the appropriate arrow image based on direction and color
        if direction == 'left' and color == 'green':
            image = self.left_green_arrow_image
        elif direction == 'right' and color == 'green':
            image = self.right_green_arrow_image
        elif direction == 'left' and color == 'red':
            image = self.left_red_arrow_image
        else:
            image = self.right_red_arrow_image

        # Calculate the position to draw the arrow, so it is centered at the specified coordinates
        screen.blit(image, (
            center[0] - image.get_width() // 2,
            center[1] - image.get_height() // 2))

    def _on_image_available(self, running_state, timestamp, frame):
        """
        Handles the event when a new image is available from the camera.

        This method is intended to be overridden in subclasses. It is called
        whenever a new frame is captured by the camera.

        :param running_state: A boolean indicating whether the recording is currently active.
        :param timestamp: The timestamp when the image was captured, usually in milliseconds.
        :param frame: The image frame captured by the camera, typically in a format suitable for processing.
        :return: None
        """
        raise NotImplementedError("Subclasses must implement this method.")


class SmoothPursuitRecorder(Recorder):
    """
    SmoothPursuitRecorder is a subclass of Recorder designed to handle
    smooth pursuit eye-tracking tasks. It manages the recording of image and gaze
    data as participants follow moving targets on the screen.

    Attributes:
        visible_rect (List): A list defining the visible area for pursuit,
                             e.g., [x, y, width, height].
        pursuit_params (dict): Parameters specific to the smooth pursuit
                               recording (e.g., target speed, distance).
        dwelling_time (float): Duration (in seconds) for which the gaze
                               should dwell on each target.
    """

    def __init__(self, camera=None, visible_rect: List = None,
                 pursuit_params=None,
                 dwelling_time: float = 2, dataset_dir: str = None,
                 subject_dir_format: str = "Pursuit_{subject_id}_{age}_{gender}_{wears_glasses}",
                 frame_name_format: str = "{frame_id:06d}_{ground_truth_x:.6f}_{ground_truth_y:.6f}.jpg"):
        super().__init__(camera, dataset_dir, subject_dir_format, frame_name_format)
        """
        Initializes the SmoothPursuitRecorder with camera settings, parameters, 
        and file formats for recording smooth pursuit eye-tracking data.

        :param camera: An optional camera object for capturing images. 
                       If None, a default camera will be used.
        :param visible_rect: A list defining the visible area for the pursuit 
                             (e.g., [x1, y1, x2, y2]).
        :param pursuit_params: Optional parameters specific to the smooth pursuit 
                               recording.
        :param dwelling_time: The duration (in seconds) for which the gaze 
                              should dwell on each target.
        :param dataset_dir: The directory where the dataset will be saved. 
                            If None, defaults to "gaze_dataset" in the current working directory.
        :param subject_dir_format: The format string for naming the subject's 
                                   directory based on participant information.
        :param frame_name_format: The format string for naming individual image frames 
                                  captured during the recording.
        """

        if pursuit_params is None:
            self.pursuit_params = {
                "freq_x": 1 / 12,
                "freq_y": 1 / 8,
                "phase_x": 0,
                "phase_y": 0
            }
        else:
            self.pursuit_params = pursuit_params

        if visible_rect is None:
            _margin = 40
            self.visible_rect = [_margin / self.screen_width, _margin / self.screen_height,
                                 (self.screen_width - _margin) / self.screen_width,
                                 (self.screen_height - _margin) / self.screen_height]
        else:
            self.visible_rect = visible_rect

        if not self._check_rect(self.visible_rect):
            raise ValueError(f"Invalid visible rect: {self.visible_rect}")

        self.dwelling_time = dwelling_time

        self.amp_x = (self.visible_rect[2] - self.visible_rect[0]) / 2
        self.amp_y = (self.visible_rect[3] - self.visible_rect[1]) / 2
        self.phase_x = self.pursuit_params["phase_x"]
        self.phase_y = self.pursuit_params["phase_y"]
        self.freq_x = self.pursuit_params["freq_x"]
        self.freq_y = self.pursuit_params["freq_y"]
        self.start_x = self.visible_rect[0]
        self.start_y = self.visible_rect[1]

        self.duration = self._gcd_lcm(
            int(round(1 / self.freq_x)), int(round(1 / self.freq_y))
        )

        self._circle_size = 36
        self._new_session()

    @staticmethod
    def _gcd_lcm(num1: int, num2: int, mode: str = "lcm") -> int:
        """
        Computes the greatest common divisor (GCD) or least common multiple (LCM)
        of two integers.

        :param num1: The first integer.
        :param num2: The second integer.
        :param mode: The mode of operation; 'gcd' for greatest common divisor
                     or 'lcm' for least common multiple. Defaults to 'lcm'.
        :return: The GCD or LCM of the two integers, depending on the specified mode.

        :raises ValueError: If the mode is not 'gcd' or 'lcm'.
        """

        # Helper function to compute GCD using Euclidean algorithm
        def gcd(a: int, b: int) -> int:
            while a != 0:
                a, b = b % a, a
            return b

        if mode == "gcd":
            return gcd(num1, num2)
        elif mode == "lcm":
            return (num1 * num2) // gcd(num1, num2)
        else:
            raise ValueError("Mode must be 'gcd' or 'lcm'")

    @staticmethod
    def _check_rect(rect: List[int]) -> bool:
        """
        Checks if a given list represents a valid rectangle.

        A rectangle is considered valid if it is represented by four integers,
        where the first two integers are the coordinates of the top-left corner
        (x1, y1) and the last two integers are the coordinates of the bottom-right
        corner (x2, y2). The coordinates must satisfy the conditions:
        x2 > x1 and y2 > y1.

        :param rect: A list of four integers representing the rectangle's coordinates.
                     Format: [x1, y1, x2, y2].
        :return: True if the rectangle is valid; False otherwise.
        """
        # Ensure the input list contains exactly four elements
        if len(rect) != 4:
            return False

        x1, y1, x2, y2 = rect

        # Check if x2 > x1 and y2 > y1 to ensure a valid rectangle
        if x2 > x1 and y2 > y1:
            return True

        return False  # Return False if rectangle is invalid

    def start(self):
        """
        Starts the recording.
        :return:
        """
        self.formal_exp = True
        response_df = self.draw(
            [
                "Formal Experiment",
                "      ",
                "A white dot will appear on the screen; please keep your gaze on it.",
                "During the experiment, the white dot will keep moving; please move your eyes to follow it.",
                "A red arrow will appear with two directions; please quickly press the F and J keys to respond.",
                "Press the SPACE to start the formal experiment."
            ]
        )

        response_df.to_excel(os.path.join(self.subject_dir, "participant_response_formal.xlsx"), index=False)
        with open(os.path.join(self.subject_dir, "screen_size.json"), 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "screen_width": self.screen_width,
                    "screen_height": self.screen_height,
                }, f
            )
        pygame.quit()
        self.camera.stop_sampling()

    def _on_image_available(self, running_state, timestamp, frame):
        """
        See the Recorder class
        :param running_state:
        :param timestamp:
        :param frame:
        :return:
        """
        if not (self.point_showing and self.formal_exp):
            return
        else:
            # Normalized ground truth point
            ground_truth_point = self.current_point
            # "{frame_id: %05d}_{ground_truth_x: %.6f}_{ground_truth_y: %.6f}.jpg"

            file_path = os.path.join(self.image_save_dir, self.frame_name_format.format(
                **{
                    "frame_id": self.n_frame,
                    "ground_truth_x": ground_truth_point[0] / self.screen_width,
                    "ground_truth_y": ground_truth_point[1] / self.screen_height,
                }
            ))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, frame)
            self.n_frame += 1

    @staticmethod
    def _generate_arrows(duration, seed):
        """
        Generates a list of arrows with specified durations and directions within a given time frame.

        Each arrow has a specific onset time, offset time, and end time, with randomly assigned
        directions (left or right). The function ensures that the arrows do not exceed the total
        duration specified.

        :param duration: The total duration for which arrows can be generated (in seconds).
        :param seed: An optional integer to set the random seed for reproducibility.
        :return: A list of tuples, each containing the onset time, offset time, end time,
                 and direction of the arrow.
        """
        if seed is not None:
            np.random.seed(seed)  # 设置随机种子，保证每次运行结果相同

        arrows = []
        time_elapsed = 0
        while time_elapsed + 3 <= duration:
            arrow_onset_time = time_elapsed
            arrow_offset_time = time_elapsed + 0.5
            trial_end_time = arrow_offset_time + np.random.uniform(1.5, 2.5)
            if trial_end_time > duration:
                break
            # 随机生成小球的方向，"left" 或 "right"
            direction = np.random.choice(["left", "right"])
            time_elapsed = trial_end_time
            arrows.append((arrow_onset_time, arrow_offset_time, trial_end_time, direction))
        return arrows

    def _new_session(self):
        """
        Set a new session.
        :return:
        """
        self.n_frame = 0
        self.elapsed_time = 0
        self.running = False
        self.point_showing = False
        self.sound_played = False
        self.current_point = [
            (self.visible_rect[0] + (self.visible_rect[2] - self.visible_rect[0]) // 2) * self.screen_width,
            (self.visible_rect[1] + (self.visible_rect[3] - self.visible_rect[1]) // 2) * self.screen_height,
        ]
        self.arrows = self._generate_arrows(self.duration, 2024)
        self._current_arrows_index = 0
        self.responses = []

    def draw(self, guidance_text):
        """
        Draw the experiment on the screen
        :param guidance_text: str, guidance text.
        :return:
        """
        self._new_session()

        # Show initial guidance screen
        self.running = True
        while self.running:
            self.screen.fill(self._color_gray)
            self.draw_guidance_text(self.screen, guidance_text)
            # Update the display
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                    pygame.quit()

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.running = False
                    self.start_time = time.time()  # Initialize the start time

        self.running = True
        self.point_showing = True
        # Show points on the screen
        while self.running:
            current_time = time.time()  # Get the current time
            self.elapsed_time = current_time - self.start_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.point_showing = False
                    self.running = False
                elif event.type == pygame.KEYDOWN and (event.key == pygame.K_f or event.key == pygame.K_j):
                    # Record response if F or J key is pressed
                    response_key = 'F' if event.key == pygame.K_f else 'J'
                    arrow_elapsed_time = time.time() - self.start_time - self.arrows[self._current_arrows_index][
                        0] - self.dwelling_time
                    if arrow_elapsed_time > 0 and (self._current_arrows_index + 1 > len(self.responses)):
                        self.responses.append({
                            'arrow_direction': self.arrows[self._current_arrows_index][-1],
                            'response_key': response_key,
                            'response_time': arrow_elapsed_time
                        })
                    else:
                        if not self.sound_played:
                            self.feedback_sound.play()
                            self.sound_played = True

            # Fill the screen with a white background
            self.screen.fill(self._color_gray)

            # Draw the current calibration point
            if self.elapsed_time > self.dwelling_time * 2 + self.duration:
                self.point_showing = False
                self.running = False

            elif self.dwelling_time < self.elapsed_time < self.duration + self.dwelling_time:
                time_elapsed_second = self.elapsed_time - self.dwelling_time
                x = self.amp_x * (math.sin(math.pi * 2 * self.freq_x * time_elapsed_second + self.phase_x) + 1)
                y = self.amp_y * (math.sin(math.pi * 2 * self.freq_y * time_elapsed_second + self.phase_y) + 1)

                self.current_point = (
                    (x + self.visible_rect[0]) * self.screen_width,
                    (y + self.visible_rect[1]) * self.screen_height,
                )

                self.draw_anti_aliased_circle(screen=self.screen, point=self.current_point)

                if self._current_arrows_index < len(self.arrows):

                    arrow_onset_time, arrow_offset_time, trial_end_time, arrow_direction = self.arrows[
                        self._current_arrows_index]

                    if arrow_onset_time <= time_elapsed_second < arrow_offset_time:
                        self.draw_arrows(screen=self.screen, center=self.current_point, direction=arrow_direction,
                                         color="green")

                    elif time_elapsed_second >= trial_end_time:
                        if len(self.responses) <= self._current_arrows_index:
                            self.responses.append({
                                'arrow_direction': self.arrows[self._current_arrows_index][-1],
                                'response_key': 'miss',
                                'response_time': -1
                            })
                        self._current_arrows_index += 1

            elif self.elapsed_time < self.dwelling_time or self.elapsed_time > self.duration + self.dwelling_time:
                self.current_point = (
                    (self.visible_rect[0] + self.amp_x) * self.screen_width,
                    (self.visible_rect[1] + self.amp_y) * self.screen_height,
                )

                self.draw_anti_aliased_circle(screen=self.screen, point=self.current_point)

            # Update the display
            pygame.display.flip()

            if self.sound_played and not pygame.mixer.get_busy():
                self.sound_played = False  # Reset sound flag to allow replay if needed

        return pd.DataFrame(self.responses,
                            columns=['arrow_direction', 'response_key',
                                     'response_time'])

    def draw_anti_aliased_circle(self, screen, point):
        """
        Draw an anti-aliased circle on the screen.

        :param screen: Pygame screen
        :param point: Circle center point (x, y)
        :return: None
        """
        # Create a new surface with per-pixel alpha transparency
        circle_surface = pygame.Surface((self._circle_size, self._circle_size), pygame.SRCALPHA)
        # Fill the surface with transparency
        circle_surface.fill((0, 0, 0, 0))
        # Draw the circle on this surface with anti-aliasing
        pygame.draw.circle(circle_surface, self._color_white, (self._circle_size // 2, self._circle_size // 2),
                           self._circle_size // 2)
        # Scale the surface slightly to make the edges smoother
        smooth_surface = pygame.transform.smoothscale(circle_surface, (self._circle_size, self._circle_size))
        # Blit the circle onto the screen
        screen.blit(smooth_surface, (point[0] - self._circle_size // 2, point[1] - self._circle_size // 2))

    # def _arrow_show(self, screen, point, arrow_direction):


class NPointRecorder(Recorder):
    """
    NPointRecorder is a class for recording face and gaze data based on a series of defined points.
    It manages the display of points on the screen and captures user gaze information
    as they dwell on each point for a specified duration.
    """

    def __init__(self, camera=None, points: List = None, dwelling_time: float = 2, dataset_dir: str = None,
                 subject_dir_format: str = "NPoint_{subject_id}_{age}_{gender}_{wears_glasses}",
                 frame_name_format: str = "{frame_id:06d}_{point_index:03d}_{ground_truth_x:.6f}_"
                                          "{ground_truth_y:.6f}.jpg"):

        """
        Initializes the NPointRecorder instance.

        :param camera: Optional; an instance of a camera class to capture images. If not provided, a default camera will be used.
        :param points: A list of points (coordinates) to be displayed on the screen for gaze tracking.
        :param dwelling_time: The duration (in seconds) for which the user should focus on each point.
        :param dataset_dir: Directory where the dataset will be stored. If None, defaults to the current working directory.
        :param subject_dir_format: Format string for generating subject-specific directories.
        :param frame_name_format: Format string for naming the recorded frames.
        """

        super().__init__(camera, dataset_dir, subject_dir_format, frame_name_format)

        if points is None:
            self.points = (
                (0.5, 0.5),  # Center
                (0.5, 0.08),  # Top center
                (0.08, 0.5),  # Left center
                (0.92, 0.5),  # Right center
                (0.5, 0.92),  # Bottom center
                (0.08, 0.08),  # Top left corner
                (0.92, 0.08),  # Top right corner
                (0.08, 0.92),  # Bottom left corner
                (0.92, 0.92),  # Bottom right corner
                (0.25, 0.25),  # Inner top left
                (0.75, 0.25),  # Inner top right
                (0.25, 0.75),  # Inner bottom left
                (0.75, 0.75),  # Inner bottom right
                (0.5, 0.5)  # Center
            )

        else:
            self.points = points

        self.dwelling_time = dwelling_time

        self.user_response = []
        self.point_directions = []

        self.pixel_points = tuple(
            [(point[0] * self.screen_width, point[1] * self.screen_height) for point in self.points])
        self._new_session()

    def _on_image_available(self, running_state, timestamp, frame):
        """
        See the Record class.
        :param running_state:
        :param timestamp:
        :param frame:
        :return:
        """
        # print("point showing: ", self.point_showing, " formal exp: ", self.formal_exp)
        if not (self.point_showing and self.formal_exp):
            return
        else:
            current_point_index = self.current_point_index
            if current_point_index >= len(self.points):
                return
            # Normalized ground truth point
            ground_truth_point = self.points[current_point_index]
            # "{frame_id: %05d}_{ground_truth_x: %.6f}_{ground_truth_y: %.6f}.jpg"

            file_path = os.path.join(self.image_save_dir, self.frame_name_format.format(
                **{
                    "frame_id": self.n_frame,
                    "point_index": current_point_index,
                    "ground_truth_x": ground_truth_point[0],
                    "ground_truth_y": ground_truth_point[1],
                }
            ))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, frame)
            self.n_frame += 1

    def _new_session(self):
        """
        Creates a new session.
        :return:
        """
        self.running = True
        self.current_point_index = 0  # Start at the first calibration point
        self.start_time = None  # Initialize the start time
        self.sound_played = False  # Flag to track if the sound has been played
        self.point_directions = []
        self.generate_point_directions()
        self.responses = []
        self.point_showing = False
        self.point_elapsed_time = 0
        self.n_frame = 0

    def generate_point_directions(self):
        """
        generate random point directions.
        :return:
        """
        num_points = len(self.points)
        # Generate lists for directions
        self.point_directions = ['left'] * (num_points // 2) + ['right'] * (num_points - num_points // 2)
        # Shuffle the list with a fixed seed for reproducibility
        np.random.seed(912)
        np.random.shuffle(self.point_directions)

    def draw_breathing_effect(self, screen, center, outer_radius: int, inner_radius: int, elapsed_time: float):
        """
        Draws a breathing light effect with a smooth color gradient towards the inner circle.

        This effect creates an animated light that pulses between the specified inner and outer radii,
        giving the appearance of breathing. The color gradient is stronger towards the inner circle.

        :param screen: The Pygame surface on which to draw the effect.
        :param center: A tuple (x, y) representing the center coordinates of the breathing effect.
        :param outer_radius: The maximum radius of the breathing effect when fully expanded.
        :param inner_radius: The minimum radius of the breathing effect at its contracted state.
        :param elapsed_time: The time in seconds since the beginning of the pulse cycle, used to animate the effect.

        :return: None
        """

        pulse_period = self.dwelling_time + 0.2  # seconds for one full pulse cycle
        pulse_amplitude = outer_radius - inner_radius  # Maximum expansion relative to inner circle
        if elapsed_time > pulse_period:
            return

        # Calculate the pulse offset to animate the gradient effect
        pulse_offset = math.sin(elapsed_time / pulse_period * math.pi / 2)  # Oscillates between 0 and 1
        current_radius = inner_radius + pulse_amplitude * (1 - pulse_offset)  # Decreases from max to min

        # Create a surface for the gradient effect with transparency
        gradient_surface = pygame.Surface((2 * current_radius, 2 * current_radius), pygame.SRCALPHA)

        # Use a higher resolution surface for anti-aliasing effect
        scale_factor = 1  # Increase the resolution by this factor
        high_res_radius = int(current_radius * scale_factor)
        high_res_surface = pygame.Surface((2 * high_res_radius, 2 * high_res_radius), pygame.SRCALPHA)

        # Draw concentric circles with varying intensity to create a gradient effect
        for i in range(high_res_radius, int(inner_radius * scale_factor), -2 * scale_factor):
            color_intensity = int(128 * ((i - inner_radius * scale_factor) / (pulse_amplitude * scale_factor)))
            gradient_color = (255, color_intensity, color_intensity, 128)  # Red gradient with varying alpha
            pygame.draw.circle(high_res_surface, gradient_color, (high_res_radius, high_res_radius), i)

        # Scale down the high-resolution surface to the original size to achieve anti-aliasing
        gradient_surface = pygame.transform.smoothscale(high_res_surface,
                                                        (2 * int(current_radius), 2 * int(current_radius)))

        # Draw the gradient surface on the screen
        screen.blit(gradient_surface, (center[0] - current_radius, center[1] - current_radius))

        # # Draw the inner white circle separately to ensure it stays the correct size
        pygame.draw.circle(screen, self._color_gray, center, inner_radius)

    def start(self):
        # pre-experiment
        response_df = self.draw(guidance_text=[
            "Practice Experiment",
            "      ",
            "During the experiment, an arrow will appear on the screen. Please keep your eyes on the arrow.",
            "The arrow will point left or right, and initially, the arrow will be red.",
            "When the arrow turns green, press the F key for the left arrow and the J key for the right arrow.",
            "Press the space bar to start the practice, and press the Esc key to exit the practice."
        ])

        response_df.to_excel(os.path.join(self.subject_dir, "participant_response_practice.xlsx"), index=False)

        self.formal_exp = True
        # formal—experiment
        response_df = self.draw(guidance_text=[
            "Formal Experiment",
            "      ",
            "During the experiment, an arrow will appear on the screen. Please keep your eyes on the arrow.",
            "The arrow will point left or right, and initially, the arrow will be red.",
            "When the arrow turns green, press the F key for the left arrow and the J key for the right arrow.",
            "Press the space bar to start the formal experiment."
        ])
        response_df.to_excel(os.path.join(self.subject_dir, "participant_response_formal.xlsx"), index=False)

        with open(os.path.join(self.subject_dir, "screen_size.json"), 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "screen_width": self.screen_width,
                    "screen_height": self.screen_height,
                }, f
            )
        self.formal_exp = False
        pygame.quit()
        self.camera.stop_sampling()

    def draw(self, guidance_text):
        """
        Draw the experiment on the screen
        :param guidance_text: str, guidance text.
        :return:
        """
        self._new_session()
        # Show initial guidance screen
        while self.running:
            self.screen.fill(self._color_gray)
            self.draw_guidance_text(self.screen, guidance_text)
            # Update the display
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                    return pd.DataFrame(self.responses,
                                        columns=['point_x', 'point_y', 'arrow_direction',
                                                 'response_key', 'response_time'])
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.running = False
                    self.start_time = time.time()  # Initialize the start time

        self.running = True
        self.point_showing = True
        points_need_to_draw = self.pixel_points
        # Show points on the screen
        while self.running:
            current_time = time.time()  # Get the current time
            self.point_elapsed_time = current_time - self.start_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    # Record response if F or J key is pressed
                    if event.key == pygame.K_f or event.key == pygame.K_j:
                        response_key = 'F' if event.key == pygame.K_f else 'J'
                        self.point_elapsed_time = time.time() - self.start_time
                        if self.current_point_index < len(points_need_to_draw):
                            self.responses.append({
                                'point_x': self.points[self.current_point_index][0],
                                'point_y': self.points[self.current_point_index][1],
                                'arrow_direction': self.point_directions[self.current_point_index],
                                'response_key': response_key,
                                'response_time': self.point_elapsed_time
                            })

                        # Advance to the next calibration point if minimum display time has passed
                        if self.point_elapsed_time >= self.dwelling_time:
                            self.current_point_index += 1
                            if self.current_point_index >= len(points_need_to_draw):
                                self.running = False  # Exit if all points are shown
                            self.start_time = time.time()  # Reset the start time for the next point
                            self.sound_played = False  # Reset sound flag
                        else:
                            if not self.sound_played:
                                self.feedback_sound.play()
                                self.sound_played = True

            # Fill the screen with a background
            self.screen.fill(self._color_gray)

            # Draw the current calibration point
            if self.current_point_index < len(points_need_to_draw):
                current_point = points_need_to_draw[self.current_point_index]
                # Decide on the direction of the arrow
                direction = self.point_directions[self.current_point_index]
                # Draw the breathing effect
                self.draw_breathing_effect(self.screen, current_point, self._arrow_image_size,
                                           self._arrow_image_size // 2,
                                           self.point_elapsed_time)
                # Draw the arrows inside the calibration point
                if self.point_elapsed_time > self.dwelling_time + 0.2:
                    self.draw_arrows(self.screen, current_point, direction, "green")
                else:
                    self.draw_arrows(self.screen, current_point, direction, "red")

            # Update the display
            pygame.display.flip()

            # Check if the sound has finished playing
            if self.sound_played and not pygame.mixer.get_busy():
                self.sound_played = False  # Reset sound flag to allow replay if needed

        self.point_showing = False
        return pd.DataFrame(self.responses,
                            columns=['point_x', 'point_y', 'arrow_direction', 'response_key',
                                     'response_time'])
