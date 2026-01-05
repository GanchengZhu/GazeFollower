# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from typing import Tuple

import cv2
import numpy
import numpy as np
import pygame


class UIBackend:
    """
    UIBackend provides an interface for graphical operations on a window,
    supporting drawing shapes, images, text, and capturing user input events.

    This class is designed to be used with graphical environments like PsychoPy
    for psychology experiments and PyGame for general-purpose game development
    or graphical applications. It defines methods for drawing geometric shapes,
    images, and text on a window or screen, as well as handling basic user input
    events like mouse positioning and clicks.

    It is intended to be subclassed or implemented with a specific graphical
    context, such as PyGame or PsychoPy, to allow rendering to a window.
    """

    def __init__(self, win):
        """
        Initialize the UI backend.

        Parameters:
            win: The window or graphical context where drawing operations will be performed.
        """
        self.win = win

    def draw_circle(self, x: int, y: int, radius: int, color: Tuple[int, int, int]):
        """
        Draw a circle on the screen.

        Parameters:
            x (int): X-coordinate of the circle center.
            y (int): Y-coordinate of the circle center.
            radius (int): Radius of the circle.
            color (Tuple[int, int, int]): RGB color of the circle (0-255).
        """
        raise NotImplementedError

    def draw_line(self, sx: int, sy: int, ex: int, ey: int, color: Tuple[int, int, int], line_width: int):
        """
        Draw a straight line on the screen.

        Parameters:
            sx (int): X-coordinate of the start point.
            sy (int): Y-coordinate of the start point.
            ex (int): X-coordinate of the end point.
            ey (int): Y-coordinate of the end point.
            color (Tuple[int, int, int]): RGB color of the line (0-255).
            line_width (int): Width of the line in pixels.
        """
        raise NotImplementedError

    def draw_image(self, img, rect: Tuple[int, int, int, int]):
        """
        Draw an image on the screen.

        Parameters:
            img (numpy.ndarray | str): The image to be drawn, either as a NumPy array or a file path.
            rect (Tuple[int, int, int, int]): Position and size of the image (x, y, width, height).
        """
        raise NotImplementedError

    def draw_rect(self, rect: Tuple[int, int, int, int], color: Tuple[int, int, int], line_width: int):
        """
        Draw a rectangle on the screen.

        Parameters:
            rect (Tuple[int, int, int, int]): Position and size of the rectangle (x, y, width, height).
            color (Tuple[int, int, int]): RGB color of the rectangle (0-255).
            line_width (int): Border width in pixels. If line_width = 0, the rectangle is filled with the color.
        """
        raise NotImplementedError

    def draw_text(self, text: str, font_name: str, font_size: int, text_color: Tuple[int, int, int],
                  rect: Tuple[int, int, int, int], align='center'):
        """
        Draw text on the screen.

        Parameters:
            text (str): The text to display.
            font_name (str): Name of the font to use.
            font_size (int): Font size.
            text_color (Tuple[int, int, int]): RGB color of the text (0-255).
            rect (Tuple[int, int, int, int]): Position and size of the text area (x, y, width, height).
            align (str): Text alignment. Options: 'center', 'left', or 'right'.
        """
        raise NotImplementedError

    def get_screen_size(self):
        """
        Get the screen size.

        Returns:
            Tuple[int, int]: Screen width and height (width, height).
        """
        raise NotImplementedError

    def listen_event(self, host, skip_event=False):
        """
        Listen for user input events.

        Parameters:
            host: The event handler object.
            skip_event (bool): If True, skip the event.
        """
        raise NotImplementedError

    def before_draw(self):
        """
        Perform any necessary operations before drawing, such as clearing the screen or setting up the drawing mode.
        """
        raise NotImplementedError

    def after_draw(self):
        """
        Perform any necessary operations after drawing, such as refreshing the screen or committing the drawing buffer.
        """
        raise NotImplementedError

    @staticmethod
    def pos_in_rect(pos, rect):
        """
        Check if a given position is inside a rectangle.

        Parameters:
            pos (Tuple[int, int]): The position (x, y) to check.
            rect (Tuple[int, int, int, int]): The rectangle (x, y, width, height).

        Returns:
            bool: True if the position is inside the rectangle, False otherwise.
        """
        x, y = pos
        rect_x, rect_y, rect_w, rect_h = rect

        return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h

    def get_mouse_pos(self):
        """
        Get the current mouse position.

        Returns:
            Tuple[int, int]: Mouse X and Y coordinates.
        """
        raise NotImplementedError

    def load_sound(self, sound_path: str, sound_id: str):
        """
        LoS a sound effect.

        Parameters:
            sound_path (str): The sound path.
            sound_id (str|int): The sound ID.
        """
        raise NotImplementedError

    def play_sound(self, sound_id):
        """
        Play a sound effect.

        Parameters:
            sound_id (str|int): The sound ID.
        """
        raise NotImplementedError

    def draw_text_on_screen_center(self, text: str, font_name: str, font_size: int, text_color=(0, 0, 0)):
        """Draw multi-line text centered on screen."""
        raise NotImplementedError

    def draw_text_in_bottom_right_corner(self, text: str, font_name: str, font_size: int, text_color=(0, 0, 0)):
        """Draw multi-line text centered on corner."""
        raise NotImplementedError

    def listen_keys(self, key: Tuple):
        """
        Listen for user keys. Return a key if the user responds.
        """
        raise NotImplementedError


class PsychoPyUIBackend(UIBackend):
    def __init__(self, win):
        super().__init__(win)
        self.font_name = "Microsoft YaHei UI Light"
        self.small_font_size = 16
        self.table_font_size = 18

        from psychopy import visual, event, sound

        self.event = event
        self.sound = sound
        # Predefine PsychoPy stimuli
        self.circle_stim = visual.ShapeStim(self.win, vertices='circle', size=(0, 0),
                                            fillColor=None, lineColor=None, colorSpace='rgb255',
                                            units='pix')
        self.line_stim = visual.ShapeStim(self.win, vertices=[(0, 0), (0, 0)],
                                          fillColor=None, lineColor=None, colorSpace='rgb255',
                                          units='pix', closeShape=False)
        self.rect_stim = visual.ShapeStim(self.win, vertices='rectangle', size=(0, 0),
                                          fillColor=None, lineColor=None, colorSpace='rgb255',
                                          units='pix', anchor='top-left')
        self.text_stim = visual.TextStim(self.win, text='', font=self.font_name, color=None, colorSpace='rgb255',
                                         units="pix")
        self.image_stim = visual.ImageStim(self.win, image=None, mask=None, colorSpace='rgb',
                                           units="pix", anchor='top-left')
        self.mouse = self.event.Mouse()
        self.win_unit = self.win.units
        self._image_cache = {}
        self._sound_cache = {}
        pygame.mixer.init()

    def draw_circle(self, x, y, radius, color):
        self.circle_stim.pos = self.pixel_to_psychopy_coordinate(x, y)
        self.circle_stim.size = (2 * radius, 2 * radius)
        self.circle_stim.lineColor = color
        self.circle_stim.fillColor = color
        self.circle_stim.draw()

    def draw_line(self, sx, sy, ex, ey, color, line_width):
        start = self.pixel_to_psychopy_coordinate(sx, sy)
        end = self.pixel_to_psychopy_coordinate(ex, ey)
        self.line_stim.vertices = [start, end]
        self.line_stim.lineColor = color
        self.line_stim.lineWidth = line_width
        self.line_stim.draw()

    def draw_image(self, img, rect: Tuple[int, int, int, int]):
        target_x, target_y, target_w, target_h = rect
        if isinstance(img, str):
            if img not in self._image_cache:
                cv_img = cv2.imread(img)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                self._image_cache[img] = cv_img
            image = self._image_cache[img]
            original_height, original_width = image.shape[:2]
        else:
            image = img
            original_height, original_width = image.shape[:2]

        aspect = original_width / original_height
        if (target_w / target_h) > aspect:
            scaled_h = int(target_h)
            scaled_w = int(scaled_h * aspect)
        else:
            scaled_w = int(target_w)
            scaled_h = int(scaled_w / aspect)

        offset_x = (target_w - scaled_w) // 2
        offset_y = (target_h - scaled_h) // 2

        draw_x = target_x + offset_x
        draw_y = target_y + offset_y

        # image = cv2.flip(image, 0)
        psychopy_pos = self.pixel_to_psychopy_coordinate(draw_x, draw_y)
        self.image_stim.pos = psychopy_pos
        self.image_stim.image = image / 255.0
        self.image_stim.size = (scaled_w, scaled_h)
        self.image_stim.flipVert = True
        self.image_stim.flipHoriz = True
        self.image_stim.draw()

    def draw_rect(self, rect: Tuple[int, int, int, int], color, line_width):
        fill_color = color if line_width == 0 else None
        line_color = color if line_width != 0 else None
        self.rect_stim.pos = self.pixel_to_psychopy_coordinate(rect[0], rect[1])
        self.rect_stim.size = (rect[2], rect[3])
        self.rect_stim.lineColor = line_color
        self.rect_stim.fillColor = fill_color
        self.rect_stim.lineWidth = line_width
        self.rect_stim.draw()

    def draw_text(self, text: str, font_name: str, font_size: int, text_color: Tuple[int, int, int],
                  rect: Tuple[int, int, int, int], align='center'):
        self.text_stim.text = text
        self.text_stim.font = font_name
        self.text_stim.height = font_size
        self.text_stim.color = text_color
        self.text_stim.pos = self.pixel_to_psychopy_coordinate(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
        self.text_stim.alignHoriz = align
        self.text_stim.draw()

    def get_screen_size(self):
        return self.win.size

    def pixel_to_psychopy_coordinate(self, x: int, y: int) -> Tuple:
        """
        Convert  PyGame pixel coordinates to PsychoPy coordinates (-1 to 1).

        Parameters:
            x (int): X-coordinate in pixels.
            y (int): Y-coordinate in pixels.
        Returns:
            tuple: Converted (x', y') in PsychoPy coordinate system (-1 to 1).
        """
        screen_width, screen_height = self.win.size
        x_psychopy = (x - screen_width // 2)
        y_psychopy = -(y - screen_height // 2)
        return x_psychopy, y_psychopy

    def listen_event(self, host, skip_event=False):
        if skip_event:
            return
        if 'space' in self.event.getKeys():
            host.running = False
        pos = self.get_mouse_pos()
        if (hasattr(host, 'stop_button_rect')
                and ((host.stop_button_rect is not None)
                     and self.pos_in_rect(pos, host.stop_button_rect)
                     and self.mouse.getPressed()[0])):
            host.running = False

    def before_draw(self):
        pass

    def after_draw(self):
        self.win.flip()

    def get_mouse_pos(self):
        x, y = self.mouse.getPos()
        screen_width, screen_height = self.win.size
        if self.win_unit == 'pix':
            x_pygame = x + screen_width // 2
            y_pygame = -y + screen_height // 2
        elif self.win_unit == 'norm':
            x_pygame = (x + 1) * (screen_width // 2)
            y_pygame = (1 - y) * (screen_height // 2)
        elif self.win_unit == 'height':
            x_pygame = x * (screen_height // 2) + (screen_width // 2)
            y_pygame = -y * (screen_height // 2) + (screen_height // 2)
        else:
            raise ValueError(f"Unsupported unit: {self.win_unit}")
        return x_pygame, y_pygame

    def load_sound(self, sound_path: str, sound_id):
        sound_file = pygame.mixer.Sound(sound_path)
        self._sound_cache[sound_id] = sound_file

    def play_sound(self, sound_id):
        self._sound_cache[sound_id].play()

    def stop_sound(self, sound_id):
        self._sound_cache[sound_id].stop()

    def draw_text_on_screen_center(self, text: str, font_name: str, font_size: int, text_color=(0, 0, 0)):
        lines = text.split('\n')
        sw, sh = self.get_screen_size()
        line_spacing = int(font_size * 0.2)
        total_h = len(lines) * font_size + (len(lines) - 1) * line_spacing
        start_y = (sh - total_h) // 2

        for idx, line in enumerate(lines):
            y_pos = start_y + idx * (font_size + line_spacing)
            self.draw_text(line, self.font_name, font_size, text_color,
                           (0, y_pos, sw, font_size), align='center')

    def draw_text_in_bottom_right_corner(self, text: str, font_name: str, font_size: int, text_color=(0, 0, 0)):
        lines = text.split('\n')
        sw, sh = self.get_screen_size()

        start_y = int(sh * 0.85)
        subregion_height = sh - start_y

        line_spacing = int(font_size * 0.2)
        total_h = len(lines) * font_size + (len(lines) - 1) * line_spacing

        y_offset = start_y + (subregion_height - total_h) // 2

        for idx, line in enumerate(lines):
            y_pos = y_offset + idx * (font_size + line_spacing)
            self.draw_text(
                text=line,
                font_name=font_name,
                font_size=font_size,
                text_color=text_color,
                rect=(0, y_pos, sw, font_size),
                align='center'
            )

    def listen_keys(self, key: Tuple):
        pressed_keys = self.event.getKeys(keyList=key)
        if pressed_keys:
            return pressed_keys[0]  # Return first matched key
        return None


class PyGameUIBackend(UIBackend):
    def __init__(self, win, bg_color=(255, 255, 255)):
        super().__init__(win)
        self._sound_cache = {}
        self._image_cache = {}
        self.bg_color = bg_color
        pygame.font.init()
        pygame.mixer.init()

    def draw_circle(self, x, y, radius, color):
        pygame.draw.circle(self.win, color, (x, y), radius)

    def draw_line(self, sx, sy, ex, ey, color, line_width):
        pygame.draw.line(self.win, color, (sx, sy), (ex, ey), line_width)

    def draw_image(self, img, rect: Tuple[int, int, int, int]):
        if isinstance(img, str):
            if img not in self._image_cache:
                image = pygame.image.load(img)
                self._image_cache[img] = image
            image = self._image_cache[img]

        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = pygame.surfarray.make_surface(img)

        original_width, original_height = image.get_size()
        target_width, target_height = rect[2], rect[3]

        aspect_ratio = original_width / original_height

        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        scaled_image = pygame.transform.smoothscale(image, (new_width, new_height))
        x = rect[0] + (target_width - new_width) // 2
        y = rect[1] + (target_height - new_height) // 2

        self.win.blit(scaled_image, (x, y))
        # image = pygame.transform.scale(image, (rect[2], rect[3]))
        # self.win.blit(image, (rect[0], rect[1]))

    def draw_rect(self, rect: Tuple[int, int, int, int], color, line_width):
        pygame.draw.rect(self.win, color, rect, line_width)

    def draw_text(self, text: str, font_name: str, font_size: int, text_color: Tuple[int, int, int],
                  rect: Tuple[int, int, int, int], align='center'):
        font = pygame.font.SysFont(font_name, font_size)
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect()
        if align == 'center':
            text_rect.center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
        elif align == 'left':
            text_rect.topleft = (rect[0], rect[1])
        elif align == 'right':
            text_rect.topright = (rect[0] + rect[2], rect[1])
        self.win.blit(text_surface, text_rect)

    def get_screen_size(self):
        return self.win.get_size()

    def before_draw(self):
        self.win.fill(self.bg_color)

    def listen_event(self, host, skip_event=False):
        for event in pygame.event.get():
            if skip_event:
                continue
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if (event.type == pygame.MOUSEBUTTONDOWN
                    and hasattr(host, 'stop_button_rect')
                    and host.stop_button_rect is not None
                    and self.pos_in_rect(event.pos, host.stop_button_rect)):
                host.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    host.running = False

    def listen_keys(self, key: Tuple):
        """
        Check for keyboard presses using PyGame's event system

        Args:
            key: Tuple of pygame key constants to listen for (e.g., (K_SPACE, K_a))

        Returns:
            int: First matching key code pressed, or None if no match
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                if key_name in key:
                    return key_name
        return None

    def after_draw(self):
        pygame.display.flip()

    def get_mouse_pos(self):
        return pygame.mouse.get_pos()

    def load_sound(self, sound_path: str, sound_id):
        sound_file = pygame.mixer.Sound(sound_path)
        self._sound_cache[sound_id] = sound_file

    def play_sound(self, sound_id):
        self._sound_cache[sound_id].play()

    def stop_sound(self, sound_id):
        self._sound_cache[sound_id].stop()

    def draw_text_on_screen_center(self, text: str, font_name: str, font_size: int, text_color=(0, 0, 0)):
        split_lines = text.split('\n')
        sw, sh = self.get_screen_size()
        line_spacing = int(font_size * 0.2)
        total_h = len(split_lines) * font_size + (len(split_lines) - 1) * line_spacing
        start_y = (sh - total_h) // 2

        for idx, line in enumerate(split_lines):
            y_pos = start_y + idx * (font_size + line_spacing)
            self.draw_text(line, font_name, font_size, text_color,
                           (0, y_pos, sw, font_size), align='center')

    def draw_text_in_bottom_right_corner(self, text: str, font_name: str, font_size: int, text_color=(0, 0, 0)):
        split_lines = text.split('\n')
        sw, sh = self.get_screen_size()

        start_y = int(sh * 0.85)
        subregion_height = sh - start_y

        line_spacing = int(font_size * 0.2)
        total_text_height = len(split_lines) * font_size + (len(split_lines) - 1) * line_spacing

        y_offset = start_y + (subregion_height - total_text_height) // 2

        for idx, line in enumerate(split_lines):
            y_pos = y_offset + idx * (font_size + line_spacing)
            self.draw_text(
                text=line,
                font_name=font_name,
                font_size=font_size,
                text_color=text_color,
                rect=(0, y_pos, sw, font_size),
                align='center')
