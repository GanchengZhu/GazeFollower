# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import platform

from .UIBackend import UIBackend, PyGameUIBackend, PsychoPyUIBackend


class BaseUI(object):
    """
    Initializes the Base UI.
        win (psychopy.visual.Window|pygame.Surface): The window to use.
        backend_name (str): The name of the backend (PsychoPy) to use for rendering, default is 'PsychoPy'.
        bg_color (Tuple): Background color for pygame screen.
    """
    def __init__(self, win, backend_name: str = "PyGame", bg_color=(255, 255, 255)):
        # backend
        self.backend_name = backend_name.lower()
        if self.backend_name == "pygame":
            self.backend: UIBackend = PyGameUIBackend(win, bg_color=bg_color)
        elif self.backend_name == "psychopy":
            self.backend: UIBackend = PsychoPyUIBackend(win)
        else:
            raise ValueError(
                f"Invalid backend name: {self.backend_name}. We only support backends, `PyGame` and `psychopy`.")
        # font name
        system = platform.system()
        if system == "Windows":
            self.font_name = "microsoftyaheiui" if self.backend_name == "pygame" else "Microsoft YaHei UI"
        elif system == "Darwin":
            self.font_name = "PingFang SC" if self.backend_name == "pygame" else "Helvetica"
        else:
            self.font_name = "WenQuanYi Zen Hei" if self.backend_name == "pygame" else "DejaVu Sans"
        # font size
        self.row_font_size = 18 if self.backend_name == "pygame" else 18
        self.button_font_size = 20 if self.backend_name == "pygame" else 20
        self.image_font_size = 24 if self.backend_name == "pygame" else 24
        self.font_size = 24 if self.backend_name == "pygame" else 24
        # colors
        self._color_white = (255, 255, 255)
        self._color_red = (255, 0, 0)
        self._color_green = (0, 255, 0)
        self._color_black = (0, 0, 0)
        self._color_blue = (0, 0, 255)
        self._color_gray = (127, 127, 127)
