# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com
from pathlib import Path

import pygame

from .BaseUI import BaseUI
from ..misc import FaceInfo


class CameraPreviewerUI(BaseUI):
    """
    Previewer for the camera device.
    """

    def __init__(self, win, backend_name: str = "PyGame", bg_color=(255, 255, 255)):
        """
        Initializes the CameraPreviewerUI with a specified backend.

        Parameters:
            win (psychopy.visual.Window|pygame.Surface): The window to use.
            backend_name (str): The name of the backend (PsychoPy) to use for rendering, default is 'PsychoPy'.
            bg_color (Tuple): Background color for pygame screen.
        """
        super().__init__(win, backend_name, bg_color)
        # layout margin
        self._margin = 25
        # face size and eye size
        self._face_image_size = (400, 400)
        self._eye_image_size = (400, 200)
        # tip text size
        self._tip_text_size = (self._face_image_size[0] + 2 * self._margin, 40)
        # image frame rect
        self.frame_rect = (self._margin * 2, self._margin * 2 + self._tip_text_size[1],
                           *self._face_image_size)
        # face and eye patch rectangles
        self.face_rect = (self._margin * 4 + self._face_image_size[0],
                          self._margin * 2 + self._tip_text_size[1], *self._face_image_size)
        self.right_eye_rect = (self._margin * 2,
                               self._face_image_size[1] + self._margin * 4 + self._tip_text_size[1] * 2,
                               *self._eye_image_size)
        self.left_eye_rect = (self._margin * 4 + self._face_image_size[0],
                              self._face_image_size[1] + self._margin * 4 + self._tip_text_size[1] * 2,
                              *self._eye_image_size)
        # text
        self._frame_text = 'Frame Image'
        self._face_text = 'Face Image'
        self._left_eye_text = 'Left Eye Image'
        self._right_eye_text = 'Right Eye Image'
        # exit button rectangle
        self.exit_button_rect = None
        # define text rectangles based on the surfaces
        self._frame_text_rect = (self._margin, self._margin,
                                 self._face_image_size[0] + self._margin, self._tip_text_size[1])
        self._face_text_rect = (self._margin * 3 + self._face_image_size[0], self._margin,
                                self._face_image_size[0] + 2 * self._margin, self._tip_text_size[1])
        self._left_eye_text_rect = (self._margin,
                                    self._face_image_size[1] + self._margin * 3 + self._tip_text_size[1],
                                    self._face_image_size[0] + 2 * self._margin, self._tip_text_size[1])
        self._right_eye_text_rect = (self._margin * 3 + self._face_image_size[0],
                                     self._face_image_size[1] + self._margin * 3 + self._tip_text_size[1],
                                     self._face_image_size[0] + 2 * self._margin, self._tip_text_size[1])
        # update _rect_list using the calculated sizes and margins
        self._rect_list = [
            (self._margin, self._margin, self._face_image_size[0] + 2 * self._margin, self._tip_text_size[1]),  # 0
            (self._margin * 3 + self._face_image_size[0], self._margin, self._face_image_size[0] + 2 * self._margin,
             self._tip_text_size[1]),  # 1
            (self._margin, self._margin + self._tip_text_size[1], self._face_image_size[0] + 2 * self._margin,
             self._face_image_size[1] + 2 * self._margin),  # 2
            (self._margin * 3 + self._face_image_size[0], self._margin + self._tip_text_size[1],
             self._face_image_size[0] + 2 * self._margin, self._face_image_size[1] + 2 * self._margin),  # 3
            (self._margin, self._margin * 3 + self._tip_text_size[1] * 1 + self._face_image_size[1],
             self._eye_image_size[0] + 2 * self._margin, self._tip_text_size[1]),  # 4
            (self._margin * 3 + self._face_image_size[0],
             self._margin * 3 + self._tip_text_size[1] * 1 + self._face_image_size[1],
             self._eye_image_size[0] + 2 * self._margin, self._tip_text_size[1]),  # 5
            (self._margin,
             self._margin * 3 + self._tip_text_size[1] * 2 + self._face_image_size[1],
             self._eye_image_size[0] + 2 * self._margin, self._eye_image_size[1] + self._margin * 2),  # 6
            (self._margin * 3 + self._face_image_size[0],
             self._margin * 3 + self._tip_text_size[1] * 2 + self._face_image_size[1],
             self._eye_image_size[0] + 2 * self._margin, self._eye_image_size[1] + 2 * self._margin),  # 7
            (self._margin - 1, self._margin - 1,
             self._face_image_size[0] * 2 + 4 * self._margin,
             self._face_image_size[1] + self._eye_image_size[1] + 2 * self._tip_text_size[1] + 4 * self._margin,),  # 8
            # 8
        ]

        face_rect_width = self.face_rect[2]
        self._table_left_top_position = (self._margin * 7 + face_rect_width * 2, self._margin * 2)

        # constants for UI element
        self._table_row_height = 50  # Define row height here
        self._table_column_width = 180  # Width of each column
        self._table_width = 2 * self._table_column_width  # Total width for two columns
        self._table_even_color = (230, 230, 230)
        self._table_odd_color = (250, 250, 250)
        self._table_text_color = self._color_black
        self._table_line_color = (150, 150, 150)  # Darker gray for clearer line visibility

        self._layout_width = (self._face_image_size[0] + self._eye_image_size[0]
                              + 9 * self._margin + self._table_column_width * 2)
        self._layout_height = (self._tip_text_size[1] * 2 + self._face_image_size[1] + self._eye_image_size[1]
                               + 6 * self._margin)
        self._layout_start_x = None
        self._layout_start_y = None
        self.stop_button_rect = None

        _current_dir = Path(__file__).parent.absolute()
        _package_dir = _current_dir.parent
        _image_asset_dir = _package_dir / "res" / "image"
        self.frame_image = str(_image_asset_dir / "frame.jpg")
        self.face_image = str(_image_asset_dir / "face.jpg")
        self.left_eye_image = str(_image_asset_dir / "left_eye.jpg")
        self.right_eye_image = str(_image_asset_dir / "right_eye.jpg")
        self._icon_path = str(_image_asset_dir / 'gazefollower.png')

        # button color and button hover color
        self._button_color = (0, 123, 255)
        self._button_hover_color = (0, 86, 179)
        # button text
        self._button_text = "Stop Previewing (Tap `Space`)"
        # button size
        self._button_size = (self._table_width + 2 * self._margin, 48)

        self.face_info_dict = {}
        # Graphics UI Running
        self.running = True
        # self.main_loop()

    def draw_rounded_button(self, rect):
        """
        Draws a button with rounded corners.

        Attributes:
            rect (): x, y, width, height
        """
        mouse_pos = self.backend.get_mouse_pos()
        is_pos_in_rect = self.backend.pos_in_rect(mouse_pos, rect)
        button_color = self._button_hover_color if is_pos_in_rect else self._button_color
        self.backend.draw_rect(rect, button_color, 0)
        self.backend.draw_text(self._button_text, self.font_name, self.button_font_size,
                               text_color=self._color_white, rect=rect)

    def update_face_info(self, face_info):
        # Example face info dictionary; replace with actual data
        self.face_info_dict = face_info.to_dict()

    def draw_table(self, data, start_pos):
        """Draws a table with alternating row colors, borders, and better alignment.
        Attributes:
            data (dict): Table rows.
            start_pos (tuple): Table left top position.
        """
        x, y = start_pos
        _num_rows = len(data)

        for i, (key, value) in enumerate(data.items()):
            row_rect = (x, y + i * self._table_row_height, self._table_column_width * 2,
                        self._table_row_height)
            row_bg_color = self._table_even_color if i % 2 == 0 else self._table_odd_color
            self.backend.draw_rect(row_rect, row_bg_color, line_width=0)

            item_key_rect = (x + 5, y + i * self._table_row_height, self._table_column_width - 10,
                             self._table_row_height)
            self.backend.draw_text(f'{key}', self.font_name, self.row_font_size,
                                   text_color=self._color_black, rect=item_key_rect, align='center')

            item_value_rect = (x + self._table_column_width + 5, y + i * self._table_row_height,
                               self._table_column_width - 10, self._table_row_height)
            self.backend.draw_text(f'{value}', self.font_name, self.row_font_size,
                                   text_color=self._color_black, rect=item_value_rect, align='center')

        _rect = (x - self._margin, y - self._margin, self._table_width + 2 * self._margin,
                 self._table_row_height * _num_rows + 5 * self._margin)
        self.backend.draw_rect(_rect, self._color_black, line_width=2)

        _button_boundary_width = self._table_width
        _button_boundary_height = self._margin * 2
        _button_boundary_x = x
        _button_boundary_y = self._table_row_height * _num_rows + y + self._margin
        self.stop_button_rect = (_button_boundary_x, _button_boundary_y,
                                 _button_boundary_width, _button_boundary_height)
        self.draw_rounded_button(self.stop_button_rect)

    def _shifting_layout(self, rect):
        """

        """
        return (self._layout_start_x + rect[0],
                self._layout_start_y + rect[1],
                rect[2], rect[3])

    def draw(self):
        """
        Draw content on the screen
        """
        _screen_width, _screen_height = self.backend.get_screen_size()

        self._layout_start_x = (_screen_width - self._layout_width) / 2
        self._layout_start_y = (_screen_height - self._layout_height) / 2

        # shifting the layout
        self.frame_rect = self._shifting_layout(self.frame_rect)
        self.face_rect = self._shifting_layout(self.face_rect)
        self.left_eye_rect = self._shifting_layout(self.left_eye_rect)
        self.right_eye_rect = self._shifting_layout(self.right_eye_rect)

        self._frame_text_rect = self._shifting_layout(self._frame_text_rect)
        self._face_text_rect = self._shifting_layout(self._face_text_rect)
        self._left_eye_text_rect = self._shifting_layout(self._left_eye_text_rect)
        self._right_eye_text_rect = self._shifting_layout(self._right_eye_text_rect)
        _tmp_x, _tmp_y = self._table_left_top_position
        self._table_left_top_position = (self._layout_start_x + _tmp_x, self._layout_start_y + _tmp_y)

        self._rect_list = [self._shifting_layout(pygame.Rect(i)) for i in self._rect_list]
        self.update_face_info(FaceInfo())

        while self.running:
            # listen event
            self.backend.listen_event(self)
            # for pygame
            self.backend.before_draw()
            # draw image previewer
            self.backend.draw_image(self.frame_image, self.frame_rect)
            self.backend.draw_image(self.face_image, self.face_rect)
            self.backend.draw_image(self.left_eye_image, self.left_eye_rect)
            self.backend.draw_image(self.right_eye_image, self.right_eye_rect)
            # draw texts
            self.backend.draw_text(self._frame_text, self.font_name, self.image_font_size,
                                   text_color=self._color_black, rect=self._frame_text_rect)
            self.backend.draw_text(self._face_text, self.font_name, self.image_font_size,
                                   text_color=self._color_black, rect=self._face_text_rect)
            self.backend.draw_text(self._left_eye_text, self.font_name, self.image_font_size,
                                   text_color=self._color_black, rect=self._left_eye_text_rect)
            self.backend.draw_text(self._right_eye_text, self.font_name, self.image_font_size,
                                   text_color=self._color_black, rect=self._right_eye_text_rect)
            # Draw face info table
            self.draw_table(self.face_info_dict, self._table_left_top_position)
            # Draw grid
            self.draw_grid_rect()
            # draw information table
            self.draw_table(self.face_info_dict, self._table_left_top_position)
            # flip the screen
            self.backend.after_draw()

    def draw_grid_rect(self):
        """
        Draw grid rects
        """
        for _rect in self._rect_list:
            self.backend.draw_rect(_rect, self._color_black, 1)

    def update_images(self, frame_image, face_image, left_eye_image, right_eye_image):
        """
        update the previewer
        :param frame_image: raw image from camera
        :param face_image: facial image
        :param left_eye_image: left eye image
        :param right_eye_image: right eye image
        :return:
        """
        if frame_image is not None:
            self.frame_image = frame_image

        if face_image is not None:
            self.face_image = face_image

        if left_eye_image is not None:
            self.left_eye_image = left_eye_image

        if right_eye_image is not None:
            self.right_eye_image = right_eye_image
