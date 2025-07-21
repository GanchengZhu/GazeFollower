# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from .DefaultConfig import *
from .Enumeration import *
from .FaceInfo import *
from .GazeInfo import *
from .Recorder import *


def clip_patch(frame, rect):
    """
    Clip a region from the frame based on the provided rectangle.

    :param frame: The input image frame as a NumPy array (height x width x channels).
    :param rect: A tuple (x, y, w, h) defining the rectangle to clip.
    :return: A NumPy array representing the clipped image patch, or None if the rectangle is invalid.
    """
    x, y, w, h = rect

    # Check for valid rectangle dimensions
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None

    # Check if the rectangle is within the frame bounds
    if x >= frame.shape[1] or y >= frame.shape[0]:
        return None

    x_end = min(x + w, frame.shape[1])
    y_end = min(y + h, frame.shape[0])

    # Clip the region from the frame
    clipped_patch = frame[y:y_end, x:x_end].copy()

    return clipped_patch


def px2cm(px_pos,
          cam_pos: Tuple[float, float],
          physical_screen_size: Tuple[float, float],
          px_screen_size: Tuple[float, float]) -> Tuple[float, float]:
    """
    Convert pixel coordinates to centimeter coordinates relative to camera position.

    Args:
        px_pos: Pixel coordinates to convert (x, y) in pixels
        cam_pos: Camera center position in centimeters (x, y)
        physical_screen_size: Physical screen dimensions (width, height) in centimeters
        px_screen_size: Screen resolution in pixels (width, height)

    Returns:
        Tuple[float, float]: Converted coordinates in centimeters (x, y)

    Example:
        >>> px2cm((960, 540), (15.0, 10.0), (30.0, 20.0), (1920, 1080))
        (0.0, 0.0)  # When camera is centered
    """
    # Calculate DPI (dots per inch) for both axes
    dpi_x = px_screen_size[0] / (physical_screen_size[0] / 2.54)
    dpi_y = px_screen_size[1] / (physical_screen_size[1] / 2.54)

    # Convert pixel coordinates to centimeters
    cm_x = px_pos[0] * 2.54 / dpi_x - cam_pos[0]
    cm_y = (px_pos[1] * 2.54 / dpi_y - cam_pos[1]) * (-1)

    return cm_x, cm_y


def cm2px(cm_pos,
          cam_pos: Tuple[float, float],
          physical_screen_size: Tuple[float, float],
          px_screen_size: Tuple[float, float]) -> Tuple[float, float]:
    """
    Convert centimeter coordinates to pixel coordinates relative to camera position.

    Args:
        cm_pos: Centimeter coordinates to convert (x, y) in cm
        cam_pos: Camera center position in centimeters (x, y)
        physical_screen_size: Physical screen dimensions (width, height) in centimeters
        px_screen_size: Screen resolution in pixels (width, height)

    Returns:
        Tuple[float, float]: Converted coordinates in pixels (x, y)

    Example:
        >>> cm2px((0.0, 0.0), (15.0, 10.0), (30.0, 20.0), (1920, 1080))
    """
    # Calculate DPI (dots per inch) for both axes
    dpi_x = px_screen_size[0] / (physical_screen_size[0] / 2.54)
    dpi_y = px_screen_size[1] / (physical_screen_size[1] / 2.54)

    # Convert centimeter coordinates to pixels
    px_x = (cm_pos[0] + cam_pos[0]) * dpi_x / 2.54
    px_y = (-cm_pos[1] + cam_pos[1]) * dpi_y / 2.54

    return px_x, px_y


def generate_points():
    """
    Generates a grid of normalized (x, y) points within a drawable area of a screen.

    The drawable area is defined by subtracting margins from the screen edges.
    The points are arranged in a grid with a specified number of rows and columns.

    Returns:
        points (ndarray): An array of (x, y) coordinates in normalized screen space.
    """
    # Define margins (in pixels) to be applied on both sides of the screen.
    x_margin = 50
    y_margin = 50

    # Calculate the drawable area's width and height as a fraction of the total screen dimensions.
    width = (1920 - x_margin * 2) / 1920
    height = (1080 - y_margin * 2) / 1080

    # Set the number of grid rows and columns.
    rows = 5
    cols = 9

    # Compute the starting offset (normalized) based on the margins.
    x_start = x_margin / 1920
    y_start = y_margin / 1080

    # Generate evenly spaced x and y coordinates across the drawable area.
    x_coordinates = np.linspace(0, width, cols)
    y_coordinates = np.linspace(0, height, rows)

    # Create a 2D meshgrid from the x and y coordinates.
    X, Y = np.meshgrid(x_coordinates, y_coordinates)
    # Flatten the grid and combine x and y coordinates into pairs.
    mesh = np.column_stack((X.flatten(), Y.flatten()))

    # Offset the mesh grid by the starting margins to get the final normalized positions.
    points = mesh + np.array([x_start, y_start])
    return points
