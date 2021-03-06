import numpy as np
from typing import Tuple, Union


def pixel2meter(coordinates: np.ndarray, dimension: np.ndarray, resolution: float) \
        -> Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]:
    """
    transforms the pixel coordinate of a map image to the corresponding metric dimensions
        NOTE: pixel coordinates start in the upper left corner with (0,0)
              metric coordinates start in the lower left corner with (0,0)

    :param coordinates: pixel coordinates (r, c)
    :param dimension: dimensions of the image in total
    :param resolution: resolution of the map [m] ( 1 pixel = ? meter )

    :return: coordinates: metric coordinates (x, y)
    """

    x = coordinates[1] * resolution  # column * resolution
    y = (dimension[0] - coordinates[0]) * resolution  # ( dim_r - r ) * resolution

    return x, y


def center2corner_pivot(box: Tuple[Tuple[Union[int, float], Union[int, float]],
                                   Tuple[Union[int, float], Union[int, float]],
                                   float]) \
        -> Tuple[Tuple[Union[int, float], Union[int, float]],
                 Tuple[Union[int, float], Union[int, float]],
                 float]:
    """
    transforms a box, defined by x and y of the center, width, height and rotation angle (rotation around center)
    into a box, defined by x and y of the lower corner, width, height and rotation angle (rotation around lower corner)

    :param box: tuple of center (tuple x, y), size (tuple width, height) and angle

    :return: box: tuple of pivot lower left corner (tuple x, y), size (tuple width, height) and angle
    """

    center, size, angle = box
    center_x, center_y = center
    width, height = size
    angle_rad = angle * np.pi / 180
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    pivot_x = center_x - cos_a * width / 2 + sin_a * height / 2
    pivot_y = center_y - cos_a * height / 2 - sin_a * width / 2

    return (pivot_x, pivot_y), (width, height), angle
