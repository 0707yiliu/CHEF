import numpy as np


def circle_sample(center_x, center_y, diameter_in, diameter_out, thickness_low, thicness_high):
    """
    create the circle shape range for target point sampling for robot
    Args:
        center: the center of the circle, depended on the base pos of the robot
        diameter_in: the inside diameter of the circle
        diameter_out: the outside diameter of the circle
        thickness: the thickness in Z direction

    Returns:
        3-dim sample point
    """
    # generate diameter and circle shape like [dia, circle-shape] for x and y
    x_y_dia_circle = np.random.uniform([0, diameter_in], [np.pi, diameter_out])

    x = center_x + x_y_dia_circle[1] * np.cos(x_y_dia_circle[0])
    y = center_y + x_y_dia_circle[1] * np.sin(x_y_dia_circle[0])
    z = np.random.uniform(thickness_low, thicness_high)

    return np.array([x, y, z])


def _normalization(data, _max, _min):
    if type(data) is not type(np.array([])):
        data = np.array(data)
    if type(_max) is not type(np.array([])):
        _max = np.array(_max)
        _min = np.array(_min)
    _range = _max - _min
    return (data - _min) / _range

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)