import numpy as np

def circle_sample(center, diameter_in, diameter_out, thickness_low, thicness_high):
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
    x_dia_circle = np.random.uniform([0, diameter_in], [2 * np.pi, diameter_out])
    y_dia_circle = np.random.uniform([0, diameter_in], [2 * np.pi, diameter_out])
    x = center + x_dia_circle[1] * np.cos(x_dia_circle[0])
    y = center + y_dia_circle[1] * np.cos(y_dia_circle[0])
    z = np.random.uniform(thickness_low, thicness_high)

    return np.array([x, y, z])
