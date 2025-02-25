import numpy as np


def make_ellipse_contours(center=(1, 1), width=1, height=.6, phi=np.pi / 5):
    """ Generate Elliptical data with noise """
    t = np.linspace(0, 2 * np.pi, 1000)
    x_noise, y_noise = np.random.rand(2, len(t))

    ellipse_x = center[0] + width * np.cos(t) * np.cos(phi) - height * np.sin(t) * np.sin(phi) + x_noise / 2.
    ellipse_y = center[1] + width * np.cos(t) * np.sin(phi) + height * np.sin(t) * np.cos(phi) + y_noise / 2.

    return [ellipse_x, ellipse_y]


def is_inside(px, py, cx, cy, width, height, phi):
    """
    Rotated ellipse eq.
    ((x-c1)*cos(phi) - (y-c2)*sin(phi))^2      ((x-c1)*sin(phi) - (y-c2)*cos(phi))^2
    -------------------------------------  +  --------------------------------------  = 1
                     a^2                                          b^2

    for center test:
        return cx - 2 < px < cx + 2 and cy - 2 < py < cy + 2
    """

    term_a = ((px - cx) * np.cos(phi) + (py - cy) * np.sin(phi)) ** 2 / width ** 2
    term_b = ((px - cx) * np.sin(phi) - (py - cy) * np.cos(phi)) ** 2 / height ** 2
    return term_a + term_b < 1


def centric_score(px, py, ellipse_param):
    cx, cy, width, height, phi = ellipse_param
    term_a = ((px - cx) * np.cos(phi) + (py - cy) * np.sin(phi)) ** 2 / width ** 2
    term_b = ((px - cx) * np.sin(phi) - (py - cy) * np.cos(phi)) ** 2 / height ** 2
    return term_a + term_b < 1

# TODO: 나중에 모두 GeoEllipse
# class GeoEllipse:
#     def __init__(self):
#         pass
#
#     # makes contours n points
#     # get point theta
