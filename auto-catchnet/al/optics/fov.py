import math

from al.maths.angles import *

"""
 (F) focal length (mm)
 - Distance over which initially collimated (parallel) rays are brought to a focus
 - How strongly the system converges or diverges light
 - en.wikipedia.org/wiki/Focal_length
 
 (D) sensor dimension (mm)
 - 
 
 (FoV) field of view (angle of view)
 - en.wikipedia.org/wiki/Angle_of_view
"""


def compute_fov(focal_length, sensor_dimension, as_radian=True):
    fov_in_rad = 2 * math.atan2(sensor_dimension, 2 * focal_length)
    fov = fov_in_rad if as_radian else rad_to_degree(fov_in_rad)
    return fov


def compute_half_fov(focal_length, sensor_dimension, as_radian=True):
    return compute_fov(focal_length, sensor_dimension, as_radian) / 2


def compute_sensor_dimension(focal_length, fov, as_radian_fov=True):
    fov_in_rad = fov if as_radian_fov else degree_to_rad(fov)
    return 2 * focal_length * math.tan(fov_in_rad / 2)
