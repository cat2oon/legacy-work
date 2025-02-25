import math

from al.maths.angles import rad_to_degree


def get_tolerance_degree(depth_in_mm, target_diff_in_mm):
    rad = math.acos(depth_in_mm / math.sqrt(depth_in_mm ** 2 + target_diff_in_mm ** 2))
    return rad_to_degree(rad)
