import numpy as np


def degree_to_rad(degree):
    return degree * np.pi / 180.0


def degrees_to_rads(*degrees):
    return map(degree_to_rad, degrees)


def rad_to_degree(radian):
    return radian * 180 / np.pi


def unit_vec_to_angles(vec):
    x, y, z = vec
    a, b, r = np.arccos(x), np.arccos(y), np.arccos(z)
    a, b, r = rad_to_degree(a), rad_to_degree(b), rad_to_degree(r)
    return 90-a, 90-b, 90-r
