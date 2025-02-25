import numpy as np
import numpy.linalg as la

from ac.common.nps import normalize
from ac.langs.decorator.bases import Reference

Reference("math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d")
Reference("gamedev.stackexchange.com/questions/20097/how-to-calculate-a-3x3-rotation-matrix-from-2-direction-vectors")
Reference("stackoverflow.com/questions/23166898/efficient-way-to-calculate-a-3x3-rotation-matrix-from-the-rotation-defined-by-tw")


def rotate_vec_as_vec_x_to_vec_y(vec_x, vec_y, vec):
    # TODO: camera coordinate 축에 맞추어 customize 해야 함
    """ Rodrigues rotation formula """
    I = np.eye(3)
    vec_x = normalize(vec_x)
    vec_y = normalize(vec_y)

    vec = np.array(vec)
    axis = normalize(np.cross(vec_x, vec_y))
    angle = angle_between_two_vector(vec_x, vec_y)
    exv = np.cross(axis, vec)
    edv = axis @ vec
    rotated_vec = (vec * np.cos(angle)) + (exv * np.sin(angle)) + (axis * edv * (1 - np.cos(angle)))

    return rotated_vec


def angle_between_two_vector(vec_a, vec_b):
    cos_angle = vec_a @ vec_b
    sin_angle = la.norm(np.cross(vec_a, vec_b))
    return np.arctan2(sin_angle, cos_angle)


"""
legacy 

if np.array_equal(x, y):
    return I
elif np.array_equal(x, -y):
    return -I
        
v = np.cross(x, y)
s = np.linalg.norm(v)
c = x @ y
k = 1 / (1 + c)

V_x = np.matrix([
    [0, -v[2], v[1]],
    [v[2], 0, -v[1]],
    [-v[1], v[0], 0]
])

# k = 1 / (1 + c)
# R = I + V_x + k * (V_x)^2 

R = I + V_x + (k * (V_x @ V_x))
"""
