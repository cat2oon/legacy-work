import cv2
import numpy as np


def convert_pose(vec):
    # en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    mat, _ = cv2.Rodrigues(np.array(vec).astype(np.float32))
    x, y, z = mat[:, 2]
    yaw = np.arctan2(x, z)
    pitch = np.arcsin(y)
    return np.array([yaw, pitch])


def convert_gaze(vec):
    x, y, z = vec
    yaw = np.arctan2(-x, -z)
    pitch = np.arcsin(-y)
    return np.array([yaw, pitch])


def point_to_scale(x, length, scale_range=None):
    if scale_range is None:
        scale_range = [-1, 1]
    return (x/length) * scale_range[1] + (1 - (x/length)) * scale_range[0]

