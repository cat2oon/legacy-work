import numpy as np

# A.K.A Camera Matrix
from al.optics.vector import Vector3D

"""
reference 
 - en.wikipedia.org/wiki/Camera_resectioning
"""


class IntrinsicParameters:
    def __init__(self,
                 focal_length_x,
                 focal_length_y,
                 optical_center_x,
                 optical_center_y,
                 skew_coefficient=0):
        self.fx = focal_length_x
        self.fy = focal_length_y
        self.cx = optical_center_x  # principle point x 주점
        self.cy = optical_center_y  # principle point x 주점
        self.skew_c = skew_coefficient
        self.ip = self._to_matrix()
        self.inv = np.linalg.inv(self.ip)

    @staticmethod
    def from_image_size(fx, fy, img_x, img_y):
        return IntrinsicParameters(fx, fy, img_x / 2, img_y / 2)

    @staticmethod
    def from_camera_matrix(m: np.ndarray):
        return IntrinsicParameters(m[0, 0], m[1, 1], m[0, 2], m[1, 2], m[0, 1])

    def _to_matrix(self):
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])

    def mat(self):
        return self.ip

    def inverse(self):
        return self.inv


class ExtrinsicParameters:
    def __init__(self, rotation_matrix, translation_vector: Vector3D):
        self.r = rotation_matrix
        self.t = translation_vector
        self.ep = self._to_augmented_matrix()

    def _to_augmented_matrix(self):
        return np.column_stack((self.r, self.t.vec())).astype(np.float32)

    def mat(self):
        return self.ep
