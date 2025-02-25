import numpy as np

from al.maths.angles import rad_to_degree

"""
[ References ]
- Homogeneous Coordinates 
- pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/homo-coor.html
- blog.daum.net/shksjy/229
- darkpgmr.tistory.com/78
"""


# TODO: 계층 구조로 묶을 것

class Vector3D:
    def __init__(self, x, y, z):
        self.v = np.array([x, y, z], dtype=np.float32)

    @staticmethod
    def from_np_vec(v: np.ndarray):
        return Vector3D(v[0], v[1], v[2])

    @staticmethod
    def wrap(vec):
        if type(vec) is np.ndarray:
            return Vector3D.from_np_vec(vec)
        if type(vec) is Vector3D:
            return vec
        return None

    @staticmethod
    def infer_unit(vec, sign=-1):
        x, y = vec[0], vec[1]
        z = sign * np.sqrt(1 - (x**2 + y**2))
        return Vector3D(x, y, z)

    def homogeneous(self, w=1.0):
        return np.hstack((self.v, w))

    def x(self):
        return self.v[0]

    def y(self):
        return self.v[1]

    def z(self):
        return self.v[2]

    def vec(self):
        return self.v

    def col_vec(self):
        return self.v.T.squeeze()

    def focal_proj(self, proj_z=1):
        v = self.v
        k_z = proj_z / v[2]
        proj_vec = np.array([v[0] * k_z, v[1] * k_z, proj_z], dtype=np.float32)
        return proj_vec

    def l2_normalize(self):
        norm = np.linalg.norm(self.v)
        self.v = self.v / norm

    def to_yaw_pitch(self):
        v = self.v
        x, y, z = v[0], v[1], v[2]
        yaw = np.arctan2(-x, -z)
        pitch = np.arcsin(y)
        yaw_in_degree = rad_to_degree(yaw)
        pitch_in_degree = rad_to_degree(pitch)
        return yaw_in_degree, pitch_in_degree

    def __sub__(self, other):
        v = self.vec() - other.vec()
        return Vector3D.from_np_vec(v)

    def __getitem__(self, idx):
        return self.v[idx]

    def __setitem__(self, idx, value):
        self.v[idx] = value

    def __str__(self):
        return "({:.2f} {:.2f} {:.2f})".format(self.x(), self.y(), self.z())


class Vector2D:
    def __init__(self, x, y):
        self.v = np.array([x, y], dtype=np.float32)

    @staticmethod
    def from_np_vec(v: np.ndarray):
        return Vector2D(v[0], v[1])

    @staticmethod
    def wrap(vec):
        if type(vec) is np.ndarray:
            return Vector2D.from_np_vec(vec)
        if type(vec) is Vector3D:
            return vec
        return None

    def homogeneous(self, w=1.0):
        return np.hstack((self.v, w))

    def vec(self):
        return self.v

    def col_vec(self):
        return self.v.T.squeeze()

    def x(self):
        return self.v[0]

    def y(self):
        return self.v[1]

    def z(self):
        return self.v[2]

    def __str__(self):
        return "({:.2f} {:.2f})".format(self.x(), self.y())

    def __getitem__(self, idx):
        return self.v[idx]

class Vector4D:
    def __init__(self, x, y, z, w=1):
        self.v = np.array([x, y, z, w], dtype=np.float32)

    @staticmethod
    def from_np_vec(v: np.ndarray):
        return Vector4D(v[0], v[1], v[2], v[3])

    def vec(self):
        return self.v

    def homogeneous(self, w=1.0):
        return np.hstack((self.v, w))

    def col_vec(self):
        return self.v.T.squeeze()
