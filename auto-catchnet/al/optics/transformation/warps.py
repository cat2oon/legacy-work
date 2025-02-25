import cv2
import numpy as np

from al.maths.angles import degrees_to_rads

cos, sin = np.cos, np.sin


class Warps:
    def __init__(self):
        pass

    @staticmethod
    def rotate(img, phi=0, theta=0, psi=0, dx=0, dy=0, is_inverse=True):
        width, height = img.shape[1], img.shape[0]
        r_phi, r_theta, r_psi = degrees_to_rads(phi, theta, psi)
        f = np.sqrt(height**2 + width**2) / (2*sin(r_psi) if sin(r_psi) != 0 else 1)
        mat = Warps.get_projection_mat(width, height, r_phi, r_theta, r_psi, dx, dy, f, f)

        flags = cv2.INTER_LINEAR
        if is_inverse:
            flags += cv2.WARP_INVERSE_MAP

        return cv2.warpPerspective(img.copy(), mat, (width, height), flags=flags)

    @staticmethod
    def get_projection_mat(width, height, phi, theta, psi, dx, dy, dz, focal_length=1):
        f, w, h = focal_length, width, height

        # Projection 2D to 3D matrix
        P23 = np.matrix([
            [1, 0, -w / 2],
            [0, 1, -h / 2],
            [0, 0, 1],
            [0, 0, 1]]).astype(np.float64)

        # Rotation matrix
        RX = np.matrix([
            [1, 0, 0, 0],
            [0, cos(phi), -sin(phi), 0],
            [0, sin(phi), cos(phi), 0],
            [0, 0, 0, 1]]).astype(np.float64)

        RY = np.matrix([
            [cos(theta), 0, -sin(theta), 0],
            [0, 1, 0, 0],
            [sin(theta), 0, cos(theta), 0],
            [0, 0, 0, 1]]).astype(np.float64)

        RZ = np.matrix([
            [cos(psi), -sin(psi), 0, 0],
            [sin(psi), cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).astype(np.float64)

        # Composed rotation matrix (X->Y->Z)
        R = RX @ RY @ RZ

        # translation matrix
        T = np.matrix([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]]).astype(np.float64)

        # Projection 3D to 2D matrix
        P32 = np.matrix([
            [f, 0, w / 2, 0],
            [0, f, h / 2, 0],
            [0, 0, 1, 0]]).astype(np.float64)

        return P32 @ T @ R @ P23
