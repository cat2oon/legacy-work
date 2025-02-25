import numpy.linalg as la

from ac.images.crop_rect import CropRect
from al.model.face.imps.candide import *


# TODO: Landmark 구현체 추상화 (candide, zensetime)
# TODO: Factory FaceModel and FaceActor
# TODO: HeadPose Model 추상화 및 injectable (좌표계 정의도 해당 구현체에서)

class FaceModel:
    def __init__(self, candide: Candide):
        self.candide = candide
        self.landmarks = None

    def match(self, img):
        self.landmarks = self.get_landmarks(img)

    def get_landmarks(self, img):
        landmarks = self.candide.get_facial_points(img)
        landmarks = np.array(landmarks)
        landmarks = landmarks[0, :, :]  # (n,2,68) -> (2,68)
        landmarks = landmarks.T         # (2,68) -> (68, 2)
        return landmarks

    def get_head_pose(self):
        # TODO : 좌표 기준 분리할 것
        def to_unity_head_pose(pitch, yaw, roll):
            u_p = pitch if pitch >= 0 else pitch + 360
            u_y = (-1 * yaw) + 180
            return u_p, u_y, roll

        head_pose = self.candide.landmark_to_head_pose(self.landmarks)
        return to_unity_head_pose(*head_pose)

    def select_both_side_landmarks(self, selector):
        l_indices, r_indices = selector()
        l_marks = self.landmarks.take(l_indices, axis=0)
        r_marks = self.landmarks.take(r_indices, axis=0)
        return l_marks, r_marks

    # TODO : crop_square, crop_rect 통합
    def get_eye_crop_square(self, length=220):
        selector = eye_landmarks_indices
        l_marks, r_marks = self.select_both_side_landmarks(selector)
        l_rect = CropRect.from_points_center_square(l_marks, length)
        r_rect = CropRect.from_points_center_square(r_marks, length)
        return l_rect, r_rect

    def get_eye_crop_rect(self, height_length=38, width_length=60):
        selector = eye_landmarks_indices
        l_marks, r_marks = self.select_both_side_landmarks(selector)
        l_rect = CropRect.from_points_center_rect(l_marks, height_length, width_length)
        r_rect = CropRect.from_points_center_rect(r_marks, height_length, width_length)
        return l_rect, r_rect

    def get_eye_center_pos(self):
        selector = eye_landmarks_for_center_indices
        l_marks, r_marks = self.select_both_side_landmarks(selector)
        l_center = np.average(l_marks, axis=0)
        r_center = np.average(r_marks, axis=0)
        return l_center, r_center

    """
    Facial Stats Measure
    """

    def get_intercanthal_width(self):
        selector = caruncle_landmarks_indices
        l_caruncle, r_caruncle = self.select_both_side_landmarks(selector)
        return la.norm(l_caruncle - r_caruncle)

    def get_fissure_lengths(self):
        selector = eye_end_landmarks_indices
        l_eye_ends, r_eye_ends = self.select_both_side_landmarks(selector)
        l_length = la.norm(l_eye_ends[0] - l_eye_ends[1])
        r_length = la.norm(r_eye_ends[0] - r_eye_ends[1])
        return l_length, r_length


