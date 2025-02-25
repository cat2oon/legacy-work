import math

"""
TODO :
1. fov를 통한 계산
2. 주어진 회전각에 대한 fov를 통한 계산
3. stereo vision (2cam)
"""


def compute_focal_length(distance, real_width, pixels_in_img):
    return pixels_in_img * distance / real_width


def compute_distance(focal_length, real_width, pixels_in_img):
    return focal_length * real_width / pixels_in_img
