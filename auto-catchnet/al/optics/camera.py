import math

# TODO: actor의 카메라 알고리즘들을 이쪽으로

"""
    fx = focal length of x (px)
    fy = focal length of y (py)
    mx = 
"""


class Camera:
    """
    1. 12.3 MP, f/2.0, 26mm (wide), 1/2.3", 1.55µm
    """

    def __init__(self):
        self.intrinsic_params = None
        self.extrinsic_params = None
        self.focal_length_in_mm = None


""" 
in1. focal length in pixels
in2. focal length in mm
out. pixels per mm
"""


def get_pixels_per_mm(fl_in_pixels, fl_in_mm):
    pass
