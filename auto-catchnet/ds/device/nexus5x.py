from al.actor.camera import Camera
from al.actor.screen import Screen
from al.optics.parameter import IntrinsicParameters

__CAM_XY_TABLE = [
    (-22, 66),  # 정중앙
    (10, 10),  # 좌상단
    (10, 66),  # 좌중앙
    (10, 122),  # 좌하단
    (-22, 10),  # 중상단
    (-22, 66),  # 정중앙
    (-22, 122),  # 중하단
    (-53, 10),  # 우상단
    (-53, 66),  # 우중앙
    (-53, 122),  # 우하단
]


def get_true_cam_xy(point_idx):
    if point_idx >= len(__CAM_XY_TABLE):
        return 0, 0
    return __CAM_XY_TABLE[point_idx]


class MobileNexus5X:
    _ppm_w = 1080 / 65
    _ppm_h = 1920 / 115
    _mpp_w = 1 / _ppm_w
    _mpp_h = 1 / _ppm_h
    _lens_to_screen_in_mm = [10, 8.5]

    @staticmethod
    def get_camera():
        screen = Screen(720, 1080)
        intrinsic_param = IntrinsicParameters(946.2, 950.3, 613.0, 326.0)
        camera = Camera(2.6, intrinsic_param, screen)

        return camera

    @staticmethod
    def pixel_to_mm(px, py):
        return [px * -MobileNexus5X._mpp_w, py * MobileNexus5X._mpp_h]

    @staticmethod
    def mm_to_pixel(mx, my):
        return [mx * -MobileNexus5X._ppm_w, my * MobileNexus5X._ppm_h]

    @staticmethod
    def pixel_to_mm_from_lens(px, py):
        x_mm, y_mm = MobileNexus5X.pixel_to_mm(px, py)
        dx_mm, dy_mm = MobileNexus5X._lens_to_screen_in_mm
        return x_mm + dx_mm, y_mm + dy_mm

    @staticmethod
    def mm_to_pixel_from_lens(mx, my):
        dx_mm, dy_mm = MobileNexus5X._lens_to_screen_in_mm
        x_mm, y_mm = mx - dx_mm, my - dy_mm
        return MobileNexus5X.mm_to_pixel(x_mm, y_mm)
