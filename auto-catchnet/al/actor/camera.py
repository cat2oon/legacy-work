import numpy as np

from ac.common.nps import normalize
from al.actor.screen import Screen
from al.optics.camera import Camera
from al.optics.parameter import IntrinsicParameters
from al.optics.transformation.projections import pixel_to_3d_camera_coord, pixel_to_normal

"""
reference 
 - @1 en.wikipedia.org/wiki/Camera_resectioning
 - @  stackoverflow.com/questions/14038002
ToDo
 - 캘리브레이션, distortion 계수
"""


class Camera:
    def __init__(self,
                 focal_length_in_mm,
                 intrinsic_param: IntrinsicParameters,  # TODO : calibration 당시 intrinsic
                 reference_screen: Screen):             # Screen은 현재 해상도에 대한 intrinsic이 필요
        self.screen = reference_screen
        self.intrinsic_param = intrinsic_param
        self.focal_length_in_mm = focal_length_in_mm
        self.focal_length_x_in_px = intrinsic_param.fx
        self.focal_length_y_in_px = intrinsic_param.fy
        self.scale_factor_x = intrinsic_param.fx / focal_length_in_mm  # mx in @1
        self.scale_factor_y = intrinsic_param.fy / focal_length_in_mm  # my in @1

    def get_normalized_uv_vec_from_px_position(self, px_x, px_y):
        # TODO: distortion 왜곡 보정한 좌표를 사용할 것

        # 1. pixel (x, y) to (u, v)
        ip = self.intrinsic_param
        uv_vec = pixel_to_normal(px_x, px_y, ip)

        # 2. to normalized vector (정규 이미지 공간이므로 거리는 1)
        vec = np.array([uv_vec.x(), uv_vec.y(), 1])
        normalized_vec = normalize(vec)

        # 4. [option] 기준 좌표 변환

        return normalized_vec

    def get_px_per_mm_in_resolution(self, target_res):
        """
        - TODO: 해상도 x만 사용하면 되는지?? (현대의 센서는 정각 아닌가?)
        - 대상 해상도에서 광학 센서 1mm 당 몇 px을 표현하는지 계산
        """
        calibration_res = 2 * self.intrinsic_param.cx   # ~ resolution when calibrated
        px_per_mm_in_res = self.scale_factor_x * (target_res / calibration_res)
        return px_per_mm_in_res

    def get_sensor_width_covered_in_mm(self, object_in_px):
        """
        - 이미지에 나타난 대상의 픽셀수를 기준으로 광학 센서 몇 mm를 사용하였는지 계산
        """
        target_resolution = self.screen.width
        px_per_mm_in_resolution = self.get_px_per_mm_in_resolution(target_resolution)
        sensor_covered_in_mm = object_in_px / px_per_mm_in_resolution
        return sensor_covered_in_mm

    def compute_z_depth_by_width(self, real_in_mm, object_in_px):
        """
        - 대상의 실제 길이와 이미지 상의 픽셀수가 주어졌을 때 z-depth 계산
        """
        sensor_in_mm = self.get_sensor_width_covered_in_mm(object_in_px)
        z_depth = (real_in_mm * self.focal_length_in_mm) / sensor_in_mm
        return z_depth

    def get_physical_pos_from_pixel_point(self, pixel_in_img, z_depth):
        """
        - 픽셀 공간의 좌표를 3차원으로 reverse projection 수행
        """
        # ip_for_this_screen = self.screen.get_intrinsic_param() # TODO : rename : intrinsic_when_calibrated
        ip_for_this_screen = self.intrinsic_param  # TODO : 원래는 스크린의 intrinsic_param 으로 해야되는 것
        return pixel_to_3d_camera_coord(pixel_in_img, ip_for_this_screen, z_depth)

    def get_pixel_point_from_physical_pos(self, l_point, r_point):
        """
        - 카메라 주점을 원점으로 하는 3차원 점을 (z=0) 픽셀 좌표로 변환
        - 스크린과 카메라의 물리적 거리까지 고려
        """
        s = self.screen
        return (0, 0), (0, 0)

    @staticmethod
    def from_nexus_5x() -> Camera:
        focal_length_in_mm = 2.6

        # TODO: 이상치 및 실제 켈리브레이션 선택 가능하도록
        intrinsic_param = IntrinsicParameters(948, 948, 326, 613)
        # intrinsic_param = IntrinsicParameters(946.3, 950.3, 720/2, 1280/2)     # 이상치로
        reference_screen = Screen(720, 1280)
        cam = Camera(focal_length_in_mm, intrinsic_param, reference_screen)

        return cam


