from ac.visualizer.plotter import render_head_pose
from ai.predictor.looks.null import NullPredictor
from ai.predictor.looks.optical_axis import OpticalAxisPredictor
from al.actor.camera import Camera
from al.model.face.face import FaceModel
from al.optics.transformation.projections import calculate_target_pos
from al.optics.vector import *


class FaceActor:
    def __init__(self,
                 camera: Camera,
                 face_model: FaceModel,
                 opt_predictor: OpticalAxisPredictor,
                 reference_intercanthal_width=38,
                 reference_fissure_length=25):
        self.frame = None
        self.l_eye_img = None
        self.r_eye_img = None

        self.l_opt_vec = None
        self.r_opt_vec = None

        """ reference property """
        self.z_depth = None
        self.l_target_gaze = None
        self.r_target_gaze = None
        self.head_pose_in_degree = None
        self.l_eye_center_in_px = None
        self.r_eye_center_in_px = None
        self.l_eye_z_depth_in_mm = None
        self.r_eye_z_depth_in_mm = None
        self.l_eye_3d_pos_in_mm = None
        self.r_eye_3d_pos_in_mm = None

        """ reference facial """
        self.facial_intercanthal_width_in_px = None
        self.facial_l_fissure_length_in_px = None
        self.facial_r_fissure_length_in_px = None

        """ standard facial stat """
        # TODO: encapsulate mean facial stat (with statistics) + profile_facial
        self.ref_fissure_length = reference_fissure_length
        self.ref_intercanthal_width = reference_intercanthal_width

        self.camera = camera
        self.model = face_model
        self.opt_predictor = opt_predictor
        if self.opt_predictor is None:
            self.opt_predictor = NullPredictor()

    """
    - Core API(s)
    """

    def match(self, face_img):
        self.frame = face_img
        self.model.match(face_img)

    def analysis(self):
        """ Notice! Order is matter """
        """ 눈 영역 이미지 """
        self.crop_eye_img()

        """ 양안 중심점 """
        l_eye_center, r_eye_center = self.get_eye_center_by_landmarks()
        self.set_reference_eye_center(l_eye_center, r_eye_center)

        """ 얼굴 카메라 z-depth 거리 """
        ic_depth, l_eye_depth, r_eye_depth = self.compute_depth()
        self.set_reference_z_depth(ic_depth, l_eye_depth, r_eye_depth)

        """ 머리 포즈 계산 """
        head_pose = self.compute_head_pose()
        self.set_reference_head_pose(head_pose)

        """ 양안 이미지 정규화 (perspective warp, unroll) """
        self.normalize_eye_img()

        """ 안축 예측 """
        l_eye_opt, r_eye_opt = self.predict_optical_vector()
        self.set_reference_eye_opt_vec(l_eye_opt, r_eye_opt)

        """ 양안 3d 위치 """
        l_eye_3d_pos, r_eye_3d_pos = self.get_eye_center_in_camera_coord()
        self.set_reference_eye_center_3d_pos(l_eye_3d_pos, r_eye_3d_pos)

    def compute_gaze_target_in_px(self, lk_vec: Vector3D, rk_vec: Vector3D):
        l_target_in_mm, r_target_in_mm = self.compute_gaze_target_pos_in_mm(lk_vec, rk_vec)
        px, py = self.camera.get_pixel_point_from_physical_pos(l_target_in_mm, r_target_in_mm)
        return px, py

    # TODO displacement 추가하기 - 현재는 candide 눈 중심이 부정확하여 k_vector 만 사용
    def compute_gaze_target_pos_in_mm(self, lk_vec: Vector3D, rk_vec: Vector3D):
        l_gaze_vec = self.l_opt_vec - lk_vec
        r_gaze_vec = self.r_opt_vec - rk_vec
        l_target, r_target = self.compute_target_pos_by(l_gaze_vec, r_gaze_vec)
        self.set_reference_target_by_gaze(l_target, r_target)

        return l_target, r_target

    """
    - Functionality API(s)
    """

    def get_eye_center_by_landmarks(self):
        return self.model.get_eye_center_pos()

    def crop_eye_img(self, image_shape=None):
        if image_shape is None:
            image_shape = self.opt_predictor.get_required_image_shape()
        height, width = image_shape[0], image_shape[1]
        l_rect, r_rect = self.model.get_eye_crop_rect(height, width)
        self.l_eye_img = l_rect.crop_image(self.frame)
        self.r_eye_img = r_rect.crop_image(self.frame)

    def normalize_eye_img(self):
        # projection.perspective_normalized(img, head_pose)
        # 처리 후 각도를 받아와야 optimize vec과 kappa 보정이 가능
        pass

    def compute_depth(self):
        # TODO: 3d rotation 펼친 landmark 길이로 따지기
        ic_in_px = self.model.get_intercanthal_width()
        l_fs_in_px, r_fs_in_px = self.model.get_fissure_lengths()

        self.set_facial_intercanthal_width(ic_in_px)
        self.set_facial_fissure_lengths(l_fs_in_px, r_fs_in_px)

        ic_width_in_mm = self.ref_intercanthal_width
        fs_length_in_mm = self.ref_fissure_length

        camera = self.camera
        ic_depth = camera.compute_z_depth_by_width(ic_width_in_mm, ic_in_px)
        l_eye_depth = camera.compute_z_depth_by_width(fs_length_in_mm, l_fs_in_px)
        r_eye_depth = camera.compute_z_depth_by_width(fs_length_in_mm, r_fs_in_px)

        return ic_depth, l_eye_depth, r_eye_depth

    def compute_head_pose(self):
        return self.model.get_head_pose()

    def predict_optical_vector(self):
        head_pose = self.head_pose_in_degree
        l_eye_img, r_eye_img = self.l_eye_img, self.r_eye_img
        r_eye_opt = self.opt_predictor.predict_img(r_eye_img, head_pose)
        l_eye_opt = self.opt_predictor.predict_img(l_eye_img, head_pose)
        return l_eye_opt, r_eye_opt

    def compute_target_pos_by(self, l_ray_vec: Vector3D, r_ray_vec: Vector3D):
        l_target_pos = calculate_target_pos(self.l_eye_3d_pos_in_mm, l_ray_vec)
        r_target_pos = calculate_target_pos(self.r_eye_3d_pos_in_mm, r_ray_vec)
        return l_target_pos, r_target_pos

    def compute_target_pos_by_opt(self):
        return self.compute_target_pos_by(self.l_opt_vec, self.r_opt_vec)

    """
    - Accessor / Mutators / Miscellaneous
    """

    def get_references(self):
        return {
            "l_opt": self.l_opt_vec,
            "r_opt": self.r_opt_vec,
            "l_tar": self.l_target_gaze,
            "r_tar": self.r_target_gaze,
            "l_pos": self.l_eye_3d_pos_in_mm,
            "r_pos": self.r_eye_3d_pos_in_mm,
            "head_pose": self.head_pose_in_degree,
            "l_fissure": self.facial_l_fissure_length_in_px,
            "r_fissure": self.facial_r_fissure_length_in_px,
            "intercanthal": self.facial_intercanthal_width_in_px,
        }

    def set_reference_head_pose(self, head_pose):
        self.head_pose_in_degree = np.array(head_pose)

    def set_reference_z_depth(self, z_depth, l_eye_z_depth, r_eye_z_depth):
        self.z_depth = z_depth
        self.l_eye_z_depth_in_mm = l_eye_z_depth
        self.r_eye_z_depth_in_mm = r_eye_z_depth

    def set_reference_eye_opt_vec(self, l_eye_opt, r_eye_opt):
        self.l_opt_vec = Vector3D.wrap(l_eye_opt)
        self.r_opt_vec = Vector3D.wrap(r_eye_opt)

    def set_reference_eye_center(self, l_center_in_px, r_center_in_px):
        self.l_eye_center_in_px = l_center_in_px
        self.r_eye_center_in_px = r_center_in_px

    def set_reference_eye_center_3d_pos(self, l_eye_3d_pos, r_eye_3d_pos):
        self.l_eye_3d_pos_in_mm = Vector3D.wrap(l_eye_3d_pos)
        self.r_eye_3d_pos_in_mm = Vector3D.wrap(r_eye_3d_pos)

    def set_reference_target_by_gaze(self, l_target_gaze, r_target_gaze):
        self.l_target_gaze = l_target_gaze
        self.r_target_gaze = r_target_gaze

    def set_facial_intercanthal_width(self, intercanthal_width):
        self.facial_intercanthal_width_in_px = intercanthal_width

    def set_facial_fissure_lengths(self, l_fissure, r_fissure):
        self.facial_l_fissure_length_in_px = l_fissure
        self.facial_r_fissure_length_in_px = r_fissure

    def get_eye_center_in_camera_coord(self):
        camera = self.camera
        l_depth, r_depth = self.l_eye_z_depth_in_mm, self.r_eye_z_depth_in_mm
        l_center, r_center = self.l_eye_center_in_px, self.r_eye_center_in_px
        l_eye_3d_pos_in_mm = camera.get_physical_pos_from_pixel_point(l_center, l_depth)
        r_eye_3d_pos_in_mm = camera.get_physical_pos_from_pixel_point(r_center, r_depth)

        return l_eye_3d_pos_in_mm, r_eye_3d_pos_in_mm

    def get_eye_imgs(self, preprocessor=None):
        if preprocessor is None:
            return self.l_eye_img, self.r_eye_img
        return preprocessor(self.l_eye_img, self.r_eye_img)

    def get_frame(self, show_head_pose=False):
        if show_head_pose:
            return render_head_pose(self.frame, *self.head_pose_in_degree)
        return self.frame

    """
    - Override
    """

    def __str__(self):
        return "head-pose ({}) lr-optimize ({}, {})".format(self.head_pose_in_degree, self.l_opt_vec, self.r_opt_vec)
