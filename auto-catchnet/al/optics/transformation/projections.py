import cv2

from ac.langs.decorator.bases import Reference
from al.optics.vector import *
from al.optics.parameter import *


#
# projection
#
def z_projected_point(vector: Vector3D, point: Vector3D):
    return calculate_target_pos(point, vector, axis=2)


def calculate_target_pos(pos_from: Vector3D, heading: Vector3D, axis=2) -> Vector3D:
    # TODO: use axis
    k = -1 * (pos_from.z() / heading.z())
    x = pos_from.x() + (k * heading.x())
    y = pos_from.y() + (k * heading.y())
    return Vector3D(x, y, .0)


#
# 3D Coordinate
#
def model_to_pixel(model_point: Vector3D, camera_mat: IntrinsicParameters, mat_rt: ExtrinsicParameters):
    # TODO: Refactor this
    if type(model_point) is not Vector3D:
        model_point = Vector3D.from_np_vec(model_point)
    ax = np.matmul(mat_rt.mat(), model_point.homogeneous())
    bx = np.matmul(camera_mat.mat(), ax)
    cx = Vector3D.from_np_vec(bx)
    return cx.focal_proj()


def world_to_camera(world_vec: Vector3D, rotation_matrix, translation_vec: Vector3D) -> Vector3D:
    if type(world_vec) is not Vector3D:
        world_vec = Vector3D.from_np_vec(world_vec)
    if type(translation_vec) is not Vector3D:
        translation_vec = Vector3D.from_np_vec(translation_vec)
    ep = ExtrinsicParameters(rotation_matrix, translation_vec)
    return world_to_camera_from_params(world_vec, ep)


def world_to_camera_from_params(world_vec: Vector3D, extrinsic_params: ExtrinsicParameters) -> Vector3D:
    camera_vec = np.matmul(extrinsic_params.mat(), world_vec.homogeneous())
    return Vector3D.from_np_vec(camera_vec)


def pixel_to_3d_camera_coord(pixel_in_img, ip: IntrinsicParameters, z_depth_in_mm) -> Vector3D:
    # (x_p - cx) * (Zc / fx) = Xc  | (Zc / fx) == (카메라 좌표에서 실제 Z(깊이) mm / fx_px)
    px, py = pixel_in_img
    u_in_px = px - ip.cx
    v_in_px = py - ip.cy
    xc_in_mm = u_in_px * z_depth_in_mm / ip.fx
    yc_in_mm = v_in_px * z_depth_in_mm / ip.fy
    return Vector3D(xc_in_mm, yc_in_mm, z_depth_in_mm)


# TODO: 무슨 용도로 만들었더라??
# def pixel_to_camera(px, py, camera_mat: IntrinsicParameters, scale=1) -> Vector3D:
#     pixel_vec = Vector3D(px, py, 1)
#     camera_vec = scale * np.matmul(camera_mat.inverse(), pixel_vec)
#     return Vector3D.from_np_vec(camera_vec)


def normal_to_camera(u, v, z_depth=1) -> Vector3D:
    return Vector3D(u * z_depth, v * z_depth, z_depth)


#
# 2D Coordinate
#
@Reference("https://darkpgmr.tistory.com/77")
def pixel_to_normal(px, py, camera_mat: IntrinsicParameters) -> Vector2D:
    u = (px - camera_mat.cx) / camera_mat.fx
    v = (py - camera_mat.cy) / camera_mat.fy
    return Vector2D(u, v)


@Reference("https://darkpgmr.tistory.com/77")
def normal_to_pixel(u, v, camera_mat: IntrinsicParameters) -> Vector2D:
    normal_vec = Vector3D(u, v, 1)
    pixel_vec = np.matmul(camera_mat.mat(), normal_vec)
    return Vector2D.from_np_vec(pixel_vec)


#
# Rodrigues
#
def rodrigues_to_rot_mat(rod_vec):
    rot_mat, jacobian = cv2.Rodrigues(rod_vec)
    return rot_mat, jacobian

# def compute_scale_factor(p, model_vec, intrinsic, extrinsic):
