import dlib
from dlib import rectangle

import cv2
import numpy as np
from scipy import optimize

from ac.langs.decorator.deprecated import DeprecatedDecorator


def line_search_func(alpha, x, d, func, args):
    # 제곱 거리 비용 계산
    r = func(x + alpha * d, *args)
    return np.sum(r ** 2)


def gauss_newton(x0, func, func_jacobian, args, max_iter=10, eps=10e-7):
    x = np.array(x0, dtype=np.float64)

    old_cost = -1
    for i in range(max_iter):  # 최적화 루프
        r = func(x, *args)
        cost = np.sum(r ** 2)  # 제곱 거리 비용 계산

        if cost < eps or abs(cost - old_cost) < eps:  # 수렴 여부 판정
            break
        old_cost = cost

        jacobian = func_jacobian(x, *args)
        grad = np.dot(jacobian.T, r)
        hessian = np.dot(jacobian.T, jacobian)
        direction = np.linalg.solve(hessian, grad)

        line_search_res = optimize.minimize_scalar(line_search_func, args=(x, direction, func, args))
        alpha = line_search_res["x"]
        x = x + alpha * direction
    return x


@DeprecatedDecorator
def steepest_descent(x0, fun, fun_jack, args, max_iter=10, eps=10e-7):
    x = np.array(x0, dtype=np.float64)

    old_cost = -1
    for i in range(max_iter):
        r = fun(x, *args)
        cost = np.sum(r ** 2)

        if cost < eps or abs(cost - old_cost) < eps:
            break
        old_cost = cost

        J = fun_jack(x, *args)
        grad = 2 * np.dot(J.T, r)
        direction = grad
        line_search_res = optimize.minimize_scalar(line_search_func, args=(x, direction, fun, args))
        alpha = line_search_res["x"]
        x = x + alpha * direction
    return x


def get_normal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    axis_x = b - a
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = c - a
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    axis_z = axis_z / np.linalg.norm(axis_z)

    return axis_z


def flip_winding(triangle):
    return [triangle[1], triangle[0], triangle[2]]


def fix_mesh_winding(mesh, vertices):
    for i in range(mesh.shape[0]):
        triangle = mesh[i]
        normal = get_normal(vertices[:, triangle])
        if normal[2] > 0:
            mesh[i] = flip_winding(triangle)
    return mesh


@DeprecatedDecorator
def get_Shape_3D(mean_3d_shape, blend_shapes, params):
    s = params[0]
    r = params[1:4]
    t = params[4:6]
    w = params[6:]

    R = cv2.Rodrigues(r)[0]
    shape_3d = mean_3d_shape + np.sum(w[:, np.newaxis, np.newaxis] * blend_shapes, axis=0)
    shape_3d = s * np.dot(R, shape_3d)
    shape_3d[:2, :] = shape_3d[:2, :] + t[:, np.newaxis]

    return shape_3d


@DeprecatedDecorator
def get_mask(rendered_img):
    mask = np.zeros(rendered_img.shape[:2], dtype=np.uint8)
    return mask


def load_3D_face_model(filename):
    face_model_file = np.load(filename)
    mean_3d_shape = face_model_file["mean3DShape"]
    mesh = face_model_file["mesh"]
    idxs_3d = face_model_file["idxs3D"]
    idxs_2d = face_model_file["idxs2D"]
    blend_shapes = face_model_file["blendshapes"]
    mesh = fix_mesh_winding(mesh, mean_3d_shape)

    return mean_3d_shape, blend_shapes, mesh, idxs_3d, idxs_2d


def get_face_key_points(img, detector, predictor, max_img_size_for_detection=640):
    img_scale = 1
    scaled_img = img

    if max(img.shape) > max_img_size_for_detection:
        img_scale = max_img_size_for_detection / float(max(img.shape))
        scaled_img = cv2.resize(img, (int(img.shape[1] * img_scale), int(img.shape[0] * img_scale)))

    dets = detector(scaled_img, 1)
    if len(dets) == 0:
        return None

    shapes_2D = []
    for det in dets:
        face_rectangle = rectangle(int(det.left() / img_scale),
                                   int(det.top() / img_scale),
                                   int(det.right() / img_scale),
                                   int(det.bottom() / img_scale))
        dlib_shape = predictor(img, face_rectangle)
        shape_2D = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        shape_2D = shape_2D.T
        shapes_2D.append(shape_2D)
    return shapes_2D


class OrthographicProjectionBlendshapes:
    nParams = 6

    def __init__(self, n_blend_shapes):
        self.n_blend_shapes = n_blend_shapes
        self.nParams += n_blend_shapes

    def residual(self, params, x, y):
        r = y - self.func(x, params)
        r = r.flatten()
        return r

    def func(self, x, params):
        s = params[0]
        r = params[1:4]
        t = params[4:6]
        w = params[6:]

        mean_3D_shape = x[0]
        blend_shapes = x[1]

        R = cv2.Rodrigues(r)[0]
        P = R[:2]
        shape_3D = mean_3D_shape + np.sum(w[:, np.newaxis, np.newaxis] * blend_shapes, axis=0)

        projected = s * np.dot(P, shape_3D) + t[:, np.newaxis]
        return projected

    def jacobian(self, params, x, y):
        s = params[0]
        r = params[1:4]
        t = params[4:6]
        w = params[6:]

        mean_3D_shape = x[0]
        blend_shapes = x[1]

        R = cv2.Rodrigues(r)[0]
        P = R[:2]
        shape3D = mean_3D_shape + np.sum(w[:, np.newaxis, np.newaxis] * blend_shapes, axis=0)

        nPoints = mean_3D_shape.shape[1]

        jacobian = np.zeros((nPoints * 2, self.nParams))
        jacobian[:, 0] = np.dot(P, shape3D).flatten()

        step_size = 10e-4
        step = np.zeros(self.nParams)
        step[1] = step_size
        jacobian[:, 1] = ((self.func(x, params + step) - self.func(x, params)) / step_size).flatten()
        step = np.zeros(self.nParams)
        step[2] = step_size
        jacobian[:, 2] = ((self.func(x, params + step) - self.func(x, params)) / step_size).flatten()
        step = np.zeros(self.nParams)
        step[3] = step_size
        jacobian[:, 3] = ((self.func(x, params + step) - self.func(x, params)) / step_size).flatten()

        jacobian[:nPoints, 4] = 1
        jacobian[nPoints:, 5] = 1

        startIdx = self.nParams - self.n_blend_shapes
        for i in range(self.n_blend_shapes):
            jacobian[:, i + startIdx] = s * np.dot(P, blend_shapes[i]).flatten()
        return jacobian

    def get_example_params(self):
        params = np.zeros(self.nParams)
        params[0] = 1
        return params

    def get_initial_params(self, x, y):
        mean_3D_shape = x.T
        shape_2D = y.T
        shape_3D_centered = mean_3D_shape - np.mean(mean_3D_shape, axis=0)
        shape_2D_centered = shape_2D - np.mean(shape_2D, axis=0)

        scale = np.linalg.norm(shape_2D_centered) / np.linalg.norm(shape_3D_centered[:, :2])
        t = np.mean(shape_2D, axis=0) - np.mean(mean_3D_shape[:, :2], axis=0)

        params = np.zeros(self.nParams)
        params[0] = scale
        params[4] = t[0]
        params[5] = t[1]

        return params


def draw_points(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)


def draw_cross(img, params, center=(470, 150), scale=30.0):
    R = cv2.Rodrigues(params[1:4])[0]  # rotation matrix
    dst, jacobian = cv2.Rodrigues(params[1:4])

    points = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    points = np.dot(points, R.T)
    points_2d = points[:, :2]

    points_2d = (points_2d * scale + center).astype(np.int32)
    cv2.line(img, (center[0], center[1]), (points_2d[0, 0], points_2d[0, 1]), (255, 0, 0), 3)
    cv2.line(img, (center[0], center[1]), (points_2d[1, 0], points_2d[1, 1]), (0, 255, 0), 3)
    cv2.line(img, (center[0], center[1]), (points_2d[2, 0], points_2d[2, 1]), (0, 0, 255), 3)

    return points_2d


def draw_mesh(img, shape, mesh, color=(255, 0, 0)):
    for triangle in mesh:
        point1 = shape[triangle[0]].astype(np.int32)
        point2 = shape[triangle[1]].astype(np.int32)
        point3 = shape[triangle[2]].astype(np.int32)

        cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]), color, 1)
        cv2.line(img, (point2[0], point2[1]), (point3[0], point3[1]), color, 1)
        cv2.line(img, (point3[0], point3[1]), (point1[0], point1[1]), color, 1)


def draw_projected_shape(x, projection, params, locked_translation=False):
    local_params = np.copy(params)

    if locked_translation:
        local_params[4] = 100
        local_params[5] = 200

    projected_shape = projection.func(x, local_params)
    return projected_shape.T


def pad(img, r_x, r_y, max_size):
    crop_r_x = 0 if r_x < 0 else r_x
    crop_r_y = 0 if r_y < 0 else r_y
    crop = img[crop_r_y:r_y + max_size, crop_r_x:r_x + max_size]

    # pad top
    if r_y < 0:
        padding = [(-r_y, 0), (0, 0), (0, 0)]
        crop = np.pad(crop, padding, 'edge')
    # pad bottom
    elif r_y + max_size > img.shape[0]:
        padding = [(0, max_size - (img.shape[0] - r_y)), (0, 0), (0, 0)]
        crop = np.pad(crop, padding, 'edge')
    # pad left
    if r_x < 0:
        padding = [(0, 0), (-r_x, 0), (0, 0)]
        crop = np.pad(crop, padding, 'edge')
    # pad right
    elif r_x + max_size > img.shape[1]:
        padding = [(0, 0), (0, max_size - (img.shape[1] - r_x)), (0, 0)]
        crop = np.pad(crop, padding, 'edge')
    return crop


def normalize(img):
    dst = np.zeros(shape=img.shape)
    img = cv2.normalize(img, dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return img


def crop_eyes_candide(img, points):
    l_eye_w = points[40] - points[46]
    l_eye_h = points[45] - min(points[43], points[39])
    l_size = l_eye_w * 1.2

    r_eye_w = points[112] - points[106]
    r_eye_h = points[111] - min(points[109], points[105])
    r_size = r_eye_w * 1.2

    max_size = max(l_size, r_size)

    l_x = int(points[46] - (max_size - l_eye_w) * 0.5)
    l_y = int(min(points[43], points[39]) - (max_size - l_eye_h) * 0.5)

    r_x = int(points[106] - (max_size - r_eye_w) * 0.5)
    r_y = int(min(points[109], points[105]) - (max_size - r_eye_h) * 0.5)

    max_size = int(max_size)

    elo = pad(img, l_x, l_y, max_size)
    ero = pad(img, r_x, r_y, max_size)

    return elo, ero


class Candide:
    def __init__(self, candide_path='./candide.npz', land_mark_path='./shape_predictor_68_face_landmarks.dat'):
        self.candide_path = candide_path
        self.face_land_marks_path = land_mark_path
        self.init_modules()

    def init_modules(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.face_land_marks_path)
        self.mean_3d_shape, self.blend_shapes, self.mesh, self.idxs_3d, self.idxs_2d = load_3D_face_model(
            self.candide_path)
        self.projection_model = OrthographicProjectionBlendshapes(self.blend_shapes.shape[0])

    def get_face_texture_coords(self, img):
        idxs_2d = self.idxs_2d
        idxs_3d = self.idxs_3d
        blend_shapes = self.blend_shapes
        mean_3D_shape = self.mean_3d_shape
        projection_model = self.projection_model

        key_points = get_face_key_points(img, self.detector, self.predictor)[0]
        model_params = projection_model.get_initial_params(mean_3D_shape[:, idxs_3d], key_points[:, idxs_2d])
        model_params = gauss_newton(model_params, projection_model.residual, projection_model.jacobian,
                                    ([mean_3D_shape[:, idxs_3d], blend_shapes[:, :, idxs_3d]], key_points[:, idxs_2d]))
        texture_coords = projection_model.func([mean_3D_shape, blend_shapes], model_params)
        return texture_coords

    def get_facial_points(self, img, max_size_for_detect=320):
        key_points = get_face_key_points(img,
                                         self.detector,
                                         self.predictor,
                                         max_img_size_for_detection=max_size_for_detect)
        return key_points

    def get_features(self, img):
        shapes2D = get_face_key_points(img, self.detector, self.predictor, max_img_size_for_detection=320)

        if shapes2D is None:
            return None

        idxs3D = self.idxs_3d
        idxs2D = self.idxs_2d
        mesh = self.mesh
        blendshapes = self.blend_shapes
        mean3DShape = self.mean_3d_shape
        projectionModel = self.projection_model
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for shape2D in shapes2D:
            modelParams = projectionModel.get_initial_params(mean3DShape[:, idxs3D], shape2D[:, idxs2D])
            modelParams = gauss_newton(modelParams, projectionModel.residual, projectionModel.jacobian,
                                       ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]))

        face_feature = modelParams.flatten()
        return face_feature

    def get_features_detail(self, img):
        shapes_2d = get_face_key_points(img, self.detector, self.predictor, max_img_size_for_detection=320)

        if shapes_2d is None:
            return None

        mesh = self.mesh
        idxs_3d = self.idxs_3d
        idxs_2d = self.idxs_2d
        blend_shapes = self.blend_shapes
        mean_3d_Shape = self.mean_3d_shape
        proj_model = self.projection_model

        model_params = None
        for shape_2d in shapes_2d:
            model_params = proj_model.get_initial_params(mean_3d_Shape[:, idxs_3d], shape_2d[:, idxs_2d])
            model_params = gauss_newton(model_params, proj_model.residual, proj_model.jacobian,
                                        ([mean_3d_Shape[:, idxs_3d], blend_shapes[:, :, idxs_3d]], shape_2d[:, idxs_2d]))
            # visualize
            draw_points(img, shape_2d.T)

        draw_cross(img, model_params, center=(img.shape[1] - 50, 50))
        projected_shape = draw_projected_shape([mean_3d_Shape, blend_shapes],
                                               proj_model,
                                               model_params,
                                               locked_translation=False)
        # drawMesh(img, projected_shape, mesh)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elo, ero = crop_eyes_candide(img_gray, projected_shape.flatten())
        face_feature = model_params.flatten()

        # s = params[0]     scale
        # r = params[1:4]   rotation matrix
        # t = params[4:6]   translate?
        # w = params[6:]    ???

        return face_feature, shapes_2d, elo, ero

