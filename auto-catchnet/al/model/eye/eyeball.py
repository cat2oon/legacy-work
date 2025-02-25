import numpy as np
from scipy import optimize

from al.model.gaze.item import GazeItem
from ac.common.constants import max_value
from al.maths.linalg.rotations import rotate_vec_as_vec_x_to_vec_y
from al.model.eye.imps.starburst import detect_iris_contour

"""
 [STATS] 
 - eyeball 
    diameter : 21 ~ 27 mm (최대 16 ~ 32 mm)  
    radius : 8 ~ 16
    mean : 24.2 mm
    
 - iris (limbic boundary)
    radius : 10.2 ~ 13 mm (넉넉 10 ~ 13 mm)
    mean : 12 mm  
    
 i-param (7)
    - eyeball_center (a, b, c)
    - eyeball_radius (r)
    - kappa (offset) vector (u, v, w) (origin : eyeball-center)
    # TODO: 다른 방식의 kappa offset 표현 방식 찾기 (theta, phi)
"""


# TODO: debug mode (for print sanity check result)
class EyeballModel:
    def __init__(self):

        # major center calibration
        self.vec_oa = (0, 0, 0)  # vec(OA) normalized
        self.vec_ob = (0, 0, 0)  # vec(OB) normalized
        self.vec_op = (0, 0, 0)  # calibration point P

        # calibration points
        self.calibrations = []

        # optimize constraint
        self.eyeball_radius_range = (11.5, 12.5)  # (8, 16)
        self.iris_center_allow_range = (-10, 10)  # mm

        # best measurements for the profile
        self.iris_radius = None
        self.kappa_offset_vector = None
        self.eyeball_radius_optimal = None
        self.eyeball_center_optimal = (0, 0, 0)

    """
    accessor & mutator
    """

    def report(self):
        R = self.eyeball_radius_optimal
        E = self.eyeball_center_optimal
        print("*** eye-ball center pos {} / radius {} ***".format(E, R))

    def set_eye_param(self, params):
        a, b, c = params  # a, b, c, r = params
        self.eyeball_radius_optimal = 12
        self.eyeball_center_optimal = np.array([a, b, c])

        if len(params) == 7:
            u, v, w = params[4:7]
            self.kappa_offset_vector = np.array([u, v, w])

    def add_calibration(self, gaze_item: GazeItem):
        self.calibrations.append(gaze_item)

    """
     API(s)
    """

    def estimate_gaze_point(self, iris_center_vec, head_pose=None, use_kappa=False):
        uv, kv = iris_center_vec, self.kappa_offset_vector
        r, E = self.eyeball_radius_optimal, self.eyeball_center_optimal

        if use_kappa:
            return EyeballModel.estimate_visual_gaze(uv, E, r, kv)
        return EyeballModel.estimate_optical_gaze(uv, E, r)

    """
    algorithm 
    """

    def compute_optical_error(self, eyeball_center, eyeball_radius):
        error, E, r = 0, eyeball_center, eyeball_radius
        for gaze_item in self.calibrations:
            iris_center_vec = gaze_item.get_oc_vec()
            ox, oy = self.estimate_optical_gaze(iris_center_vec, E, r)
            error += gaze_item.get_dist_error(ox, oy)
        return error

    # TODO: !!! Global Minima를 찾으면 안된다!!!
    # TODO: Local Minima 중에 진짜 global Minima가 있다!!!
    # TODO: kappa와 렌즈 굴절 모델이 뒤에 있기 때문에
    # 1. 대강의 범위를 찾고 그 범위로 range 조합을 만든 다음에
    # 그 조합을 루프 돌면서 최적값 찾기
    # 2. eyeball radius 12mm로 고정
    def optimize_eyeball_params(self):
        r = 12

        def eyeball_center_and_radius_optimizer(params):
            a, b, c = params  # a, b, c, r = params
            error = self.compute_optical_error(np.array([a, b, c]), r)
            # TODO: hold minimum nth params
            return error

        iar = self.iris_center_allow_range
        radius_range = self.eyeball_radius_range
        # res = optimize.minimize(eyeball_center_and_radius_optimizer,
        #                         [0, 0, 200, 12], bounds=(iar, iar, (100, 400), radius_range))
        res = optimize.minimize(eyeball_center_and_radius_optimizer, [0, 0, 200], bounds=(iar, iar, (100, 400)))

        if res.success:
            print(res)
            self.set_eye_param(res.x)

    def optimize_one_shot(self):
        """
        [ optimize params i-param(7) ]
        - eyeball_center (a, b, c)
        - eyeball_radius (r)
        - kappa (offset) vector (u, v, w) (origin : eyeball-center)
        """

        def opt_gaze_point(a, b, c, i, j, k, t):
            opt_gaze_x = a - c * (t * i - a) / (t * k - c)
            opt_gaze_y = b - c * (t * j - b) / (t * k - c)
            return opt_gaze_x, opt_gaze_y

        def to_optical_vec(a, b, c, i, j, k, t):
            return (t * i) - a, (t * j) - b, (t * k) - c

        def to_iris_center_pos(i, j, k, t):
            return [t * i, t * j, t * k]

        def compute_fovea_pos(a, b, c, kappa_vec, opt_vec):
            kappa_vec = rotate_vec_as_vec_x_to_vec_y((0., 0., 1.), opt_vec, kappa_vec)
            fovea_pos = (a, b, c) + kappa_vec
            return fovea_pos

        def vis_gaze_point(iris_center_pos, fovea_pos):
            fx, fy, fz = fovea_pos
            vx, vy, vz = iris_center_pos - fovea_pos
            vis_gaze_x = fx - vx * (fz / vz)
            vis_gaze_y = fy - vy * (fz / vz)
            return vis_gaze_x, vis_gaze_y

        def f(params):
            error_sum = 0
            a, b, c, r, u, v, w = params
            for cp in self.calibrations:
                i, j, k = cp[0]
                lx, ly, _ = cp[1]

                _, t, _ = self.compute_intersection(np.array([a, b, c]), np.array([i, j, k]), r)
                opt_vec = to_optical_vec(a, b, c, i, j, k, t)
                iris_pos = to_iris_center_pos(i, j, k, t)

                # fovea_pos = compute_fovea_pos(a, b, c, (u, v, w), opt_vec)
                # vx, vy = vis_gaze_point(iris_pos, fovea_pos)
                gx, gy = opt_gaze_point(a, b, c, i, j, k, t)

                err = np.linalg.norm((gx - lx, gy - ly))
                # err = np.linalg.norm((vx - lx, vy - ly))
                # err = (vx - lx)**2 + (vy - ly)**2
                error_sum += err

            return error_sum

        # constraints
        def kappa_constraint(params):
            a, b, c, r, u, v, w = params
            return u ** 2 + v ** 2 + w ** 2 - r ** 2

        cons = [
            # {'type': 'eq', 'fun': kappa_constraint}
        ]

        # TODO: 사람 평균 kappa offset 초기값 설정
        res = optimize.minimize(f, [0, 0, 300, 12, 0, 0, 0],
                                bounds=((-10, 10), (-10, 10), (100, 400), (8, 16), (-16, 16), (-16, 16), (-16, 16)),
                                constraints=cons)

        if res.success:
            print(res)
            self.set_eye_param(res.x)

    @staticmethod
    def estimate_optical_gaze(iris_center_vec, eyeball_center_pos, eyeball_radius):
        E, uv, r = eyeball_center_pos, iris_center_vec, eyeball_radius
        vertex_pos, d_scalar, det = EyeballModel.compute_intersection(E, uv, r)

        if det < 0:
            return 10000, 10000  # TODO:  None, None으로 하고 iris failed 밖에서 처리

        vx, vy, _ = EyeballModel.compute_optical_gaze_point(vertex_pos, E)
        return vx, vy

    @staticmethod
    def estimate_visual_gaze(iris_center_vec, eyeball_center_pos, eyeball_radius, kappa_offset_vec):
        E, uv, r, kv = eyeball_center_pos, iris_center_vec, eyeball_radius, kappa_offset_vec
        vertex_pos, d_scalar, det = EyeballModel.compute_intersection(E, uv, r)

        if det < 0:
            return None, None

        opt_vec = EyeballModel.compute_optical_vec(E, vertex_pos)
        fovea_pos = EyeballModel.compute_fovea_pos(E, opt_vec, kv)
        vx, vy, _ = EyeballModel.compute_visual_gaze_point(vertex_pos, fovea_pos)
        return vx, vy

    @staticmethod
    def to_iris_center_pos(iris_center_vec, magnitude):
        t = magnitude
        i, j, k = iris_center_vec
        return np.array([t * i, t * j, t * k])

    @staticmethod
    def compute_intersection(eyeball_center_pos, iris_center_vec, eyeball_radius):
        c, l, r = eyeball_center_pos, iris_center_vec, eyeball_radius
        det = (l @ c) ** 2 - ((c @ c) - r ** 2)
        if det < 0:
            return np.array([1000, 1000, 1000]), 0, det
        magnitude = l @ c - np.sqrt(det)  # 작은 값 선택
        intersection = magnitude * l  # 교점 중 카메라에 가까운 것
        return intersection, magnitude, det

    @staticmethod
    def compute_optical_vec(eyeball_center_pos, vertex_pos):
        return vertex_pos - eyeball_center_pos

    @staticmethod
    def compute_fovea_pos(eyeball_center_pos, opt_vec, kappa_offset_vec):
        kappa_offset = rotate_vec_as_vec_x_to_vec_y((0., 0., 1.), opt_vec, kappa_offset_vec)
        return eyeball_center_pos + kappa_offset

    @staticmethod
    def compute_visual_gaze_point(vertex_pos, fovea_pos):
        return EyeballModel.compute_point_with_vec_a_to_b(fovea_pos, vertex_pos)

    @staticmethod
    def compute_optical_gaze_point(vertex_pos, eyeball_center_pos):
        return EyeballModel.compute_point_with_vec_a_to_b(eyeball_center_pos, vertex_pos)

    @staticmethod
    def compute_point_with_vec_a_to_b(vec_a, vec_b):
        a, b, c = vec_a
        i, j, k = vec_b
        mag = c / (c - k)
        x = a + mag * (i - a)
        y = b + mag * (j - b)
        return x, y, 0

    @staticmethod
    def estimate_optical_gaze_legacy(eyeball_center_3d, eyeball_radius, uv_vec_oc):
        """
        E = (a, b, c)
        UV = (i, j, k)
        OPT = (ti - a, tj - b, tk - c)  (optical axis vec)
        OGP = (a - c(ti-a)/(tk-c), b - c(tj-b)/(tk-c), 0)
        """
        iris_center_3d = EyeballModel.compute_iris_center_3d(eyeball_center_3d, eyeball_radius, uv_vec_oc)
        optical_axis_vec = iris_center_3d - eyeball_center_3d
        E, OAV = eyeball_center_3d, optical_axis_vec
        d = E[2] / OAV[2]
        optical_gaze_point = E - d * OAV

        return optical_gaze_point, optical_axis_vec

    @staticmethod
    def compute_iris_center_3d(eyeball_center_3d, eyeball_radius, uv_vec_oc):
        """
        Solve the problem that, given a 3D eyeball center pos E, radius ER and
        uv vec (to iris center), find an estimate for iris center 3d pos.

        v) sphere intersection 방식 (홍채 중심) *
        - uv_vec_oc : 카메라 중심에서 IRIS 중심을 향하는 normal vec
        - uv_vec_oc의 scalar 값을 최적화를 통해 구하고 홍채의 3차원 좌표를 얻음

        +) iris radius 측정한 경우 (이걸로도 구현해 볼 것)
        """

        def f(t):
            r = eyeball_radius
            a, b, c = eyeball_center_3d
            i, j, k = uv_vec_oc
            return (t * i - a) ** 2 + (t * j - b) ** 2 + (t * k - c) ** 2 - r ** 2

        res = optimize.fsolve(f, [-10000, 10000])
        vec_scalar = res[0]
        iris_center_3d = uv_vec_oc * vec_scalar

        return iris_center_3d

    @staticmethod
    def get_iris_contour_points(eye_image, iris_center, eyeball_center):
        # TODO: deep learning 모델로 대체
        return detect_iris_contour(eye_image, iris_center, eyeball_center)

    """ 
    Legacy 
    """

    def stage1(self, iter_unit=0.1, verbose=False):
        # optimize range
        self.eyeball_center_range = (-100, 200)
        radius_candidates = np.arange(*self.eyeball_radius_range, iter_unit)
        center_to_radius_candidates = []  # radius : 3d center

        # 1. estimate eye center for each (w/ center and p point)
        for r in radius_candidates:
            ta = self.estimate_eyeball_center(r, self.vec_oa, self.vec_ob, self.vec_op)
            center_3d_pos = ta * self.vec_oa
            center_to_radius_candidates.append((center_3d_pos, r))

        # TODO: 각 이미지별로 eyeball-center-pos 재계산 할 것
        # 2. find best fit eyeball radius (each candidate all calibration frames)
        min_idx, min_error = -1, max_value(dtype=np.float32)
        for i, center_to_radius in enumerate(center_to_radius_candidates):
            error = self.compute_optical_error(*center_to_radius)
            if error < min_error:
                min_idx, min_error = i, error
            if verbose:
                print("error {} center {} rad {}".format(error, center_to_radius[0], center_to_radius[1]))
        if verbose:
            print("MIN ERROR CASE", min_error, center_to_radius_candidates[min_idx])

        ec, er = center_to_radius_candidates[min_idx]
        # self.eyeball_center_stage1 = ec
        # self.eyeball_radius_stage1 = er

    def stage2(self, verbose=False):
        ec = self.eyeball_center_stage1
        optimal_ec, optimal_r, min_err = None, None, max_value(dtype=np.float32)
        radius_range = np.arange(*self.eyeball_radius_range, 0.25)

        # 1. iterative refinement eyeball center depth
        # TODO: 좀 더 다른 방식의 최적화
        # One-shot 최적화
        # Binary Search 접근 방식 (선형적인가??)
        for dz in np.arange(*self.eyeball_center_range, 2):
            ec_candidate = (ec[0], ec[1], ec[2] + dz)
            r, error = self.find_best_radius_and_error(ec_candidate, radius_range)
            if error < min_err:
                min_err, optimal_ec, optimal_r = error, ec_candidate, r
        print(min_err)

        # 2. iterative refinement eyeball center up/down
        for dx in np.arange(-25, 25, 0.2):
            ec_candidate = (optimal_ec[0], optimal_ec[1] + dx, optimal_ec[2])
            error = self.compute_optical_error(ec_candidate, optimal_r)
            if error < min_err:
                min_err, optimal_ec = error, ec_candidate

        print(min_err)
        self.eyeball_center_optimal = optimal_ec
        self.eyeball_radius_optimal = optimal_r

        # 3. find kappa angle offset
        return None

    def find_best_radius_and_error(self, eyeball_center, radius_range):
        min_r, min_error = None, max_value(dtype=np.float32)
        for r in radius_range:
            error = self.compute_optical_error(eyeball_center, r)
            if error < min_error:
                min_error, min_r = error, r
        return min_r, min_error

    @staticmethod
    def estimate_eyeball_center(eyeball_radius, vec_oe, vec_oi, vec_op):
        """
        Solve the problem that, given a 3D eyeball radius r
        find an estimate for the 3D eyeball center E.
        """
        r = eyeball_radius
        A, B, P = vec_oe, vec_oi, vec_op
        AB = A @ B

        X = np.cross(np.cross(A, B), A) @ P  # (AxBxA) @ p
        Y = np.cross(np.cross(A, B), A) @ B  # (AxBxA) @ B
        Z = np.cross(np.cross(A, B), P) @ B  # (AxBxP) @ B

        S = Y ** 2
        T = (-2 * Y * Z) + (-2 * AB * X * Y)
        U = Z ** 2 + (2 * AB * X * Z) - (r ** 2 * Y ** 2) + X ** 2
        V = 2 * r ** 2 * Y * Z
        W = -(r ** 2) * Z ** 2

        def object_func(x, s, t, u, v, w):
            return (s * x ** 4) + (t * x ** 3) + (u * x ** 2) + (v * x) + w

        # QE의 해가 항상 존재하는지? 없으면 최소 자승법으로...
        # res = optimize.least_squares(object_func, 1., args=(S, T, U, V, W), bounds=(-1000, 1000))
        # ta = res.x

        res = optimize.fsolve(object_func, [-1000, 1000], args=(S, T, U, V, W))
        ta = max(*res)  # 양수는 OE, 음수는 OB의 depth 절대값
        # tb = (ta * X) / ((ta * Y) - Z)
        # r = lin.norm(A * ta - B * tb) (for sanity check)

        return ta

    @staticmethod
    def estimate_iris_center_depth(eyeball_center_3d,
                                   eyeball_radius,
                                   iris_center_3d,
                                   iris_radius):
        cx, cy = iris_center_3d
        ex, ey, ez = eyeball_center_3d
        er, ir = eyeball_radius, iris_radius

        def object_func(k):
            return er ** 2 - ir ** 2 + (ex - cx) ** 2 + (ey - cy) ** 2 + (ez - k) ** 2

        res = optimize.fsolve(object_func, [0, 1000])
        iris_depth = res.mean()

        return iris_depth

    @staticmethod
    # @Deprecated
    def solve_intersect_scalar(a, b, c, r, i, j, k):
        ## 이건 그냥 analytic 해 찾는 것으로 변경하였음
        def object_func(t):
            return (t * i - a) ** 2 + (t * j - b) ** 2 + (t * k - c) ** 2 - r ** 2

        res = optimize.fsolve(object_func, [-10000, 10000])
        t_scalar = res[0]  # 정석은 항상 작은 것 선택 min(*res); 작은게 먼저 나오는 듯
        return t_scalar

    # @Deprecated
    def apply_kappa(self, optical_vec):
        ## # optical axis rodrigues 변환으로 처리하였음
        """
        - rotate eye-ball
        - phi z-x ccw angle, rho z-y ccw angle
        """
        phi_deg, rho_deg = self.kappa_offset_vector
        sin, cos = np.sin, np.cos
        phi, rho = np.deg2rad(phi_deg), np.deg2rad(rho_deg)

        rot_p = np.matrix([
            [cos(phi), 0, sin(phi)],
            [0, 1, 0],
            [-sin(phi), 0, cos(phi)],
        ])

        rot_r = np.matrix([
            [1, 0, 0],
            [0, cos(rho), -sin(rho)],
            [0, sin(rho), cos(rho)],
        ])

        rot_mat = rot_r @ rot_p
        rotated = np.matmul(rot_mat, optical_vec).A1

        return rotated[0], rotated[1], rotated[2]
