from al.optics.vector import *


class FacialModel:
    def __init__(self):
        self.index_2d = None
        self.index_3d = None
        self.mean_vec_3d = None
        self.index_mapper = None
        self.reverse_mapper = None
        self.target_feature_indexes = None
        self.target_3d_model_indexes = None

        self.define_indexes()
        self.define_mean_3d()
        self.define_target_landmarks()

    def get_feature_idx_from(self, idx_3d):
        return self.reverse_mapper.get(int(idx_3d), -1)

    def get_feature_indexes_from(self, *indexes_3d):
        indexes = []
        for index in indexes_3d:
            indexes.append(self.get_feature_idx_from(index))
        return indexes

    def get_3d_idx_from(self, feature_idx):
        return self.index_mapper.get(int(feature_idx), -1)

    def get_3d_indexes_from(self, *feature_indexes):
        indexes = []
        for index in feature_indexes:
            indexes.append(self.get_3d_idx_from(index))
        return indexes

    def get_3d_point(self, feature_idx):
        idx = self.get_3d_idx_from(feature_idx)
        if idx == -1:
            return None
        return self.mean_vec_3d[:, idx]

    def get_target_facial_3d_points(self):
        points = []
        for idx in self.target_3d_model_indexes:
            point_3d = self.mean_vec_3d[:, idx]
            points.append(point_3d)
        return np.array(points)

    def get_target_feature_indexes(self):
        return self.target_feature_indexes

    def get_eye_crop_rect(self, landmarks, length=240):
        l_marks, r_marks = self.select_eye_regions(landmarks)
        l_rect = self.crop_square_rect(l_marks, length)
        r_rect = self.crop_square_rect(r_marks, length)
        return l_rect, r_rect

    def crop_square_rect(self, points, length):
        center = np.average(points, axis=0)
        half = length / 2
        up_left_xy = np.array([center[0] - half, center[1] - half])
        down_right_xy = np.array([center[0] + half, center[1] + half])
        return up_left_xy, down_right_xy

    def crop_rect(self, points, margin_factor=2):
        # (x, y) == (w, h) == (horizontal, vertical)
        center = np.average(points, axis=0)
        up_left_xy = [center[0], center[1]]
        down_right_xy = [center[0], center[1]]

        for pt in points:
            x, y = pt[0], pt[1]
            if x < up_left_xy[0]:
                up_left_xy[0] = x
            if y < up_left_xy[1]:
                up_left_xy[1] = y
            if x > down_right_xy[0]:
                down_right_xy[0] = x
            if y > down_right_xy[1]:
                down_right_xy[1] = y

        # set crop point applied margin factor
        up_left_xy, down_right_xy = np.asarray(up_left_xy), np.asarray(down_right_xy)
        up_left_xy = (margin_factor * up_left_xy) + ((1 - margin_factor) * center)
        down_right_xy = (margin_factor * down_right_xy) + ((1 - margin_factor) * center)

        return up_left_xy, down_right_xy

    def select_eye_regions(self, landmarks):
        l_indices = self.l_eye_indices
        r_indices = self.r_eye_indices
        l_landmarks = landmarks.take(l_indices, axis=0)
        r_landmarks = landmarks.take(r_indices, axis=0)
        return l_landmarks, r_landmarks

    def compute_eye_centers(self, landmarks):
        l_landmarks = landmarks.take([37, 38, 40, 41], axis=0)
        r_landmarks = landmarks.take([43, 44, 46, 47], axis=0)
        l_center = np.average(l_landmarks, axis=0)
        r_center = np.average(r_landmarks, axis=0)
        return l_center, r_center

    def select_eye_inners(self, landmarks):
        # TODO : 고개 돌렸을 때 rotation vec 적용 계산
        l_inner = landmarks.take(39, axis=0)
        r_inner = landmarks.take(42, axis=0)
        return l_inner, r_inner

    def define_target_landmarks(self):
        # 실제로 매핑된 지점들 중 선정
        self.l_eye_indices = (36, 37, 38, 39, 40, 41)
        self.r_eye_indices = (42, 43, 44, 45, 46, 47)

        target_feature_indexes = [
            28, 30,  # 코 라인
            31, 32, 33, 34, 35,  # 콧볼 라인
            36, 37, 38, 39, 40, 41,  # 왼쪽 눈
            42, 43, 44, 45, 46, 47,  # 오른 눈
            48, 49, 51, 53, 54, 55, 57, 59,  # 외측 입술
            60, 62, 64, 66,  # 내측 입술
            0, 2, 4, 7, 8, 9, 12, 14, 16  # 턱 라인
        ]

        target_3d_indexes = [
            94, 5,
            59, 112, 6, 111, 26,
            53, 98, 104, 56, 110, 100,
            23, 103, 97, 20, 99, 109,
            64, 80, 7, 79, 31, 85, 8, 86,
            89, 87, 88, 40,
            62, 61, 63, 65, 10, 32, 30, 28, 29,
        ]

        self.target_feature_indexes = target_feature_indexes
        self.target_3d_model_indexes = target_3d_indexes

    def define_indexes(self):
        index_3d = [
            26, 59, 5, 6, 94, 111, 112, 53, 98, 104, 56, 110,
            100, 23, 103, 97, 20, 99, 109, 48, 49, 50, 17, 16,
            15, 64, 7, 31, 8, 79, 80, 85, 86, 89, 87, 88, 40,
            65, 10, 32, 62, 61, 63, 29, 28, 30
        ]

        index_2d = [
            35, 31, 30, 33, 28, 34, 32, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 17, 19, 21, 22, 24,
            26, 48, 51, 54, 57, 53, 49, 55, 59, 60, 62, 64,
            66, 7, 8, 9, 0, 2, 4, 16, 14, 12
        ]

        index_mapper = {}
        for i, face_idx in enumerate(index_2d):
            index_mapper[face_idx] = index_3d[i]

        self.index_2d = index_2d
        self.index_3d = index_3d
        self.index_mapper = index_mapper
        self.reverse_mapper = {v: k for k, v in index_mapper.items()}

    def define_mean_3d(self):
        self.mean_vec_3d = np.array([[
            0.00169, 0.175274, 0.000858, 0.000443, 0.000339, -0.000354, -0.000422,
            -0.000664, -0.000838, -0.000941, -0.001357, 0.218654, 0.458447, 0.435996,
            0.610858, 0.522442, 0.391595, 0.130443, 0.391512, 0.304358, 0.470235,
            0.304325, 0.304194, 0.130236, 0.304165, 0.10875, 0.173611, 0.38684,
            0.549601, 0.609235, 0.469044, 0.245265, 0.172711, 0.042369, -0.172726,
            0.000858, 0.000443, 0.000339, -0.000354, -0.000422, -0.000734, -0.000838,
            -0.000941, -0.001357, -0.215345, -0.455552, -0.434002, -0.609141, -0.521557,
            -0.390404, -0.129557, -0.390487, -0.303641, -0.469764, -0.303675, -0.303805,
            -0.129764, -0.303834, -0.10925, -0.174388, -0.387159, -0.550397, -0.608764,
            -0.470955, -0.246734, -0.175288, -0.043631, 0.348318, 0.348183, -0.347681,
            -0.347816, 0.265318, 0.265183, -0.264681, -0.264817, 0.07965, -0.08035,
            0.022339, -0.021661, 0.122347, -0.123653, 0.099266, -0.100734, 0.099266,
            -0.100734, 0.122191, -0.123809, -0.000734, 0.199266, -0.200734, 0.356265,
            -0.357734, 0.065045, -0.064955, 0.000108, 0.38732, -0.386679, 0.387296,
            -0.386703, 0.3872, -0.386799, 0.387186, -0.386813, 0.21732, -0.21668,
            0.217296, -0.216703, 0.2172, -0.216799, 0.217186, -0.216813, 0.119578, -0.120422
        ], [
            -1.060999, -0.799722, -0.538999, -0.278, -0.213, 0.222, 0.265,
            0.416999, 0.525999, 0.590999, 0.851999, -1.038653, -0.908271, -0.625306,
            -0.538028, -0.277168, -0.373377, -0.277793, -0.321377, -0.224516, -0.147251,
            -0.203516, -0.121516, -0.147793, -0.103516, 0.157173, 0.244277, 0.100616,
            0.250876, -0.14703, 0.600748, 0.461391, 0.809276, 0.396068, -0.800276,
            -0.538999, -0.278, -0.213, 0.222, 0.265, 0.460999, 0.525999,
            0.590999, 0.851999, -1.039344, -0.909727, -0.626692, -0.539971, -0.278831,
            -0.374622, -0.278207, -0.322622, -0.225484, -0.148748, -0.204484, -0.122484,
            -0.148207, -0.104484, 0.156826, 0.243723, 0.099384, 0.249124, -0.14897,
            0.599251, 0.460608, 0.808722, 0.395931, -0.199446, -0.114446, -0.200554,
            -0.115554, -0.199578, -0.114578, -0.200422, -0.115422, 0.220127, 0.219872,
            -0.212965, -0.213035, 0.410195, 0.409804, 0.461159, 0.46084, 0.461159,
            0.46084, 0.508195, 0.507803, 0.460999, 0.461318, 0.460681, 0.461568,
            0.460431, -0.027896, -0.028103, -0.068, -0.200383, -0.201616, -0.185383,
            -0.186616, -0.125383, -0.126616, -0.116383, -0.117616, -0.200654, -0.201345,
            -0.185654, -0.186345, -0.125654, -0.126345, -0.116654, -0.117345, 0.265191,
            0.264809
        ], [
            0.371, 0.024, -0.085, -0.107, -0.085, -0.21, -0.124,
            -0.142, -0.15, -0.107, -0.063, 0.371, 0.328, 0.111,
            0.328, 0.111, -0.03, -0.107, -0.03, 0.002, 0.111,
            0., 0., 0., 0., -0.037, -0.037, 0.045,
            0.328, 0.328, 0.328, 0., 0., -0.15, 0.024,
            -0.085, -0.107, -0.085, -0.21, -0.124, -0.124, -0.15,
            -0.107, -0.063, 0.371, 0.328, 0.111, 0.328, 0.111,
            -0.03, -0.107, -0.03, 0.002, 0.111, 0., 0.,
            0., 0., -0.037, -0.037, 0.045, 0.328, 0.328,
            0.328, 0., 0., -0.15, 0.03, 0.03, 0.03,
            0.03, 0.03, 0.03, 0.03, 0.03, -0.15, -0.15,
            -0.063, -0.063, -0.063, -0.063, -0.05, -0.05, -0.05,
            -0.05, -0.063, -0.063, -0.124, 0.024, 0.024, 0.05,
            0.05, -0.05, -0.05, -0.1, 0.056, 0.056, 0.056,
            0.056, 0.056, 0.056, 0.067, 0.067, 0.013, 0.013,
            0.013, 0.013, 0.013, 0.013, 0.024, 0.024, -0.1, -0.1
        ]])
