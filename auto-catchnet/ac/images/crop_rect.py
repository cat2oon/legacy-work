import numpy as np

from ac.langs.sequences import abs_seq


class CropRect:
    def __init__(self, top_left_xy, bottom_right_xy):
        self.top_left_x = int(top_left_xy[0])
        self.top_left_y = int(top_left_xy[1])
        self.bottom_right_x = int(bottom_right_xy[0])
        self.bottom_right_y = int(bottom_right_xy[1])

    @staticmethod
    def from_points_center_square(points, side_length):
        return CropRect.from_points_center_rect(points, side_length, side_length)

    @staticmethod
    def from_points_center_rect(points, height_length, width_length):
        height_half = height_length / 2
        width_half = width_length / 2
        center = np.average(points, axis=0)
        top_left = ([center[0] - width_half, center[1] - height_half])
        bottom_right = ([center[0] + width_half, center[1] + height_half])
        return CropRect(top_left, bottom_right)

    @staticmethod
    def from_center_radius_rect(center, radius, far_factor):
        height_half, width_half = radius * far_factor, radius * far_factor
        top_left = [center[0] - width_half, center[1] - height_half]
        bottom_right = [center[0] + width_half, center[1] + height_half]
        return CropRect(top_left, bottom_right)

    @staticmethod
    def least_cover_eye_box(points, ints, cars, height_margin_ratio, width_margin_ratio):
        def get_px_diff():
            int_left, _, _, _ = CropRect.get_corners_from_points(ints)
            car_left, _, _, _ = CropRect.get_corners_from_points(cars)
            return abs(int_left - car_left)

        unit_px = get_px_diff()
        x_min, y_min, x_max, y_max = CropRect.get_corners_from_points(points)
        width_margin = unit_px * width_margin_ratio
        height_margin = unit_px * height_margin_ratio
        top_left_xy = abs_seq((x_min - width_margin, y_min - height_margin))
        bottom_right_xy = abs_seq((x_max + width_margin, y_max + height_margin))
        return CropRect(top_left_xy, bottom_right_xy)

    @staticmethod
    def get_corners_from_points(points):
        points = points[:, :2]
        left_x, top_y = points[:, 0].min(), points[:, 1].min()
        right_x, bottom_y = points[:, 0].max(), points[:, 1].max()
        return left_x, top_y, right_x, bottom_y

    def random_shift(self, margin):
        lx, ly = self.get_length()
        max_shift = (lx / 2) - margin
        sx, sy = np.random.random_integers(-max_shift, max_shift, size=(2,))
        self.shift(sx, sy)

    def shift(self, sx, sy):
        self.top_left_x += sx
        self.top_left_y += sy
        self.bottom_right_x += sx
        self.bottom_right_y += sy

    def get_xs(self):
        return self.top_left_x, self.bottom_right_x

    def get_ys(self):
        return self.top_left_y, self.bottom_right_y

    def get_top(self):
        return self.top_left_x, self.top_left_y

    def get_bottom(self):
        return self.bottom_right_x, self.bottom_right_y

    def get_center(self):
        cx = (self.top_left_x + self.bottom_right_x) // 2
        cy = (self.top_left_y + self.bottom_right_y) // 2
        return cx, cy

    def get_length(self):
        lx = abs(self.bottom_right_x - self.top_left_x)
        ly = abs(self.bottom_right_y - self.top_left_y)
        return lx, ly

    # TODO: 영역 처리 수정할 것
    # 1. 음수 영역을 0 처리 (사이즈 축소됨) 
    # 2. 음수 영역 만큼 패딩 처리
    # 3. 0을 기준으로 crop length 유지 처리
    def crop_image(self, img, preserve_length=False):
        h, w = img.shape[:2]
        x0, x1 = self.get_xs()
        y0, y1 = self.get_ys()
        
        if x0 < 0:
            x0, x1 = 0, x1 - x0
        if y0 < 0:
            y0, y1 = 0, y1 - y0
        if x1 > w:
            x0, x1 = x0-x1 + w, w
        if y1 > h:
            y0, y1 = y0-y1 + h, h

        if len(img.shape) == 2:
            return img[y0:y1, x0:x1]

        return img[y0:y1, x0:x1, :]

    def __str__(self):
        return "top_xy ({},{}) bottom_xy ({},{})".format(
            self.top_left_x, self.top_left_y,
            self.bottom_right_x, self.bottom_right_y)
