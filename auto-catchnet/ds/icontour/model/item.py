import numpy as np
from typing import List

from ac.common.images import decode_img
from ac.filesystem.paths import path_join, mkdir
from ac.images.crop_rect import CropRect
from ac.images.filters.filters import bgr_to_rgb
from ac.visualizer.plotter import show_image, make_ellipse, draw_point
from al.optimize.circle.lsc import fit_circle
from al.optimize.ellipse.ellipse import fit_ellipse

"""
# 안구 한쪽 기준
# zt (19개) / unity (32개) / high occlusion : 7/19,  대략 1/3 지점만 안 가려짐
# zt는 레이블 정확성 검증은 따로 하지 않았음 (가끔 완전 딴 곳 보는 이미지 있음)

# === 실험0 (N개) ===
# 레이블 좌표에서 타원 피팅 파라미터 추출 

# === 실험1 (24개) ===
# 레이블 좌표에서 타원 피팅 후 24개를 추출

# vcset zense time 512 샘플링 
# fail / total 비율 : 21 / 512 (96%, 4)
# (추정 : 이전 프레임 정보의 값들을 사용하는 듯) 
"""


class IrisContourItem:
    def __init__(self):
        self.ellipse_param = None
        self.frame_encoded = None
        self.frame_origin_shape = None

        # crop info holder
        self.crop_rect = None

    def calculate_ellipse(self, points):
        param = fit_ellipse(points)
        self.ellipse_param = param

    def calculate_circle(self, points, center_estimate):
        center, radius = fit_circle(points, center_estimate)
        self.ellipse_param = (center, radius, radius, 0)

    def get_decoded_frame(self):
        return decode_img(self.frame_encoded)

    def get_cropped_img(self, far_factor):
        x, y, w, h, _ = self.get_ellipse_param()
        max_radius = max(w, h)
        self.crop_rect = cr = CropRect.from_center_radius_rect((x, y), max_radius, far_factor)
        cr.random_shift(1)
        return self.crop_rect.crop_image(self.get_decoded_frame())

    def get_cropped_params(self, out_shape):
        cr = self.crop_rect
        xc, yc, width, height, phi = self.get_ellipse_param()

        # to cropped coordinate
        cr_top = cr.get_top()
        xc, yc = xc - cr_top[0], yc - cr_top[1]

        # to scaled coordinate
        lx, ly = cr.get_length()
        k = out_shape[0] / lx  # scale factor
        xc = (out_shape[0] / 2) + k * (xc - (lx / 2))
        yc = (out_shape[1] / 2) + k * (yc - (ly / 2))
        width, height = k * width, k * height

        cropped_param = [xc, yc, width, height, phi]
        return cropped_param

    def get_ellipse_param(self):
        (xc, yc), width, height, phi = self.ellipse_param
        return xc, yc, width, height, phi

    def get_meta(self):
        return [self.frame_origin_shape, self.ellipse_param]

    def set_meta(self, meta):
        self.frame_origin_shape = meta[0]
        self.ellipse_param = meta[1]

    """
    exporter & importer
    """

    @staticmethod
    def npz_path(out_dir_path, npz_idx):
        return path_join(out_dir_path, "ic-{:05d}".format(npz_idx))

    @staticmethod
    def to_npz(out_dir_path, items: List['IrisContourItem'], npz_idx):
        metas = np.asarray([item.get_meta() for item in items])
        images = np.asarray([item.frame_encoded for item in items])
        images = images.transpose()

        mkdir(out_dir_path)
        path = IrisContourItem.npz_path(out_dir_path, npz_idx)
        np.savez_compressed(path, metas=metas, images=images)

    @staticmethod
    def from_npz(npz, pos_range=(None, None)) -> List['IrisContourItem']:
        metas, images = npz['metas'], npz['images']
        if pos_range is not None:
            metas = metas[pos_range[0]:pos_range[1]]
            images = images[pos_range[0]:pos_range[1]]
        return [IrisContourItem.from_npz_dict(images[i], metas[i, :]) for i in range(images.shape[0])]

    @staticmethod
    def from_npz_dict(img, meta) -> 'IrisContourItem':
        item = IrisContourItem()
        item.frame_encoded = img
        item.set_meta(meta)
        return item

    """
    visualize
    """

    def plot(self, fig_size=(10, 10)):
        ax = show_image(bgr_to_rgb(self.get_decoded_frame()), fig_size=fig_size)
        (x, y), w, h, p = self.ellipse_param
        plot_center = (x, self.frame_origin_shape[0] - y)
        draw_point(ax, *plot_center)
        e = make_ellipse(plot_center, height=2 * h, width=2 * w, phi=-p, edge_color="blue")
        ax.add_patch(e)

    def __str__(self):
        return "center: {} w: {} h: {} phi: {}".format(*self.ellipse_param)
