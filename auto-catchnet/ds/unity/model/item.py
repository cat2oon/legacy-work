import hashlib
from typing import List

import numpy as np

from ac.common.images import decode_img
from ac.common.prints import pretty_print_pairs
from ac.filesystem.paths import mkdir, path_join, exists
from ac.images.crop_rect import CropRect
from ac.images.filters.filters import bgr_to_rgb
from ac.visualizer.plotter import show_image, draw_point
from ds.unity.model.eye_params import eye_param_to_class_num


class UnityItem:
    def __init__(self):
        self.uid = None  # image md5
        self.meta_keys = None  # exportable
        self.image_shape = None     # shape in hw (480, 600) (720, 1280)
        self.image_encoded = None   # non meta attr

        """ 
        - unity 이미지 생성 시점에만 획득 가능
        - unity eye_x, y와 달리 상단 우측 양수 좌표계
        """
        self.eye_yaw = None
        self.eye_pitch = None

        # from json
        self.iris = None
        self.head_pose = None
        self.look_vec = None
        self.pupil_size = None
        self.iris_size = None
        self.interiors = None
        self.caruncles = None
        self.iris_texture = None
        self.lighting_details = None
        self.eye_region_details = None

        self.set_export_meta_keys()

    def set_export_meta_keys(self):
        field_to_val = vars(self)
        meta_keys = list(field_to_val.keys())
        meta_keys.sort()
        meta_keys.remove('image_encoded')  # TODO: abstract filter non meta attrs
        meta_keys.remove('meta_keys')  # TODO: abstract filter non meta attrs
        self.meta_keys = ["meta_keys"] + meta_keys

    def set_eye_param(self, eye_x, eye_y):
        self.eye_yaw = -1 * eye_y
        self.eye_pitch = -1 * eye_x

    def get_eye_class(self):
        return eye_param_to_class_num(self.eye_pitch, self.eye_yaw)

    def set_uid(self, encoded_image):
        md5 = hashlib.md5()
        md5.update(encoded_image)
        self.uid = md5.hexdigest()

    def set_image(self, encoded_image, img_shape):
        if img_shape is not None:
            self.image_shape = img_shape
        self.image_encoded = encoded_image

    def set_from_meta(self, meta):
        meta_keys = meta[0]
        for i, key in enumerate(meta_keys):
            setattr(self, key, meta[i])

    def get_ordered_exportable_metas(self):
        return [self[key] for key in self.meta_keys]

    def report(self):
        attrs = ['uid', 'image_shape', 'eye_pitch', 'eye_yaw', 'look_vec', 'head_pose']
        pairs = [(key.upper(), vars(self)[key]) for key in attrs]
        pretty_print_pairs(pairs)

    """
    - eye crop rect
    """
    def get_cropped_eye_img(self):
        points = np.vstack([self.interiors, self.caruncles])
        cr = CropRect.least_cover_eye_box(points, self.interiors, self.caruncles, 1, 1)
        return cr.crop_image(self.get_decoded_frame())

    def get_decoded_frame(self):
        return decode_img(self.image_encoded)

    """
    override
    """

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return "UnityItem uid[{}]".format(self.uid)

    """
    export & import
    """

    def set_from_json(self, json):
        self.iris = self.to_points(json["iris_2d"])
        self.caruncles = self.to_points(json["caruncle_2d"])
        self.interiors = self.to_points(json["interior_margin_2d"])
        self.set_from_head_pose_json(json["head_pose"])
        self.set_from_eye_details_json(json["eye_details"])
        self.lighting_details = json["lighting_details"]
        self.eye_region_details = json["eye_region_details"]

    def to_points(self, tuple_points_json):
        points = [eval(point) for point in tuple_points_json]
        points = np.array(points)
        points[:, 1] = np.subtract(self.image_shape[0], points[:, 1])
        return points

    def set_from_head_pose_json(self, head_pose):
        head_pose = eval(head_pose)
        self.head_pose = np.array(head_pose)

    def set_from_eye_details_json(self, eye_details):
        look_vec = eval(eye_details["look_vec"])
        self.look_vec = np.array(look_vec)
        self.iris_size = float(eye_details["iris_size"])
        self.pupil_size = float(eye_details["pupil_size"])
        self.iris_texture = eye_details["iris_texture"]

    """
    exporter & importer
    """

    @staticmethod
    def npz_path(out_dir_path, npz_idx):
        return path_join(out_dir_path, "unity-{:05d}".format(npz_idx))

    @staticmethod
    def exists_npz(out_dir_path, npz_idx):
        npz_path = UnityItem.npz_path(out_dir_path, npz_idx) + ".npz"
        return exists(npz_path)

    @staticmethod
    def to_npz(out_dir_path, items, npz_idx):
        metas = np.asarray([item.get_ordered_exportable_metas() for item in items])
        images = np.asarray([item.image_encoded for item in items])
        images = images.transpose()

        mkdir(out_dir_path)
        path = UnityItem.npz_path(out_dir_path, npz_idx)
        np.savez_compressed(path, metas=metas, images=images)

    @staticmethod
    def from_npz_path(npz_path) -> List['UnityItem']:
        npz = np.load(npz_path)
        return UnityItem.from_npz(npz)

    @classmethod
    def from_npz(cls, npz, pos_range=(None, None)) -> List['UnityItem']:
        metas, images = npz['metas'], npz['images']
        if pos_range is not None:
            metas = metas[pos_range[0]:pos_range[1]]
            images = images[pos_range[0]:pos_range[1]]
        return [UnityItem.from_npz_dict(images[i], metas[i, :]) for i in range(images.shape[0])]

    @staticmethod
    def from_npz_dict(img, meta) -> 'UnityItem':
        ui = UnityItem()
        ui.set_image(img, None)
        ui.set_from_meta(meta)
        return ui

    @staticmethod
    def from_img_and_json(img, json, img_shape) -> 'UnityItem':
        ui = UnityItem()
        ui.set_uid(img)
        ui.set_image(img, img_shape)
        ui.set_from_json(json)
        return ui

    """
    visualize
    """
    def plot(self):
        ax = show_image(bgr_to_rgb(self.get_decoded_frame()))
        for p in self.iris:
            draw_point(ax, p[0], self.image_shape[0] - p[1], color='red')
