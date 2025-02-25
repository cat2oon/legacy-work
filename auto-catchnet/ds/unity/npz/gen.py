from typing import List

from ac.common.images import normalize_img, scale_abs, resize_and_pad
from ac.images.augs.augmentors import make_basic
from ac.images.filters.filters import *
from ac.langs.funcs import apply
from ds.core.npz.gen import *
from ds.unity.model.eye_params import eye_param_to_class_num
from ds.unity.model.item import UnityItem


class UnityEyePoseGenerator(NPZGenerator):
    def __init__(self,
                 npz_base_path,
                 batch_size=128,
                 purpose: Purpose = Purpose.TRAIN,
                 use_aug=True,
                 use_postprocess=False,
                 is_item_mode=False):
        self.is_item_mode = is_item_mode
        self.use_postprocess = use_postprocess
        self.augmentor = self.build_augmentor(use_aug)
        super().__init__(npz_base_path, batch_size, purpose)

    def transform(self, npz, start, end):
        items = UnityItem.from_npz(npz, (start, end))

        if self.is_item_mode:
            return items

        imgs = self.transform_image(items)
        apps = self.transform_appendix(items)
        preds = self.transform_label(items)
        return [imgs, apps], preds

    @staticmethod
    def transform_label(items):
        def get_class_indexes():
            return [eye_param_to_class_num(item.eye_pitch, item.eye_yaw) for item in items]

        def get_eye_poses():
            return [[item.eye_pitch, item.eye_yaw] for item in items]

        def get_look_vecs():
            return [(item.look_vec[0], item.look_vec[1]) for item in items]

        return np.asarray(get_look_vecs())

    def transform_image(self, items: List['UnityItem']):
        pairs = [(item.get_cropped_eye_img(), item) for item in items]
        images = apply(self.preprocess_img, pairs)
        images = self.augmentation(images)
        images = apply(self.postprocess_img, images)
        images = apply(normalize_img, images)
        images = np.asarray(images)
        images = np.expand_dims(images, axis=3)
        return images

    @staticmethod
    def transform_appendix(items):
        looks = [np.array(item.head_pose) for item in items]
        return np.asarray(looks)

    """
    - augmentations 
    """

    @staticmethod
    def build_augmentor(use_augmentation):
        if not use_augmentation:
            return None
        return make_basic()

    @staticmethod
    def preprocess_img(img_item_pair):
        img, item = img_item_pair
        img = color_to_grey(img)
        img = scale_abs(img)
        return img

    def postprocess_img(self, img):
        if self.use_postprocess:
            img = apply_sobel_x(img)
        img = resize_and_pad(img, (56, 112), over_mode=False)
        return img

    def augmentation(self, images):
        if self.augmentor is not None:
            return self.augmentor.augment_images(images)
        return images

