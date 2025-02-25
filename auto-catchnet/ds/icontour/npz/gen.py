from typing import List

from ac.common.images import *
from ac.images.makes.class_map import *
from ac.images.augs.augmentors import for_iris_contour
from ac.langs.funcs import apply
from ds.core.npz.gen import *
from ds.icontour.model.item import IrisContourItem


class IrisContourGenerator(NPZGenerator):
    def __init__(self,
                 npz_base_path,
                 out_shape=(112, 112),
                 batch_size=512,
                 purpose: Purpose = Purpose.TRAIN,
                 use_aug=True,
                 use_softmax_pred=False,
                 is_item_mode=False,
                 is_ellipse_mode=True):

        self.augmentor = None
        self.out_shape = out_shape
        self.is_item_mode = is_item_mode
        self.is_ellipse_mode = is_ellipse_mode
        self.use_softmax_pred = use_softmax_pred
        if use_aug:
            self.augmentor = for_iris_contour()

        super().__init__(npz_base_path, batch_size, purpose)

    def transform(self, npz, start, end):
        items = IrisContourItem.from_npz(npz, (start, end))

        if self.is_item_mode:
            return items

        imgs = self.transform_image(items)

        if self.is_ellipse_mode:
            labels = self.transform_formula_label(items)
        else:
            labels = self.transform_label(items)

        return imgs, labels

    def transform_label(self, items: List['IrisContourItem']):
        labels = []
        for item in items:
            ellipse_param = item.get_cropped_params(self.out_shape)

            if self.use_softmax_pred:
                labels.append(make_softmax_map_from_ellipse(self.out_shape, ellipse_param))
            else:
                labels.append(make_class_map_from_ellipse(self.out_shape, ellipse_param))

        labels = np.array(labels)
        if self.use_softmax_pred:
            labels = np.reshape(labels, [len(items), self.out_shape[0], self.out_shape[1], 2])

        return labels

    def transform_image(self, items: List['IrisContourItem']):
        images = apply(self.preprocess_img, items)
        images = self.augmentation(images)
        images = apply(self.postprocess_img, images)
        images = apply(normalize_img, images)
        images = np.asarray(images)
        images = np.squeeze(images)
        return images

    """
    augmentations 
    """

    @staticmethod
    def preprocess_img(item: IrisContourItem):
        img = item.get_cropped_img(far_factor=3.0)
        img = scale_abs(img)
        return img

    def postprocess_img(self, img):
        return resize_and_pad(img, self.out_shape, over_mode=False)

    def augmentation(self, images):
        if self.augmentor is None:
            return images
        return self.augmentor.augment_images(images)

    def transform_formula_label(self, items: List['IrisContourItem']):
        labels = []
        for item in items:
            label = item.get_cropped_params(self.out_shape)
            labels.append(label[:2])
        return np.array(labels)
