import cv2

from ac.common.images import byte_arr_to_img, resize_and_pad, normalize_img
from ac.images.crop_rect import CropRect
from ds.core.npz.gen import *


class EveryEyeClusterGenerator(NPZGenerator):
    def __init__(self,
                 npz_base_path,
                 batch_size=1024,
                 purpose: Purpose = Purpose.ALL):
        super().__init__(npz_base_path, batch_size, purpose)

    def transform(self, npz, start, end):
        images = npz['images']
        images = images[start:end]
        images = [normalize_img(img) for img in images]
        images = [img.flatten() for img in images]
        images = np.array(images, dtype=np.float32)
        return images

    def item_by_index(self, item_idx):
        batch_idx = item_idx // self.batch_size
        idx_delta = (item_idx % self.batch_size) - 1
        images = self[batch_idx]
        return images[idx_delta]

