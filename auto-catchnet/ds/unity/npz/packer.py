from typing import List
from ac.common.images import read_encoded_img
from ac.common.jsons import load_json
from ac.common.randoms import RandomSeed
from ac.filesystem.greps import *
from ac.filesystem.paths import *
from ds.unity.model.eye_params import class_num_to_eye_param
from ds.unity.model.item import UnityItem


class UnityNPZPacker:
    def __init__(self, out_dir_path, src_base_path, image_shape_hw):
        self.image_shape = image_shape_hw   # (480, 640) (720, 1280)
        self.out_dir_path = out_dir_path
        self.src_base_path = src_base_path
        self.random_seed = RandomSeed.alpha_numeric()

    def pack(self):
        dir_paths = grep_dirs(self.src_base_path)
        for path in dir_paths:
            print("*** processing items in {} ***".format(path))
            items = self.load_items_in(path)
            npz_idx = int(basename_in_path(path))
            UnityItem.to_npz(self.out_dir_path, items, npz_idx)

    def load_items_in(self, dir_path) -> List[UnityItem]:
        items = []
        for img_path, json_path in grep_pairs(dir_path, "jpg", "json"):
            # TODO : parent folder name idx 에서 class_num 추출하기
            eye_x, eye_y = class_num_to_eye_param(class_num)

            try:
                img = read_encoded_img(img_path)
                json = load_json(json_path)
                item = UnityItem.from_img_and_json(img, json, self.image_shape)
                item.set_eye_param(eye_x, eye_y)
                items.append(item)
            except Exception as e:
                print("{}, {}".format(json_path, e))
        return items


