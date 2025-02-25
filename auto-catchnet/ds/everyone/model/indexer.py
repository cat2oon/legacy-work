import json

from ac.filesystem.paths import *
from ac.langs.sequences import *
from ds.everyone.model.item_index import *


def format_id(id_value):
    if type(id_value) is int:
        return "%05d" % id_value
    return id_value


class EveryoneIndexer:
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.profile_ids = []
        self.item_index_per_profile = {}

    def get_index(self, profile_id):
        return self.item_index_per_profile[profile_id]

    def index(self):
        print(">>> index everyone from: {}".format(self.source_dir))  # annotation
        for profile_id in os.listdir(self.source_dir):
            self.index_profile(profile_id)
            self.profile_ids.append(profile_id)
            num_item_in_profile = len(self.item_index_per_profile[profile_id])
            print("indexing done at profile [{}] num item:{}".format(profile_id, num_item_in_profile))
        print(">>> indexing all profiles complete")

        print(">>> validating items... ")
        self.select_only_valid_index()

        num_total_items = sum([len(items) for items in self.item_index_per_profile.values()])
        print(">>> validating items complete. total:{}".format(num_total_items))

    def select_only_valid_index(self):
        for profile_id, item_indexes in self.item_index_per_profile.items():
            valid_item_index = [ii for ii in item_indexes if ii.validate()]
            self.item_index_per_profile[profile_id] = valid_item_index

    def index_profile(self, profile_id):
        item_indexes = []
        index_meta = self.get_profile_index_meta(profile_id)
        for file_name in self.get_coexist_file_names(profile_id):
            item_index = ItemIndex.make(file_name, index_meta)
            item_indexes.append(item_index)
        self.item_index_per_profile[profile_id] = item_indexes

    def get_coexist_file_names(self, profile_id):
        frames = os.listdir(self.get_frame_base_path(profile_id))
        left_eyes = os.listdir(self.get_eye_base_path(profile_id, False))
        right_eyes = os.listdir(self.get_eye_base_path(profile_id, True))

        coexist = intersect_list(frames, left_eyes, right_eyes)
        return list(filter(lambda x: x.endswith("jpg"), coexist))

    def get_profile_index_meta(self, profile_id):
        return {
            "pid": profile_id,
            "profile_path": self.get_profile_path(profile_id),
            "info": self.get_json_info(profile_id, "info.json"),
            "dots": self.get_json_info(profile_id, "dotInfo.json"),
            "grid": self.get_json_info(profile_id, "faceGrid.json"),
            "face": self.get_json_info(profile_id, "appleFace.json"),
            "frames": self.get_json_info(profile_id, "frames.json"),
            "screens": self.get_json_info(profile_id, "screen.json"),
            "left_eye": self.get_json_info(profile_id, "appleLeftEye.json"),
            "right_eye": self.get_json_info(profile_id, "appleRightEye.json"),
        }

    def get_json_info(self, profile_id, filename):
        base_path = self.get_profile_path(profile_id)
        with open(path_join(base_path, filename)) as f:
            data = json.load(f)
        return data

    def get_profile_path(self, profile_id):
        return path_join(self.source_dir, format_id(profile_id))

    def get_eye_base_path(self, profile_id, choose_right=True):
        which = "appleLeftEye"
        if choose_right:
            which = "appleRightEye"
        return path_join(self.get_profile_path(profile_id), which)

    def get_frame_base_path(self, profile_id):
        base_path = self.get_profile_path(profile_id)
        return path_join(base_path, "frames")

