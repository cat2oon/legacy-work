import os

from ac.common.images import *
from ac.filesystem.paths import path_join
from al.actor.face import FaceActor
from ds.everyone.model.item_index import ItemIndex


def format_pid(num_profile_id):
    return "{:05d}".format(num_profile_id)


def is_target_npz(name, pid):
    # everyone-pid01995-so1-idx0
    return name.startswith('everyone-pid{}-so'.format(pid)) and name.endswith('npz')


class EveryoneFaceProfileGenerator:
    def __init__(self,
                 data_dir,
                 face: FaceActor,
                 profile_id=120,
                 purpose='train'):

        self.iter_idx = -1
        self.face = face
        self.purpose = purpose
        self.data_dir = data_dir
        self.pid = format_pid(profile_id)
        self.items = self.load_npz()

    def load_npz(self):
        files = os.listdir(self.data_dir)
        files = [name for name in files if is_target_npz(name, self.pid)]

        profile_items = []
        for file in files:
            path = path_join(self.data_dir, file)
            with np.load(path) as npz:
                images = np.array(npz['images'])
                metas = np.array(npz['metas'])
                items = ItemIndex.from_npz(metas, images)
                profile_items = profile_items + items

        return profile_items

    def get_purpose_range(self):
        num_items = len(self.items)
        if self.purpose == 'train':
            return range(0, int(num_items * 0.7))
        elif self.purpose == 'valid':
            return range(int(num_items * 0.7), int(num_items * 0.9))
        elif self.purpose == 'test':
            return range(int(num_items * 0.9), num_items)

    def __len__(self):
        ar = self.get_purpose_range()
        return ar[-1] - ar[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_idx > len(self):
            raise StopIteration
        else:
            self.iter_idx += 1
            return self[self.iter_idx]

    def __getitem__(self, index):
        item = self.items[index]
        # return self.__for_k_model(item)
        return self.__for_eye_crop(item)

    def __for_k_model(self, item):
        frame = byte_arr_to_img(item.image_frame)
        self.face.match(frame)
        self.face.analysis()
        return item, self.face.get_references()

    def __for_eye_crop(self, item):
        frame = byte_arr_to_img(item.image_frame)
        try:
            self.face.match(frame)
            self.face.crop_eye_img()
            return self.face.l_eye_img, self.face.r_eye_img
        except Exception:
            print("parsing error on {}/{}".format(item.pid, item.uid))
        return None, None

