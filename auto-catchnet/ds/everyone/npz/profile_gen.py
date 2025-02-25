import os

from ac.common.images import *
from ac.filesystem.paths import path_join
from ds.core.purpose import Purpose
from ds.everyone.model.item_index import ItemIndex


def format_pid(num_profile_id):
    return "{:05d}".format(num_profile_id)


def is_target_npz(name, pid):
    return name.startswith('everyone-pid{}-so'.format(pid)) and name.endswith('npz')


class EveryoneProfileGenerator:
    def __init__(self,
                 data_dir,
                 profile_id=120,
                 purpose=Purpose.TRAIN):

        self.iter_idx = -1
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
        purpose = self.purpose
        num_items = len(self.items)

        if purpose.for_train():
            return range(0, int(num_items * 0.7))
        elif purpose.for_valid():
            return range(int(num_items * 0.7), int(num_items * 0.9))
        elif purpose.for_test():
            return range(int(num_items * 0.9), num_items)
        elif purpose.for_all():
            return num_items

    def reset_iterator(self):
        self.iter_idx = -1

    def __len__(self):
        ar = self.get_purpose_range()
        return ar[-1] - ar[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_idx > len(self):
            self.reset_iterator()
            raise StopIteration
        else:
            self.iter_idx += 1
            return self[self.iter_idx]

    def __getitem__(self, index)-> ItemIndex:
        return self.items[index]
