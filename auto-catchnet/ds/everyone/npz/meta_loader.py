import os
import sys
import glob

import numpy as np

from ds.everyone.model.item_index import ItemIndex


class MetaLoader:
    def __init__(self, path, grep_pattern="*"):
        self.archive = {}
        self.npz_dir_path = path
        self.grep_pattern = grep_pattern

    def load_npz(self, npz_path):
        npz = np.load(npz_path)
        metas = npz['metas']
        return ItemIndex.from_npz(metas, None)

    def get_block_files(self):
        glob_pattern = "item-{}.npz".format(self.grep_pattern)
        return glob.glob(os.path.join(self.npz_dir_path, glob_pattern))

    def load(self):
        blocks = self.get_block_files()
        print("load npz blocks num: {}".format(len(blocks)))

        archive = {}
        for i, block_path in enumerate(blocks):
            indexes = self.load_npz(block_path)
            for index in indexes:
                archive[index.uid] = index
            if i % 100 == 0:
                sys.stdout.write("loading blocks[{}] ...\n".format(i))

        self.archive = archive
        print("total item-indexes: {}\n".format(len(archive)))

    def __getitem__(self, uid):
        if type(uid) is list:
            uid = str(uid[0])
        if type(uid) is int:
            uid = str(uid)
        if uid not in self.archive:
            return None
        return self.archive[uid]

    def get_all(self, uids):
        items = [self[uid] for uid in uids]
        for item in items:
            item.load_images()
            item.decode_images()
        return {item.uid: item for item in items}
