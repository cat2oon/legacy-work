import os
import sys

import numpy as np

from ac.common.jsons import load_json, write_json
from ac.common.regexes import get_number_seq
from ac.filesystem.greps import grep_files
from ac.filesystem.paths import basename_in_path


class NPZMeta:
    def __init__(self, base_path):
        self.meta_name = "ds.meta"
        self.base_path = base_path

        """ properties """
        self.version = 0
        self.npz_keys = []
        self.npz_file_index = {}  # (idx, npz_name)
        self.num_npz_blocks = 0
        self.num_total_items = 0
        self.num_items_in_block = 0
        self.dataset_name = "Unknown"

    def check_exist(self):
        return os.path.exists(self.get_meta_path())

    def scan_npzs(self):
        npzs = grep_files(self.base_path, "*.npz")
        self.num_npz_blocks = len(npzs)
        npzs = sorted(npzs)

        npz_index = {}
        count_total, max_num_items = 0, 0
        for npz_path in npzs:
            npz_filename = basename_in_path(npz_path)
            idx = int(get_number_seq(npz_filename))
            npz_index[idx] = npz_filename

            sys.stdout.write("\nload npz index[{:05d}]".format(idx))
            npz = np.load(npz_path)
            items = npz[npz.keys()[0]]
            num_items = items.shape[0]
            count_total += num_items
            if num_items > max_num_items:
                max_num_items = num_items

        self.num_total_items = count_total
        self.num_items_in_block = max_num_items
        self.npz_file_index = npz_index
        print("*** complete build meta total:{} / npzs:{} ***".format(count_total, len(npzs)))

    def build_from_base_path(self):
        print("*** no dataset meta exist. build it now ***")
        self.scan_npzs()
        self.save()

    def verify(self):
        for idx, npz_name in self.npz_file_index.items():
            npz_path = self.get_npz_path(idx)
            if not os.path.exists(npz_path):
                return False
        print("*** meta verify complete [{}] ***".format(self.dataset_name))
        return True

    def get_npz_path(self, idx: int):
        file_path = self.npz_file_index.get(str(idx), None)
        if file_path is None:
            return None
        return os.path.join(self.base_path, file_path)

    def load(self):
        meta = load_json(self.get_meta_path())
        assert meta is not None, "*** dataset meta is none ***"

        self.version = meta["version"]
        self.npz_keys = meta["npz_keys"]
        self.dataset_name = meta["dataset_name"]
        self.num_npz_blocks = meta["num_npz_blocks"]
        self.num_total_items = meta["num_total_items"]
        self.num_items_in_block = meta["num_items_in_block"]
        self.npz_file_index = meta["npz_file_index"]

    def save(self):
        write_json(self.get_meta_path(), self.to_json())

    def to_json(self):
        return {
            "version": self.version,
            "npz_keys": self.npz_keys,
            "dataset_name": self.dataset_name,
            "num_npz_blocks": self.num_npz_blocks,
            "num_total_items": self.num_total_items,
            "num_items_in_block": self.num_items_in_block,
            "npz_file_index": self.npz_file_index
        }

    def get_meta_path(self):
        return os.path.join(self.base_path, self.meta_name)

    def get_num_npz_blocks(self):
        return self.num_npz_blocks

    def get_num_items_in_npz(self):
        return self.num_items_in_block

    def get_num_total_items(self):
        return self.num_items_in_block
