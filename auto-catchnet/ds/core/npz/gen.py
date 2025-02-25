import math
from abc import *

import numpy as np
from keras.utils.data_utils import Sequence

from ac.common.nps import clone_npz
from al.maths.block_seq import get_block_range
from ds.core.npz.meta import NPZMeta
from ds.core.purpose import Purpose


class NPZGenerator(ABC, Sequence):
    def __init__(self,
                 npz_base_path,
                 batch_size=128,
                 purpose: Purpose = Purpose.TRAIN):
        self.meta = None
        self.block_size = 0
        self.num_batches = 0
        self.purpose = purpose
        self.batch_size = batch_size
        self.npz_base_path = npz_base_path

        """ cache block """
        self.cache_path = None
        self.cache_npz = {}

        self.scan_npz()
        self.compute_batches()

    def scan_npz(self):
        meta = NPZMeta(self.npz_base_path)
        if not meta.check_exist():
            meta.build_from_base_path()
        meta.load()
        self.meta = meta
        self.block_size = meta.num_items_in_block
        assert meta.verify(), "*** [ERROR] broken dataset ***"

    def __len__(self):
        return self.num_batches  # 나머지 마지막 잉여 배치는 버림

    def compute_batches(self):
        num_items = self.meta.num_total_items
        start, last = self.get_purpose_portion()
        purpose_item_range = range(int(num_items * start), int(num_items * last))
        num_items_in_purpose = len(purpose_item_range)

        if num_items_in_purpose == 0:
            num_batches = 0
        elif num_items_in_purpose < self.batch_size:
            num_batches = 1
        else:
            num_batches = int(math.floor(num_items_in_purpose / self.batch_size))

        self.num_batches = num_batches

    def get_purpose_portion(self):
        purpose = self.purpose
        if purpose.for_train():
            return 0.0, 0.7
        elif purpose.for_valid():
            return 0.7, 0.9
        elif purpose.for_test():
            return 0.9, 1.0
        elif purpose.for_all():
            return 0.0, 1.0
        return None

    def load_npz(self, batch_index):
        blocks, start, end = get_block_range(self.block_size, self.batch_size, batch_index)
        npz_paths = [self.meta.get_npz_path(block_idx) for block_idx in blocks]
        npz_paths = list(filter(None.__ne__, npz_paths))

        assert len(npz_paths) != 0, "failed to find npz path in {}".format(batch_index)

        self.load_npz_cache(npz_paths[0])
        if len(npz_paths) == 2:
            npz_to_append = np.load(npz_paths[1])
            self.append_npz_cache(npz_to_append)
        npz = self.cache_npz

        return npz, start, end

    def load_npz_cache(self, npz_path):
        if npz_path == self.cache_path:
            return
        with np.load(npz_path) as npz:
            self.cache_path = npz_path
            self.cache_npz = clone_npz(npz)

    def append_npz_cache(self, npz):
        npz_cached = self.cache_npz
        npz_to_append = clone_npz(npz)
        npz_merged = {}
        for key in npz_cached.keys():
            npz_merged[key] = np.concatenate([npz_cached[key], npz_to_append[key]])
        self.cache_npz = npz_merged

    def __getitem__(self, batch_index):
        npz, start, end = self.load_npz(batch_index)
        return self.transform(npz, start, end)

    @abstractmethod
    def transform(self, npz, start, end):
        pass
