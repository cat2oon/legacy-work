import os

import numpy as np
from keras.utils.data_utils import Sequence


class MPIIGenerator(Sequence):
    def __init__(self, data_dir, batch_size=128, out_hw=(80, 80), purpose='train'):

        self.block_num = 15-1
        self.block_size = 3000
        self.batch_size = batch_size
        self.dataset_dir = data_dir

    def __len__(self):
        """num batches per epoch"""
        total_length = self.block_num * self.block_size
        return int(np.floor(total_length / self.batch_size))

    def __getitem__(self, batch_idx):
        """get one batch"""
        batch_size = self.batch_size
        item_idxs = list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        block_range = self.compute_block(item_idxs)
        return self.__data_generation(block_range)

    def compute_block(self, item_indexes):
        """block id and pos range"""
        size = self.block_size
        first, last = item_indexes[0], item_indexes[-1]
        first_idx, last_idx = int(first / size), int(last / size)
        first_pos, last_pos = first % size, last % size

        if first_idx >= self.test_idx:
            first_idx = first_idx + 1
        if last_idx >= self.test_idx:
            last_idx = last_idx + 1

        if first_idx != last_idx:
            return [(first_idx, (first_pos, size)), (last_idx, (0, last_pos + 1))]
        return [(first_idx, (first_pos, last_pos + 1))]

    def __data_generation(self, block_range):
        """read batch items from npz"""
        images = poses = gazes = None
        for block_id, pos_range in block_range:
            path = os.path.join(self.dataset_dir, 'p{:03}-exp.npz'.format(block_id))
            with np.load(path) as fin:
                i = fin['image'][pos_range[0]:pos_range[1]]
                p = fin['pose'][pos_range[0]:pos_range[1]]
                g = fin['gaze'][pos_range[0]:pos_range[1]]
            if images is None:
                images = i
                poses = p
                gazes = g
            else:
                images = np.vstack((images, i))
                poses = np.vstack((poses, p))
                gazes = np.vstack((gazes, g))

        return [images, poses], gazes


class MPIIGazeValGeneratorExp(Sequence):
    def __init__(self, dataset_dir, test_block_id=3, batch_size=128):
        self.block_size = 3000
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.test_idx = test_block_id

    def __len__(self):
        return int(np.floor(self.block_size / self.batch_size))

    def __getitem__(self, batch_idx):
        idx_range = [batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size]
        return self.__data_generation(idx_range)

    def __data_generation(self, idx_range):
        path = os.path.join(self.dataset_dir, 'p{:03}-exp.npz'.format(self.test_idx))
        with np.load(path) as fin:
            images = fin['image'][idx_range[0]:idx_range[1]]
            poses = fin['pose'][idx_range[0]:idx_range[1]]
            gazes = fin['gaze'][idx_range[0]:idx_range[1]]
        return [images, poses], gazes

