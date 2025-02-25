import os

from keras.utils.data_utils import Sequence

from ac.common.images import *
from ac.filesystem.paths import path_join
from ds.everyone.model.item_index import ItemIndex


def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data.astype('float32') / 255.
    data = data - np.mean(data, axis=0)
    return np.reshape(data, shape)


def prepare_data(data):
    eye_left, eye_right, face, face_mask, y = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    return [eye_left, eye_right, face, face_mask, y]


class EveryoneGenerator(Sequence):

    def __init__(self, data_dir, batch_size=128, block_size=512, purpose='train'):
        self.purpose = purpose
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_blocks = self.count_npz()
        self.purpose_range = self.compute_range()

    def count_npz(self):
        files = os.listdir(self.data_dir)
        count = len([name for name in files if name.endswith('npz')])
        count /= 2  # metrics + item
        count = count - 1  # discard last item
        return count

    def compute_range(self):
        num = self.num_blocks
        if self.purpose == 'train':
            return range(0, int(num * 0.7))
        elif self.purpose == 'valid':
            return range(int(num * 0.7), int(num * 0.9))
        elif self.purpose == 'test':
            return range(int(num * 0.9), num)

    def __len__(self):
        """ num batches per epoch """
        num_total = len(self.purpose_range) * self.block_size
        return int(math.floor(num_total / self.batch_size))

    def __getitem__(self, index):
        index += int((self.purpose_range[0]) * self.block_size / self.batch_size)
        pos = index % int(self.block_size / self.batch_size)
        block_id = int(index * self.batch_size / self.block_size)
        item_block, heads, dists, eyes = self.__load_block(block_id, pos)
        return self.__generation(item_block, heads, dists, eyes)

    def __load_block(self, block_id, pos):
        name = "item-{:05d}.npz".format(block_id)
        path = path_join(self.data_dir, name)
        npz = np.load(path)

        stride = self.batch_size
        start = pos * stride
        end = start + stride

        name = "item-metrics-{:05d}.npz".format(block_id)
        path = path_join(self.data_dir, name)
        metrics_npz = np.load(path)

        images = npz['images'][:, start:end]  # (3,  512->128)
        indexes = npz['indexes'][:, start:end]  # (18, 512->128)

        heads = metrics_npz['head'][start:end, :]  # (512->128, 2)
        dists = metrics_npz['dist'][start:end]  # (512->128, 1)
        eyes = metrics_npz['eye'][start:end]  # (512->128, 4->x)

        return ItemIndex.from_npz(indexes, images), heads, dists, eyes

    def __generation(self, item_block, heads, dists, eyes):
        y = []
        eyes_in = []
        heads_in = []
        dists_in = []
        eye_grids_in = []

        for idx, item in enumerate(item_block):
            eye, head, dist, eye_grid = self.item_to_x(item, idx, heads, dists, eyes)
            eyes_in.append(eye)
            heads_in.append(head)
            dists_in.append(dist)
            eye_grids_in.append(eye_grid)
            y.append([item.camera_x, item.camera_y])

        x = [np.asarray(eyes_in),
             np.asarray(heads_in),
             np.asarray(dists_in),
             np.asarray(eye_grids_in)]
        return x, np.asarray(y)

    def item_to_x(self, item, idx, heads, dists, eyes):
        le = byte_arr_to_img(item.image_left_eye)
        re = byte_arr_to_img(item.image_right_eye)
        dual_eye = make_dual_channel_image(le, re, shape=(36, 60))

        head_pose = heads[idx]
        dist = dists[idx]
        eye = eyes[idx]

        # screen_size = [item.screen_w, item.screen_h]
        if eye == [-1, -1]:
            eye = [-1, -1, -1, -1]

        return dual_eye, head_pose, dist, eye

