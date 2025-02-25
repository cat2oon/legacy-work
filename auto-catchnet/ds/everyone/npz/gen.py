import os

from keras.utils.data_utils import Sequence

from ac.common.images import *
from ac.filesystem.paths import path_join
from ds.everyone.model.item_index import ItemIndex


def normalize(img):
    shape = img.shape
    img = np.reshape(img, (shape[0], -1))
    img = img.astype(np.float32) / 255.
    img = img - np.mean(img, axis=0)
    return np.reshape(img, shape)


def item_to_x(item):
    left_eye = byte_arr_to_img(item.image_left_eye)
    right_eye = byte_arr_to_img(item.image_right_eye)
    face = byte_arr_to_img(item.image_face)
    face_mask = item.get_face_grid()

    left_eye = cv2.resize(left_eye, (64, 64))
    # right_eye = cv2.resize(right_eye, (64, 64))
    face = cv2.resize(face, (64, 64))

    left_eye = normalize(left_eye)
    # right_eye = normalize(right_eye)
    face = normalize(face)

    return left_eye, right_eye, face, face_mask


def generation(item_block):
    y = []
    # faces, face_masks = [], []
    # eye_lefts, eye_rights = [], []
    eye_rights = []

    for item in item_block:
        eye_left, eye_right, face, face_mask = item_to_x(item)
        # eye_lefts.append(eye_left)
        eye_rights.append(eye_right)
        # faces.append(face)
        # face_masks.append(face_mask)
        y.append([item.camera_x, item.camera_y])

    # x = [np.asarray(eye_lefts), np.asarray(eye_rights),
    #      np.asarray(faces), np.asarray(face_masks)]
    x = np.asarray(eye_rights)
    y = np.asarray(y)

    return x, y


class EveryoneGenerator(Sequence):

    def __init__(self, data_dir, batch_size=128, block_size=512, purpose='train'):
        self.cache_path = {}
        self.cache_metas = None
        self.cache_images = None

        self.purpose = purpose
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_blocks = self.count_npz()
        self.purpose_range = self.compute_range()

    def count_npz(self):
        files = os.listdir(self.data_dir)
        count = len([name for name in files if name.endswith('npz')])
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
        num_total = len(self.purpose_range) * self.block_size
        return int(math.floor(num_total / self.batch_size))

    def __getitem__(self, index):
        index += int((self.purpose_range[0]) * self.block_size / self.batch_size)
        pos = index % int(self.block_size / self.batch_size)
        block_id = int(index * self.batch_size / self.block_size)
        item_block = self.__load_block(block_id, pos)

        return generation(item_block)

    def __load_block(self, block_id, pos):
        def load_npz(path):
            if path != self.cache_path:
                del self.cache_images
                del self.cache_metas
                with np.load(path) as npz:
                    self.cache_path = path
                    self.cache_images = np.array(npz['images'])
                    self.cache_metas = np.array(npz['metas'])
            return self.cache_images, self.cache_metas

        name = "item-{:05d}.npz".format(block_id)
        path = path_join(self.data_dir, name)

        stride = self.batch_size
        start = pos * stride
        end = start + stride

        images, metas = load_npz(path)
        images = images[:, start:end]
        metas = metas[:, start:end]

        return ItemIndex.from_npz(metas, images)
