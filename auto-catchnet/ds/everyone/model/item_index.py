from ac.common.images import *
from ac.common.utils import make_counter
from ac.filesystem.paths import path_join, mkdir
from ac.langs.sequences import set_attrs_from_seq
from ac.langs.logics import no_none, is_not_empty

from ds.device.infos import get_cam_to_screen
from ds.everyone.model.measures import KEY_TO_ATTR
from ds.everyone.model.measures import MeasureKeys as Keys


class ItemIndex:
    counter = make_counter()

    def __init__(self, measures=None, metas=None, images=None):
        self.uid = -1
        self.iid = None
        self.pid = None
        self.device = None

        self.point_x = None
        self.point_y = None
        self.camera_x = None
        self.camera_y = None
        self.camera_to_screen_x = None
        self.camera_to_screen_y = None
        self.screen_h = None
        self.screen_w = None
        self.orientation = None
        self.face_grid_x = None
        self.face_grid_y = None
        self.face_grid_w = None
        self.face_grid_h = None
        self.face_grid_valid = None

        self.path_frame = None
        self.path_left_eye = None
        self.path_right_eye = None
        self.path_face = None

        self.image_frame = None
        self.image_left_eye = None
        self.image_right_eye = None
        self.image_face = None

        if measures is not None:
            self.init_from_measures(measures)
        elif no_none(metas, images):
            self.init_from_npz(metas, images)

    def init_from_measures(self, measures):
        for key, value in measures.items():
            if value is None:
                continue
            attr = KEY_TO_ATTR.get(key, None)
            if attr is None:
                continue
            setattr(self, attr, value)
        ItemIndex.counter()
        self.uid = ItemIndex.counter.count

    def init_from_npz(self, metas, images):
        self.load_writable_metas(metas)
        if is_not_empty(images):
            self.load_writable_images(images)

    def load_images(self):
        # notice: image encoded byte array
        self.image_face = read_img_to_byte_arr(self.path_face)
        self.image_frame = read_img_to_byte_arr(self.path_frame)
        self.image_left_eye = read_img_to_byte_arr(self.path_left_eye)
        self.image_right_eye = read_img_to_byte_arr(self.path_right_eye)

    def decode_images(self):
        self.image_face = byte_arr_to_img(self.image_face)
        self.image_frame = byte_arr_to_img(self.image_frame)
        self.image_left_eye = byte_arr_to_img(self.image_left_eye)
        self.image_right_eye = byte_arr_to_img(self.image_right_eye)

    def validate(self):
        # if not is_readable_images(self.path_frame, self.path_left_eye, self.path_right_eye):
        # log.info(">>> broken image [%s-%s]" % (self.pid, self.iid))
        # return False
        if is_screen_out(self.screen_w, self.screen_h, self.point_x, self.point_y):
            print(">>> screen out [{}, {},] hw({},{}), yx({},{}) ".format(self.pid, self.iid, self.screen_h,
                                                                          self.screen_w, self.point_y, self.point_x))
            return False
        return True     # return self.face_grid_valid

    def select_writable_metas(self):
        return [self[key] for key in Keys.writable_meta_keys()]

    def select_writable_images(self):
        return [self[key] for key in Keys.writable_image_keys()]

    def load_writable_metas(self, metas):
        set_attrs_from_seq(self, metas, Keys.writable_meta_keys())

    def load_writable_images(self, images):
        set_attrs_from_seq(self, images, Keys.writable_image_keys())

    def get_face_grid(self):
        face_grid = np.zeros(shape=(25, 25, 1), dtype=np.float32)
        grid_x = int(self.face_grid_x)
        grid_y = int(self.face_grid_y)
        grid_xe = grid_x + int(self.face_grid_w)
        grid_ye = grid_y + int(self.face_grid_h)
        face_grid[grid_y:grid_ye, grid_x:grid_xe, 0] = 1
        return face_grid

    def clear(self):
        self.image_frame = None
        self.image_left_eye = None
        self.image_right_eye = None
        self.image_face = None

    """ sugars """
    def is_pad(self):
        return "IPAD" in self.device.upper()

    def __getitem__(self, item):
        if type(item) is Keys:
            item = KEY_TO_ATTR.get(item, None)
        if item is not None and hasattr(self, item):
            return getattr(self, item)
        return None

    def __setitem__(self, key, value):
        if type(key) is not Keys:
            return
        attr = KEY_TO_ATTR.get(key, None)
        if attr is not None:
            setattr(self, attr, value)

    def __str__(self):
        return "ID:{} path:{}".format(self.uid, self.path_frame)

    @staticmethod
    def make(file_name, index_meta):
        measures = ItemIndex.to_measures(file_name, index_meta)
        return ItemIndex(measures)

    @staticmethod
    def to_measures(file_name, meta):
        measures = Keys.make_empty_measures()

        face = meta['face']  # 크롭 이미지 있어서 사용 안 함
        left_eye = meta['left_eye']  # 크롭 이미지 있어서 사용 안함
        right_eye = meta['right_eye']  # 크롭 이미지 있어서 사용 안함

        info = meta["info"]
        dots = meta['dots']
        grid = meta['grid']
        frames = meta['frames']
        screens = meta['screens']

        idx = frames.index(file_name)
        profile_id = meta['pid']
        profile_path = meta['profile_path']

        device_name = info["DeviceName"]
        sx, sy = get_cam_to_screen(device_name)
        measures[Keys.CAMERA_TO_SCREEN_X] = sx
        measures[Keys.CAMERA_TO_SCREEN_Y] = sy

        measures[Keys.IID] = idx
        measures[Keys.PID] = profile_id
        measures[Keys.DEVICE_NAME] = device_name

        measures[Keys.POINT_X] = dots["XPts"][idx]
        measures[Keys.POINT_Y] = dots["YPts"][idx]
        measures[Keys.CAMERA_X] = dots["XCam"][idx]
        measures[Keys.CAMERA_Y] = dots["YCam"][idx]
        measures[Keys.SCREEN_H] = screens["H"][idx]
        measures[Keys.SCREEN_W] = screens["W"][idx]
        measures[Keys.FACE_GRID_X] = grid["X"][idx]
        measures[Keys.FACE_GRID_Y] = grid["Y"][idx]
        measures[Keys.FACE_GRID_W] = grid["W"][idx]
        measures[Keys.FACE_GRID_H] = grid["H"][idx]
        measures[Keys.FACE_VALID] = grid["IsValid"][idx]
        measures[Keys.ORIENTATION] = screens["Orientation"][idx]

        measures[Keys.FRAME_PATH] = path_join(profile_path, "frames", file_name)
        measures[Keys.EYE_LEFT_PATH] = path_join(profile_path, "appleLeftEye", file_name)
        measures[Keys.EYE_RIGHT_PATH] = path_join(profile_path, "appleRightEye", file_name)
        measures[Keys.FACE_PATH] = path_join(profile_path, "appleFace", file_name)

        return measures

    @staticmethod
    def to_npz(out_dir, idx, item_block):
        mkdir(out_dir)
        metas = [item.select_writable_metas() for item in item_block]
        metas = np.asarray(metas)
        metas = metas.transpose()

        images = [item.select_writable_images() for item in item_block]
        images = np.asarray(images)
        images = images.transpose()

        path = path_join(out_dir, "item-{:05d}".format(idx))
        np.savez_compressed(path, metas=metas, images=images)

    @staticmethod
    def to_profile_npz(out_dir, profile_id, orientation_id, idx, items):
        mkdir(out_dir)
        metas = [item.select_writable_metas() for item in items]
        metas = np.asarray(metas)
        metas = metas.transpose()

        images = [item.select_writable_images() for item in items]
        images = np.asarray(images)
        images = images.transpose()

        path = path_join(out_dir, "everyone-pid{}-so{}-idx{}".format(profile_id, orientation_id, idx))
        np.savez_compressed(path, metas=metas, images=images)

    @staticmethod
    def from_npz(metas, images):
        item_block = []
        for i in range(metas.shape[1]):
            item_metas = metas[:, i]
            item_images = images[:, i] if images is not None else []
            item = ItemIndex(metas=item_metas, images=item_images)
            item_block.append(item)
        return item_block

