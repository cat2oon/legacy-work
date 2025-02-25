from __future__ import division, print_function, absolute_import

import os
import glob
import json
import math
import logging
import cv2 as cv
import numpy as np
import tensorflow as tf

from collections import OrderedDict


"""
    Device Screen Camera Difference (cm)
"""
loc_meta = {
    'iphone 6s plus':   [ 23.54  ,8.66   ,68.36  ,121.54 ],
    'iphone 6s':        [ 18.61  ,8.04   ,58.49  ,104.05 ],
    'iphone 6 plus':    [ 23.54  ,8.65   ,68.36  ,121.54 ],
    'iphone 6':         [ 18.61  ,8.03   ,58.5   ,104.05 ],
    'iphone 5s':        [ 25.85  ,10.65  ,51.7   ,90.39  ],
    'iphone 5c':        [ 25.85  ,10.64  ,51.7   ,90.39  ],
    'iphone 5':         [ 25.85  ,10.65  ,51.7   ,90.39  ],
    'iphone 4s':        [ 14.96  ,9.78   ,49.92  ,74.88  ],
    'ipad mini':        [ 60.7   ,8.7    ,121.3  ,161.2  ],
    'ipad air 2':       [ 76.86  ,7.37   ,153.71 ,203.11 ],
    'ipad air':         [ 74.4   ,9.9    ,149    ,198.1  ],
    'ipad 4':           [ 74.5   ,10.5   ,149    ,198.1  ],
    'ipad 3':           [ 74.5   ,10.5   ,149    ,198.1  ],
    'ipad 2':           [ 74.5   ,10.5   ,149    ,198.1  ],
    'ipad pro':         [ 98.31  ,10.69  ,196.61 ,262.15 ]
}


"""
    Exclude Profiles
"""
exclude_profiles = [
    "00117", "00124", "00133", "00198", "00207", "00229", "00251", "00252", "00258", "00266",
    "00322", "00383", "00463", "00500", "00521", "00549", "00595", "00597", "00653", "00696",
    "00740", "00748", "00779", "00808", "00828", "00861", "00876", "00880", "00890", "00939",
    "00930", "00932", "00955", "00960", "00976", "00998", "01001", "01029", "01030", "01066",
    "01099", "01109", "01122", "01126", "01134", "01185", "01206", "01224", "01225", "01267",
    "01282", "01350", "01366", "01367", "01372", "01392", "01432", "01443", "01474", "01544",
    "01556", "01661", "01676", "01702", "01805", "01809", "01819", "01859", "01876", "01896",
    "01939", "02002", "02027", "02032", "02033", "02048", "02117", "02119", "02155", "02165",
    "02174", "02190", "02194", "02223", "02243", "02353", "02364", "02417", "02456", "02526",
    "02533", "02542", "02551", "02622", "02739", "02840", "02976", "02984", "03007", "03039",
    "03059", "03060", "03212", "03224", "03239", "03380", "03389", "03474"   
]


"""
    Free functions
"""
def grep_recur(base_path, pattern="*.*"):
    sub_greps = list(chain(*[grep_recur(dp, pattern) for dp in grep_dirs(base_path)]))
    return grep_files(base_path, pattern) + sub_greps

def grep_files(base_path, pattern="*.*"):
    return glob.glob("{}/{}".format(base_path, pattern))

def grep_dirs(base_path):
    file_paths = [path_join(base_path, name) for name in os.listdir(base_path)]
    return [p for p in file_paths if os.path.isdir(p)]

def byte_arr_to_img(byte_arr):
    dt = np.dtype(np.int8)  # 255
    dt = dt.newbyteorder('>')  # depend on mach arch
    np_arr = np.frombuffer(byte_arr, dt)
    return cv.imdecode(np_arr, 1)

def setup_logger():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    
"""
    GazeCaptureNpzSequence
"""
class GazeCaptureNpzSequence(tf.keras.utils.Sequence):

    @classmethod
    def create(cls, ctx, data_tag):
        return GazeCaptureNpzSequence(data_tag, 
                                      ctx.npz_root_path, 
                                      ctx.resource_path, 
                                      ctx.batch_size,
                                      ctx.seed,
                                      exclude_profiles)
    
    def __init__(self, 
                 data_tag, 
                 npz_root_path, 
                 resource_path, 
                 batch_size=64, 
                 seed=1234,
                 exclude_profiles=None, 
                 custom_getter=None):
        
        """ TODO: props 통합하기 """
        self.tag = data_tag
        self.batch_size = batch_size
        self.custom_getter = custom_getter
        self.npz_root_path = npz_root_path
        self.resource_path = resource_path
        self.exclude_pids = exclude_profiles

        self.setup_props()
        self.setup_random_seed(seed)
        self.prepare_index()
        self.prepare_batch_index(batch_size)
        self.summary_index()

    def setup_props(self):
        self.index = None
        self.num_cache = 3
        self.npz_cache = {}
        self.batch_index = []
        self.num_min_items = 150
        self.undistort = Undistorter()
        self.index_pattern = "{}-index.json"
        self.npz_pattern = "profile-recode-{:05d}.npz"
        self.split_json_path = 'gazecapture_split.json'
        
    def setup_random_seed(self, seed):
        np.random.seed(seed)

    def __len__(self):
        return len(self.batch_index)
    
    def get_batch_size(self):
        return self.batch_size

    def get_num_items(self):
        return self.index['num_items']
    
    """
        Batch Iterator
    """
    def get_index_iterator(self):
        for pid in self.index:
            for iid in self.index[pid]:                   
                yield pid, iid
        return None, None

    def get_batch_iterator(self, batch_size):
        gen = self.get_index_iterator()
        while True:
            bag = {}
            for i in range(batch_size):
                pid, iid = next(gen)
                if pid is None:
                    yield None
                if pid not in bag:
                    bag[pid] = []
                bag[pid].append(iid)
            yield bag
    
    def prepare_batch_index(self, batch_size):
        batch_iter = self.get_batch_iterator(batch_size)
        while True:
            try:
                batch = next(batch_iter) 
                self.batch_index.append(batch)
            except:
                break
        logging.info("Complete build %s batch index", self.tag)
    
    """
        Index
    """
    def summary_index(self):
        """ 훈련 데이터셋 통계 내기"""
        # for pid, items in self.index.items():
        #     print("Profile %s items[%d]" % (pid, len(items)))
        pass
    
    def prepare_index(self, overwrite=False):
        split_path = self.get_split_info_path()
        split_info = self.load_spilt_info(split_path)
        index_path = self.get_index_path(self.tag)
        
        if os.path.exists(index_path) and not overwrite:
            self.index = self.load_index(index_path)
            return
        
        num_items = 0
        item_index = OrderedDict()
        profiles = split_info[self.tag]
        profiles = [pid for pid in profiles if not pid in self.exclude_pids]
        for pid in profiles:
            items = self.index_valid_items(pid)
            if len(items) < self.num_min_items:
                logging.info("Ignored profile[%s] too small items[%d]" % (pid, len(items)))
            else:
                """ TODO: calibration set 랜덤으로 분리해두기 """
                item_index[pid] = items
                num_items = num_items + len(items)
        
        item_index['num_items'] = num_items
        self.write_index(index_path, item_index)
        self.index = item_index

    def index_valid_items(self, pid):
        npz = np.load(self.get_npz_path(pid), mmap_mode='r', allow_pickle=True)
        summary = npz['summary'].tolist()
        device = summary['device'].lower()
        
        if device not in loc_meta:
            logging.info("can't find device info %s" % pid)
            return []

        items = []
        metas = npz['metas']
        for idx in range(len(metas)):
            m = metas[idx]
            if not m['left_eye_valid'] or not m['right_eye_valid']:
                logging.info("invalid eye info at profile:%s item_idx:%d" % (pid, idx))
                continue
            # TODO: 음수 face_rect 포함들 
            items.append(idx)
        return items

    
    """
        File System
    """
    def get_split_info_path(self):
        return os.path.join(self.resource_path, self.split_json_path)
    
    def load_spilt_info(self, path):
        with open(path, 'r') as f:
            split_info = json.load(f)
        return split_info
    
    def get_index_path(self, tag):
        return os.path.join(self.npz_root_path, self.index_pattern.format(tag))
    
    def load_index(self, index_path):
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_meta = json.load(f)
                return index_meta
    
    def write_index(self, index_path, index):
        with open(index_path, 'w') as out:
            json.dump(index, out)
            
    def get_npz_path(self, profile_id):
        return os.path.join(self.npz_root_path, self.npz_pattern.format(int(profile_id)))
    
    def load_npz(self, npz_path):
        return np.load(npz_path, mmap_mode='r', allow_pickle=True)
    
    def recycle_cache(self, num_max_cache):
        num_cache = len(self.npz_cache)
        if num_cache < num_max_cache:
            return
        for key in list(self.npz_cache.keys())[:-int(num_cache/2)]:
            del self.npz_cache[key]

    def fetch_npz(self, profile_id):
        if str(profile_id) in self.npz_cache:
            cache = self.npz_cache[profile_id]
            return cache
        
        self.recycle_cache(self.num_cache)
        npz = self.load_npz(self.get_npz_path(profile_id))
        self.npz_cache[profile_id] = npz
        return npz

    
    """
        Item
    """
    def preprocess_image(self, image, resize_wh):
        image = cv.resize(image, resize_wh, interpolation=cv.INTER_AREA)
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        image = 2.0 * image / 255.0 - 1
        return image
    
    def eye_corner_pos_to_uv(self, eye_corner, camera_mat):
        # eye corner landmark [ x1 x2 x3 x4 y1 y2 y3 y4 ]
        ec, cm = eye_corner, camera_mat
        fx, fy, cx, cy = cm[0, 0], cm[1, 1], cm[0, 2], cm[1, 2]
        xs, ys = ec[0:4], ec[4:]
        xs = (xs - cx) / fx    
        ys = (ys - cy) / fy    
        return np.hstack([xs, ys])
    
    def __getitem__(self, idx):
        if self.custom_getter is not None:
            return custom_getter(self, idx)
        return self.fetch_batch(idx)
    
    def npz_to_batch_items(self, npz, pid, item_ids):
        metas = npz['metas']
        frames = npz['frames']
        summary = npz['summary'].tolist()
        device = summary['device'].lower()
        loc = loc_meta[device]

        batch_items = []
        for item_id in item_ids:
            batch_item = self.get_batch_item(pid, item_id, metas, frames, device, loc)
            batch_items.append(batch_item)

        return batch_items
    
    def get_batch_item(self, pid, id, metas, frames, device, loc):
        meta = metas[id]
        frame = frames[id]
        face_rect = meta['face_rect']
        le_rect = meta['left_eye_rect']
        re_rect = meta['right_eye_rect']
        orientation = meta['orientation']
        
        frame = byte_arr_to_img(frame)
        frame = self.undistort(frame, meta['origin_camera_param'], meta['camera_distortion'])
        ec = get_eye_corner(face_rect, le_rect, re_rect, frame.shape[:2])
        loc = self.eye_corner_pos_to_uv(ec, meta['origin_camera_param'])
        target_xy = np.array([meta['target_dist']['x'], meta['target_dist']['y']])
        le_img, re_img = get_eye_left_right_patch(frame, face_rect, le_rect, re_rect)
        
        """flip left eye """
        le_img = np.fliplr(le_img)

        try:
            le_img = self.preprocess_image(le_img, (64, 64))
            re_img = self.preprocess_image(re_img, (64, 64))
        except Exception as e:
            print("Pid[%s] id[%d]" % (pid, id), e)

        # batch_item = [ pid, le_img, re_img, target_xy, orientation ]
        batch_item = [ le_img, re_img, loc, target_xy ]
        return batch_item
    
    def fetch_batch(self, batch_idx):
        """ TODO 연산 최적화 """
        batch_items = []
        batch = self.batch_index[batch_idx]
        for pid in batch.keys():
            item_ids = batch[pid]
            npz = self.fetch_npz(pid)
            items = self.npz_to_batch_items(npz, pid, item_ids)
            batch_items = batch_items + items
        return self.encode_batch_named_dict(batch_items)

    def encode_batch_named_dict(self, batch_items):
        le, re, ec, xy = [], [], [], []
        num_batch = len(batch_items)
        for i in range(num_batch):
            record = batch_items[i]
            le.append(record[0])
            re.append(record[1])
            ec.append(record[2])
            xy.append(record[3])
                
        targets = np.array(xy)
#         inputs = {
#             'left_eye_patch': np.array(le),
#             'right_eye_patch': np.array(re),
#             'eye_corner_landmark': np.array(ec) }
        inputs = (np.array(le), np.array(re), np.array(ec))
        return inputs, targets
        

"""
    Free function (TODO: 실수 있는지 체크)
    
    'screen_hw': {'h': 568, 'w': 320},
    'right_eye_rect': {'h': 103.10, 'w': 103.10, 'x': 32.64, 'y': 89.34}
"""
def rect_to_tl_br(rect):
    h, w, y, x = rect['h'], rect['w'], rect['y'], rect['x']
    h, w, y, x = round(h), round(w), round(y), round(x)
    h, w, y, x = max(0, h), max(0, w), max(0, y), max(0, x)  # Profile 33번 음수 버그 있음
    return y, x, y+h, x+w

def get_eye_corner(face_rect, left_eye_rect, right_eye_rect, image_hw):
    # h, w = image_hw
    top, left, bottom, right = rect_to_tl_br(face_rect)
    ltop, lleft, lbottom, lright = rect_to_tl_br(left_eye_rect)
    rtop, rleft, rbottom, rright = rect_to_tl_br(right_eye_rect)

    # convert to frame base
    ltop, lleft, lbottom, lright = top+ltop, left+lleft, top+lbottom, left+lright
    rtop, rleft, rbottom, rright = top+rtop, left+rleft, top+rbottom, left+rright

    # normalized eye corner landmark  --> 이 부분은 txm 레이어에서 수행
    # ltop, lleft, lbottom, lright = ltop/h, lleft/w, lbottom/h, lright/w
    # rtop, rleft, rbottom, rright = rtop/h, rleft/w, rbottom/h, rright/w

    # [ x x x x y y y y ]
    return [ltop, lbottom, rtop, rbottom, lleft, lright, rleft, rright]

def get_eye_left_right_patch(frame, face_rect, left_rect, right_rect):
    top, left, bottom, right = rect_to_tl_br(face_rect)
    ltop, lleft, lbottom, lright = rect_to_tl_br(left_rect)
    rtop, rleft, rbottom, rright = rect_to_tl_br(right_rect)

    #     face = frame[top:bottom, left:right]       # Height x Width x Channels
    #     le_img = face[ltop:lbottom, lleft:lright]
    #     re_img = face[rtop:rbottom, rleft:rright]
    le_img = frame[top+ltop:top+lbottom, left+lleft:left+lright, :]
    re_img = frame[top+rtop:top+rbottom, left+rleft:left+rright, :]

    return le_img, re_img

    
    
"""
    Undistorter
"""
class Undistorter:
    _map = None
    _prev_param = None

    def should_parameter_update(self, all_params):
        return self._prev_param is None or len(self._prev_param) != len(all_params) \
            or not np.allclose(all_params, self._prev_param)

    def update_undistort_map(self, cam_mat, dist_coef, img_wh, all_params):
        self._map = cv.initUndistortRectifyMap(cam_mat, dist_coef, R=None,
                                               size=img_wh, m1type=cv.CV_32FC1, 
                                               newCameraMatrix=cam_mat)
        self._prev_param = np.copy(all_params)

    def __call__(self, image, camera_matrix, distortion):
        h, w, _ = image.shape
        all_params = np.concatenate([camera_matrix.flatten(), distortion.flatten(), [h, w]])

        if self.should_parameter_update(all_params):
            self.update_undistort_map(camera_matrix, distortion, (w, h), all_params)

        return cv.remap(image, self._map[0], self._map[1], cv.INTER_LINEAR)
 