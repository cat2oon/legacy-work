import os
import time
import glob
import pickle
import random

import cv2 as cv
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from collections import namedtuple as nt


"""
    Free Function
"""
def grep_recur(base_path, pattern="*.*"):
    from itertools import chain
    sub_greps = list(chain(*[grep_recur(dp, pattern) for dp in grep_dirs(base_path)]))
    return grep_files(base_path, pattern) + sub_greps

def grep_files(base_path, pattern="*.*"):
    import glob
    return glob.glob("{}/{}".format(base_path, pattern))

def grep_dirs(base_path):
    file_paths = [os.path.join(base_path, name) for name in os.listdir(base_path)]
    return [p for p in file_paths if os.path.isdir(p)]

def timeit(func):
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('%r  %2.2f s' % (func.__name__, (end - start)))
        return result
    return timed

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

    # [ x x x x y y y y ]
    return [ltop, lbottom, rtop, rbottom, lleft, lright, rleft, rright]

def get_eye_left_right_patch(frame, face_rect, left_rect, right_rect):
    top, left, bottom, right = rect_to_tl_br(face_rect)
    ltop, lleft, lbottom, lright = rect_to_tl_br(left_rect)
    rtop, rleft, rbottom, rright = rect_to_tl_br(right_rect)
    le_img = frame[top+ltop:top+lbottom, left+lleft:left+lright, :]
    re_img = frame[top+rtop:top+rbottom, left+rleft:left+rright, :]

    return le_img, re_img

def get_face_patch(frame, face_rect):
    top, left, bottom, right = rect_to_tl_br(face_rect)
    face = frame[top:bottom, left:right, :]
    return face

def is_valid_item(m):
    return m['face_valid'] and m['face_grid_valid']  and m['left_eye_valid'] and m['right_eye_valid']

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

    

"""
    DataProvider
"""
class DataProvider():
    
    def __init__(self, ctx, pid, use_gc_eye=False):
        self.pid = pid
        self.ctx = ctx
        self.use_gc_eye = use_gc_eye
        self.setup_random_seed(ctx.seed)
        self.define_type()
        self.setup_dtype(ctx)
        self.load_data()
        self.index_data()
        # self.undistort = Undistorter()
        
    def setup_random_seed(self, seed):
        np.random.seed(seed)
        
    def setup_dtype(self, ctx):
        self.int_dtype = np.int64 if ctx.use_64bits else np.int32
        self.float_dtype = np.float64 if ctx.use_64bits else np.float32
        
    def load_profile_meta(self):
        with open(self.profile_meta_path, 'rb') as f:
            return pickle.load(f)
            
    def load_data(self):
        metas = self.load_profile_meta()
        self.metas = metas['items']
        self.summary = metas['summary']
        
        frame_npz = np.load(self.profile_frame_path, mmap_mode='r')
        self.frames = frame_npz['frames']
        self.frame_index = list(frame_npz['index'])
        self.remove_missing_data()
            
    def remove_missing_data(self):
        missing_ids = []
        for i, m in enumerate(self.metas):
            if m['frame_name'][:-4] not in self.frame_index:
                missing_ids.append(i)
        for id in missing_ids:
            self.metas.pop(id)
            print("pop meta item", id)

    def get_filtered_item_ids(self, shuffle=True, orientation=None):
        ids = list(range(0, len(self.metas)))
        if self.use_gc_eye:
            ids = [id for id in ids if is_valid_item(self.metas[id])]  
        if orientation is not None:
            ids = [id for id in ids if self.metas[id]['orientation'] == orientation]  
        # print("num filtered id: ", len(ids))
        if shuffle:
            np.random.shuffle(ids)
        return ids
        
    def index_data(self):
        num_items_per_shot = self.ctx.num_k_shot + self.ctx.num_valid_shot
        num_items_per_batch = self.ctx.batch_size * num_items_per_shot

        batch_index = []
        index = self.get_filtered_item_ids(shuffle=True)
        for bi in range(0, len(index), num_items_per_batch):
            batch = []
            item_ids = index[bi:bi+num_items_per_batch]
            if len(item_ids) != num_items_per_batch:
                continue
            for k in range(0, len(item_ids), num_items_per_shot):
                batch.append(item_ids[k:k+num_items_per_shot])
            batch_index.append(batch)
        np.random.shuffle(batch_index)
        self.batch_index = batch_index
        
    def index_data_ori(self):
        num_items_per_shot = self.ctx.num_k_shot + self.ctx.num_valid_shot
        num_items_per_batch = self.ctx.batch_size * num_items_per_shot
        
        batch_index = []
        orientations = [1, 2, 3, 4]    
        for o in orientations:
            index = self.get_filtered_item_ids(shuffle=True, orientation=o)
            for bi in range(0, len(index), num_items_per_batch):
                batch = []
                item_ids = index[bi:bi+num_items_per_batch]
                if len(item_ids) != num_items_per_batch:    # drop remain
                    continue
                for k in range(0, len(item_ids), num_items_per_shot):
                    batch.append(item_ids[k:k+num_items_per_shot])
                batch_index.append(batch)
        np.random.shuffle(batch_index)
        self.batch_index = batch_index

    def __len__(self):
        return len(self.batch_index) - 1

    def __getitem__(self, batch_idx):
        ids_in_batch = self.batch_index[batch_idx]
        train, valid = self.to_tensor_batch(ids_in_batch)
        return self.to_named_tasks(train, valid)
      
    def define_type(self):
        dataset_type = ['tr', 'val']
        attribute_names = ['le', 're', 'uv_ec', 'csd', 'ori', 'gaze_o', 'z_scale', 
                           'landmark', 'rot_mat', 'norm_rot_mat', 'output']
        tuples = []
        for ds in dataset_type:
            for attr in attribute_names:
                tuples.append("{}_{}".format(ds, attr))
            
        self.Task = nt("task", tuples)
        self.Item = nt('Item', attribute_names)

    def to_named_tasks(self, train, valid):
        tasks = []
        for i in range(self.ctx.batch_size):
            t = []
            for elem in train + valid:
                t.append(elem[i])
            tasks.append(self.Task(*t)) 
        return tasks
    
    def to_tensor_batch(self, ids_in_batch):
        assert len(ids_in_batch) == self.ctx.batch_size, "broken batch index"
        tr = [[],[],[],[],[],[],[],[],[],[],[]]
        vl = [[],[],[],[],[],[],[],[],[],[],[]]
        for ids_in_task in ids_in_batch:
            for seq_id, item_id in enumerate(ids_in_task):
                batch_router = tr if seq_id < self.ctx.num_k_shot else vl
                process_item = self.processed_item(item_id)
                for ei, input_val in enumerate(process_item):
                    batch_router[ei].append(input_val.astype(self.float_dtype))

        train_shapes = (self.ctx.batch_size, self.ctx.num_k_shot)
        valid_shapes = (self.ctx.batch_size, self.ctx.num_valid_shot)
        train = [np.array(e) for e in tr]
        valid = [np.array(e) for e in vl]
        train = [np.reshape(e, train_shapes + e.shape[1:]) for e in train] 
        valid = [np.reshape(e, valid_shapes + e.shape[1:]) for e in valid] 
        # shape [ in1:(num_task, k_shot, in1_shape...), output:(num_task, k_shot, 2) ]
        
        return train, valid

    """
        Process
    """
    def to_gray(self, img, as_float=True):
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.equalizeHist(gray_img)
        img = img.reshape(img.shape + (1,))
        if as_float:
          img = img / 255.0 - 1
        return img.astype(self.float_dtype)
      
    def processed_item(self, id):
        m = self.metas[id]
        frame = self.frame_by(m)
        target = m['target_dist'] 
        norm_info = m['normalized'][0]
        
        face = self.warp_frame(frame, m)
        nr, rc = face.shape[:2]
        rt, rb = int(nr/2 - 32), int(nr/2 + 32)
        face_l = face[rt:rb,:128,:]
        face_r = face[rt:rb,128:,:]
        face_l, face_r = self.to_gray(face_l), self.to_gray(face_r)
        
        """ support """
        uv_ec = self.get_uv_eye_corner(m, frame)
        csd = np.array(m['cam_to_screen_dist'])
        ori = np.array([m['orientation']], dtype=self.float_dtype)
        z_scale = np.array([norm_info['z_scale']])
        gaze_origin = np.array(m['gaze_origin'])
        landmark_3d = np.array(m['landmarks_3d']).flatten()
        rot_mat = np.array(m['rotate_mat']).flatten()
        norm_rot_mat = np.array(norm_info['forward_rot_mat']).flatten()
        
        """ label """
        target_xy = np.array([target['x'], target['y']])
        target_gaze = norm_info['norm_gaze'].A1
        label = np.hstack((target_gaze, target_xy))
        
        item = self.Item(face_l, face_r, uv_ec, csd, ori, gaze_origin, 
                         z_scale, landmark_3d, rot_mat, norm_rot_mat, label)
        return item
    
    def warp_frame(self, frame, meta, norm_slot=0):
        ni = meta['normalized'][norm_slot]
        return cv.warpPerspective(frame, ni['warp_mat'], ni['size_wh'])
    
    def preprocess_image(self, image, resize_wh):
        image = cv.resize(image, resize_wh, interpolation=cv.INTER_AREA)
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        image = image / 255.0 - 1
        return image
    
    def get_uv_eye_corner(self, m, frame):
        face_rect = m['face_rect']
        le_rect = m['left_eye_rect']
        re_rect = m['right_eye_rect']
        ec = get_eye_corner(face_rect, le_rect, re_rect, frame.shape[:2])
        ec = self.eye_corner_pos_to_uv(ec, m['cam_matrix'])
        return ec
    
    def eye_corner_pos_to_uv(self, eye_corner, camera_mat):
        ec, cm = eye_corner, camera_mat
        fx, fy, cx, cy = cm[0, 0], cm[1, 1], cm[0, 2], cm[1, 2]
        xs, ys = ec[0:4], ec[4:]
        xs = (xs - cx) / fx    
        ys = (ys - cy) / fy    
        return np.hstack([xs, ys])


    """
        Property
    """
    @property
    def profile_frame_path(self):
        return os.path.join(self.ctx.npy_root_path, "frames", "frame-{}.npz".format(self.pid))
    
    @property
    def profile_meta_path(self):
        return os.path.join(self.ctx.npy_root_path, "metas", "meta-{}.pkl".format(self.pid))
    
    def frame_by(self, meta_item):
        frame_name = meta_item['frame_name'][:-4]
        frame_idx = self.frame_index.index(frame_name)
        frame = self.frames[frame_idx]
        return tf.io.decode_jpeg(frame).numpy()
        
        
    """
         Visualize
    """
    def visualize(self, idx, wh_size=None, norm_slot=0):
        m = self.metas[idx]

        norm_info = m['normalized'][norm_slot]
        W = norm_info['warp_mat']
        ow, oh = norm_info['size_wh']
        norm_gaze = norm_info['gaze_pitch_yaw']
        norm_head = norm_info['head_pose_pitch_yaw']
        
        wh_size = (ow, oh) if wh_size is None else wh_size
        frame = self.frame_by(m)
        img = cv.warpPerspective(frame, W, wh_size)
        print(img.shape)

        img = draw_gaze(img, (0.5 * ow, 0.25 * oh), norm_gaze, length=120.0, thickness=2)
        plt.figure(figsize=(6,6))
        plt.subplot(1, 2, 1)
        plt.imshow(frame)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    
def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx,
                                   eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_out



"""
    Original Gaze Capture
"""
def load_split(base_path):
    with open(os.path.join(base_path, 'split-origin.pkl'), 'rb') as f:
        split = pickle.load(f)
    return split
