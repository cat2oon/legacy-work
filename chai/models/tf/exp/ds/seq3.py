import os
import glob
import json
import pickle
import logging
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

    
"""
    NPY Sequence
"""
class Sequence(tf.keras.utils.Sequence):
    
    def __init__(self, 
                 data_tag, 
                 pid,
                 npy_root_path, 
                 batch_size=64, 
                 seed=1234):
        self.pid = pid 
        self.batch_size = batch_size
        self.npy_root_path = npy_root_path
        self.setup_random_seed(seed)
        self.load_data()
        self.prepare_index(data_tag, batch_size)
        
    def setup_random_seed(self, seed):
        np.random.seed(seed)

    def __len__(self):
        return len(self.batch_index)
    
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

    def prepare_index(self, data_tag, batch_size):
        item_ids = list(range(0, len(self.metas)))
        if data_tag == 'none':
            item_ids = item_ids
        elif data_tag == 'train':
            np.random.shuffle(item_ids)
            item_ids = item_ids[:int(len(item_ids)*0.65)]
        else:
            np.random.shuffle(item_ids)
            item_ids = item_ids[int(len(item_ids)*0.65):]
        
        batch_index = []
        for i in range(0, len(item_ids), batch_size):
            batch_item_ids = item_ids[i:i+batch_size]
            batch_index.append(batch_item_ids)
        self.batch_index = batch_index 

    def __len__(self):
        return len(self.batch_index) - 1

    def __getitem__(self, batch_idx):
        ids_in_batch = self.batch_index[batch_idx]
        return self.to_tensor_batch(ids_in_batch)
    
    def to_tensor_batch(self, ids_in_batch):
        inputs, targets = [[],[],[],[],[],[],[]], []
        for item_id in ids_in_batch:
            processed_item = self.processed_item(item_id) 
            for input_seq, val in enumerate(processed_item):
                if input_seq == len(processed_item)-1:
                    targets.append(val)
                else:
                    inputs[input_seq].append(val)        

        inputs_shape = (self.batch_size, )
        inputs  = [np.array(e) for e in inputs]
        inputs  = [np.reshape(e, inputs_shape + e.shape[1:]) for e in inputs] 
        targets = np.array(targets)
                    
        return inputs, targets
    
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
        
        face_l = self.preprocess_image(face_l, use_gray=True)
        face_r = self.preprocess_image(face_r, use_gray=True)
        
        """ support """
        uv_ec = self.get_uv_eye_corner(m, frame)
        z_scale = np.array(norm_info['z_scale'])
        csd = np.array(m['cam_to_screen_dist'])
        ori = np.array([m['orientation']], dtype=np.float64)
        head_pose = np.array(norm_info['forward_rot_mat']).flatten()
        rot_mat = np.array(m['rotate_mat']).flatten()
        landmark_3d = np.array(m['landmarks_3d']).flatten()
        gaze_origin = np.array(m['gaze_origin']).flatten()
        
        """ label """
        target_xy = np.array([target['x'], target['y']])
        target_gaze = norm_info['norm_gaze'].A1
        label = np.hstack((target_gaze, target_xy))
        
        return face_l, face_r, rot_mat, head_pose, z_scale, uv_ec, gaze_origin, label
    
    def warp_frame(self, frame, meta, norm_slot=0):
        ni = meta['normalized'][norm_slot]
        return cv.warpPerspective(frame, ni['warp_mat'], ni['size_wh'])
    
    def preprocess_image(self, image, resize_wh=None, use_gray=False):
        if resize_wh is not None:
            image = cv.resize(image, resize_wh, interpolation=cv.INTER_AREA)
            
        if use_gray:
            gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            image = cv.equalizeHist(gray_img)
            image = image.reshape(image.shape + (1,))
        else:
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
        return os.path.join(self.npy_root_path, "frames", "frame-{}.npz".format(self.pid))
    
    @property
    def profile_meta_path(self):
        return os.path.join(self.npy_root_path, "metas", "meta-{}.pkl".format(self.pid))
    
    def frame_by(self, meta_item):
        frame_name = meta_item['frame_name'][:-4]
        frame_idx = self.frame_index.index(frame_name)
        frame = self.frames[frame_idx]
        return tf.io.decode_jpeg(frame).numpy()

    
    """
         Visualize
    """
    def visualize(self, idx):
        m = self.metas[idx]
        face_l, face_r, support, target = self.processed_item(idx)
        
        print('face_l, r shape:', face_l.shape, face_r.shape)
        
        plt.figure(figsize=(6,6))
        plt.subplot(1, 2, 1)
        plt.imshow(face_l, cmap='gray')
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.subplot(1, 2, 2)
        plt.imshow(face_r, cmap='gray')
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
