import os
import time
import glob
import pickle
import random

import cv2 as cv
import numpy as np
import tensorflow as tf

from ds.data_utils import *
from matplotlib import pyplot as plt



"""
    DataProvider
"""
class DataProvider():
    
    def __init__(self, ctx, pid):
        self.pid = pid
        self.ctx = ctx
        self.setup_random_seed(ctx.seed)
        self.define_type()
        self.setup_dtype(ctx)
        self.load_data()
        self.index_data(ctx.shuffle)
        
    def setup_random_seed(self, seed):
        np.random.seed(seed)
        
    def setup_dtype(self, ctx):
        self.float_dtype = np.float64 if ctx.use_64bits else np.float32
        
    def load_profile_meta(self):
        with open(self.profile_meta_path, 'rb') as f:
            return pickle.load(f)
    
    def load_landmark(self):
        with open(self.profile_landmark_path, 'rb') as f:
            return json.load(f)
    
    def load_3dmm(self):
        return np.load(self.profile_3dmm_path, allow_pickle=True)
        
    def load_calibration_candidate_ids(self):
        with open(self.calibration_info_path, "r") as f:
            pid_to_calis = json.load(f)
            return pid_to_calis[self.pid]
            
    def load_data(self):
        mm3d = self.load_3dmm()
        landmark = self.load_landmark()
        metas = self.load_profile_meta()
        cali_frame_ids = self.load_calibration_candidate_ids()
        
        self.mm3d = mm3d['items']
        self.landmark = landmark
        self.metas = metas['items']
        self.summary = metas['summary']
        self.cali_fids = cali_frame_ids
        
        frame_npz = np.load(self.profile_frame_path, mmap_mode='r')
        self.frames = frame_npz['frames']
        self.frame_index = list(frame_npz['index'])
    
    def build_frame_id_to_meta_idx(self, shuffle):
        if shuffle:
            np.random.shuffle(self.metas)
        self.frame_id_to_meta_idx = {m['frame_id']:i for i,m in enumerate(self.metas)}
        self.cali_fids = [fid for fid in self.cali_fids if -1 != self.meta_idx_from(fid)]
        
    def index_data(self, shuffle=False):
        self.metas = self.remove_invalid_and_noisy()
        self.build_frame_id_to_meta_idx(shuffle)
        self.batch_index = self.index_batch(self.ctx.use_calib_pick)
    
    def index_batch(self, use_cal_items):
        if use_cal_items:
            num_items_per_shot = self.ctx.num_valid_shot
        else:
            num_items_per_shot = self.ctx.num_k_shots + self.ctx.num_valid_shot
        num_items_per_batch = self.ctx.batch_size * num_items_per_shot
        
        batch_index = []
        index = list(range(0, len(self.metas)))
        for bi in range(0, len(index), num_items_per_batch):
            batch, meta_idxs = [], index[bi:bi+num_items_per_batch]
            if len(meta_idxs) != num_items_per_batch:
                continue
            for k in range(0, len(meta_idxs), num_items_per_shot):
                shots = meta_idxs[k:k+num_items_per_shot]
                if use_cal_items:
                    shots += self.pick_calibration_ids(self.ctx.num_k_shots)
                batch.append(shots)
            batch_index.append(batch)
        return batch_index
    
    def pick_calibration_ids(self, num_pick):
        picked_fids = np.random.choice(self.cali_fids, num_pick, replace=False)
        return [self.meta_idx_from(fid) for fid in picked_fids]
    

    def __len__(self):
        if self.batch_index is None:
            return 0
        return len(self.batch_index)

    def __getitem__(self, batch_idx):
        ids_in_batch = self.batch_index[batch_idx]
        train, valid = self.to_tensor_batch(ids_in_batch)
        return self.to_named_tasks(train, valid)
      
    def define_type(self):
        self.item_attrs = ['le', 're', 'eye_dual', # 'face', 
                           't_vec', 'eye_pos_l', 'eye_pos_r', 
                           'face_faze_R', 'face_xu_R', 'inv_nR', 
                           't_xy_mm', 'gaze_l', 'gaze_r', 'ngaze_l', 'ngaze_r', 
                           'pid', 'id' ]
        tuples = []
        for ds in ['cal', 'val']:
            for attr in self.item_attrs:
                tuples.append("{}_{}".format(ds, attr))
        self.Task = nt("task", tuples)
        self.Item = nt('item', self.item_attrs)

    def to_named_tasks(self, train, valid):
        tasks = []
        for i in range(self.ctx.batch_size):
            t = []
            for elem in train + valid:
                t.append(elem[i])
            tasks.append(self.Task(*t)) 
        return tasks
    
    def create_item_attr_list(self):
        return [[] for _ in range(len(self.item_attrs))] 
    
    def to_tensor_batch(self, ids_in_batch):
        n_batch = self.ctx.batch_size
        num_k_shots, num_valid = self.ctx.num_k_shots, self.ctx.num_valid_shot
        assert len(ids_in_batch) == n_batch, "broken batch index"
        
        cal = self.create_item_attr_list()
        val = self.create_item_attr_list()
        for ids_in_task in ids_in_batch:
            for seq_id, item_id in enumerate(ids_in_task):
                bucket = val if seq_id < num_valid else cal
                self.put_processed_item(bucket, item_id)
                    
        calib = self.to_tensor(cal, (n_batch, num_k_shots))
        valid = self.to_tensor(val, (n_batch, num_valid))
        return calib, valid
    
    def put_processed_item(self, bucket, item_id):
        item = self.processed_item(item_id)
        for attr_idx, value in enumerate(item):
            if value.dtype is not np.int64: 
                value = value.astype(self.float_dtype)
            bucket[attr_idx].append(value)
    
    def to_tensor(self, item_list, shape):
        ts = [np.array(e) for e in item_list]
        return [np.reshape(e, shape + e.shape[1:]) for e in ts] 

    
    """
        Process
    """
    @classmethod
    def get_both_eye_patch(cls, frame):
        # (128, 256, 3) -> (48, 96, 1)
        eye_half = int(frame.shape[0]/2) - 24   
        o_margin = 8
        in_margin = 128-24
        le = frame[eye_half:-eye_half,o_margin:in_margin,:]
        re = frame[eye_half:-eye_half,-in_margin:-o_margin,:]
        return le, re
    
    def processed_item(self, id):
        m = self.metas[id]
        frame_id = m['frame_id']
        frame = self.frame_by(m)
        landmark_2d = self.landmark_by(m)
        
        """ normalize """
        norm_dist = 400
        norm_img_wh = (256, 128)
        norm_cam_mat = make_cam_mat(64*14, 64*14, norm_img_wh[0]/2, norm_img_wh[1]/2)
        norm_info = (norm_dist, norm_img_wh, norm_cam_mat)
        norm_res = normalize_niqab(m, norm_info, landmark_2d=landmark_2d)
        nRs, nSs, nWs, Gots, nGots, eye_pos_both, t_xy, face_3d, face_R = norm_res
        
        """ frame """
        # patch = to_gray_niqab(cv2.warpPerspective(frame, nWs[0], norm_img_wh))
        # face = resize_image(self.face_by(m, frame), (224,224))
        niqab = cv2.warpPerspective(frame, nWs[0], norm_img_wh)
        patch_l, patch_r = self.get_both_eye_patch(niqab)
        patch_l, patch_r = to_gray_center(patch_l), to_gray_center(patch_r)
        patch_lr = to_dual_channel(patch_l, patch_r)
        
        """ support """
        inv_nR = nRs[0].T
        face_xu_R = face_R.flatten()
        t_vec = m['head_pose'][3:]
        faze_face_R = m['rotate_mat'].flatten()
        face_mm_R = self.rotate_mat_by(m)
        eye_pos_l, eye_pos_r = eye_pos_both 
        
        """ Select R """
        face_R = faze_face_R
        face_R = face_mm_R
        
        """ label """
        t_xy_mm = t_xy
        ngaze_l, ngaze_r = nGots
        ngaze_l, ngaze_r = to_np_uvec(ngaze_l), to_np_uvec(ngaze_r)
        gaze_l, gaze_r = Gots
        gaze_l, gaze_r = to_np_uvec(gaze_l), to_np_uvec(gaze_r)
        
        """ debug """
        pid = np.array([self.pid], dtype=np.int64)
        item_id = np.array([frame_id], dtype=np.int64)
        
        item = self.Item(patch_l, patch_r, patch_lr, # face,
                         t_vec, eye_pos_l, eye_pos_r, 
                         face_R, face_xu_R, inv_nR,
                         t_xy_mm, gaze_l, gaze_r, ngaze_l, ngaze_r, 
                         pid, item_id)
        return item

    
    
    """
        Validator
    """
    def check_missing(self, m):
        frame_name = m['frame_name'][:-4]
        if frame_name not in self.frame_index:
            return False
        if frame_name not in self.landmark:
            return False
        return True
    
    def get_blurry_indexes(self, metas, cut_score, cut_sigma):
        scores = []
        for idx in list(range(0, len(metas))):
            score = variance_of_laplacian(self.frame_by(metas[idx]), as_gray=False)
            scores.append(score)

        scores = np.array(scores)
        zscore = stats.zscore(scores)
        abs_cut_idx = np.argwhere(scores < cut_score).flatten()
        sig_cut_idx = np.argwhere(zscore < cut_sigma).flatten()
        
        abs_cut_idx = set() if abs_cut_idx.shape[0] == 0 else set(abs_cut_idx.tolist())
        sig_cut_idx = set() if sig_cut_idx.shape[0] == 0 else set(sig_cut_idx.tolist())
        cut_idx = abs_cut_idx | sig_cut_idx 
        
        return list(cut_idx)
        
    def remove_invalid_and_noisy(self):
        # remove missing item
        metas = [m for m in self.metas if self.check_missing(m)] 
        
        # remove invalid case 
        if self.ctx.use_valid_only:
            metas = [m for m in metas if is_valid_item(m)] 
        
        # remove blurry frame
        if self.ctx.remove_blurry:
            blurry_meta_indexes = self.get_blurry_indexes(self.metas, 100, -1.0)
            metas = [m for i, m in enumerate(metas) if i not in blurry_meta_indexes] 

        # TODO: remove extreme yaw, pitch, roll case
        return metas

    

    """
        Property
    """
    @property
    def profile_frame_path(self):
        return os.path.join(self.ctx.npy_root_path, "frames", "frame-{}.npz".format(self.pid))
    
    @property
    def profile_meta_path(self):
        return os.path.join(self.ctx.npy_root_path, "metas", "meta-{}.pkl".format(self.pid))
    
    @property
    def profile_landmark_path(self):
        return os.path.join(self.ctx.landmark_root_path, "lm-{}.json".format(self.pid))
    
    @property
    def profile_3dmm_path(self):
        return os.path.join(self.ctx.dmm_root_path, "3dmm-{}.npz".format(self.pid))
    
    @property
    def calibration_info_path(self):
        return os.path.join(self.ctx.npy_root_path, "cali-sample.json")
    
    def meta_idx_from(self, frame_id):
        if frame_id not in self.frame_id_to_meta_idx:
            return -1
        return self.frame_id_to_meta_idx[frame_id]
    
    def item_by(self, item_id, as_meta_id=False):
        meta_idx = item_id if as_meta_id else self.frame_id_to_meta_idx[item_id]
        
        m = self.metas[meta_idx]
        frame = self.frame_by(m)
        face = self.face_by(m, frame, margin=0)
        lm = self.landmark_by(m)
        return m, frame, face, lm
    
    def frame_by(self, meta_item):
        frame_name = meta_item['frame_name'][:-4]
        frame_idx = self.frame_index.index(frame_name)
        frame = self.frames[frame_idx]
        return tf.io.decode_jpeg(frame).numpy()
    
    def landmark_by(self, meta_item):
        frame_name = meta_item['frame_name'][:-4]
        lm = self.landmark[frame_name]
        nose_bridge, nose_bottom = lm[27:31], lm[33:34]
        jaws, eye_right, eye_left = lm[7:10], lm[36:42], lm[42:48]
        # return np.vstack((eye_right, eye_left, nose_bridge, nose_bottom, jaws))
        return np.vstack((eye_right, eye_left, nose_bridge, nose_bottom))

    def face_by(self, m, frame, margin=10):
        top, left, bottom, right = rect_to_tl_br(m['face_rect'])
        face = frame[top-margin:bottom+margin, left-margin:right+margin, :]
        return face
    
    def rotate_mat_by(self, meta_item):
        def to_rotation_matrix(pitch, yaw, roll):
            cos, sin = np.cos, np.sin
            phi, gamma, theta = pitch, yaw, roll    # angle x, y, z

            Rx = np.matrix([[1.0,      0.0,       0.0], 
                            [0.0,  cos(phi), sin(phi)], 
                            [0.0, -sin(phi), cos(phi)]])

            Ry = np.matrix([[cos(gamma), 0.0, -sin(gamma)], 
                            [0.0,        1.0,         0.0], 
                            [sin(gamma), 0.0,  cos(gamma)]])

            Rz = np.matrix([[ cos(theta), sin(theta), 0.0], 
                            [-sin(theta), cos(theta), 0.0],
                            [        0.0,        0.0, 1.0]])
            R = Rx * Ry * Rz

            return R
        
        frame_name = meta_item['frame_name'][:-4]
        item = [x for x in self.mm3d if x['frame_id'] == frame_name][0]
        pitch, yaw, roll = item['pitch'], item['yaw'], item['roll']
        R = to_rotation_matrix(pitch, yaw, roll)
        return R
    

    """
        TF Decoder
    """
    @classmethod
    def get_decoder(cls):
        features = {
            "le"        : tf.io.VarLenFeature(dtype=tf.string),
            "re"        : tf.io.VarLenFeature(dtype=tf.string),
            "frame"     : tf.io.VarLenFeature(dtype=tf.string),
            
            "eye_pos_l" : tf.io.FixedLenFeature([3], dtype=tf.float32),
            "eye_pos_r" : tf.io.FixedLenFeature([3], dtype=tf.float32),

            "face_faze_R" : tf.io.FixedLenFeature([9], dtype=tf.float32),
            "face_xu_R"   : tf.io.FixedLenFeature([9], dtype=tf.float32),
            "inv_nR"      : tf.io.FixedLenFeature([9], dtype=tf.float32),

            "t_xy_mm"  : tf.io.FixedLenFeature([2], dtype=tf.float32),
            "gaze_l"   : tf.io.FixedLenFeature([3], dtype=tf.float32),
            "gaze_r"   : tf.io.FixedLenFeature([3], dtype=tf.float32),
            "ngaze_l"  : tf.io.FixedLenFeature([3], dtype=tf.float32),
            "ngaze_r"  : tf.io.FixedLenFeature([3], dtype=tf.float32),

            "pid" : tf.io.FixedLenFeature([1], dtype=tf.int64),
            "id"  : tf.io.FixedLenFeature([1], dtype=tf.int64),
        } 
    
        def decode(serialized_item):
            parsed = tf.io.parse_example(serialized_item, features)
            
            """ inputs """
            le = parsed['le'].values[0]
            re = parsed['re'].values[0]
            frame = parsed['frame'].values[0]
            
            le, re = tf.io.decode_jpeg(le), tf.io.decode_jpeg(re)
            le = tf.image.convert_image_dtype(le, tf.float32)
            re = tf.image.convert_image_dtype(re, tf.float32)
            
            frame = tf.io.decode_jpeg(frame)
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            
            lep, rep = parsed['eye_pos_l'], parsed['eye_pos_r']
            
            face_R = parsed['face_faze_R']
            face_R = tf.reshape(face_R, (3, 3))
            
            """ outputs """
            t_xy_mm = parsed['t_xy_mm']
            gaze_l, gaze_r = parsed['gaze_l'], parsed['gaze_r']
            
            """ debug """
            pid, item_id = parsed['pid'], parsed['id']
            
            ins = (le, re, frame, lep, rep, face_R)
            outs = (gaze_l, gaze_r, t_xy_mm)
            return ins, outs   #, (pid, item_id)
        return decode
    
    @classmethod
    def get_tf_dataset(cls, tf_paths, num_batch=16, num_parallel=16):
        decoder = cls.get_decoder()
        ds = tf.data.TFRecordDataset(tf_paths)
        seq = ds.map(decoder, num_parallel_calls=num_parallel) \
                .prefetch(tf.data.experimental.AUTOTUNE)       \
                .batch(num_batch)
        return seq
 