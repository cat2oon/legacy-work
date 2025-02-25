import os
import glob
import json
import pickle
import logging
import cv2 as cv
import numpy as np
import tensorflow as tf

from ds.data_utils import *
from collections import namedtuple as nt

    
"""
    Niqab Sequence
"""
class Sequence(tf.keras.utils.Sequence):
    
    def __init__(self, ctx, pid):
        self.ctx = ctx
        self.pid = pid 
        self.setup_random_seed(ctx.seed)
        self.define_type()
        self.load_data()
        self.index_data(shuffle=True)
        
    def setup_random_seed(self, seed):
        np.random.seed(seed)

    def __len__(self):
        return len(self.batch_index)
    
    def load_profile_meta(self):
        with open(self.profile_meta_path, 'rb') as f:
            return pickle.load(f)
    
    def load_landmark(self):
        with open(self.profile_landmark_path, 'rb') as f:
            return json.load(f)
            
    def load_data(self):
        landmark = self.load_landmark()
        metas = self.load_profile_meta()
        
        self.landmark = landmark
        self.metas = metas['items']
        self.summary = metas['summary']
        
        frame_npz = np.load(self.profile_frame_path, mmap_mode='r')
        self.frames = frame_npz['frames']
        self.frame_index = list(frame_npz['index'])
        self.remove_missing_data()
            
    def remove_missing_data(self):
        missing_ids = []
        for i, m in enumerate(self.metas):
            frame_name = m['frame_name'][:-4]
            if frame_name not in self.frame_index or frame_name not in self.landmark:
                missing_ids.append(i)
        missing_ids.reverse()
        for id in missing_ids:
            self.metas.pop(id)

    def filter_noisy_data(self, orientation=None):
        ids = list(range(0, len(self.metas)))
        """ 02065는 모든 프레임이 eye detect fail """
        invalid_ids = [id for id in ids if not is_valid_item(self.metas[id])]  
        sorted(invalid_ids)
        invalid_ids.reverse()
        for id in invalid_ids:
            self.metas.pop(id)
        if len(self.metas) > 0:
            self.remove_blurry_frame()
            
    def get_blurry_frame_ids(self):
        scores = []
        for idx in list(range(0, len(self.metas))):
            m = self.metas[idx]
            score = clear_score(self.frame_by(m))
            scores.append(score)

        scores = np.array(scores)
        zscore = stats.zscore(scores)
        abs_cut_ids = np.argwhere(scores < 30).flatten()
        sig_cut_ids = np.argwhere(zscore < -1.0).flatten()
        
        abs_cut_ids = set() if abs_cut_ids.shape[0] == 0 else set(abs_cut_ids.tolist())
        sig_cut_ids = set() if sig_cut_ids.shape[0] == 0 else set(sig_cut_ids.tolist())
        cut_ids = abs_cut_ids | sig_cut_ids 
        
        return cut_ids
    
    def remove_blurry_frame(self):
        cut_ids = get_blurry_frame_ids()
        if len(cut_ids) == 0:
            return
        for idx in range(len(self.metas), 0, -1):  # 괴상한 sorted 버그 있어서 이렇게 함
            if idx in cut_ids:
                self.metas.pop(idx)
        
    def index_data(self, shuffle=False):
        self.filter_noisy_data()
        item_ids = list(range(0, len(self.metas)))
        
        if shuffle:
            np.random.shuffle(item_ids)
        self.batch_index = item_ids

    def __getitem__(self, seq_id):
        item_id = self.batch_index[seq_id]
        return self.processed_item(item_id)

    def define_type(self):
        self.item_attrs = ['le', 're', 'frame', 'eye_pos_l', 'eye_pos_r',
                           'face_faze_R', 'face_xu_R', 'inv_nR', 
                           't_xy_mm', 'gaze_l', 'gaze_r', 'ngaze_l', 'ngaze_r',
                           'pid', 'id' ]
        self.Item = nt('item', self.item_attrs)


    """
        Process
    """
    def processed_item(self, id):
        return self.process_niqab(id)
    
    def process_niqab(self, id):
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
        use_norm = False
        niqab = cv2.warpPerspective(frame, nWs[0], norm_img_wh)
        frame = to_gray(niqab, norm=use_norm)
        patch_l, patch_r = get_both_eye_patch(niqab)
        patch_l, patch_r = to_gray_center(patch_l, norm=use_norm), to_gray_center(patch_r, norm=use_norm)
        
        """ image to tf """
        frame = tf.io.encode_jpeg(frame, quality=100).numpy()
        patch_l = tf.io.encode_jpeg(patch_l, quality=100).numpy()
        patch_r = tf.io.encode_jpeg(patch_r, quality=100).numpy()
        
        """ support """
        inv_nR = nRs[0].T.flatten()
        face_xu_R = face_R.flatten()
        face_faze_R = m['rotate_mat'].flatten()
        eye_pos_l, eye_pos_r = eye_pos_both 
        
        """ label """
        t_xy_mm = t_xy
        ngaze_l, ngaze_r = nGots
        ngaze_l, ngaze_r = to_np_uvec(ngaze_l), to_np_uvec(ngaze_r)
        gaze_l, gaze_r = Gots
        gaze_l, gaze_r = to_np_uvec(gaze_l), to_np_uvec(gaze_r)
        
        """ debug """
        item_id = np.array([frame_id], dtype=np.int32)
        pid = np.array([self.pid], dtype=np.int32)
        
        item = self.Item(patch_l, patch_r, frame, eye_pos_l, eye_pos_r, 
                         face_faze_R, face_xu_R, inv_nR, 
                         t_xy_mm, gaze_l, gaze_r, ngaze_l, ngaze_r,
                         pid, item_id)
        return item
    
    def get_both_eye_patch(frame):
        # (128, 256, 3) -> (48, 96, 1)
        eye_half = int(frame.shape[0]/2) - 24   
        o_margin = 8
        in_margin = 128-24
        le = frame[eye_half:-eye_half,o_margin:in_margin,:]
        re = frame[eye_half:-eye_half,-in_margin:-o_margin,:]
        return le, re


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
        

    

    
