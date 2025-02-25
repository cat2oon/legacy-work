import os
import glob
import json
import pickle
import logging
import cv2 as cv
import numpy as np
import tensorflow as tf

from ds.data_utils import *

    
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
            item_ids = item_ids[:int(len(item_ids)*0.8)]
        else:
            np.random.shuffle(item_ids)
            item_ids = item_ids[int(len(item_ids)*0.8)+1:]
        
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
        inputs, targets = [[],[],[],[],[],[],[],[],[],[]], []
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
      
    def to_gray(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img = cv.equalizeHist(gray_img)
        img = img.reshape(img.shape + (1,))
        return img
    
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
        face_l = tf.io.encode_jpeg(face_l, quality=100).numpy()
        face_r = tf.io.encode_jpeg(face_r, quality=100).numpy()
        
        """ support """
        uv_ec = self.get_uv_eye_corner(m, frame)
        csd = np.array(m['cam_to_screen_dist'])
        ori = np.array([m['orientation']], dtype=np.float64)
        z_scale = np.array([norm_info['z_scale']])
        gaze_origin = np.array(m['gaze_origin'])
        landmark_3d = np.array(m['landmarks_3d']).flatten()
        rot_mat = np.array(m['rotate_mat']).flatten()
        norm_rot_mat = np.array(norm_info['forward_rot_mat']).flatten()
          
        """ label """
        target_xy = np.array([target['x'], target['y']])
        target_gaze = norm_info['norm_gaze'].A1
        label = np.hstack((target_gaze, target_xy))
        
        return face_l, face_r, uv_ec, csd, ori, gaze_origin, z_scale, landmark_3d, rot_mat, norm_rot_mat, label 
    
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
        return os.path.join(self.npy_root_path, "frames", "frame-{}.npz".format(self.pid))
    
    @property
    def profile_meta_path(self):
        return os.path.join(self.npy_root_path, "metas", "meta-{}.pkl".format(self.pid))
    
    def frame_by(self, meta_item):
        frame_name = meta_item['frame_name'][:-4]
        frame_idx = self.frame_index.index(frame_name)
        frame = self.frames[frame_idx]
        return tf.io.decode_jpeg(frame).numpy()

    
