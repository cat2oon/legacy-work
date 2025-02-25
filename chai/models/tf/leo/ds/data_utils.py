import cv2
import time
import copy
import pickle
import numpy as np
import os, json, glob
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import stats
from tqdm import tqdm
from bunch import Bunch
from tensorflow import keras
from matplotlib.pyplot import figure
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


"""
    Calibration Set Selector
"""
def is_primary_pose(meta):
    R = meta['rotate_mat']
    xtc = R[:, 0]
    ytc = R[:, 1]
    ztc = R[:, 2]
    
    r = np.rad2deg(np.arctan2(R[1,0], R[0,0]))
    y = np.rad2deg(np.arccos(xtc @ np.array([1,0,0])))
    
    if r < -15 or r > 15:
        return False
    if y < -15 or y > 15:
        return False
    
    return True

def is_proper_depth(meta):
    fx, fy, fz = meta['head_pose'][3:]
    if 250 < fz < 350:
        return True
    return False

def is_both_eye_open(landmark):
    lm = landmark
    lh1 = np.abs(lm[5][1] - lm[1][1])
    lh2 = np.abs(lm[4][1] - lm[2][1])
    
    rh1 = np.abs(lm[7][1] - lm[11][1])
    rh2 = np.abs(lm[8][1] - lm[10][1])
    
    if None in [lh1, lh2, rh1, rh2]:
        return False
    
    lo = np.mean([lh1, lh2])
    ro = np.mean([rh1, rh2])
    
    if lo < 6.0 or ro < 6.0:
        return False
    return True

def get_rounded_target_xy(meta):
    x, y, _ = meta['gaze_target']
    return round(float(x)), round(float(y))

def get_blurry_frame_ids(items, cut_score, cut_sigma):
    scores = []
    for idx, meta_face_frame in enumerate(items):
        meta, face, frame, _ = meta_face_frame 
        if np.prod(face.shape) == 0:
            scores.append(clear_score(frame) - 50.0)
        else:
            scores.append(clear_score(face))

    scores = np.array(scores)
    zscore = stats.zscore(scores)
    abs_cut_ids = np.argwhere(scores < cut_score).flatten()
    sig_cut_ids = np.argwhere(zscore < cut_sigma).flatten()
    
    abs_cut_ids = set() if abs_cut_ids.shape[0] == 0 else set(abs_cut_ids.tolist())
    sig_cut_ids = set() if sig_cut_ids.shape[0] == 0 else set(sig_cut_ids.tolist())
    cut_ids = list(abs_cut_ids | sig_cut_ids)
    
    return cut_ids
    
def calc_calibration_candidates(items, blur_cut_score, blur_cut_sigma):
    items = filter(lambda x: is_valid_item(x[0], strong=True), items)
    items = filter(lambda x: is_proper_depth(x[0]), items)
    items = filter(lambda x: is_primary_pose(x[0]), items)
    items = filter(lambda x: is_both_eye_open(x[3]), items)
    items = list(items)
    
    cut_ids = get_blurry_frame_ids(items, blur_cut_score, blur_cut_sigma)
    if len(cut_ids) != 0:
        for idx in range(len(items), -1, -1):
            if idx in cut_ids:
                items.pop(idx)
                
    return items, list(cut_ids)



"""
    NP utility
"""
def to_np_uvec(vec):
    return vec/np.linalg.norm(vec)

# en.wikipedia.org/wiki/
# Rotation_formalisms_in_three_dimensions#Conversion_formulae_between_formalisms
""" PITCH, YAW, ROLL """
def to_yaw_pitch_roll(rotate_mat, to_deg=True):  
    r = rotate_mat
    r32, r33 = r[2,1], r[2,2]
    
    roll  = np.arctan2(r[1,0], r[0,0])
    yaw   = np.arctan2(-r[2,0], np.sqrt(r32**2+r33**2))
    pitch = np.arctan2(r32, r33)
    
    if not to_deg:
        return yaw, pitch, roll 
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)

def decompose_rotate_mat(rotate_mat, to_deg=True):
    R = rotate_mat
    y_rot = np.arcsin(R[2,0])
    x_rot = np.arccos(R[2,2]/np.cos(y_rot))
    z_rot = np.arctan2(R[1,0], R[0,0])
    # z_rot = np.arccos(R[0][0]/np.cos(y_rot))
    
    if not to_deg:
        return x_rot, y_rot, z_rot
    return np.rad2deg(x_rot), np.rad2deg(y_rot), np.rad2deg(z_rot)

def decompose_rotate_mat_v1(rotate_mat, to_deg=True):
    r = rotate_mat
    r32, r33 = r[2,1], r[2,2]
    
    pitch = np.arctan2(r32, r33)
    # pitch = np.arcsin(r[1,2]) 
    yaw   = np.arctan2(r[0,2], r[2,2])
    roll  = np.arctan2(r[1,0], r[0,0])
    
    if not to_deg:
        return yaw, pitch, roll 
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)

def calc_mlp_size(num_in, l1, l2, l3):
    a, b, c, d = num_in, l1, l2, l3
    return a*b + b*c + c*d + b+c+d

def euler_to_quat(yaw, pitch, roll, from_degree=True):
    if from_degree:
        yaw   = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll  = np.deg2rad(roll) 
        
    a, b, c = roll/2.0, pitch/2.0, yaw/2.0
    A, B = np.cos(a), np.sin(a)
    C, D = np.cos(b), np.sin(b)
    E, F = np.cos(c), np.sin(c)
    
    qw = A*C*E + B*D*F
    qx = B*C*E - A*D*F
    qy = A*D*E + B*C*F
    qz = A*C*F - B*D*E
    
    return qw, qx, qy, qz



"""
    Normalize Config
"""
norm_info = {
    "face": {
        'dist' : 300,
        'fx' : 200,
        'img_size' : (96, 48)   
    },
    "eye": {
        'dist' : 300,
        'fx' : 700,
        'img_size' : (96, 96)   
    }
}
   
def make_cam_mat(fx, fy, cx, cy, skewed=1.0):
    cm = np.array([
        [fx,  0,     cx],
        [0,  fy,     cy],
        [0,   0, skewed]], dtype=np.float64)
    return np.around(cm, decimals=5)



"""
    Normalize Operation
"""
def get_standard_face_3d_points():
    data = {
      "eye_right": [
            [-45.161,  -34.500, 35.797],
            [-39.287,  -39.759, 28.830],
            [-28.392,  -39.432, 27.056],
            [-19.184,  -33.718, 28.925],
            [-28.547,  -30.282, 29.844],
            [-38.151,  -30.684, 30.856]],
      "eye_left": [
            [ 19.184,  -33.718, 28.925],
            [ 28.392,  -39.432, 27.056],
            [ 39.287,  -39.759, 28.830],
            [ 45.161,  -34.500, 35.797],
            [ 38.151,  -30.684, 30.856],
            [ 28.547,  -30.282, 29.844]],
      "nose": [
            [  0.000,  -33.762, 16.068],
            [  0.000,  -22.746, 10.370],
            [  0.000,  -12.328,  3.639],
            [  0.000,  -0.000,  3.077]],
      "nose_bottom": [
            [  0.000, 14.869, 13.731]],
      "jaws": [
            [-14.591, 67.050, 27.722],
            [  0.000, 69.735, 26.787],
            [ 14.591, 67.050, 27.722]]
    }
    
    eye_right   = np.array(data['eye_right'],   dtype='float32')
    eye_left    = np.array(data['eye_left'],    dtype='float32')
    nose        = np.array(data['nose'],        dtype='float32')
    nose_bottom = np.array(data['nose_bottom'], dtype='float32')
    jaws        = np.array(data['jaws'],        dtype='float32')

    # return np.vstack((eye_right, eye_left, nose, nose_bottom, jaws))
    return np.vstack((eye_right, eye_left, nose, nose_bottom))

def estimate_head_pose(img_pts, obj_pos, cam_mat, distortion):
    img_pts, obj_pos = np.array(img_pts), np.array(obj_pos)
    ret, rvec, tvec = cv2.solvePnP(obj_pos, img_pts, cam_mat, 
                                   distortion, flags=cv2.SOLVEPNP_EPNP)
    ret, rvec, tvec = cv2.solvePnP(obj_pos, img_pts, cam_mat, 
                                   distortion, rvec, tvec, True)
    return rvec, tvec

def compute_landmark_3d(landmark_2d, obj_positions, cam_mat, distortion):
    rvec, tvec = estimate_head_pose(landmark_2d, obj_positions, cam_mat, distortion)
    rotate_mat, _ = cv2.Rodrigues(rvec)
    rotated = np.matmul(rotate_mat, obj_positions.T).T
    landmarks_3d = rotated + tvec.T
    return landmarks_3d, rotate_mat, rvec, tvec

def get_meta_attrs(meta_item):
    m = meta_item
    attrs = [key for key in m.keys()]
    norm_info = [key for key in m['normalized'][0]]
    return attrs + norm_info

def get_scale_mat(target_pos, norm_dist):
    S = np.eye(3, dtype=np.float64)
    z_scale = norm_dist / np.linalg.norm(target_pos)
    S[2, 2] = z_scale
    return S

def get_warp_mat(dest_cam_mat, scale_mat, norm_rot_mat, origin_cam_mat):
    return np.dot(np.dot(dest_cam_mat, scale_mat), 
                  np.dot(norm_rot_mat, np.linalg.inv(origin_cam_mat)))

def get_norm_R(rot_mat, origin_3d):
    x_axis = rot_mat[:, 0]
    distance = np.linalg.norm(origin_3d)
    forward = (origin_3d / distance)
    down = np.cross(forward, x_axis)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)
    R = np.c_[right, down, forward].T
    return R

def get_both_eyes_mats(m, eye_pos_both, rotate_mat, norm_cam_mat, norm_dist):
    # rotate matrix
    le_pos, re_pos = eye_pos_both
    R_le = get_norm_R(rotate_mat, le_pos)
    R_re = get_norm_R(rotate_mat, re_pos)
    Rs = (R_le, R_re)
    
    # scale matrix
    S_le = get_scale_mat(le_pos, norm_dist)
    S_re = get_scale_mat(re_pos, norm_dist)
    Ss = (S_le, S_re)
    
    # warp matrix
    cam_mat = m['cam_matrix']
    W_le = get_warp_mat(norm_cam_mat, S_le, R_le, cam_mat)
    W_re = get_warp_mat(norm_cam_mat, S_re, R_re, cam_mat)
    Ws = (W_le, W_re)
    
    return Rs, Ss, Ws

def get_gaze_vec_and_norm_vec(eye_pos_both, t_xy, Rs):
    le_pos, re_pos = eye_pos_both
    tx, ty = t_xy
    target_pos = np.array([tx, ty, 0], dtype=np.float64)
    gaze_lt = target_pos - le_pos  
    gaze_rt = target_pos - re_pos 
    norm_gaze_lt = Rs[0] @ gaze_lt
    norm_gaze_rt = Rs[1] @ gaze_rt
    gaze_ots = (gaze_lt, gaze_rt)
    norm_gaze_ots = (norm_gaze_lt, norm_gaze_rt)
    
    return gaze_ots, norm_gaze_ots

def normalize(m, norm_info, landmark_3d=None, landmark_2d=None, undistort=False):
    norm_dist, norm_img_wh, norm_cam_mat = norm_info
    
    # use landmark_2d
    if landmark_2d is not None:
        """ TODO: 턱선 없는 거라 rvec 부정확 할 듯 -> frame 레이어로 대체? """
        cam_mat, distort = m['cam_matrix'], m['distort_params']
        obj_3ds = get_standard_face_3d_points()
        face, rot_mat, rvec, tvec = compute_landmark_3d(landmark_2d, obj_3ds, cam_mat, distort)
        le_pos = np.mean(face[0:6], axis=0)
        re_pos = np.mean(face[6:12], axis=0)
        # print(le_pos, re_pos)
    else:
        # face key points 3d position
        face_m = m['landmarks_3d'] if landmark_3d is None else landmark_3d
        rot_mat = m['rotate_mat']
        le_pos_m = np.mean([face[9],  face[10]], axis=0)
        re_pos_m = np.mean([face[11], face[12]], axis=0)
        # print(le_pos_m, re_pos_m)
    
    # normalize transforms
    eye_pos_both = (le_pos, re_pos)
    nRs, nSs, nWs = get_both_eyes_mats(m, eye_pos_both, rot_mat, norm_cam_mat, norm_dist)

    """ gaze vector  NOTE: negative gaze-catpure coord -> xu cong zhang """
    target_xy = m['target_dist']                      # cm
    t_xy = (target_xy['x']*-10, target_xy['y']*-10)   # mm / flip 
    gaze_ots, norm_gaze_ots = get_gaze_vec_and_norm_vec(eye_pos_both, t_xy, nRs)
         
    return nRs, nSs, nWs, gaze_ots, norm_gaze_ots, eye_pos_both, np.array(t_xy), face, rot_mat


""" niqab normalize """
def normalize_niqab(m, norm_info, landmark_2d):
    obj_3ds = get_standard_face_3d_points()
    norm_dist, norm_img_wh, norm_cam_mat = norm_info
    cam_mat, distort = m['cam_matrix'], m['distort_params']
    
    face_3d, rot_mat, rvec, tvec = compute_landmark_3d(landmark_2d, obj_3ds, cam_mat, distort)
    le_pos = np.mean(face_3d[0:6], axis=0)
    re_pos = np.mean(face_3d[6:12], axis=0)
    
    eye_pos_both = (le_pos, re_pos)
    nR, nS, nW = get_niqab_eyes_mats(m, eye_pos_both, rot_mat, norm_cam_mat, norm_dist)

    target_xy = m['target_dist']                      # cm
    t_xy = (target_xy['x']*-10, target_xy['y']*-10)   # mm / flip 
    Gs, nGs= get_gaze_vec_and_norm_vec(eye_pos_both, t_xy, (nR, nR))
    
    nRs, nSs, nWs = (nR, nR), (nS, nS), (nW, nW)
    return nRs, nSs, nWs, Gs, nGs, eye_pos_both, np.array(t_xy), face_3d, rot_mat

def get_niqab_eyes_mats(m, eye_pos_both, rotate_mat, norm_cam_mat, norm_dist):
    cam_mat = m['cam_matrix']
    le_pos, re_pos = eye_pos_both
    eye_center_pos = np.mean([le_pos, re_pos], axis=0)
    
    R = get_norm_R(rotate_mat, eye_center_pos)
    S = get_scale_mat(eye_center_pos, norm_dist)
    W = get_warp_mat(norm_cam_mat, S, R, cam_mat)
    
    return R, S, W


"""
    GazeCapture PreProcess
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

    # [ x x x x y y y y ]
    return [ltop, lbottom, rtop, rbottom, lleft, lright, rleft, rright]

def get_eye_left_right_patch(frame, face_rect, left_rect, right_rect):
    top, left, bottom, right = rect_to_tl_br(face_rect)
    ltop, lleft, lbottom, lright = rect_to_tl_br(left_rect)
    rtop, rleft, rbottom, rright = rect_to_tl_br(right_rect)
    le_img = frame[top+ltop:top+lbottom, left+lleft:left+lright, :]
    re_img = frame[top+rtop:top+rbottom, left+rleft:left+rright, :]

    return le_img, re_img

def get_face_patch(frame, face_rect, margin=10):
    top, left, bottom, right = rect_to_tl_br(face_rect)
    face = frame[top-margin:bottom+margin, left-margin:right+margin, :]
    return face

def get_face_patch_from(frame, m):
    return get_face_patch(frame, m['face_rect'])


def is_valid_item(m, strong=False):
    if strong:
        return m['face_valid'] and m['left_eye_valid'] and m['right_eye_valid']
    return m['face_valid']



"""
    Visualizer
"""
def draw_eye(m, frame, norm_slot_idx=0):
    norm_info = m['normalized'][norm_slot_idx]
    W = norm_info['warp_mat']
    ow, oh = norm_info['size_wh']
    norm_gaze = norm_info['gaze_pitch_yaw']
    norm_head = norm_info['head_pose_pitch_yaw']

    img = cv2.warpPerspective(frame, W, (ow, oh))
    # print(img.shape)
    r = img.shape[0]
    c = img.shape[1]
    # print(r,c)
    r,c = img.shape[0:2]

    le = img[int(r/2-32):int(r/2+16), 8:96, :]
    re = img[int(r/2-32):int(r/2+16), 128+32:-8, :]

    # print(le.shape)
    plt.figure(figsize=(12,12))
    plt.subplot(1, 3, 1)
    plt.imshow(le)
    plt.subplot(1, 3, 2)
    plt.imshow(re)
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    
def draw_both_eye(le, re, figsize=(12, 12)):
    # print('le blurry', variance_of_laplacian(le))
    # print('re blurry', variance_of_laplacian(re))
    plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    cm = None if le.shape[2] == 3 else 'gray'
    le = np.squeeze(le)
    plt.imshow(le, cmap=cm)
    plt.subplot(1, 3, 2)
    cm = None if re.shape[2] == 3 else 'gray'
    re = np.squeeze(re)
    plt.imshow(re, cmap=cm)
    
def show_landmarks_3d(frame, meta, figsize=(7,7)):
    def to_img_pos(pos):
        x, y, _ = pos
        x, y = 2*x, 2*y
        x, y = x + frame.shape[1]/2, y + frame.shape[0]/2
        return x, y
    
    fs = meta['landmarks_3d']
    re = np.mean(fs[9:11, :], axis=0)
    le = np.mean(fs[11:13,:], axis=0)
    
    plt.figure(figsize=figsize)
    plt.imshow(frame)
    
    for i, p in enumerate(fs):
        x, y = to_img_pos(p)
        plt.plot(x, y, 'o')
        plt.text(x, y, str(i), color="red", fontsize=12)
    
    rx, ry = to_img_pos(re) 
    lx, ly = to_img_pos(le) 
    plt.plot(rx, ry, 'x')
    plt.plot(lx, ly, 'x')

    
    
"""
    Visualizer - 3D
    { x-axis:빨강, y-axis:초록, z-axis:파랑 }
"""
def vis_rotate_mat(rot_mat, norm_system=True, figsize=(7,7)):
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    
    zero = np.zeros(3)
    lines = ['-r', '-g', '-b']
    norm_R = coord_system_converter('')
    
    for i, axis in enumerate(decompose_axes(rot_mat)):
        n_axis = (norm_R @ axis.T).A1
        x, y, z = zip(zero, n_axis)
        plt.plot(x, y, z, lines[i], linewidth=3)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_zticks(ax.get_zticks()[::2])
    # ax.grid(False)
    
def decompose_axes(rot_mat):
    # axis -> plural -> axes
    return rot_mat[:, 0], rot_mat[:, 1], rot_mat[:, 2]

def coord_system_converter(case=None):
    if case is None:
        return np.eye(3)
    # if from 3d-max to xucong  
    # if direct-x to opengl 
    return rmat_3dmax_to_xucong()

def rmat_3dmax_to_xucong():
    """ 좌표계 변환 (3d-max -> 주콩장) """
    return np.matrix([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float64)



"""
    Image Processing
"""
def resize_image(image, resize_wh):
    # TODO: interpolation img.shape[0]*img.shape[1] <=> resize_wh[0]*[1]
    return cv2.resize(image, resize_wh, interpolation=cv2.INTER_AREA)

def to_dual_channel(gray_img_x, gray_img_y):
    dual_channel_img = np.concatenate((gray_img_x, gray_img_y), axis=2)
    return dual_channel_img

def to_gray(img, norm=True, from_rgb=True):
    from_to = cv2.COLOR_RGB2GRAY if from_rgb else cv2.COLOR_BGR2GRAY
    gray_img = cv2.cvtColor(img, from_to)
    img = cv2.equalizeHist(gray_img)
    img = img.reshape(img.shape + (1,))   # single channel 
    if norm:
        img = img / 255.0 - 1
    return img.astype(np.float64)

def to_gray_center(img, norm=True, from_rgb=True):
    from_to = cv2.COLOR_RGB2GRAY if from_rgb else cv2.COLOR_BGR2GRAY
    gray_img = cv2.cvtColor(img, from_to)
    
    eye_half = int(img.shape[0]/2) - 16
    in_margin = 12
    out_margin = 84
    
    r = [eye_half,-eye_half,in_margin,out_margin]
    roi = gray_img[r[0]:r[1], r[2]:r[3]]
    gray_img[r[0]:r[1], r[2]:r[3]] = cv2.equalizeHist(roi)

    img = gray_img
    img = img.reshape(img.shape + (1,))   # single channel 
    
    if norm:
        img = img / 255.0 - 1
    return img.astype(np.float64)

def to_gray_niqab(img, norm=True, from_rgb=True):
    from_to = cv2.COLOR_RGB2GRAY if from_rgb else cv2.COLOR_BGR2GRAY
    gray_img = cv2.cvtColor(img, from_to)
    
    eye_half = int(img.shape[0]/2) - 14
    in_margin = 16
    out_margin = 88
    
    rl = [eye_half,-eye_half,in_margin,out_margin]
    rr = [eye_half,-eye_half,-out_margin,-in_margin]
    
    l_eye_roi = gray_img[rl[0]:rl[1], rl[2]:rl[3]]
    r_eye_roi = gray_img[rr[0]:rr[1], rr[2]:rr[3]]

    gray_img[rl[0]:rl[1], rl[2]:rl[3]] = cv2.equalizeHist(l_eye_roi)
    gray_img[rr[0]:rr[1], rr[2]:rr[3]] = cv2.equalizeHist(r_eye_roi)

    img = gray_img
    img = img.reshape(img.shape + (1,))   # single channel 
    
    if norm:
        img = img / 255.0 - 1
    return img.astype(np.float64)
    
def variance_of_laplacian(image, as_gray=False):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    if as_gray:
        iamge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def clear_score(image, as_gray=False):
    return round(variance_of_laplacian(image, as_gray))


class Undistorter:
    _map = None
    _prev_param = None

    def should_parameter_update(self, all_params):
        return self._prev_param is None or len(self._prev_param) != len(all_params) \
            or not np.allclose(all_params, self._prev_param)

    def update_undistort_map(self, cam_mat, dist_coef, img_wh, all_params):
        self._map = cv2.initUndistortRectifyMap(cam_mat, dist_coef, R=None,
                                               size=img_wh, m1type=cv2.CV_32FC1, 
                                               newCameraMatrix=cam_mat)
        self._prev_param = np.copy(all_params)

    def __call__(self, image, camera_matrix, distortion):
        h, w, _ = image.shape
        all_params = np.concatenate([camera_matrix.flatten(), distortion.flatten(), [h, w]])

        if self.should_parameter_update(all_params):
            self.update_undistort_map(camera_matrix, distortion, (w, h), all_params)

        return cv2.remap(image, self._map[0], self._map[1], cv2.INTER_LINEAR)



"""
    TF Record convert (정리 안 했음 그냥 백업만)
"""
def as_list(value):
    return value if isinstance(value, list) else [value]

def feature_bytes(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=as_list(value)))

def feature_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def feature_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def load_split_and_summary(base_dir_path):
    split_path = os.path.join(base_dir_path, 'split-origin.pkl')
    summary_path = os.path.join(base_dir_path, 'summary.pkl')
    with open(split_path, 'rb') as f:
        split = pickle.load(f)
    with open(summary_path, 'rb') as f:
        summary = pickle.load(f)
    train_info = split['train']
    valid_info = split['valid']
    
    return summary, train_info, valid_info

def convert(pid_to_items, ctx, is_valid, shuffle_stride=20):
    no_meta_pids = []
    for pid in pid_to_items:
        meta_path = os.path.join(ctx.npy_root_path, 
                                 'metas', 'meta-{}.pkl'.format(pid))
        if not os.path.exists(meta_path):
            no_meta_pids.append(pid)

    pids = list(pid_to_items.keys())
    pids = [p for p in pids if p not in no_meta_pids]     # filter no recode_meta
    sorted(pids)
    
    for block_idx, i in enumerate(range(0, len(pids), shuffle_stride)):
        block_pids = pids[i: i+shuffle_stride]
        items = load_items(block_pids, ctx)
        writer = make_writer(ctx.out_root_path, block_idx, is_valid)
        write_block_items(writer, items, block_idx)
        
def item_to_example(item):
    feature_dict = {
        'le'          : feature_bytes(item.le),
        're'          : feature_bytes(item.re),
        'frame'       : feature_bytes(item.frame),
        
        'eye_pos_l'   : feature_float(item.eye_pos_l),
        'eye_pos_r'   : feature_float(item.eye_pos_r),
        
        'face_faze_R' : feature_float(item.face_faze_R),
        'face_xu_R'   : feature_float(item.face_xu_R),
        'inv_nR'      : feature_float(item.inv_nR),

        't_xy_mm'     : feature_float(item.t_xy_mm),
        'gaze_l'      : feature_float(item.gaze_l),
        'gaze_r'      : feature_float(item.gaze_r),
        'ngaze_l'     : feature_float(item.ngaze_l),
        'ngaze_r'     : feature_float(item.ngaze_r),

        'pid'         : feature_int64(item.pid),
        'id'          : feature_int64(item.id),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature_dict)) 

def load_items(pids, ctx):
    from ds.seq import Sequence
    items = []
    for pid in pids:
        dp = Sequence(ctx, pid)
        if len(dp) <= 0:
            print("zero item profile", pid)
            continue
        for item in dp:
            items.append(item)
    return items

def make_writer(out_path, subject_key, is_valid):
    prefix = 'valid' if is_valid else 'train'
    path = os.path.join(out_path, '{:s}-{:05d}.tf'.format(prefix, subject_key))
    writer = tf.io.TFRecordWriter(path=path)
    return writer
    
def write_block_items(writer, items, block_idx):
    examples = []
    for item in items:
        examples.append(item_to_example(item))
    for e in examples:
        writer.write(e.SerializeToString())
    writer.close()
    print(">>> wirte block is done [", block_idx, "]  len(", len(items), ")")





















"""
    DB
"""
device_data = {
    'iphone se': -1,
    'iphone 4': -1,
    'iphone 4s': {
        'matrix': [606.59362793, 609.2008667, 236.86116028, 312.28497314],
        'distortion': [ 0.24675941, -0.65499198,  0.00301733, -0.00097767]
    },
    'iphone 5': {
        'matrix': [623.28759766, 626.64154053, 236.86317444, 316.909729  ],
        'distortion': [ 0.03760624, -0.043609, -0.00114902,  0.00269194]
    },
    'iphone 5c': {
        'matrix': [585.13171387, 588.14447021, 242.18914795, 321.20614624],
        'distortion': [ 0.01302955, -0.10349616, -0.0009803,  0.00301618]
    },
    'iphone 5s': {
        'matrix': [585.13171387, 588.14447021, 242.18914795, 321.20614624],
        'distortion': [ 0.01302955, -0.10349616, -0.0009803,  0.00301618]
    },
    'iphone 6': {
        'matrix': [592.50164795, 595.66986084, 236.12217712, 327.50753784],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone 6 plus': {
        'matrix': [592.50164795, 595.66986084, 236.12217712, 327.50753784],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone 6s': {
        'matrix': [592.50164795, 595.66986084, 236.12217712, 327.50753784],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone 6s plus': {
        'matrix': [592.50164795, 595.66986084, 236.12217712, 327.50753784],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone 7': {
        'matrix': [592.50164795, 595.66986084, 236.12217712, 327.50753784],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone 7 plus': {
        'matrix': [592.50164795, 595.66986084, 236.12217712, 327.50753784],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone 8': {
        'matrix': [580.34485, 581.34717, 239.41379, 319.58548],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone 8 plus': {
        'matrix': [580.34485, 581.34717, 239.41379, 319.58548],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone x': {
        'matrix': [592.16473, 593.1875, 242.00687, 320.23456],
        'distortion': [ 0.0822313, -0.18398251, -0.00631323, -0.00075782]
    },
    'iphone xs': -1,
    'iphone xs max global': -1,
    'iphone xr': -1,

    'ipad air': {
        'matrix': [578, 578, 240, 320],
        'distortion': [0.124, -0.214, 0, 0]
    },  # ipad air from Web
    'ipad air 2': {
        'matrix': [592.35223389, 595.9105835, 234.15885925, 313.48773193],
        'distortion': [ 1.93445340e-01, -5.54507077e-01,  6.13935478e-03,  3.40262457e-04]
    },
    'ipad 2': {
        'matrix': [621.54315186, 624.44012451, 233.66329956, 313.44387817],
        'distortion': [-0.0243901, -0.10230259, -0.00513017,  0.00057966]
    },
    'ipad 6': -1,
    'ipad pro 2 (10.5-inch': -1,
    'ipod touch 6': -1,
    'ipad mini': {
        'matrix': [623.28759766, 626.64154053, 236.86317444, 316.909729],
        'distortion': [ 0.03760624, -0.043609, -0.00114902,  0.00269194]
    },
}

