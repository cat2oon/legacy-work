import os 
import cv2
import json
import glob
import h5py
import pickle
import numpy as np
import tensorflow as tf

from itertools import chain
from collections import namedtuple
from matplotlib import pyplot as plt


"""
    Device Screen Camera Difference (cm)
"""
cam_screen_dist_meta = {
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
    Faze supplementary
"""
def load_supplymentary(supplymentary_path):
    with h5py.File(supplymentary_path, 'r') as f:
        data = {}
        for pid, group in f.items():
            p_data = []
            for i in range(next(iter(group.values())).shape[0]):
                item = to_supply_item(group, i)
                p_data.append(item)
            data[pid] = p_data
    return data 

def to_supply_item(group, idx):
    filename = group['file_name'][idx]
    head_pose = group['head_pose'][idx, :]
    gaze_3d = group['3d_gaze_target'][idx, :]
    cam_params = group['camera_parameters'][idx, :]
    distort_params = group['distortion_parameters'][idx, :]
    item_idx = int(filename.decode('utf-8')[-9:-4])
    named_type = namedtuple(
        "item", ['i', 'gaze_3d', 'head_pose', 'cam_params', 'distort_params'])
    return named_type(item_idx, gaze_3d, head_pose, cam_params, distort_params)

def to_pickle(supply, out_dir_path):
    for key in supply.keys():
        ps_arr = []
        for nt in supply[key]:
            ps_arr.append({
                'i' : nt.i, 
                'gaze_3d' : nt.gaze_3d, 
                'head_pose' : nt.head_pose, 
                'cam_params': nt.cam_params, 
                'distort_params' : nt.distort_params
            })

        out_path = os.path.join(out_dir_path, '{:s}-supply.pkl'.format(key))
        with open(out_path, 'wb') as f:
            pickle.dump(ps_arr, f)
            
def load_supply_pickle(base_dir_path, profile_id, as_task=True):
    path = os.path.join(base_dir_path, '{}-supply.pkl'.format(profile_id))
    with open(path, 'rb') as f:
        x = pickle.load(f)
    if not as_task:
        return x
    
    supply_items = []
    ItemType = namedtuple(
        "item", ['i', 'gaze_3d', 'head_pose', 'cam_params', 'distort_params'])
    for item in x:
        supply_items.append(ItemType(*item.values()))
    return supply_items


"""
    File System
"""
def grep_recur(base_path, pattern="*.*"):
    sub_greps = list(chain(*[grep_recur(dp, pattern) for dp in grep_dirs(base_path)]))
    return grep_files(base_path, pattern) + sub_greps

def grep_files(base_path, pattern="*.*"):
    return glob.glob("{}/{}".format(base_path, pattern))

def grep_pairs(base_path, ext_x, ext_y, check_missing=False):
    xs = grep_files(base_path, "*.{}".format(ext_x))
    ys = grep_files(base_path, "*.{}".format(ext_y))
    xs.sort()
    ys.sort()
    return list(zip(xs, ys))

def grep_dirs(base_path):
    file_paths = [os.path.join(base_path, name) for name in os.listdir(base_path)]
    return [p for p in file_paths if os.path.isdir(p)]


"""
    Utility
"""
def timeit(func):
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('%r  %2.2f s' % (func.__name__, (end - start)))
        return result
    return timed

def preprocess_image(image, resize_wh):
    image = cv.resize(image, resize_wh, interpolation=cv.INTER_AREA)
    ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
    image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
    image = 1.0 * image / 255.0 - 1
    # image = np.rollaxis(image, 2, 0)
    return image


"""
    Device Info
"""
def get_camera_matrix(fx, fy, cx, cy, skewed=1.0):
    cm = np.array([
        [fx, 0,     cx],
        [0, fy,     cy],
        [0,  0, skewed],
    ])
    return np_around(cm)

def get_camera_distortion(distortion_coef):
    return np_around(np.hstack((distortion_coef, 0)))

def np_around(x, d=5):
    return np.around(x, decimals=d)


"""
    GazeCapture Face
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

def eye_corner_pos_to_uv(eye_corner, camera_mat):
    # eye corner landmark [ x1 x2 x3 x4 y1 y2 y3 y4 ]
    ec, cm = eye_corner, camera_mat
    fx, fy, cx, cy = cm[0, 0], cm[1, 1], cm[0, 2], cm[1, 2]
    xs, ys = ec[0:4], ec[4:]
    xs = (xs - cx) / fx    
    ys = (ys - cy) / fy    
    return np.hstack([xs, ys])


"""
    Recode
"""
def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def recode_meta(idx, s, meta, summary, face_model, norm_spec):
    m = meta[idx]
    frame_idx = int(m['frame_name'][:-4])
    device = summary['device']
    csd = cam_screen_dist_meta[device]
    
    assert s.i == idx == frame_idx, "inconsistent idx"
    assert 'size_wh' in norm_spec, "missing normalize info"
    assert 'distance' in norm_spec, "missing normalize info"
    assert 'focal_length' in norm_spec, "missing normalize info"
    
    """ supplymentary 제공 정보 """
    cam_mat = get_camera_matrix(*s.cam_params)
    cam_distort = get_camera_distortion(s.distort_params)
    
    """ rotation matrix """
    rvec = s.head_pose[:3].reshape(3,1)
    tvec = s.head_pose[3:].reshape(3,1)
    rotate_mat, _ = cv2.Rodrigues(rvec)
    
    """ face 3d positions """
    landmarks_3d_orientation = np.matmul(rotate_mat, face_model.T).T
    landmarks_3d = landmarks_3d_orientation + tvec.T
    
    """ gaze-origin and gaze target"""
    gaze_o = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
    gaze_o = gaze_o.reshape(3, 1)
    gaze_t = s.gaze_3d[:].reshape(3, 1)
    gaze = gaze_t - gaze_o
    gaze_unit = gaze / np.linalg.norm(gaze)
    
    """ actual distance between gaze origin and original camera """
    real_dist_cam_to_face = np.linalg.norm(gaze_o)
    z_scale = norm_spec['distance'] / real_dist_cam_to_face
    scale_mat = np.eye(3, dtype=np.float64)
    scale_mat[2, 2] = z_scale
    
    """ 바라보는 축을 기준으로 좌표축 설정 """
    hR_x_axis = rotate_mat[:, 0]
    forward = (gaze_o / real_dist_cam_to_face).reshape(3)
    down = np.cross(forward, hR_x_axis)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R
    
    """ 변환 행렬 """
    w, h = norm_spec['size_wh']
    fxy = norm_spec['focal_length']
    norm_cam_mat = get_camera_matrix(fxy, fxy, w/2, h/2)
    warp_mat = np.dot(np.dot(norm_cam_mat, scale_mat),
                      np.dot(R, np.linalg.inv(cam_mat)))
    
    """ 시선 방향 정규화 후 head pose """
    R = np.asmatrix(R)
    head_pose_pitch_yaw = np.array([np.arcsin(rotate_mat[1, 2]),
                                    np.arctan2(rotate_mat[0, 2], rotate_mat[2, 2])])
    head_mat = R * rotate_mat
    norm_head_pose_pitch_yaw = np.array([np.arcsin(head_mat[1, 2]), 
                                         np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    """ 정규화 후 gaze """
    norm_gaze = R * gaze_unit
    norm_gaze /= np.linalg.norm(norm_gaze)
    norm_gaze_pitch_yaw = vector_to_pitchyaw(-norm_gaze.T).flatten()
    
    norm_meta = {
        'z_scale' : z_scale,
        'warp_mat' : warp_mat,
        'forward_rot_mat' : R,
        'norm_cam_mat' : norm_cam_mat,
        'norm_gaze' : norm_gaze,
        'gaze_pitch_yaw' : norm_gaze_pitch_yaw,
        'head_pose_pitch_yaw' : norm_head_pose_pitch_yaw,
        'real_dist_cam_to_face' : real_dist_cam_to_face,
    }
    norm_meta.update(norm_spec)
    
    item = {
        'frame_id' : s.i,
        'head_pose' : np_around(s.head_pose),
        'head_pose_pitch_yaw' : head_pose_pitch_yaw,
        'rotate_mat' : np_around(rotate_mat),
        'cam_matrix'  : cam_mat,
        'distort_params' : cam_distort,
        'landmarks_3d' : landmarks_3d,
        'landmarks_3d_orientation' : landmarks_3d_orientation,
        'cam_to_screen_dist' : csd,
        
        'gaze_origin' : gaze_o,
        'gaze_target' : gaze_t,
        'normalized' : [norm_meta]
    }
    item.update(m)
    
    return item

def warp_frame(frame, norm_meta, norm_slot=0):
    norm_info = norm_meta['normalized'][norm_slot]
    ow, oh = norm_info['size_wh']
    W = norm_info['warp_mat']
    frame = tf.io.decode_jpeg(frame).numpy()
    patch = cv2.warpPerspective(frame, W, (ow, oh))
    return patch, frame
    
def recode_profile_meta(npz, supply, face_model, norm_spec):
    meta = npz['meta']
    summary = npz['summary'].tolist()
    
    recoded_meta = { 'summary':summary, 'items':[] }
    for s_item in supply: # supply가 기준이 되어야 함
        meta_item = recode_meta(s_item.i,  s_item, meta, 
                                summary, face_model, norm_spec)
        recoded_meta['items'].append(meta_item)
    return recoded_meta

def recode_all_meta(out_dir_path, npz_paths, supply_dir_path, face_model, norm_spec):
    for npz_path in npz_paths:
        try:
            pid = npz_path[-9:-4]
            supply = load_supply_pickle(supply_dir_path, pid)
            npz = np.load(npz_path, allow_pickle=True)
            norm_meta = recode_profile_meta(npz, supply, face_model, norm_spec)

            out_path = os.path.join(out_dir_path, 'meta-{:s}.pkl'.format(pid))
            with open(out_path, 'wb') as f:
                pickle.dump(norm_meta, f)
        except Exception as e:
            print("Failed at profile[{}]\n".format(pid), e)
            
def recode_all_meta_parallel(out_dir_path, npz_paths, supply_dir_path, face_model_path, norm_spec):
    face_model = np.load(face_model_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
        for i in range(0, len(npz_paths), 25):
            paths = npz_paths[i:i+25]
            e.submit(recode_all_meta, out_dir_path, paths, supply_dir_path, face_model, norm_spec)
            
def recode_all_summary(out_dir_path, npz_paths):
    summary = {}
    for npz_path in npz_paths:
        pid = npz_path[-9:-4]
        npz = np.load(npz_path, allow_pickle=True)
        
        profile_summary = npz['summary'].tolist()
        summary[pid] = profile_summary
        oris, counts = np.unique([m['orientation'] for m in npz['meta']], return_counts=True) 
        
        co = list(zip(oris, counts))
        if 2 not in oris:
            co = co + [(2, 0)]
        profile_summary['count_orientations'] = co
            
    out_path = os.path.join(out_dir_path, 'summary.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(summary, f)            

"""
    Visualize
"""
def visualize_profile(npz_paths, nps_dir_path, pid, idx=0):
    target_path = [p for p in npz_paths if pid in p][0]
    npz = np.load(target_path)
    with open(os.path.join(nps_dir_path, 'meta-pickle', 'meta-{}.pkl'.format(pid)), 'rb') as f:
        meta_norm = pickle.load(f)
    img = visualize(npz, meta_norm, idx)
    print(img.shape)

def visualize(npz, norm_meta, idx, norm_slot=0):
    m = norm_meta['items'][idx]
    
    norm_info = m['normalized'][norm_slot]
    ow, oh = norm_info['size_wh']
    norm_gaze = norm_info['gaze_pitch_yaw']
    norm_head = norm_info['head_pose_pitch_yaw']
    frame = npz['frames'][idx]
    img, origin = warp_frame(frame, m)
    
    img = draw_gaze(img, (0.5 * ow, 0.25 * oh), norm_gaze, length=120.0, thickness=2)
    plt.figure(figsize=(6,6))
    plt.subplot(1, 2, 1)
    plt.imshow(origin)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    return img 

def draw_landmarks(image, landmarks):
    for pt in landmarks:
        pos = tuple([pt[0].astype(int), pt[1].astype(int)])
        cv2.circle(image, pos, 3, (0, 0, 255), -1)
        
    image = image[:,:,::-1]
    plt.imshow(image)
    figure(num=None, figsize=(800, 600), dpi=80, facecolor='w', edgecolor='k')
    plt.show()

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx,
                                   eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out












"""
    Legacy 
"""
def legacy_recode_item(npz, idx):
    cam_param, distor_coef = get_focal_length_and_distortion(device)   
    
    if cam_param is None:
        # print("can't not find device. [{0} at {1}]".format(device, profile_id))
        return
    
    camera_distortion = np.hstack((distor_coef, 0))
    camera_matrix = get_square_camera_matrix(cam_param[0], cam_param[3])
     
    landmarks = get_landmarks_from(profile_id, frame_name)
    if landmarks is None:
        # print("No landmarks {0} at {1}".format(profile_id, frame_name))
        return
    
    landmarks, face_3d = remove_outside_image(landmarks, copy.deepcopy(standardFace))
    num_points = len(landmarks)
    
    lookat = np.array([data['XCam'], data['YCam'], 0])
    lookat = lookat * 10             # cm to mm
    
    # undistort landmark points and image
    landmarks = cv2.undistortPoints(landmarks.reshape(num_points, 1, 2), camera_matrix, camera_distortion, P=camera_matrix)
    image_undistorted = cv2.undistort(image, camera_matrix, camera_distortion)

    face_3d = face_3d.reshape(num_points, 1, 3)
    landmarks = landmarks.reshape(num_points, 1, 2)
    r_vec, t_vec = estimate_headpose(landmarks, face_3d, camera_matrix, camera_distortion)
    
    face_3d = face_3d.reshape(num_points, 3).T
    t_vec = t_vec.reshape((3, 1))
    r_vec = cv2.Rodrigues(r_vec)[0]

    # warped image, rotated gaze vector, face R, S, W, 3D face points
    image_warped, optical_vecs, R, S, W, FACE = normalize_frame(image_undistorted, face_3d, r_vec, t_vec, camera_matrix, lookat)
    landmarks = cv2.perspectiveTransform(landmarks, W)
    image_cut, landmarks_e = cut_image(image_warped, landmarks.reshape(num_points, 2))
    
def normalize_frame(image, face_3d, r_vec, t_vec, camera_mat, look_vec):
    t_vec = t_vec.reshape((3, 1))
    look_vec = look_vec.reshape((3, 1))
    
    hR = cv2.Rodrigues(r_vec)[0]
    Fc = np.dot(hR, face_3d) + t_vec
    origin = Fc[:, 15].reshape((3, 1))     # nose_tip
    real_distance = np.linalg.norm(origin)
    
    # std_cam = get_standard_camera_matrix()
    S = get_scale_at_standard_dist(real_distance)
    R = get_rotation_for_coord_system(hR, origin, real_distance)
    
    W = np.dot(np.dot(std_cam, S), np.dot(R, np.linalg.inv(camera_mat)))  # transformation matrix
    img_warped = cv2.warpPerspective(image, W, get_standard_roi_size())   # image normalization

    eye_right = np.array([sum(x) for x in Fc[:, 0:6]]) / 6
    eye_left = np.array([sum(x) for x in Fc[:, 6:12]]) / 6
    eye_center = (eye_right + eye_left) / 2.0
    gaze_dirs = get_optical_vecs(eye_right, eye_left, eye_center);
    
    return img_warped, np.array(gaze_dirs), R, S, W, Fc.T

def get_optical_vecs(*eyes):
    dir_vecs = []
    for eye in eyes:
        eye = eye.reshape((3, 1))
        g = look_vector*np.array([[-1], [-1], [1]]) - eye
        g = np.dot(R, g)
        g = g / np.linalg.norm(g)
        dir_vecs.append(g)
    return dir_vecs

def get_rotation_for_coord_system(R, origin, distance):
    hRx = R[:, 0]
    forward = (origin / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    return np.column_stack[right, down, forward].T

def cut_image(image, landmarks):
    width = image.shape[1]
    height = image.shape[0]

    min_x = min(landmarks[:, 0])
    max_x = max(landmarks[:, 0])
    min_y = min(landmarks[:, 1])
    max_y = max(landmarks[:, 1])

    w = max_x - min_x
    h = max_y - min_y

    img_xs = int(max(min_x - 0.4 * w, 0))
    img_xe = int(min(max_x + 0.4 * w, width))
    img_ys = int(max(min_y - 0.5 * h, 0))
    img_ye = int(min(max_y + 0.1 * h, height))

    # if jaws are removed
    if len(landmarks) == 17:
        img_ys = int(max(min_y - 1.5 * h, 0))
        img_ye = int(min(max_y + 2.0 * h, height))

    return image[img_ys:img_ye, img_xs:img_xe], landmarks - [img_xs, img_ys]

def remove_outside_image(landmarks, face3d):
    # removing jaws only
    isoutside = False
    width = image.shape[1]
    height = image.shape[0]

    for i in range(17, 20):
        if landmarks[i][0] > width or landmarks[i][1] > height:
            isoutside = True
            break

    if isoutside:
        return landmarks[0:17], face3d[0:17]
    else:
        return landmarks, face3d
    
def get_standard_3d_face_points():
    mode_path = 'standard3DFace.json'
    if not os.path.isfile(mode_path):
        print(mode_path, 'not found!')
        return None

    with open(mode_path, 'r') as f:
        data = json.load(f)

    eye_right   = np.array( data['eye_right'],   dtype='float32')
    eye_left    = np.array( data['eye_left'],    dtype='float32')
    nose        = np.array( data['nose'],        dtype='float32')
    nose_bottom = np.array( data['nose_bottom'], dtype='float32')
    jaws        = np.array( data['jaws'],        dtype='float32')

    return np.vstack((eye_right, eye_left, nose, nose_bottom, jaws))
    
def estimate_headpose(refined_landmarks, position, camera_matrix, camera_distortion):
    ret, rvec, tvec = cv2.solvePnP(position, 
                                   refined_landmarks,
                                   camera_matrix, 
                                   camera_distortion, flags=cv2.SOLVEPNP_EPNP)
    ret, rvec, tvec = cv2.solvePnP(position, 
                                   refined_landmarks, 
                                   camera_matrix, 
                                   camera_distortion, rvec, tvec, True)
    return rvec, tvec

def get_landmarks_from_json(json_data):
    if 'vcft' not in json_data:
        return None
    if 'face_landmarks' not in json_data['vcft']:
        return None
    
    jsonarr = np.array(json_data['vcft']['face_landmarks'])
    jsonarr = jsonarr.reshape(int(len(jsonarr) / 2), 2)
    
    return pick_effective_landmarks(jsonarr)

def pick_effective_landmarks(raw_landmarks):
    jaws = raw_landmarks[7:10]
    eye_right = raw_landmarks[36:42]
    eye_left = raw_landmarks[42:48]
    nose_bridge = raw_landmarks[27:31]
    nose_bottom = raw_landmarks[33:34]

    return np.vstack((eye_right, eye_left, nose_bridge, nose_bottom, jaws))
