import os
import glob
import time
import json
import cv2 as cv
import numpy as np
import tensorflow as tf
import multiprocessing
import concurrent.futures

from itertools import chain


"""
    Feature Types
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


"""
    Free functions
"""
def grep_recur(base_path, pattern="*.*"):
    sub_greps = list(chain(*[grep_recur(dp, pattern) for dp in grep_dirs(base_path)]))
    return grep_files(base_path, pattern) + sub_greps

def grep_files(base_path, pattern="*.*"):
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

@timeit
def collect_tablet(npz_paths, num_min_items=200):
    small_pids, tablet_pids = [], []
    for p in npz_paths:
        pid = p[-9:-4]
        npz = np.load(p, allow_pickle=True)
        summary = npz['summary'].tolist()
        if 'pad' in summary['device'].lower():
            tablet_pids.append(pid)
        if int(summary['num_frames']) < num_min_items:
            small_pids.append(pid)
    return tablet_pids, small_pids


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

device_data = {
    'iphone se': -1,
    'iphone 4': -1,
    'iphone 4s': {
        'matrix': [606.59362793, 609.2008667, 236.86116028, 312.28497314],
        'distortion': [ 0.24675941, -0.65499198,  0.00301733, -0.00097767]
    },
    'iphone 5': {
        'matrix': [623.28759766, 626.64154053, 236.86317444, 316.909729],
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
    },  
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

def get_camera_matrix(fx, fy, cx, cy, skewed=1.0):
    return np.array([
        [fx, 0,     cx],
        [0, fy,     cy],
        [0,  0, skewed],
    ])

def get_camera_param_and_distortion(device_name):
    device_name = device_name.lower()
    
    if not(device_name in device_data):
        return None, None
    
    if device_data[device_name] == -1:
        return None, None
    
    di = device_data[device_name]
    camera_distortion = np.hstack((di['distortion'], 0))
    camera_matrix = get_camera_matrix(*di['matrix'])
    
    return camera_matrix, camera_distortion

"""
    Exclude Profiles
"""
def get_exclude_pids():
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

    tablet_profiles =  [
        '01383', '02522', '00801', '03117', '01085', '02298', '00842', '01618', '02878', '02342', 
        '01200', '01901', '01243', '01295', '01151', '02204', '00986', '01090', '01726', '00853',
        '01508', '02118', '00326', '01039', '02883', '01582', '00010', '00325', '01052', '01269',
        '02272', '01158', '02193', '00827', '00578', '01183', '00295', '01843', '00791', '01283',
        '02198', '00135', '03027', '03266', '00509', '00894', '01862', '02087', '01025', '01266',
        '01627', '00757', '02413', '02805', '00756', '03413', '01727', '00138', '00317', '01905',
        '01373', '02267', '01221', '02436', '02869', '02857', '03277', '02450', '02734', '01232',
        '00850', '00718', '01173', '01717', '02761', '01104', '00208', '00546', '00741', '00927',
        '01128', '00980', '01233', '00613', '00358', '00028', '00825', '02613', '00666', '01038',
        '02370', '02700', '01022', '03442', '01091', '00210', '01524', '00796', '02967', '00619',
        '02349', '01352', '00806', '00921', '02902', '00241', '02879', '01088', '02368', '01384',
        '01089', '01152', '01390', '01060', '02961', '00891', '02480', '02414', '00934', '00981', 
        '00804', '01941'
    ]

    small_profiles =  [
        '00627', '00841', '00984', '00243', '00693', '00003', '01755', '01421', '00005', '00354', 
        '01483', '00807', '03463', '03023', '01167', '01462', '00578', '00024', '00728', '00295', 
        '00006', '00687', '00103', '00311', '01542', '00584', '00819', '01168', '00317', '00542', 
        '01087', '00670', '00649', '01275', '01213', '00997', '01260', '01393', '02956', '00087', 
        '00307', '00825', '01038', '00982', '00774', '00002', '00303', '00089', '01022', '01276', 
        '01482', '00339', '00467', '00289', '00632', '00675', '01705', '00544', '02989', '01102', 
        '00621', '01169', '00633'
    ]    
    
    return exclude_profiles + tablet_profiles + small_profiles


"""
    Writer
"""
def make_writer(out_path, subject_key):
    path = os.path.join(out_path, 'ps-{:05d}.tf'.format(subject_key))
    # options = tf.io.TFRecordOptions(compression_type='GZIP')
    # writer = tf.io.TFRecordWriter(path=path, options=options)
    writer = tf.io.TFRecordWriter(path=path)
    return writer

def is_valid_item(m):
    return m['face_valid'] and m['face_grid_valid'] and m['left_eye_valid'] and m['right_eye_valid']

@timeit
def split_by_orientation(npz):
    summary = npz['summary'].tolist()
    device = summary['device'].lower()
    csd = cam_screen_dist_meta[device]
    num_frames = int(summary['num_frames'])
    
    items_per_ori = {}
    items_per_ori[1] = []
    items_per_ori[2] = []
    items_per_ori[3] = []
    items_per_ori[4] = []
    
    items_per_ori = [[], [], [], []]
    for i in range(num_frames-1):
        m = npz['meta'][i]
        o = m['orientation']
        if not is_valid_item(m):
            continue

#     items_per_ori = {}
#     for i in range(num_frames-1):
#         m = npz['meta'][i]
#         if not is_valid_item(m):
#             continue
#         f = npz['frames'][i]
#         o = m['orientation']
#         if o not in items_per_ori:
#             items_per_ori[o] = []
#         items_per_ori[o].append({
#             'meta':m, 'frame':f, 'cam_screen_dist':csd, 'device':device
#         })
        
    return items_per_ori

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

def preprocess_image(image, resize_wh):
    image = cv.resize(image, resize_wh, interpolation=cv.INTER_AREA)
    ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
    image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
    image = 2.0 * image / 255.0 - 1
    return image
    
undistort = Undistorter()

def encode_item(item, idx, prefix):
    m = item['meta']
    f = item['frame']
    d = item['device']
    csd = item['cam_screen_dist']
    prefix = prefix + str(idx) + '_'
    cam_param, distort = get_camera_param_and_distortion(d)
    
    # 프레임 
    frame = tf.io.decode_jpeg(f).numpy()
    # u_frame = undistort(frame, cam_param, distort)

    # eye corner region landmark
    face_rect = m['face_rect']
    le_rect, re_rect = m['left_eye_rect'], m['right_eye_rect']
    eye_corner = get_eye_corner(face_rect, le_rect, re_rect, frame.shape[:2])
    eye_loc = eye_corner_pos_to_uv(eye_corner, cam_param)
    
    # left/right eye image
    le_img, re_img = get_eye_left_right_patch(frame, face_rect, le_rect, re_rect)
    le_img = preprocess_image(le_img, (64, 64))
    re_img = preprocess_image(re_img, (64, 64))
    le_img = tf.io.encode_jpeg(le_img, quality=100).numpy()
    re_img = tf.io.encode_jpeg(re_img, quality=100).numpy()
        
    # support info
    o = np.array([m['orientation']])
    
    # label 
    target_xy = np.array([m['target_dist']['x'], m['target_dist']['y']])
    
    e = {}
    e['le_img'] = feature_bytes(le_img)
    e['re_img'] = feature_bytes(re_img)
    e['eye_loc'] = feature_float(eye_loc)
    e['target_xy'] = feature_float(target_xy)
    
    e['orientation'] = feature_int64(o)
    e['cam_screen_dist'] = feature_float(csd)
    
    """ candidate feature """
    # e['cam_extrinsic']
    # e['cam_param']
    
    return {str(prefix+k):v for k, v in e.items()} 

def make_example(items, num_supports, num_targets):
    feature_dict = {}
    supports, targets = items[:num_supports], items[num_supports:] 
    for i, s in enumerate(supports):
        feature_dict.update(encode_item(s, i, 's'))
    for i, t in enumerate(targets):
        feature_dict.update(encode_item(t, i, 't'))
    return tf.train.Example(features=tf.train.Features(feature=feature_dict)) 
        
@timeit
def examples_from_npz(npz, num_shots, num_targets):
    examples = []
    num_pack = num_shots + num_targets
    items_per_ori = split_by_orientation(npz)
    for o, items in items_per_ori.items():
        np.random.shuffle(items)
        for i in range(0, len(items)-num_pack+1, num_pack):
            batch = items[i:num_pack]
            e = make_example(batch, num_shots, num_targets)
            examples.append(e)
    return examples

def write_block(writer, npz_paths, num_k_shots, num_targets, i):
    examples = []
    path_to_npz = {p:np.load(p, allow_pickle=True) for p in npz_paths}
    for npz_path, npz in path_to_npz.items():
        print(">>> Load:", npz_path)
        es = examples_from_npz(npz, num_k_shots, num_targets)
        examples += es
        
    np.random.shuffle(examples)
    for e in examples:
        writer.write(e.SerializeToString())
    writer.close()
    print(">>> wirte is done [", i, "]")

def convert(npz_root_path, out_root_path, num_k_shots=5, num_targets=5, block_size=10, seed=1234):
    np.random.seed(seed)
    npz_paths = grep_recur(npz_root_path, "*.npz")
    
    # 태블릿 제외, 200장 미만 제외
    # tablets, smalls = collect_tablet(npz_paths)
    exclude_pids = get_exclude_pids()
    npz_paths = [path for path in npz_paths if path[-9:-4] not in exclude_pids]
    np.random.shuffle(npz_paths)
    print("최종 선정 프로파일 개수:", len(npz_paths))
    
    def convert_task(paths, i):
        print("Executor[", i, "] running")
        writer = make_writer(out_root_path, i) 
        print(paths)
        write_block(writer, paths, num_k_shots, num_targets, i)
        print("Task[",i,"] complete")
    
    num_workers = multiprocessing.cpu_count() / 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as e:
        for idx, i in enumerate(range(0, len(npz_paths), block_size)): 
            paths = npz_paths[i:i+block_size]
            e.submit(convert_task, paths, idx)
            break