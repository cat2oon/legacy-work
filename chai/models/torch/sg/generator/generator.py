import os
import glob
import json
import h5py
import torch
import logging
import cv2 as cv
import numpy as np

from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, Subset


missing_list = [
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
    "03059", "03060", "03212", "03224", "03239", "03380", "03389", "03474"   ]


"""DeviceCameraToScreenXMm, DeviceCameraToScreenYMm, DeviceScreenWidthMm, DeviceScreenHeightMm"""
loc_meta = {
    'iphone 6s plus':	[23.54	,8.66	,68.36	,121.54	],
    'iphone 6s':		[18.61	,8.04	,58.49	,104.05	],
    'iphone 6 plus':	[23.54	,8.65	,68.36	,121.54	],
    'iphone 6':			[18.61	,8.03	,58.5	,104.05	],
    'iphone 5s':		[25.85	,10.65	,51.7	,90.39	],
    'iphone 5c':		[25.85	,10.64	,51.7	,90.39	],
    'iphone 5':			[25.85	,10.65	,51.7	,90.39	],
    'iphone 4s':		[14.96	,9.78	,49.92	,74.88	],
    'ipad mini':		[60.7	,8.7	,121.3	,161.2	],
    'ipad air 2':		[76.86	,7.37	,153.71	,203.11	],
    'ipad air':			[74.4	,9.9	,149	,198.1	],
    'ipad 4':			[74.5	,10.5	,149	,198.1	],
    'ipad 3':			[74.5	,10.5	,149	,198.1	],
    'ipad 2':			[74.5	,10.5	,149	,198.1	],
    'ipad pro':			[98.31	,10.69	,196.61	,262.15	]
}

"""
{
    'frame_name': '00074.jpg', 
    'face_valid': True,  'face_grid_valid': True,
    'left_eye_valid': True, 'right_eye_valid': True, 
    'face_rect': {'h ': 378.559999093, 'w': 378.559999093, 'x': 57.7200004533, 'y': 57.7200004533}, 
    'face_grid_rect': {'h': 15, 'w': 15, 'x': 3, 'y': 6}, 
    'left_eye_rect': {'h': 113.568005371, 'w': 113.568005371, 'x': 177.923213117, 'y': 47.3199787252}, 
    'right_eye_rect': {'h': 113.568005371, 'w': 113.568005371, 'x': 35.9632064034, 'y': 41.6415784566}, 
    'target_id': 17, 
    'target_pts': {'x': 166.117594242, 'y': 157.682228088}, 
    'target_dist': {'x': 3.70855094076, 'y': 0.0374465024471}, 
    'target_time': 0.447148, 'screen_hw': {'h': 320, 'w': 568}, 'orientation': 3, 
    'norm_gaze_pitchyaw': array([-0.01114615,  0.27524275], dtype=float32), 
    'norm_head_pose': array([0.36828426, 0.28148794], dtype=float32), 
    'norm_rot_matrix': array([[ 0.9890568 ,  0.03299534, -0.14379804], [-0.05040688,  0.9915927 , -0.11917635], 
                                [ 0.13865682,  0.1251206 ,  0.98240477]], dtype=float32), 
    'gaze_direction': array([[-0.13004518], [ 0.13943431], [-0.9816549 ]], dtype=float32), 
    'gaze_origin': array([[-46.09127 ], [-38.199337], [314.88806 ]], dtype=float32), 
    'gaze_target': array([[-87.806206], [  6.52738 ], [ -0.      ]], dtype=float32), 
    'head_pose': array([0.23939598, 0.143912  ], dtype=float32), 
    'origin_extrinsic': array([-2.4010064e- 01,  1.4133327e-01,  1.5601543e-03, 
                                -4.5725677e+01, -3.8964321e+01,  3.1811200e+02], dtype=float32), 
    'origin_camera_param': array([
        [588.1445 ,   0.     , 318.79385],
        [  0.     , 585.1317 , 242.18915],
        [  0.     ,   0.     ,   1.     ]], dtype=float32), 
    'camera_distortion': array([ 0.01302955, -0.10349616,  0.00301618, -0.0009803 ] , dtype=float32)
}
"""


"""
 Undistorter 
"""
class Undistorter:
    _map = None
    _prev_param = None
    
    def should_parameter_update(self, all_params):
        return self._prev_param is None or len(self._prev_param) != len(all_params) \
            or not np.allclose(all_params, self._prev_param)
    
    def update_undistort_map(self, cam_mat, dist_coef, img_wh, all_params, is_gazecapture):
        new_cm = cam_mat if is_gazecapture else None
        self._map = cv.initUndistortRectifyMap(cam_mat, dist_coef, R=None, 
                                               size=img_wh, m1type=cv.CV_32FC1, newCameraMatrix=new_cm)
        self._prev_param = np.copy(all_params)
            
    def __call__(self, image, camera_matrix, distortion, is_gazecapture=True):
        h, w, _ = image.shape
        all_params = np.concatenate([camera_matrix.flatten(), distortion.flatten(), [h, w]])
        
        if self.should_parameter_update(all_params):
            self.update_undistort_map(camera_matrix, distortion, (w, h), all_params, is_gazecapture)

        return cv.remap(image, self._map[0], self._map[1], cv.INTER_LINEAR)


"""
 Free functions
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
    file_paths = [path_join(base_path, name) for name in os.listdir(base_path)]
    return [p for p in file_paths if os.path.isdir(p)]

def worker_init_fn(worker_id):
    # Custom worker init to not repeat pairs
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def byte_arr_to_img(byte_arr):
    dt = np.dtype(np.int8)  # 255
    dt = dt.newbyteorder('>')  # depend on mach arch
    np_arr = np.frombuffer(byte_arr, dt)
    return cv.imdecode(np_arr, 1)


"""
 NPZ Dataset
"""
class NPZDataset(Dataset):
    
    def __init__(self, npz_root_path,  data_tag,  profiles, item_selector_fn=None):
        assert os.path.isdir(npz_root_path)
        self.tag = data_tag
        self.num_cache = 8
        self.profiles = profiles 
        self.undistort = Undistorter()
        self.npz_root_path = npz_root_path
        self.item_selector_fn = item_selector_fn
        self.npz_cache = OrderedDict()     # 스레드 내 재활용 목적 TODO: global thread safe
        self.prepare_index(profiles)
        
    def prepare_index(self, profiles):
        index_to_query = []
        meta_path = os.path.join(self.npz_root_path, "{}-meta.json".format(self.tag))
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                dataset_meta = json.load(f)
                self.index_to_query = dataset_meta['index_to_query']
                return
            
        for pid in profiles:
            if pid in missing_list:
                continue
                
            path = os.path.join(self.npz_root_path, "profile-recode-{:05d}.npz".format(int(pid)))
            npz = np.load(path, mmap_mode='r', allow_pickle=True)
            pid = int(npz['summary'].tolist()['profile_id'])
            metas = npz['metas']
            summary = npz['summary'].tolist()
            
            device = summary['device'].lower()
            if device not in loc_meta:
                print(">>>>>> NO %s" % summary['device'])
                continue
            
            indexes = []
            for idx in range(len(metas)):
                m = metas[idx]
                if m['left_eye_valid'] and m['right_eye_valid']:
                    indexes.append((pid, idx))
                
            index_to_query += indexes 
       
        data = {'index_to_query': index_to_query}
        with open(meta_path, 'w') as out:
            json.dump(data, out)
            
        self.index_to_query = index_to_query

    def __len__(self):
        return len(self.index_to_query)
    
    def npz_to_cache_obj(self, npz, profile_id, seq_idx):
        cache = {}
        
        # profile based
        metas = npz['metas']
        frames = npz['frames']
        summary = npz['summary'].tolist()
        cache['summary'] = summary
        device = summary['device'].lower()
        loc = np.array(loc_meta[device])
        num_items = len(npz['norm_face'])
        
        # each item
        for i in range(num_items):
            meta = metas[i]
            frame = frames[i]
            
            if not meta['left_eye_valid'] or not meta['right_eye_valid']:
                continue
            
            face_rect = meta['face_rect']
            le_rect = meta['left_eye_rect']
            re_rect = meta['right_eye_rect']
            orientation = meta['orientation']
            cam_mat = meta['origin_camera_param']
            distortion_param = meta['camera_distortion']        
        
            frame = byte_arr_to_img(frame) 
            frame = self.undistort(frame, cam_mat, distortion_param)
            ec = get_eye_corner(face_rect, le_rect, re_rect, frame.shape[:2])
            le_img, re_img = get_eye_left_right_patch(frame, face_rect, le_rect, re_rect)
        
            """flip left eye """ 
            le_img = np.fliplr(le_img)
        
            try:
                le_img = self.preprocess_image(le_img, (64, 64))
                re_img = self.preprocess_image(re_img, (64, 64))
            except:
                print("pid:", profile_id, i, seq_idx, le_img.shape, re_img.shape)
            
            target_xy = np.array([meta['target_dist']['x'], meta['target_dist']['y']]) 
            
            cache[i] = { 
                'loc' : loc,
                'seq_idx' : seq_idx,
                'pid' : profile_id,
                'device' : device,
                'eye_corner' : ec,
                'left_eye' : le_img,
                'right_eye' : re_img,
                'target_xy' : target_xy,
                'cam_mat' : cam_mat,
                'orientation': orientation
            }
            
        return cache
        

    def load_recode_profile_npz(self, profile_id, seq_idx):
        base_dir_path = self.npz_root_path
        npz_name = "profile-recode-{:05d}.npz".format(int(profile_id))
        
        if npz_name in self.npz_cache:
            cache = self.npz_cache[profile_id]
            return cache
        
        if len(self.npz_cache) > self.num_cache:
            num_items = len(self.npz_cache.keys())
            for key, cache in list(self.npz_cache.items())[:-int(num_items/2)]:
                del cache
                
        p = os.path.join(base_dir_path, npz_name)
        npz = np.load(p, mmap_mode='r', allow_pickle=True)   
        cache = self.npz_to_cache_obj(npz, profile_id, seq_idx)
        self.npz_cache[profile_id] = cache
        return cache
         
    
    """
         Get Item
    """
    def getitem(self, idx, selector_fn):
        return selector_fn(self, idx)
    
    def retrieve_item(self, idx):
        profile_id, item_idx = self.index_to_query[idx]
        profile_idx = self.profiles.index("{:05d}".format(profile_id))
        
        cache = self.load_recode_profile_npz(profile_id, idx)
        cache_entry = cache[item_idx]
        return self.preprocess_entry(cache_entry)
 
    def __getitem__(self, idx):
        if self.item_selector_fn is not None:
            return self.item_selector_fn(self, idx)
        return self.retrieve_item(idx)
    
    def preprocess_image(self, image, resize_wh):
        image = cv.resize(image, resize_wh, interpolation=cv.INTER_AREA)
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])
        image = 2.0 * image / 255.0 - 1
        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            if 'target' in key:    # ALG - 236
                pass
            elif isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.int16, requires_grad=False)
        return entry
 
"""
    Free function (TODO: 실수 있는지 체크)
    'right_eye_rect': {'h': 103.10, 'w': 103.10, 'x': 32.64, 'y': 89.34}
    'screen_hw': {'h': 568, 'w': 320},
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
    return np.array([ltop, lbottom, rtop, rbottom, lleft, lright, rleft, rright])

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
 NPZ Dataset Generator Builder
"""
class NPZDatasetGenerator():
    
    def __init__(self, ctx, item_selector_fn=None, shuffle_train=True, pin_memory=False):
        self.config = ctx 
        self.all_data = None
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.npz_root_path = ctx.npz_root_path
        self.resource_path = ctx.resource_path
        self.item_selector_fn = item_selector_fn
 
    def load_spilt_info(self):
        with open(os.path.join(self.resource_path, 'gazecapture_split.json'), 'r') as f:
            split_info = json.load(f)
        return split_info
    
    def prepare_train(self, data_bag, split_info):
        profiles = split_info['train']
        dataset = NPZDataset(self.npz_root_path, 
                             'train',
                             profiles=profiles,
                             item_selector_fn=self.item_selector_fn)
        
        loader = DataLoader(dataset, 
                            drop_last=True, 
                            shuffle=self.shuffle_train, 
                            batch_size=self.config.batch_size,
                            pin_memory=self.pin_memory,
                            num_workers=self.config.num_data_loaders) 
                            # worker_init_fn=worker_init_fn)
        data_bag['gc/train'] = { 'dataset': dataset, 'dataloader': loader }
        
    def prepare_valid(self, data_bag, split_info):
        tag_to_profiles = [ 
            ('gc/val',  self.npz_root_path, split_info['val']), 
            ('gc/test', self.npz_root_path, split_info['test']),
        ]
        
        for tag, dataset_path, profiles in tag_to_profiles:
            dataset = NPZDataset(dataset_path, 
                                 tag[3:],
                                 profiles=profiles, 
                                 item_selector_fn=self.item_selector_fn)
            
            loader = DataLoader(dataset, 
                                num_workers=2, 
                                shuffle=False,
                                pin_memory=False, 
                                batch_size=self.config.batch_size, 
                                worker_init_fn=worker_init_fn)
            data_bag[tag] = { 'dataset': dataset, 'dataloader': loader }
        
    def generate(self, verbose=True):
        all_data = OrderedDict()
        split_info = self.load_spilt_info()
        self.prepare_train(all_data, split_info)
        self.prepare_valid(all_data, split_info)
        if verbose:
            self.print_stats(all_data)
            
        return all_data
    
    def print_stats(self, data):
        logging.info('')
        logging.info('>>> Data-Generator prepared <<<')
        for tag, val in data.items():
            tag = '[%s]' % tag
            dataset = val['dataset']
            origin_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
            num_origin_items = len(origin_ds)
            num_people = len(origin_ds.profiles)
            logging.info('%10s full set size:           %7d' % (tag, num_origin_items))
            logging.info('%10s current set size:        %7d' % (tag, len(dataset)))
            logging.info('%10s num people:              %7d' % (tag, num_people))
            logging.info('%10s mean entries per person: %7d' % (tag, num_origin_items / num_people))
            logging.info('---' * 12)
   
    
