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
	"00117",  "00124",  "00133",  "00198", "00207", "00229", "00251", "00252", "00258", "00266",
	"00322", "00383", "00463", "00500", "00521", "00549", "00595", "00597", "00653", "00696",
	"00740", "00748", "00779", "00808", "00828", "00861", "00876", "00880", "00890", "00939",
        "00930", "00932", "00955",
	"00960", "00976", "00998", "01001", "01029", "01030", "01066", "01099", "01109", "01122", "01126",
	"01134", "01185", "01206", "01224", "01225", "01267", "01282", "01350", "01366", "01367",
	"01372", "01392", "01432", "01443", "01474", "01544", "01556", "01661", "01676", "01702",
	"01805", "01809", "01819", "01859", "01876", "01896", "01939", "02002", "02027", "02032", "02033",
	"02048", "02117", "02119", "02155", "02165", "02174", "02190", "02194", "02223", "02243", "02353",
	"02364", "02417", "02456", "02526", "02533", "02542", "02551", "02622", "02739", "02840", "02976",
	"02984", "03007", "03039", "03059", "03060", "03212", "03224", "03239", "03380", "03389", "03474"
    ]


"""
 File Systems
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

    
"""
 NPZ Dataset Free function
"""
def get_num_items_in_profile(npz):
    return npz['norm_face'].shape[0]

def byte_arr_to_img(byte_arr):
    dt = np.dtype(np.int8)  # 255
    dt = dt.newbyteorder('>')  # depend on mach arch
    np_arr = np.frombuffer(byte_arr, dt)
    return cv.imdecode(np_arr, 1)

 # Functions to calculate relative rotation matrices for gaze dir. and head pose
def R_x(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    return np.array([
        [1., 0., 0.],
        [0., cos_, -sin_],
        [0., sin_, cos_]
    ]). astype(np.float32)

def R_y(phi):
    sin_ = np.sin(phi)
    cos_ = np.cos(phi)
    return np.array([
        [cos_, 0., sin_],
        [0., 1., 0.],
        [-sin_, 0., cos_]
    ]). astype(np.float32)

def calculate_rotation_matrix(e):
    return np.matmul(R_y(e[1]), R_x(e[0]))

def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = -np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out
 
    
"""
 NPZ Dataset Generator Builder
"""
class NPZDatasetGenerator():
    
    def __init__(self, ctx, shuffle_train=True, pin_memory=False):
        self.config = ctx 
        self.all_data = None
        self.npz_root_path = ctx.npz_root_path
        self.resource_path = ctx.resource_path
        
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
 
    def load_spilt_info(self):
        with open(os.path.join(self.resource_path, 'gazecapture_split.json'), 'r') as f:
            split_info = json.load(f)
        return split_info
    
    def prepare_train(self, data_bag, split_info):
        profiles = split_info['train']
        dataset = NPZDataset(self.npz_root_path, 
                             'train',
                             profiles=profiles, 
                             get_2nd_sample=True)
        
        loader = DataLoader(dataset, 
                            drop_last=True, 
                            shuffle=self.shuffle_train, 
                            batch_size=self.config.batch_size,
                            pin_memory=self.pin_memory,    # PIN_MEMORY 시스템 메모리 에러
                            num_workers=self.config.num_data_loaders) 
                            # worker_init_fn=worker_init_fn)
        data_bag['gc/train'] = { 'dataset': dataset, 'dataloader': loader }
        
    def prepare_valid(self, data_bag, split_info):
        tag_to_profiles = [ 
            ('gc/val',  self.npz_root_path, split_info['val']), 
            ('gc/test', self.npz_root_path, split_info['test']),
            # ('mpi', .mpiigaze_file, None),
        ]
        
        for tag, dataset_path, profiles in tag_to_profiles:
            dataset = NPZDataset(dataset_path, 
                                 tag[3:],
                                 profiles=profiles, 
                                 get_2nd_sample=True)
            
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
        logging.info('\n>>> Data-Generator prepared <<<')
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
            logging.info('')

   
    
"""
 NPZ Dataset
"""
class NPZDataset(Dataset):
    
    def __init__(self, npz_root_path, data_tag, profiles=None, get_2nd_sample=False):
        assert os.path.isdir(npz_root_path)
        self.tag = data_tag
        self.profiles = profiles 
        self.npz_root_path = npz_root_path
        self.get_2nd_sample = get_2nd_sample
        self.num_cache = 8
        self.npz_cache = OrderedDict()
        
        self.prepare_index(profiles)
        
    def prepare_index(self, profiles):
        """ 여기서 페어 조합 뻥튀기 가능 """
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
            """ WARN: use norm_face not frames"""
            num_items_profile = npz['norm_face'].shape[0]
            indexes = [(pid, i) for i in range(num_items_profile)]
            index_to_query += indexes 
       
        data = {'index_to_query': index_to_query}
        with open(meta_path, 'w') as out:
            json.dump(data, out)
            
        self.index_to_query = index_to_query

    def __len__(self):
        return len(self.index_to_query)

    def preprocess_image(self, image):
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1
        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            """ ALG - 236 """
            if 'target' in key:
                pass
            elif isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.int16, requires_grad=False)

        return entry
    
    def retrieve_input_from_npz(self, npz, idx):
        meta = npz['metas'][idx]
        face_jpeg = npz['norm_face'][idx]
        face_img = byte_arr_to_img(face_jpeg)

        eyes = self.preprocess_image(face_img)
        norm_gaze = meta['norm_gaze_pitchyaw']    
        norm_head = meta['norm_head_pose']
        t_x = meta['target_dist']['x']
        t_y = meta['target_dist']['y']

        return eyes, norm_gaze, norm_head, t_x, t_y

    def load_recode_profile_npz(self, base_dir_path, profile_id):
        npz_name = "profile-recode-{:05d}.npz".format(int(profile_id))
        
        if npz_name in self.npz_cache:
            return self.npz_cache[npz_name]
        
        if len(self.npz_cache) > self.num_cache:
            num_items = len(self.npz_cache.keys())
            for key, npz in list(self.npz_cache.items())[:-int(num_items/2)]:
                del npz

        p = os.path.join(base_dir_path, npz_name)
        npz = np.load(p, mmap_mode='r', allow_pickle=True)   
        self.npz_cache[npz_name] = npz
        
        return npz
         
    
    """
     this method is evaluated at runtime
    """
    def getitem(self, idx, selector_fn):
        pid_a, item_idx_a = self.index_to_query[idx]
        return selector_fn(self, idx)
    
    def __getitem__(self, idx):
        """
        key_a, idx_a ==> profile_id, item_id 하지 왜 이름을 저러케해?
        """
        profile_id, item_idx_a = self.index_to_query[idx]    
        key_idx = self.profiles.index("{:05d}".format(profile_id))
        npz_a = self.load_recode_profile_npz(self.npz_root_path, profile_id)
        npz_b = npz_a

        """
        # 셀프 매칭은 왜 제외하는 건가? Identity를 학습하기 위해 좋은 샘플아닌가?
        # -> 과거에는 프레임 당 딱 한번 매칭 정책이므로 당연한 코드였음
        # 이러면 프레임 당 딱 한 쌍씩 매칭하겠다는 거인데 수량이 괜찮은가?
        """
        num_items_in_profile = get_num_items_in_profile(npz_a)
        all_indices = list(range(num_items_in_profile))
        all_indices_but_a = np.delete(all_indices, item_idx_a)
        item_idx_b = np.random.choice(all_indices_but_a)

        """ ALG """
        self.dbg_pid = profile_id
        self.dbg_idxa = item_idx_a
        self.dbg_idxb = item_idx_b

        # Grab 1st (input) entry
        """ original version
        entry = {
            'key': profile_id, 'key_index': key_idx, 'image_a': eyes_a,
            'gaze_a': norm_gaze_a, 'head_a': norm_head_a,
            'R_gaze_a': calculate_rotation_matrix(norm_gaze_a),
            'R_head_a': calculate_rotation_matrix(norm_head_a),
        }
        """ 
        eyes_a, norm_gaze_a, norm_head_a, t_x_a, t_y_a = self.retrieve_input_from_npz(npz_a, item_idx_a)
        entry = {
            'pid': profile_id, 
            'key_idx': key_idx, 
            'image_a': eyes_a,
            'gaze_a': norm_gaze_a, 
            'head_a': norm_head_a,
            'item_idx_a': item_idx_a, 
            'rot_gaze_a': calculate_rotation_matrix(norm_gaze_a),
            'rot_head_a': calculate_rotation_matrix(norm_head_a),
            'target_x_a' : t_x_a,
            'target_y_a' : t_y_a,
        }

        if not self.get_2nd_sample:
            return self.preprocess_entry(entry)

        eyes_b, norm_gaze_b, norm_head_b, t_x_b, t_y_b = self.retrieve_input_from_npz(npz_b, item_idx_b)
        entry['image_b'] = eyes_b
        entry['gaze_b'] = norm_gaze_b
        entry['head_b'] = norm_head_b
        entry['item_idx_b'] = item_idx_b
        entry['rot_gaze_b'] = calculate_rotation_matrix(entry['gaze_b'])
        entry['rot_head_b'] = calculate_rotation_matrix(entry['head_b'])
        entry['target_x_b'] = t_x_b
        entry['target_y_b'] = t_y_b

        return self.preprocess_entry(entry)
