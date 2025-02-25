"""
 Faze preprocess
"""
import os
import sys
import time
import h5py
import cv2 as cv
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt


"""
 Normalize Sugar API
"""
def run_concurrent(out_dir_path, input_base_path, supplementary_path):
    num_profiles = 1366
    
    def worker(worker_idx, num_workers):
        supple_path = supplementary_path.format(worker_idx)
        fn = FazeNormalizer(out_dir_path, input_base_path, supple_path)
        num_batch = int(num_profiles / num_workers)
        start = worker_idx * num_batch
        last = start + num_batch - 1
        print(">>> worker[{}] ({} ~ {})".format(worker_idx, start, last))
        fn.run_preprocess(idx_from=start, idx_to=last)
        
    num_workers = 8
    jobs = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(i,num_workers))
        jobs.append(p)
        p.start()

    time.sleep(5)
    for j in jobs:
        print(">>> waiting {}".format(j))
        j.join()


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
 FazeBaseNormalizer
"""
class FazeNormalizer:
    
    def __init__(self, out_dir_path, input_base_path, supplementary_path, face_model_path):
        self.out_dir_path = out_dir_path
        self.input_base_path = input_base_path
        
        self.num_profiles = 0
        self.undistort = Undistorter()
        self.ctx = self.prepare_context(face_model_path)
        self.load_supplementary(supplementary_path)

    def prepare_context(self, face_model_path): 
        norm_cam_info = {
            'distance': 600, 'focal_length': 1300,  'size': (256, 64)
        } 

        fx = norm_cam_info['focal_length']
        fy = norm_cam_info['focal_length'] 
        cx = 0.5 * norm_cam_info['size'][0]
        cy = 0.5 * norm_cam_info['size'][1]

        global_context = {
            'face_3d' : self.load_3d_face_model(face_model_path), 
            'norm_cam_info' : norm_cam_info, 
            'norm_cam_mat' : self.make_camera_param(fx, fy, cx, cy)
        }

        return global_context
            
    def load_supplementary(self, supplementary_path):
        self.supplementary = h5py.File(supplementary_path, 'r')
        self.num_profiles = len(list(self.supplementary.items()))
            
    def load_3d_face_model(self, model_path='./resources/sfm_face_coordinates.npy'):
        face_model_3d_coordinates = np.load(model_path)
        if face_model_3d_coordinates is None:
            print(">>> failed to load sfm face")
            sys.exit()
        return face_model_3d_coordinates
    
    def make_camera_param(self, fx, fy, cx, cy):
        return np.array([[fx, 0, cx], [0, fy, cy], [0,  0,  1]], dtype=np.float64) 
    
    def __len__(self):
        return self.num_profiles
    
    def __del__(self):
        # self.supplementary.close()
        return
        

    
    """
        API
    """
    def run_preprocess(self, idx_from=0, idx_to=None):
        if idx_to is None or idx_to > len(self): 
            idx_to = len(self) - 1
            
        person_to_record = list(self.supplementary.items())
        for idx in range(idx_from, idx_to):
            person_id, record = person_to_record[idx]
            self.process_profile(person_id, record, idx)
            
    def process_profile(self, person_id, record, idx):
        print(">>> [{}] Process profile {}\n".format(idx, person_id))
        npz = self.load_profile_npz(self.input_base_path, person_id)
        to_write = self.data_normalization(npz, record)
        self.write(to_write, npz, self.out_dir_path)
    
    """
        Normalization Process
    """
    def data_normalization(self, npz, profile_record):
        to_write = {}
        def add(key, value):
            if key not in to_write:
                to_write[key] = [value]
            else:
                to_write[key].append(value)

        num_invalid = 0
        num_entries = next(iter(profile_record.values())).shape[0]

        for i in range(num_entries):
            processed_entry = self.data_normalization_entry(npz, profile_record, i)        
            if processed_entry is None:
                num_invalid = num_invalid + 1
                continue

            # Gather all of the person's data
            add('pixels',              processed_entry['patch'])    # undistorted face normalized image
            add('norm_gaze',           processed_entry['normalized_gaze_direction'])
            add('norm_head_pose',      processed_entry['normalized_head_pose'])
            add('norm_rot_matrix',     processed_entry['normalization_matrix'])
            add('gaze_direction',      processed_entry['gaze_direction'])
            add('gaze_origin',         processed_entry['gaze_origin'])
            add('gaze_target',         processed_entry['gaze_target'])
            add('head_pose',           processed_entry['head_pose'])
            add('origin_extrinsic',    processed_entry['origin_extrinsic'])
            add('origin_camera_param', processed_entry['origin_camera_parameter'])
            add('camera_distortion',   processed_entry['camera_distortion'])

        if len(to_write) == 0:
            return

        # to numpy
        for key, values in to_write.items():
            to_write[key] = np.asarray(values)

        return to_write

    def data_normalization_entry(self, npz, group, i):
        # print(">>> In profile [{}]".format(i))
        
        # Form original camera matrix
        fx, fy, cx, cy = group['camera_parameters'][i, :]
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        image = self.get_frame_from_npz(npz, i)
        if image is None:
            return

        image = self.undistort(image, camera_matrix, group['distortion_parameters'][i, :])
        image = image[:, :, ::-1]  # BGR to RGB

        # Calculate rotation matrix and euler angles
        rvec = group['head_pose'][i, :3].reshape(3, 1)
        tvec = group['head_pose'][i, 3:].reshape(3, 1)
        rotate_mat, _ = cv.Rodrigues(rvec)

        face_model_3d_coordinates = self.ctx['face_3d']

        # Take mean face model landmarks and get transformed 3D positions
        landmarks_3d = np.matmul(rotate_mat, face_model_3d_coordinates.T).T
        landmarks_3d += tvec.T

        """
        - Gaze-origin (g_o) ==> between 2 eyes (동공 랜드마크가 아닌 양안 중간)
        - Gaze-target (g_t) ==> 3d gaze target (from supplementary)
        """
        g_o = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
        g_o = g_o.reshape(3, 1)
        g_t = group['3d_gaze_target'][i, :].reshape(3, 1)
        g = g_t - g_o
        g /= np.linalg.norm(g)

        # Code below is an adaptation of code by Xucong Zhang
        # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

        # actual distance between gaze origin and original camera
        distance = np.linalg.norm(g_o)
        z_scale = self.ctx['norm_cam_info']['distance'] / distance
        S = np.eye(3, dtype=np.float64)
        S[2, 2] = z_scale

        hRx = rotate_mat[:, 0]
        forward = (g_o / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R

        # transformation matrix
        W = np.dot(np.dot(self.ctx['norm_cam_mat'], S), np.dot(R, np.linalg.inv(camera_matrix)))

        ow, oh = self.ctx['norm_cam_info']['size']
        patch = cv.warpPerspective(image, W, (ow, oh))  # image normalization

        # Correct head pose
        R = np.asmatrix(R)
        h = np.array([np.arcsin(rotate_mat[1, 2]), np.arctan2(rotate_mat[0, 2], rotate_mat[2, 2])])
        head_mat = R * rotate_mat
        n_h = np.array([np.arcsin(head_mat[1, 2]), 
                        np.arctan2(head_mat[0, 2], head_mat[2, 2])
                       ])

        # Correct gaze
        n_g = R * g
        n_g /= np.linalg.norm(n_g)
        n_g = self.vector_to_pitchyaw(-n_g.T).flatten()

        # if i % 50 == 0:
        #    self.visualize()

        return {
            'patch': patch.astype(np.uint8),
            'gaze_direction': g.astype(np.float32),
            'gaze_origin': g_o.astype(np.float32),
            'gaze_target': g_t.astype(np.float32),
            'head_pose': h.astype(np.float32),
            'normalization_matrix': np.transpose(R).astype(np.float32),
            'normalized_gaze_direction': n_g.astype(np.float32),
            'normalized_head_pose': n_h.astype(np.float32),
            'origin_camera_parameter': camera_matrix.astype(np.float32),
            'camera_distortion': group['distortion_parameters'][i, :].astype(np.float32),
            'origin_extrinsic':group['head_pose'][i,:].astype(np.float32)
        }
    
    
    """
        OPs
    """
    def vector_to_pitchyaw(self, vectors):
        """ Convert given gaze vectors to yaw (theta) and pitch (phi) angles """
        n = vectors.shape[0]
        out = np.empty((n, 2))
        vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
        out[:, 0] = np.arcsin(vectors[:, 1])                  # theta
        out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
        return out
   
    
    """
        Output writer
    """
    def write(self, to_write, origin_npz, out_dir, write_mode='npz'):
        if write_mode is 'npz':
            self.write_as_npz(to_write, origin_npz, out_dir)        

        if write_mode is "hdf":
            self.write_as_hdf(to_write)

    def write_as_hdf(self, to_write):
        with h5py.File(output_path, 'a' if os.path.isfile(output_path) else 'w') as f:
            if person_id in f:
                del f[person_id]
            group = f.create_group(person_id)
            for key, values in to_write.items():
                chunks = (tuple([1] + list(values.shape[1:])) if isinstance(values, np.ndarray) else None)
                group.create_dataset(key, data=values, chunks=chunks, compression='lzf')    
                
    def write_as_npz(self, to_write, origin_npz, out_dir_path):      
        metas = origin_npz['meta']
        frames = origin_npz['frames']
        summary = origin_npz['summary'].tolist()
        profile_id = summary['profile_id']

        norm_faces = []
        norm_metas = []

        num_item = to_write['pixels'].shape[0]
        for i in range(num_item):
            item = self.get_ith_data(to_write, i)

            # encode jpeg image
            face = item['pixels']
            jpeg = raw_img_to_jpeg(face)
            norm_faces.append(jpeg)

            # additional meta
            m = metas[i]
            m['norm_gaze_pitchyaw']= item['norm_gaze']   # gaze vectors to yaw (theta) and pitch (phi) angles
            m['norm_head_pose']    = item['norm_head_pose']
            m['norm_rot_matrix']   = item['norm_rot_matrix'] 
            m['gaze_direction']    = item['gaze_direction'] 
            m['gaze_origin']       = item['gaze_origin'] 
            m['gaze_target']       = item['gaze_target'] 
            m['head_pose']         = item['head_pose']
            m['origin_extrinsic']    = item['origin_extrinsic'] 
            m['origin_camera_param'] = item['origin_camera_param'] 
            m['camera_distortion']   = item['camera_distortion']
            norm_metas.append(m)

        n_faces = np.asarray(norm_faces).transpose()

        out_path = os.path.join(out_dir_path, "profile-recode-{}".format(profile_id))
        np.savez_compressed(out_path, summary=summary, metas=norm_metas, norm_face=n_faces, frames=frames)
   
    
    """
        Accessors
    """
    def load_profile_npz(self, input_root_path, profile_id):
        p = os.path.join(input_root_path, "profile-{:05d}.npz".format(int(profile_id)))
        return np.load(p, allow_pickle=True) 

    def get_frame_from_npz(self, npz, idx):
        meta = npz['meta'][idx]
        file_idx = int(meta['frame_name'][:-4])

        if idx != file_idx:
            print(">>> file index not matched. idx:{} summary:{}".format(idx, meta))
            sys.exit()

        img = npz['frames'][file_idx]
        return byte_arr_to_img(img)

    def get_ith_data(self, to_write, i):
        item = {}
        for k in to_write.keys():
            item[k] = to_write[k][i]
        return item
    
    
    """
        Visualizer & Sanity Checker
    """    
    def visualize(self):
        to_visualize = cv.equalizeHist(cv.cvtColor(patch, cv.COLOR_RGB2GRAY))
        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.25 * oh), n_g, length=80.0, thickness=1)
        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.75 * oh), n_h, length=40.0, thickness=3, color=(0, 0, 0))
        to_visualize = draw_gaze(to_visualize, (0.5 * ow, 0.75 * oh), n_h, length=40.0, thickness=1, color=(255, 255, 255))
        cv.imshow('normalized_patch', to_visualize)
        cv.waitKey(1)

    def show_image(self, img, cmap=None, extent=None, title=None, fig_size=(10, 10)):
        if cmap is None:
            cmap = select_colormap_by_shape(img.shape)
        if extent is None:
            extent = get_image_extent(img)

        img = squeeze_if_gray(img)
        fig = plt.figure(figsize=fig_size)

        plt.axis('off')
        if title is not None:
            plt.title(title)
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap=cmap, interpolation='none', extent=extent)
        plt.axis('image')

        return ax

    def sanity_check_npz_recode(self, profile_id, idx, root_path="/Volumes/SSD3/everyone-faze/"):
        npz = np.load(os.path.join(root_path, "profile-recode-{}.npz".format(profile_id)), allow_pickle=True)
        print(npz.files)

        face = npz['norm_face'][idx]
        img = byte_arr_to_img(face)
        print(img.shape)
        show_image(img)
        
    def draw_gaze(self, image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
        """Draw gaze angle on given image with a given eye positions."""
        image_out = image_in
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[0])
        cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                       tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color, thickness, cv.LINE_AA, tipLength=0.2)
        return image_out

    def show_supplementary(self):
        with h5py.File('./resources/GazeCapture_supplementary.h5', 'r') as f:
            person, group = None, None

            count = 0
            for person_id, g in f.items():
                person, group = person_id, g
                count = count + 1
                if count == 4:
                    break

            num_entries = next(iter(group.values())).shape[0]
            print("profile-{} num: {}".format(person, num_entries))

            for i in range(0, 5): # range(num_entries):
                fx, fy, cx, cy = group['camera_parameters'][i, :]
                image_path = '%s' % (group['file_name'][i].decode('utf-8'))
                rvec = group['head_pose'][i, :3].reshape(3, 1)
                tvec = group['head_pose'][i, 3:].reshape(3, 1)
                distor = group['distortion_parameters'][i, :]
                gaze_t = group['3d_gaze_target'][i, :].reshape(3, 1)

                # print(rvec) # print(tvec) # print(distor)
                print(image_path)
                print(fx, fy, cx, cy)
                print(gaze_t)
                print(rvec)
                print("")

    
"""
 Free Function
""" 
def select_colormap_by_shape(shape):
    if len(shape) is 3 and shape[2] == 1:
        return 'gray'
    elif len(shape) is 2:
        return 'gray'
    return 'viridis'

def get_image_extent(img):
    h, w = shape_to_hw(img.shape)
    return 0, h, 0, w

def shape_to_hw(shape):
    return shape[1], shape[0]

def squeeze_if_gray(img):
    if len(img.shape) == 3 and img.shape[2] == 1:
        return np.squeeze(img)
    return img

def byte_arr_to_img(byte_arr):
    dt = np.dtype(np.int8)  # 255
    dt = dt.newbyteorder('>')  # depend on mach arch
    np_arr = np.frombuffer(byte_arr, dt)
    return cv.imdecode(np_arr, 1)

def raw_img_to_jpeg(frame):
    _, frame = cv.imencode('.JPEG', frame)
    return frame


        