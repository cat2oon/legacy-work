import os
import cv2
import numpy as np

from bunch import Bunch
from scipy.io import loadmat

"""
<<< 3D Morphable Model >>>
- gravis.dmi.unibas.ch/publications/Sigg99/morphmod2.pdf (color, illum)
- 다른 프로젝트에서 py로 포팅한 코드 찾음 (github.com/reshow/PRNet-PyTorch/data.py)
- 연산량을 위해 np.float32

BFM model: 
    nver = 53215, ntri = 105840
    nver: number of vertices  (num_vertex)
    ntri: number of triangles (num_triangles)

    shapeMU  : [3*nver, 1]
    shapePC  : [3*nver, 199]
    shapeEV  : [199, 1]
    expMU'   : [3*nver, 1]
    expPC'   : [3*nver, 29]
    expEV'   : [29, 1]
    texMU'   : [3*nver, 1]
    texPC'   : [3*nver, 199]
    texEV'   : [199, 1]
    tri      : [ntri, 3] (start from 1)
    kpt_ind  : [68,]     (start from 1, facial key point index)
    trim_index: BFM 메쉬 모델에서 실제 사용된 정점들에 대한 인덱스 (start from 1)

PAPER:
    S_mu    : mu_shape + mu_expression
    A_shape : shape principle basis
    A_exp   : expression principle basis
    S = S_mu + (A_shape * α_shape) + (A_exp * α_exp) (3DMM)
    
R:
    x: pitch. positive for looking down 
    y: yaw. positive for looking left
    z: roll. positive for tilting head right
    
PORTING:    
    WARN: reshape 순서가 numpy와 다름 np.reshape(order='F')가 matlab 방식 
    NOTE: NP는 broadcasting 지원하므로 repmat은 가급적 브로드캐스팅으로 대체
"""


class MorphableModel:
    
    def __init__(self, model_dir_path):
        self.model_dir_path = model_dir_path
        self.load_model_mat()
        self.template = self.generate_template()  
    
    def load_model_mat(self):
        self.model_exp = loadmat(self.model_exp_path)
        self.model_info = loadmat(self.model_info_path)
        self.model_morphable = loadmat(self.model_morphable_path)

    def generate_template(self):
        mm, mi, me = self.model_morphable, self.model_info, self.model_exp
        
        trim_index = mi['trimIndex'].astype(np.int32)    # (53215,  1)
        shape_mu = mm['shapeMU'].astype(np.float32)      # (160470, 1)
        shape_pc = mm['shapePC'].astype(np.float32)      # (160470, 199)
        shape_ev = mm['shapeEV'].astype(np.float32)      # (199, 1)
        texture_mu = mm['texMU'].astype(np.float32)      # (160470, 1)
        texture_pc = mm['texPC'].astype(np.float32)      # (160470, 199)
        expression_mu = me['mu_exp'].astype(np.float32)  # (159645, 1)     # 실제 사용된 점
        expression_w  = me['w_exp'].astype(np.float32)   # (159645, 29)    # 실제 사용된 점
        
        trim_index_3d = np.array([3*trim_index-3, 3*trim_index-2, 3*trim_index-1])  # (3, 53215, 1)
        index_flatten = trim_index_3d.T.flatten()                                   # (159645,)

        # 실제 사용된 정점들만 선택
        mu_exp = expression_mu                   # (159645, 1)
        mu_shape = shape_mu[index_flatten]       # (159645, 1)
        mu_texture = texture_mu[index_flatten]   # (159645, 1)
        
        sigma_shape = shape_ev                   # (199, 1)
        w_shape = shape_pc[index_flatten]        # (159645, 199) 
        w_texture = texture_pc[index_flatten]    # (159645, 199)
        w_exp = expression_w                     # (159645, 29)
        
        tri_index = (mi['tri'] - 1).T    # (105840, 3)   # zero-base
        keypoints = mi['keypoints']      # (1, 68) landmark template
        
        t = Bunch()
        t.mu_face = mu_shape + mu_exp
        t.w_exp = w_exp
        t.w_shape = w_shape
        t.mu_texture = mu_texture
        t.w_texture = w_texture
        t.tri_index = tri_index
        t.keypoints = keypoints
        
        return t


    
    """
        Parse Param
    """
    def parse_pose_param(self, p): 
        # [[phi(pitch), gamma(yaw), theta(roll), tx, ty, tz, scale]]
        pose_param = p.pose_param[0]  
        p.t_vec = pose_param[3:6]
        p.scale = pose_param[6]
        
        pitch, yaw, roll = pose_param[0:3]
        p.pitch, p.yaw, p.roll = (pitch, yaw, roll)
        p.R = self.to_rotation_matrix(pitch, yaw, roll)
        p.proj_2d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
    def parse_color_param(self, p):
        cp = p.color_param[0]
        gain_r, gain_g, gain_b = cp[:3]
        offset_r, offset_g, offset_b = cp[3:6]
        offsets = np.array([offset_r, offset_g, offset_b])
        
        p.color_c = cp[6]  # don't know what it is even though read ref paper (but usually 1.0)
        p.color_M = np.array([[0.3, 0.59, 0.11],[0.3, 0.59, 0.11], [0.3, 0.59, 0.11]])
        p.color_gain = np.diag(np.array([gain_r, gain_g, gain_b]))
        
        """ TODO: 활용처에서 브로드캐스팅 대체 여부 확인 """
        # num_mesh = p.face_vertex.shape[1]  # (3, 53215) -> 53215
        p.color_offset = offsets   # p.offset_vec = np.repeat(offsets, num_mesh)   
        
    def parse_illumination_param(self, p):
        ip = p.illum_param[0]
        amb_r, amb_g, amb_b = ip[:3]
        dir_r, dir_g, dir_b = ip[3:6]
        theta_l, phi_l = ip[6:8]
        ks, v = ip[8], ip[9]
        l = np.array([np.cos(theta_l) * np.sin(phi_l), 
                      np.sin(theta_l), 
                      np.cos(theta_l) * np.cos(phi_l)]); 
        h = l + np.array([0, 0, 1])
        h = h / np.linalg.norm(h)
        
        p.illum_amb = np.diag(np.array([amb_r, amb_g, amb_b]))
        p.illum_dir = np.diag(np.array([dir_r, dir_g, dir_b])) # direction of illumination
        p.illum_light = l          # 광원 벡터 (approx)
        p.illum_highlight = h      # 정반사에서 카메라로 들어오는 및 (approx)
        p.illum_shininess = ks     # surface shininess  (가끔 0인 경우는?)
        p.illum_v = v     # meaning what? 

    def to_face_params(self, item_mat):
        m = item_mat
        p = Bunch()
        p.roi = m['roi']               # crop region on the original image
        p.lmk_2d = m['pt2d']           # 2D 랜드마크 68점
        
        p.texture_param = m['Tex_Para']   # 텍스쳐
        p.illum_param = m['Illum_Para']   # 조명
        p.shape_param = m['Shape_Para']   # 3DMM shape param
        p.exp_param = m['Exp_Para']       # 3DMM expression param
        p.pose_param = m['Pose_Para']     # head pose
        p.color_param = m['Color_Para']   # 색상
        
        return p
           
        
    
    """
        Handle Item
    """
    def generate_vertices(self, param, template):
        p, t = param, template
        vertices = t.mu_face + (t.w_shape @ p.shape_param) + (t.w_exp @ p.exp_param)
        vertices = vertices.reshape(3, -1, order='F').T
        
        return vertices

    def generate_texture(self, param, template):
        p, t = param, template
        colors = t.mu_texture + (t.w_texture @ p.texture_param)
        colors = colors.reshape(3, -1, order='F').T / 255.0
        
        return colors
        
    def process_item(self, item_id):
        img, mat = self.item(item_id)
        
        p = self.to_face_params(mat)
        self.parse_pose_param(p)
        self.parse_color_param(p)
        self.parse_illumination_param(p)
        
        t = self.template
        texture = self.generate_texture(p, t)
        vertices = self.generate_vertices(p, t)
        norm_dirs = self.calc_norm_direction(vertices, t.tri_index) 
        
        trans_vertices = self.transform_vertex(vertices, p) 
        texture_color = self.calc_texture_color(texture, norm_dirs, p)
        
        return p, vertices, texture_color, norm_dirs

    def item(self, item_id):
        img_path = os.path.join(self.model_dir_path, "{:05d}.jpg".format(item_id))
        mat_path = os.path.join(self.model_dir_path, "{:05d}.mat".format(item_id))
        return cv2.imread(img_path), loadmat(mat_path)   

    
    
    """
        Ops
    """
    def transform_vertex(self, vertices, p):
        return (p.scale * vertices @ p.R) + p.t_vec
         
    def calc_texture_color(self, texture, norm, p):
        """ phong 조명 모델 수식 
        - sigg99 논문의 조명 모델은 phong 모델의 근사 모델인 듯 수식에 차이가 있음
        
        PORTING:
            matlab: n_l = max(l' * norm,0) 
            numpy : array.clip(min=0)
        """
        i_amb, i_dir =  p.illum_amb, p.illum_dir
        l, h = p.illum_light, p.illum_highlight
        ks, v = p.illum_shininess, p.illum_v
        
        n_l = norm.dot(l).clip(min=0)  # (53215, )
        n_l = np.tile(n_l, (3, 1))     # (3, 53215)
        n_h = norm.dot(h).clip(min=0)  # (53215, )
        n_h = np.tile(n_h, (3, 1))     # (3, 53215)
        
        ambient = i_amb.dot(texture.T)        # 간접 조명 
        diffuse = i_dir.dot(n_l * texture.T)  # 난반사   
        specular = (ks * i_dir).dot(n_h ** v) # 정반사   
        light = ambient + diffuse + specular  # phong 조명값
        
        offset = p.color_offset
        g, c, M = p.color_gain, p.color_c, p.color_M
        CT = g.dot(c * np.eye(3) + (1 - c) * M)
        texture_color = CT.dot(light).T + offset.T 
        texture_color = texture_color.clip(min=0, max=1)
        
        p.illum_light = l          # 광원 벡터 (approx)
        p.illum_highlight = h      # 정반사에서 카메라로 들어오는 및 (approx)
        p.illum_shininess = ks     # surface shininess  (가끔 0인 경우는?)
        p.illum_v = v     # meaning what? 

        return texture_color
    
    def to_rotation_matrix(self, pitch, yaw, roll):
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
    
    def calc_norm_direction(self, vertex, tri):
        """ matlab 원본 코드는 아래 함수가 binary 파일로 되어 있어서
            github.com/reshow/PRNet-PyTorch에서 코드 가져옴 
        """
        def Tnorm_VnormC(norm_t, tri, n_tri, n_ver):
            norm_v = np.zeros((n_ver, 3))
            for i in range(n_tri):
                pt = tri[i]
                for j in range(3):
                    for k in range(3):
                        norm_v[pt[j]][k] += norm_t[i][k]
            return norm_v

        pt1 = vertex[tri[:, 0], :]
        pt2 = vertex[tri[:, 1], :]
        pt3 = vertex[tri[:, 2], :]
        n_tri = np.cross(pt1 - pt2, pt1 - pt3)

        N = Tnorm_VnormC(n_tri, tri, tri.shape[0], vertex.shape[0])
        mag = np.sum(N * N, axis=1)
        co = np.nonzero(mag == 0)
        mag[co] = 1
        N[co, 0] = np.ones((len(co)))
        mag2 = np.tile(mag, (3, 1)).T
        N = N / np.sqrt(mag2)
        N = -N
        
        return N
    
    
    
    """
        Property
    """
    @property
    def model_exp_path(self):
        return os.path.join(self.model_dir_path, "model-exp.mat")
    
    @property
    def model_info_path(self):
        return os.path.join(self.model_dir_path, "model-info.mat")
    
    @property
    def model_shape_path(self):
        return os.path.join(self.model_dir_path, "model-shape.mat")
    
    @property
    def model_morphable_path(self):
        return os.path.join(self.model_dir_path, "morphable-model.mat")
        
    