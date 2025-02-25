import os
import sys
import numpy as np
import scipy.io as sio
from skimage import io
from scipy.io import loadmat
import time
import math
import skimage
from math import sin, cos, asin, acos, atan, atan2




"""
    IMPORT <loader>
    
    model: (nver = 53215, ntri = 105840)
    nver: number of vertices
    ntri: number of triangles.
    
    'shapeMU': [3*nver, 1]
    'shapePC': [3*nver, 199]
    'shapeEV': [199, 1]
    'expMU': [3*nver, 1]
    'expPC': [3*nver, 29]
    'expEV': [29, 1]
    'texMU': [3*nver, 1]
    'texPC': [3*nver, 199]
    'texEV': [199, 1]
    'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
    'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles)
    'kpt_ind': [68,] (start from 1)
"""
def load_bfm(model_path):
    model = loadmat(model_path) 

    model['shapeMU'] = (model['shapeMU']).astype(np.float32)
    model['shapePC'] = model['shapePC'].astype(np.float32)
    model['shapeEV'] = model['shapeEV'].astype(np.float32)
    model['expMU'] = (model['expMU']).astype(np.float32)
    model['expEV'] = model['expEV'].astype(np.float32)
    model['expPC'] = model['expPC'].astype(np.float32)
    model['texMU'] = (model['texMU']).astype(np.float32)
    model['texPC'] = model['texPC'].astype(np.float32)
    model['texEV'] = model['texEV'].astype(np.float32)

    # matlab start with 1. change to 0 in python.
    model['tri'] = model['tri'].T.copy(order = 'C').astype(np.int32) - 1
    model['tri_mouth'] = model['tri_mouth'].T.copy(order = 'C').astype(np.int32) - 1
    
    # kpt ind
    model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

    return model





"""
    IMPORT <morphable_model>
    
    nver: number of vertices 
    ntri: number of triangles
    *: must have
    ~: can generate ones array for place holder
    
    'shapeMU': [3*nver, 1]. *
    'shapePC': [3*nver, n_shape_para]. *
    'shapeEV': [n_shape_para, 1]. ~
    'expMU': [3*nver, 1]. ~ 
    'expPC': [3*nver, n_exp_para]. ~
    'expEV': [n_exp_para, 1]. ~
    'texMU': [3*nver, 1]. ~
    'texPC': [3*nver, n_tex_para]. ~
    'texEV': [n_tex_para, 1]. ~
    'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
    'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
    'kpt_ind': [68,] (start from 1). ~
"""

class MorphabelModel():
    
    def __init__(self, model_path):
        self.model = load_bfm(model_path)

        # fixed attributes
        self.nver = self.model['shapePC'].shape[0] / 3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        # self.n_tex_para = self.model['texMU'].shape[1]
        self.n_tex_para = self.model['texPC'].shape[1]

        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']
        self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))

    # shape: represented with mesh(vertices & triangles(fixed))
    def get_shape_para(self, type='random'):
        if type == 'zero':
            sp = np.random.zeros((self.n_shape_para, 1))
        elif type == 'random':
            sp = np.random.rand(self.n_shape_para, 1) * 1e04
        return sp

    def get_exp_para(self, type='random'):
        if type == 'zero':
            ep = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            ep = -1.5 + 3 * np.random.random([self.n_exp_para, 1])
            ep[6:, 0] = 0

        return ep

    def generate_vertices(self, shape_para, exp_para):
        """
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1) 
        Returns:
            vertices: (nver, 3)
        """
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape_para) + self.model['expPC'].dot(exp_para) + \
                   self.model['expMU']
        vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T

        return vertices

    # -------------------------------------- texture: here represented with rgb value(colors) in vertices.
    def get_tex_para(self, type='random'):
        if type == 'zero':
            tp = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            tp = np.random.rand(self.n_tex_para, 1)
        return tp

    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''
        # colors = self.model['texMU'] + self.model['texPC'].dot(tex_para*self.model['texEV'])
        # refer to the code of AFLW in matlab, ignore texEV
        colors = self.model['texMU'] + self.model['texPC'].dot(tex_para)
        colors = np.reshape(colors, [int(3), int(len(colors) / 3)], 'F').T / 255.

        return colors

    # ------------------------------------------- transformation
    # -------------  transform
    def rotate(self, vertices, angles):
        ''' rotate face
        Args:
            vertices: [nver, 3]
            angles: [3] x, y, z rotation angle(degree)
            x: pitch. positive for looking down 
            y: yaw. positive for looking left
            z: roll. positive for tilting head right
        Returns:
            vertices: rotated vertices
        '''
        return mesh.transform.rotate(vertices, angles)

    def transform(self, vertices, s, angles, t3d):
        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    def transform_3ddfa(self, vertices, s, angles, t3d):  # only used for processing 300W_LP data
        R = mesh.transform.angle2matrix_3ddfa(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    # --------------------------------------------------- fitting
    def fit(self, x, X_ind, max_iter=4, isShow=False):
        ''' fit 3dmm & pose parameters
        Args:
            x: (n, 2) image points
            X_ind: (n,) corresponding Model vertex indices
            max_iter: iteration
            isShow: whether to reserve middle results for show
        Returns:
            fitted_sp: (n_sp, 1). shape parameters
            fitted_ep: (n_ep, 1). exp parameters
            s, angles, t
        '''
        if isShow:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points_for_show(x, X_ind, self.model, n_sp=self.n_shape_para,
                                                                    n_ep=self.n_exp_para, max_iter=max_iter)
            angles = np.zeros((R.shape[0], 3))
            for i in range(R.shape[0]):
                angles[i] = mesh.transform.matrix2angle(R[i])
        else:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points(x, X_ind, self.model, n_sp=self.n_shape_para,
                                                           n_ep=self.n_exp_para, max_iter=max_iter)
            angles = mesh.transform.matrix2angle(R)
        return fitted_sp, fitted_ep, s, angles, t

    def generate_offset(self, shape_para, exp_para):
        """
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1)
        Returns:
            vertices: (nver, 3)
        """
        vertices = self.model['shapePC'].dot(shape_para) + self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T
        return vertices.astype(np.float32)

    def get_mean_shape(self):
        vertices = self.model['shapeMU'] + self.model['expMU']
        vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T
        return vertices.astype(np.float32)

    
    
    
    
    
    
    
    
    
    
    
    
    
    


#  global data
# bfm = MorphabelModel('data/Out/BFM.mat')
default_init_image_shape = np.array([450, 450, 3])
default_cropped_image_shape = np.array([256, 256, 3])
default_uvmap_shape = np.array([256, 256, 3])
# face_mask_np = io.imread('uv-data/uv_face_mask.png') / 255.
# face_mask_mean_fix_rate = (256 * 256) / np.sum(face_mask_np)


def process_uv(uv_coordinates):
    [uv_h, uv_w, uv_c] = default_uvmap_shape
    uv_coordinates[:, 0] = uv_coordinates[:, 0] * (uv_w - 1)
    uv_coordinates[:, 1] = uv_coordinates[:, 1] * (uv_h - 1)
    uv_coordinates[:, 1] = uv_h - uv_coordinates[:, 1] - 1
    uv_coordinates = np.hstack((uv_coordinates, np.zeros((uv_coordinates.shape[0], 1))))  # add z
    return uv_coordinates


def readUVKpt(uv_kpt_path):
    file = open(uv_kpt_path, 'r', encoding='utf-8')
    lines = file.readlines()
    # txt is inversed
    x_line = lines[1]
    y_line = lines[0]
    uv_kpt = np.zeros((68, 2)).astype(int)
    x_tokens = x_line.strip().split(' ')
    y_tokens = y_line.strip().split(' ')
    for i in range(68):
        uv_kpt[i][0] = int(float(x_tokens[i]))
        uv_kpt[i][1] = int(float(y_tokens[i]))
    return uv_kpt


#  global data
# uv_coords = faceutil.morphable_model.load.load_uv_coords('data/Out/BFM_UV.mat')
# uv_coords = process_uv(uv_coords)
# uv_kpt = readUVKpt('uv-data/uv_kpt_ind.txt')
# uvmap_place_holder = np.ones((256, 256, 1))


def getLandmark(ipt):
    # from uv map
    kpt = ipt[uv_kpt[:, 0], uv_kpt[:, 1]]
    return kpt


def bfm2Mesh(bfm_info, image_shape=default_init_image_shape):
    """
    generate mesh data from 3DMM (bfm2009) parameters
    :param bfm_info:
    :param image_shape:
    :return: meshe data
    """
    [image_h, image_w, channel] = image_shape
    pose_para = bfm_info['Pose_Para'].T.astype(np.float32)
    shape_para = bfm_info['Shape_Para'].astype(np.float32)
    exp_para = bfm_info['Exp_Para'].astype(np.float32)
    tex_para = bfm_info['Tex_Para'].astype(np.float32)
    color_Para = bfm_info['Color_Para'].astype(np.float32)
    illum_Para = bfm_info['Illum_Para'].astype(np.float32)

    # 2. generate mesh_numpy
    # shape & exp param
    vertices = bfm.generate_vertices(shape_para, exp_para)
    # texture param
    tex = bfm.generate_colors(tex_para)
    norm = NormDirection(vertices, bfm.model['tri'])

    # color param
    [Gain_r, Gain_g, Gain_b, Offset_r, Offset_g, Offset_b, c] = color_Para[0]
    M = np.array([[0.3, 0.59, 0.11], [0.3, 0.59, 0.11], [0.3, 0.59, .11]])

    g = np.diag([Gain_r, Gain_g, Gain_b])
    o = [Offset_r, Offset_g, Offset_b]
    o = np.tile(o, (vertices.shape[0], 1))

    # illum param
    [Amb_r, Amb_g, Amb_b, Dir_r, Dir_g, Dir_b, thetal, phil, ks, v] = illum_Para[0]
    Amb = np.diag([Amb_r, Amb_g, Amb_b])
    Dir = np.diag([Dir_r, Dir_g, Dir_b])
    l = np.array([math.cos(thetal) * math.sin(phil), math.sin(thetal), math.cos(thetal) * math.cos(phil)]).T
    h = l + np.array([0, 0, 1]).T
    h = h / math.sqrt(h.T.dot(h))

    # final color
    n_l = l.T.dot(norm.T)
    n_h = h.T.dot(norm.T)
    n_l = np.array([max(x, 0) for x in n_l])
    n_h = np.array([max(x, 0) for x in n_h])
    n_l = np.tile(n_l, (3, 1))
    n_h = np.tile(n_h, (3, 1))
    L = Amb.dot(tex.T) + Dir.dot(n_l * tex.T) + (ks * Dir).dot((n_h ** v))
    CT = g.dot(c * np.eye(3) + (1 - c) * M)
    tex_color = CT.dot(L) + o.T
    tex_color = np.minimum(np.maximum(tex_color, 0), 1).T

    # transform mesh_numpy
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]

    # 3ddfa-R: radian || normal transform - R:degree
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
    image_vertices = projected_vertices.copy()
    # should not -1
    image_vertices[:, 1] = image_h - image_vertices[:, 1]
    mesh_info = {'vertices': image_vertices, 'triangles': bfm.full_triangles,
                 'full_triangles': bfm.full_triangles,
                 'colors': tex_color}
    # 'landmarks': bfm_info['pt3d_68'].T
    return mesh_info


def UVmap2Mesh(uv_position_map, uv_texture_map=None, only_foreface=True, is_extra_triangle=False):
    """
    if no texture map is provided, translate the position map to a point cloud
    :param uv_position_map:
    :param uv_texture_map:
    :param only_foreface:
    :return: mesh data
    """
    [uv_h, uv_w, uv_c] = default_uvmap_shape
    vertices = []
    colors = []
    triangles = []
    if uv_texture_map is not None:
        for i in range(uv_h):
            for j in range(uv_w):
                if not only_foreface:
                    vertices.append(uv_position_map[i][j])
                    colors.append(uv_texture_map[i][j])
                    pa = i * uv_h + j
                    pb = i * uv_h + j + 1
                    pc = (i - 1) * uv_h + j
                    pd = (i + 1) * uv_h + j + 1
                    if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                        triangles.append([pa, pb, pc])
                        triangles.append([pa, pc, pb])
                        triangles.append([pa, pb, pd])
                        triangles.append([pa, pd, pb])

                else:
                    if face_mask_np[i, j] == 0:
                        vertices.append(np.array([0, 0, 0]))
                        colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        vertices.append(uv_position_map[i][j])
                        colors.append(uv_texture_map[i][j])
                        pa = i * uv_h + j
                        pb = i * uv_h + j + 1
                        pc = (i - 1) * uv_h + j
                        pd = (i + 1) * uv_h + j + 1
                        if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                            if is_extra_triangle:
                                pe = (i - 1) * uv_h + j + 1
                                pf = (i + 1) * uv_h + j
                                if (face_mask_np[i, j + 1] > 0) and (face_mask_np[i + 1, j + 1] > 0) and (face_mask_np[i + 1, j] > 0) and (
                                        face_mask_np[i - 1, j + 1] > 0 and face_mask_np[i - 1, j] > 0):
                                    triangles.append([pa, pb, pc])
                                    triangles.append([pa, pc, pb])
                                    triangles.append([pa, pc, pe])
                                    triangles.append([pa, pe, pc])
                                    triangles.append([pa, pb, pe])
                                    triangles.append([pa, pe, pb])
                                    triangles.append([pb, pc, pe])
                                    triangles.append([pb, pe, pc])

                                    triangles.append([pa, pb, pd])
                                    triangles.append([pa, pd, pb])
                                    triangles.append([pa, pb, pf])
                                    triangles.append([pa, pf, pb])
                                    triangles.append([pa, pd, pf])
                                    triangles.append([pa, pf, pd])
                                    triangles.append([pb, pd, pf])
                                    triangles.append([pb, pf, pd])

                            else:
                                if not face_mask_np[i, j + 1] == 0:
                                    if not face_mask_np[i - 1, j] == 0:
                                        triangles.append([pa, pb, pc])
                                        triangles.append([pa, pc, pb])
                                    if not face_mask_np[i + 1, j + 1] == 0:
                                        triangles.append([pa, pb, pd])
                                        triangles.append([pa, pd, pb])
    else:
        for i in range(uv_h):
            for j in range(uv_w):
                if not only_foreface:
                    vertices.append(uv_position_map[i][j])
                    colors.append(np.array([64, 64, 64]))
                    pa = i * uv_h + j
                    pb = i * uv_h + j + 1
                    pc = (i - 1) * uv_h + j
                    if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                        triangles.append([pa, pb, pc])
                else:
                    if face_mask_np[i, j] == 0:
                        vertices.append(np.array([0, 0, 0]))
                        colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        vertices.append(uv_position_map[i][j])
                        colors.append(np.array([128, 0, 128]))
                        pa = i * uv_h + j
                        pb = i * uv_h + j + 1
                        pc = (i - 1) * uv_h + j
                        if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                            if not face_mask_np[i, j + 1] == 0:
                                if not face_mask_np[i - 1, j] == 0:
                                    triangles.append([pa, pb, pc])
                                    triangles.append([pa, pc, pb])

    vertices = np.array(vertices)
    colors = np.array(colors)
    triangles = np.array(triangles)
    # verify_face = mesh.render.render_colors(verify_vertices, verify_triangles, verify_colors, height, width,
    #                                         channel)
    mesh_info = {'vertices': vertices, 'triangles': triangles,
                 'full_triangles': triangles,
                 'colors': colors}
    return mesh_info


def mesh2UVmap(mesh_data):
    """
    generate uv map from mesh data
    :param mesh_data:
    :return: uv position map and corresponding texture
    """
    [uv_h, uv_w, uv_c] = default_uvmap_shape
    vertices = mesh_data['vertices']
    colors = mesh_data['colors']
    triangles = mesh_data['full_triangles']
    # colors = colors / np.max(colors)
    # model_image = mesh.render.render_colors(vertices, bfm.triangles, colors, image_h, image_w) # only for show

    uv_texture_map = mesh.render.render_colors(uv_coords, triangles, colors, uv_h, uv_w, uv_c)
    position = vertices.copy()
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z
    uv_position_map = mesh.render.render_colors(uv_coords, triangles, position, uv_h, uv_w, uv_c)
    return uv_position_map, uv_texture_map


def renderMesh(mesh_info, image_shape=None):
    if image_shape is None:
        image_height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
        image_width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
    else:
        [image_height, image_width, image_channel] = image_shape
    mesh_image = mesh.render.render_colors(mesh_info['vertices'],
                                           mesh_info['triangles'],
                                           mesh_info['colors'], image_height, image_width)
    mesh_image = np.clip(mesh_image, 0., 1.)
    return mesh_image


def getTransformMatrix(s, angles, t, height):
    x, y, z = angles[0], angles[1], angles[2]

    Rx = np.array([[1, 0, 0],
                   [0, cos(x), sin(x)],
                   [0, -sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, -sin(y)],
                   [0, 1, 0],
                   [sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), sin(z), 0],
                   [-sin(z), cos(z), 0],
                   [0, 0, 1]])
    # rotate
    R = Rx.dot(Ry).dot(Rz)
    R = R.astype(np.float32)
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[3, 3] = 1.
    # scale
    S = np.diagflat([s, s, s, 1.])
    T = S.dot(T)
    # offset move
    M = np.diagflat([1., 1., 1., 1.])
    M[0:3, 3] = t.astype(np.float32)
    T = M.dot(T)
    # revert height
    # x[:,1]=height-x[:,1]
    H = np.diagflat([1., 1., 1., 1.])
    H[1, 1] = -1.0
    H[1, 3] = height
    T = H.dot(T)
    return T.astype(np.float32)


def getColors(image, posmap):
    [h, w, _] = image.shape
    [uv_h, uv_w, uv_c] = posmap.shape
    # tex = np.zeros((uv_h, uv_w, uv_c))
    around_posmap = np.around(posmap).clip(0, h - 1).astype(np.int)

    tex = image[around_posmap[:, :, 1], around_posmap[:, :, 0], :]
    return tex