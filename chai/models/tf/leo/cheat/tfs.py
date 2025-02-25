import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from tensorflow import losses
from tensorflow.keras import Model
from tensorflow.keras.layers import *
                                     


"""
    Random Seed
"""
def set_random_seed(seed):
    assert seed is not None, "random seed is none"
    np.random.seed(seed)
    tf.random.set_seed(seed)  

    
    
"""
    Gradients
"""
def clip_gradients(grads, grad_threshold, grad_norm_threshold):
    if grad_threshold > 0:
        grads = [tf.clip_by_value(g, -grad_threshold, grad_threshold) for g in grads]
    if grad_norm_threshold > 0:
        grads = [tf.clip_by_norm(g, grad_norm_threshold) for g in grads]
    return grads



"""
    General ops
"""
def channel_shuffle(x):
    """ static shape version """
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, [-1, h, w, 2, c // 2])      # 반반 채널 쪼개기
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])     # 반반 채널 스왑
    x = tf.reshape(x, [-1, h, w, c])              # 반반 채널 통합
    return x



"""
    Makers
"""
def random_normal(shape, mean=0.0, stddev=0.1):
    return tf.random_normal(shape, mean, stddev)

def variable(x, name, trainable=True):
    return tf.Variable(x, name=name, trainable=trainable)

def var_with_init(initializer, name, trainable=True):
    return tf.Variable(initializer, name=name, trainable=trainable)

def var_with_shape(shape, name, seed, trainable=True):
    init_w = tf.keras.initializers.glorot_uniform(seed=1234)(shape=shape)
    return tf.Variable(init_w, name=name, trainable=trainable)

def var_tile(value, shape, name, dtype=tf.float32):
    return tf.Variable(np.tile(value, shape), name=name, dtype=dtype)



"""
    Vector ops
"""
def to_vec_3d(vec_2d):
    num_batch = tf.shape(vec_2d)[0]
    z = -tf.ones([num_batch, 1])
    return tf.concat([vec_2d, z], 1)

def to_unit_vec(vec, axis=1):
    return tf.math.l2_normalize(vec, axis=axis)

def to_unit_vec_3d(vec_2d):
    return to_unit_vec(to_vec_3d(vec_2d))

def flip_y_axis(vec):
    return vec * tf.constant([1, -1, 1], dtype=tf.float32)

def flip_x_axis(vec):
    return vec * tf.constant([-1, 1, 1], dtype=tf.float32)
    
def split_pair_vec(tensors):
    if type(tensors) == tuple: 
        x, y = tensors
        return x, y
    num_half = int(tensors.shape[1] / 2)
    return tf.split(tensors, [num_half, num_half], axis=1)

def drop_z_in_vec(tensors, batch=True):
    if batch:
        return tf.slice(tensors, [0,0], [tensors.shape[0],2])
    return tf.slice(tensors, [0], [2])

def calc_target(vec, eye_pos, some_R=None):
    if some_R is not None:
        some_R = tf.reshape(some_R, (-1, 3, 3))
        vec = tf.matmul(some_R, tf.reshape(vec, (-1, 3, 1)))
    gaze_vec = tf.reshape(vec, (-1, 3))
    factor = -tf.math.divide_no_nan(eye_pos[:,2], gaze_vec[:,2])
    factor = tf.expand_dims(factor, axis=1)
    
    target = eye_pos + (factor * gaze_vec)
    return target

def deg2rad(degree):
    return degree * np.pi / 180.0

def rad2deg(radian):
    return radian * 180.0 / np.pi

def to_visual_axis(theta, phi, alpha, beta):
    sin, cos = tf.math.sin, tf.math.cos
    x = sin(theta + alpha) * cos(phi + beta)
    y = -sin(phi + beta)                             # xu-cong-zhang
    z = -cos(theta + alpha) * cos(phi + beta)
    visual_axis = tf.concat([x, y, z], axis=1)
    return visual_axis

def pitch_to_R(pitch):
    ones = tf.ones_like(pitch)
    zeros = tf.zeros_like(pitch)
    sin_p, cos_p = tf.math.sin(pitch), tf.math.cos(pitch)
    R = tf.concat([ones,   zeros, zeros, 
                   zeros,  cos_p, sin_p,
                   zeros, -sin_p, cos_p], axis=1)
    R = tf.reshape(R, (-1, 3, 3))
    return R
    

    
"""
    Tensor ops
"""
def filter_tensors(tensors, tensor_names):
    return [v for v in tensors if v.name not in tensor_names]
    
    
    
"""
    Layer ops 
"""
def dense(x, w, b=None, act=tf.nn.selu):
    x = tf.einsum("ij,jk->ik", x, w)
    if b is not None:
        x = x + b
    if act is not None:
        x = act(x)
    return x 
      
def dense_from(x, theta, shape_i, shape_j, act=tf.nn.relu, bias=True):
    if not bias:
        k = tf.reshape(theta, (shape_i, shape_j))
        return dense(x, k, None, act)
        
    k, b = tf.split(theta, [shape_i*shape_j, shape_j*1], axis=1)
    k = tf.reshape(k, (shape_i, shape_j))
    b = tf.reshape(b, (shape_j, ))
    return dense(x, k, b, act)
    

def mlp_from(x, theta, kernel_dims, last_act=tf.nn.tanh):
    layers = split_mlp_theta(theta, kernel_dims, axis=1)
    for i, theta in enumerate(layers):
        w, b = theta
        if i == len(layers)-1:
            x = dense(x, w, b, act=last_act)
        else:
            x = dense(x, w, b)
    return x

def get_layer_size(kernel_dims):
    split, shapes = [], []
    prev = kernel_dims[0]
    for d in kernel_dims[1:]:
        dim_w, dim_b = prev * d, d
        split.append(dim_w)
        split.append(dim_b)
        shapes.append((prev, d))
        shapes.append((d,))      # NOTE: NO (d, 1) 
        prev = d
    return split, shapes

def split_mlp_theta(theta, kernel_dims, axis):
    split, shapes = get_layer_size(kernel_dims)
    thetas = tf.split(theta, split, axis=axis)
    layers = []
    for i in range(0, len(thetas), 2):
        w = tf.reshape(thetas[i+0], shapes[i+0])
        b = tf.reshape(thetas[i+1], shapes[i+1])
        layers.append((w, b))
    return layers 



"""
    Activations
"""
def act(x, name):
    assert hasattr(tf.nn, name), "wrong activation name"
    act_fn = getattr(tf.nn, name)
    return act_fn(x)



"""
    Losses & Metrics
"""
def gaze_loss(y_true, y_pred):
    return tf.reduce_mean(tf.losses.mse(y_true, y_pred))

def cos_loss(y_true, y_pred):
    return tf.keras.losses.CosineSimilarity(axis=1)(y_true, y_pred) + 1

def euclidean_dist(y_true, y_pred):
    square = tf.math.square(y_pred - y_true)
    reduce_sum = tf.math.reduce_sum(square, axis=1)
    dists = tf.math.sqrt(reduce_sum)
    return tf.math.reduce_mean(dists)



"""
    Sampling Distribution 
"""
def sample_dist(statistics, no_bias=False, use_norm_std=True, stddev_offset=0.0, mean_scale=None):
    means, unnormalized_stddev = tf.split(statistics, 2, axis=-1)
    
    if mean_scale is not None:
        means = mean_scale * means
    if no_bias:
        return means, tf.constant(0.0)
    
    if use_norm_std:
        stddev = tf.exp(unnormalized_stddev)
        stddev -= (1. - stddev_offset)
        stddev = tf.maximum(stddev, 1e-10)
    else:
        stddev = unnormalized_stddev
        
    distribution = tfp.distributions.Normal(loc=means, scale=stddev)
    samples = distribution.sample()
    return samples, kl_divergence(samples, distribution)

def kl_divergence(samples, normal_distribution):
    random_prior = tfp.distributions.Normal(loc=tf.zeros_like(samples), scale=tf.ones_like(samples))
    kl = tf.reduce_mean(normal_distribution.log_prob(samples) - random_prior.log_prob(samples))
    return kl



"""
    Regularizers
"""
def l2_reg(factor=0.01):
    return tf.keras.regularizers.l2(factor)

def orthogonality_regularize_term(w, penalty):
    w2 = tf.matmul(w, w, transpose_b=True)
    wn = tf.norm(w, ord=2, axis=1, keepdims=True) + 1e-32
    correlation_matrix = w2 / tf.matmul(wn, wn, transpose_b=True)
    matrix_size = correlation_matrix.get_shape().as_list()[0]
    identity = tf.eye(matrix_size)
    weight_corr = tf.reduce_mean(tf.math.squared_difference(correlation_matrix, identity))
    return tf.multiply(penalty, weight_corr, "orthogonality")



"""
    Name alias
"""
BatchNorm = BatchNormalization
InstanceNorm = tfa.layers.InstanceNormalization
CosineDecayRestarts = tf.keras.experimental.CosineDecayRestarts



"""
    Keras Layers
"""

""" WTF? 왜 이걸로 호출하면 dilation이 안 먹히는 거지?? """
def conv3x3(num_channels, name, dl=(1,1), strides=(1,1), 
            bias=False, pad='same', l2_reg_factor=0.01, act='selu'):
    return Conv2D(filters=num_channels, kernel_size=(3,3), padding=pad, 
                  strides=strides, dilation_rate=dl, use_bias=bias, 
                  kernel_regularizer=l2_reg(l2_reg_factor), activation=act, name=name)

def dw_conv3x3(name, strides=(1,1), bias=False, pad='same', act=None):
    return DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding=pad, 
                           use_bias=bias, activation=act, name=name)

def pw_conv1x1(num_channels, name, bias=False, act=None):
    return Conv2D(filters=num_channels, kernel_size=(1,1), use_bias=bias, activation=act, name=name)


    
    
    
    
    