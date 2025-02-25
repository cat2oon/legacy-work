import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from os.path import *
from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


"""
    Free Functions
"""
def conv_block(net, depth_ksize=3, depth_strides=1, conv_filters=16, conv_ksize=1, conv_strides=1):
    shortcut = net
    net = DepthwiseConv2D(kernel_size=depth_ksize, strides=depth_strides, padding='same')(net)
    net = DepthwiseConv2D(kernel_size=depth_ksize, strides=1, padding='same')(net)
    net = Conv2D(filters=conv_filters, kernel_size=conv_ksize, strides=conv_strides, padding='same')(net)
    net = Add()([shortcut, net])
    net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

    return net

def branch_block(net, depth_ksize=3, depth_strides=2, conv_filters=16, conv_ksize=1, conv_strides=1, pad=True):
    branch_1 = DepthwiseConv2D(kernel_size=depth_ksize, strides=depth_strides, padding='same')(net)
    branch_1 = Conv2D(filters=conv_filters, kernel_size=conv_ksize, strides=conv_strides, padding='same')(branch_1)
    branch_2 = MaxPool2D(pool_size=2)(net)
    branch_2 = Conv2D(filters=conv_filters, kernel_size=conv_ksize, strides=conv_strides, padding='same')(branch_2)

    net = Add()([branch_1, branch_2])
    net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)

    return net


"""
    Utils
"""
def euclidean_dist(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=1))

def cos_sim():
    return tf.keras.losses.CosineSimilarity(axis=1)

def to_3d_vec(vec_2d):
    nb = tf.shape(vec_2d)[0]
    z = -tf.ones([nb, 1])
    return tf.concat([vec_2d, z], 1)

def unit_vector(vector):
    norm = tf.sqrt(tf.reduce_sum(tf.square(vector), axis=1, keepdims=True))
    res = tf.math.divide_no_nan(vector, norm)
    return res

@tf.function
def modify_eye(input):
    original_eye = input[0]   # eye_3d
    offset = tf.reshape(input[1], [-1, 3, 1])
    R = tf.reshape(input[2], [-1, 3, 3])
    invR = tf.linalg.inv(R)
    return original_eye + tf.reshape(tf.linalg.matmul(invR, offset), [-1, 3])

def adjust_kappa_angle_with_eye(input):
    pred_vector = input[0]
    pred_unit_vector = unit_vector(pred_vector)
    
    x_pred = tf.transpose(pred_unit_vector)[0]
    y_pred = tf.transpose(pred_unit_vector)[1]
    z_pred = tf.transpose(pred_unit_vector)[2]
    theta_pred = tf.asin(y_pred)
    phi_pred = tf.atan((x_pred + K.epsilon())/(-z_pred + K.epsilon()))
    
    
    true_eye = tf.reshape(input[1], [-1, 3])
    true_gaze_2d = tf.reshape(input[2], [-1, 3])
    R = tf.linalg.inv(tf.reshape(input[3], [-1, 3, 3]))
    true_vector = tf.reshape(true_gaze_2d - true_eye, [-1, 3, 1])
    true_vector = tf.reshape(tf.linalg.matmul(R, true_vector), [-1, 3])
    true_unit_vector = unit_vector(true_vector)
    
    x_true = K.transpose(true_unit_vector)[0]
    y_true = K.transpose(true_unit_vector)[1]
    z_true = K.transpose(true_unit_vector)[2]
    theta_true = tf.asin(y_true)
    phi_true = tf.atan((x_true + K.epsilon())/(-z_true + K.epsilon()))

    dtheta = tf.clip_by_value(tf.reduce_mean(theta_true - theta_pred), -0.1, 0.1)
    dphi = tf.clip_by_value(tf.reduce_mean(phi_true - phi_pred), -0.18, 0.18)

    theta = theta_pred + tf.math.maximum(tf.math.minimum(dtheta, 0.1), -0.1)
    phi = phi_pred + tf.math.maximum(tf.math.minimum(dphi, 0.18), -0.18)

    ty_pred = tf.math.sin(theta)
    tx_pred = tf.math.multiply(tf.math.cos(theta), tf.math.sin(phi))
    tz_pred = -tf.math.multiply(tf.math.cos(theta), tf.math.cos(phi))

    tv_pred = K.reshape(K.concatenate([K.reshape(tx_pred, (-1, 1)),
                                        K.reshape(ty_pred, (-1, 1)),
                                        K.reshape(tz_pred, (-1, 1))], 1), (-1, 3))
    return tv_pred

@tf.function
def calculate_gaze_2d(input): #gaze_vector, eye_position, invR):
    gaze_vector = input[0]
    gaze_vector = tf.reshape(gaze_vector, [-1, 3, 1])
    eye_position = tf.reshape(input[1], [-1, 3, 1])
    invR = tf.reshape(input[2], [-1, 3, 3])
    
    gaze_vector = tf.linalg.matmul(invR, gaze_vector)
    multiplier = - tf.math.divide_no_nan(eye_position[:, 2], gaze_vector[:, 2] + tf.keras.backend.epsilon())
    multiplier = tf.reshape(tf.tile(multiplier, [1, 3]), [-1, 3, 1])
    
    target = tf.reshape(tf.add(eye_position, tf.multiply(gaze_vector, multiplier)), [-1, 3])
    return target / 10


"""
    Elon's Gaze Vector Model (ALG-250-2)
"""
class M4():
    
    def __init__(self, config, show_summary=False):
        self.config = config
        self.make_model()
        self.compile_model(show_summary)
                
    def load(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)
        print("Model loaded")
        
    def save(self, checkpoint_path):
        if self.config.get('save_weight') == True:
            self.model.save_weights(checkpoint_path)
        if self.config.get('save_weight') == False:
            self.model.save_model(checkpoint_path)
        print("Model saved")
        
        
    def make_model(self):
        eye_img_h, eye_img_w, c = self.config.img_size_y, self.config.img_size_x, self.config.n_channel
        input_re = Input(shape=(eye_img_h, eye_img_w, c), name='in_re')
        input_le = Input(shape=(eye_img_h, eye_img_w, c), name='in_le')
        input_we = Input(shape=(self.config.whole_eye_image_y, self.config.whole_eye_image_x, c), name='in_we')
        
        input_eyes       = Input(shape=(6), name='in_eyes')        # 차이? eyes and eye_poses   input_eyes -> eye_3d (3d position maybe?)  
        input_eye_poses  = Input(shape=(9), name='in_eye_poses')   #  eye_poses -> transform/rotation matrix(?)
        input_right_invR = Input(shape=(9), name='in_right_invR')
        input_left_invR  = Input(shape=(9), name='in_left_invR')
        input_right_true = Input(shape=(3), name='in_right_truth')
        input_left_true  = Input(shape=(3), name='in_left_truth')
        
        eye_r_3d   = tf.slice(input_eyes, [0, 0], [-1, 3])
        eye_l_3d   = tf.slice(input_eyes, [0, 3], [-1, 3])
        eye_pose_r = tf.slice(input_eye_poses, [0, 0], [-1, 3])
        eye_pose_l = tf.slice(input_eye_poses, [0, 3], [-1, 3])
        
        """ Construct """
        re_fv = self.build_eye_model('re')(input_re)
        le_fv = self.build_eye_model('le')(input_le)
        face_fv = self.build_frame_model('frame')(input_we)
        
        eye_pose_r_emb = Dense(12, activation='sigmoid')(eye_pose_r)
        eye_pose_l_emb = Dense(12, activation='sigmoid')(eye_pose_l)
        
        gaze_vec_r = self.build_gaze_module(re_fv, eye_pose_r_emb)
        gaze_vec_l = self.build_gaze_module(le_fv, eye_pose_l_emb)
        r_offset, l_offset = self.build_eye_offset(input_eyes, face_fv, eye_pose_r_emb, eye_pose_l_emb)
        
        gaze_xy_r, adjusted_vec_r = self.build_gaze_xy(eye_r_3d, r_offset, gaze_vec_r, input_right_invR, input_right_true, is_left=False)
        gaze_xy_l, adjusted_vec_l = self.build_gaze_xy(eye_l_3d, l_offset, gaze_vec_l, input_left_invR,  input_left_true,  is_left=True)
        gaze_xy = Lambda(lambda x : tf.reduce_mean(x, 0), name='center')([gaze_xy_r, gaze_xy_l])
        
        """ Make Model """
        inputs = [input_re, input_le, input_we, input_eyes, input_eye_poses, input_right_invR, input_left_invR, input_right_true, input_left_true]
        outputs = [adjusted_vec_r, adjusted_vec_l, gaze_xy_r, gaze_xy_l, gaze_xy]
        self.model = Model(inputs=inputs, outputs=outputs)

        
    def compile_model(self, show_summary):
        metrics = {'center':euclidean_dist}
        losses  = {'rv':cos_sim(), 'lv':cos_sim(), 'right':'MSE', 'left':'MSE', 'center':'MSE'}
        loss_weights = {'rv':0, 'lv':0, 'right':1, 'left':1, 'center':16}
        
        optimizer = keras.optimizers.Adam(lr=self.config.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.001)
        self.model.compile(loss=losses, loss_weights=loss_weights, metrics=metrics, optimizer=optimizer)
        if show_summary:
            self.model.summary()
    
    
    """
        Module Build
    """ 
    def build_eye_model(self, name_prefix): 
        input_image = Input(shape=(self.config.img_size_y, self.config.img_size_x, self.config.n_channel))

        net = Conv2D(filters=12, kernel_size=5, strides=2, padding='same')(input_image)
        net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)            # (32 x 48 x 12)
        
        net = Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(net)
        net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)            # (32 x 48 x 18)
        
        net = branch_block(net, conv_filters=18)
        net = conv_block(net, conv_filters=18)                                     # (16 x 24 x 24)
        
        net = branch_block(net, conv_filters=24)
        net = conv_block(net, conv_filters=24)                                     # (8 x 12 x 32)
    
        net = branch_block(net, conv_filters=28)                                   # (4 x 6 x 36)
        net = branch_block(net, conv_filters=28)                                   # (2 x 3 x 36)
        
        net = MaxPool2D(pool_size=2, padding='valid', strides=1)(net)
        net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)            # (1 x 2 x 36)
        net = Lambda(lambda x: tf.reshape(x, [-1, 56]))(net)                       # (72)
        
        net = Dense(24, activation='tanh')(net)
        
        return Model(inputs=[input_image], outputs=net, name=name_prefix)
    
    
    def build_frame_model(self, name_prefix):
        input_image = Input(shape=(self.config.whole_eye_image_y, self.config.whole_eye_image_x, self.config.n_channel))
        
        net = Conv2D(filters=12, kernel_size=5, strides=2, padding='same')(input_image)
        net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)               # (24 x 48 x 16)
        
        net = Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(net)
        net = PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(net)               # (24 x 48 x 16)
        
        net = branch_block(net, conv_filters=16)                                      # (12 x 24 x 16)
        
        net = branch_block(net, conv_filters=16)
        net = conv_block(net, conv_filters=16)                                        # (6 x 12 x 16)

        net = branch_block(net, conv_filters=24)                                      # (3 x 6 x 24)
        
        net = GlobalAveragePooling2D()(net)                                           # (1 x 1 x 24)
        
        net = Lambda(lambda x: tf.reshape(x, [-1, 24]))(net)
        net = Dropout(0.2) (net)
        
        return Model(inputs=input_image, outputs=net, name=name_prefix)


    def build_gaze_module(self, eye_net, eye_pose_emb):
        x = Concatenate(axis=1)([eye_net, eye_pose_emb])
        x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = Dropout(0.4)(x)
        x = Dense(16, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(x)
        
        gaze_vec = Dense(2, activation='linear', kernel_regularizer=regularizers.l2(0.0001))(x)
        gaze_vec = Lambda(to_3d_vec)(gaze_vec)
        
        return gaze_vec
    
    def build_eye_offset(self, input_eyes, face_fv, eye_pose_r_emb, eye_pose_l_emb):
        eye = Dense(18, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001))(input_eyes)
        eye = Dropout(0.4)(eye)
        
        x = Concatenate(axis=1)([face_fv, eye, eye_pose_r_emb, eye_pose_l_emb])
        x = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = Dropout(0.3)(x)
        eye_offset = Dense(6, kernel_initializer='zeros', bias_initializer='zeros')(x)
        
        return eye_offset[:, :3], eye_offset[:, 3:]
    
    def build_gaze_xy(self, eye_3d, offset, gaze_vec, in_invR, in_true, is_left):
        vec_name = 'lv' if is_left else 'rv'
        gaze_name = 'left' if is_left else 'right'
        eye_3d = Lambda(modify_eye)([eye_3d, offset, in_invR])
        vec_adjusted = Lambda(adjust_kappa_angle_with_eye, name=vec_name)([gaze_vec, eye_3d, in_true, in_invR])
        gaze_xy = Lambda(calculate_gaze_2d, name=gaze_name)([vec_adjusted, eye_3d, in_invR])
        
        return gaze_xy, vec_adjusted
        
        