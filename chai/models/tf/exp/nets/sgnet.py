from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (Add, 
                                     Concatenate, 
                                     Conv2D, 
                                     Dense,  
                                     Flatten,  
                                     Input, Lambda, LeakyReLU, 
                                     AveragePooling2D, MaxPool2D, UpSampling2D, ZeroPadding2D)

    
"""
    SGNet
"""
class SGNet(tf.keras.Model):
    
    @classmethod
    def create(cls, activation='relu'):
        # left/right eye patch input
        input_ec = Input(shape=(8), name="eye_corner_landmark")
        input_le = Input(shape=(64, 64, 3), name='left_eye_patch')
        input_re = Input(shape=(64, 64, 3), name='right_eye_patch')
        
        # landmark 
        ecx = Dense(100, activation=activation, name='ec_fc1')(input_ec)
        ecx = Dense(16,  activation=activation, name='ec_fc2')(ecx)
        ecx = Dense(16,  activation=activation, name='ec_fc3')(ecx)
        
        # shared left/right eye
        eye_net = GimpleNet()
        lex = eye_net(input_le)
        rex = eye_net(input_re)
        lex = tf.squeeze(lex, axis=[1,2])
        rex = tf.squeeze(rex, axis=[1,2])
        
        # final
        z = Concatenate(name='bottleneck')([lex, rex, ecx])
        z = Dense(16, activation=activation, name='final_fc1', use_bias=False)(z)
        z = Dense(2,  activation='linear',   name='final_fc2', use_bias=False)(z)
        
        model = Model(inputs=[input_le, input_re, input_ec], outputs=z)
        return model
    
    def __init__(self, activation='relu', **kwargs):
        super().__init__(**kwargs)
        
        # left/right eye patch input
        self.input_ec = Input(shape=(8), name="eye_corner_landmark")
        self.input_le = Input(shape=(64, 64, 3), name='left_eye_patch')
        self.input_re = Input(shape=(64, 64, 3), name='right_eye_patch')
        
        # landmark 
        self.ec_fc1 = Dense(100, activation=activation, name='ec_fc1')
        self.ec_fc2 = Dense(16,  activation=activation, name='ec_fc2')
        self.ec_fc3 = Dense(16,  activation=activation, name='ec_fc3')
        
        self.eye_net = GimpleNet()
        self.final_fc1 = Dense(16, activation=activation, name='final_fc1')
        self.final_fc2 = Dense(2,  activation='linear',   name='final_fc2')
    
    def call(self, le, re, ec):
        ecx = self.ec_fc1(ec)
        ecx = self.ec_fc2(ecx)
        ecx = self.ec_fc3(ecx)
        
        # shared left/right eye
        lex = self.eye_net(le)
        rex = self.eye_net(re)
        lex = tf.squeeze(lex, axis=[1,2])
        rex = tf.squeeze(rex, axis=[1,2])
        
        # final
        z = Concatenate(name='bottleneck')([lex, rex, ecx])
        z = self.final_fc1(z)
        z = self.final_fc2(z)
        return z
    
    
"""
    GimpleNet
"""
class GimpleNet(tf.keras.Model):
    
    def __init__(self, activation='relu', **kwargs):
        super().__init__(**kwargs)
        
        self.avg_pool1 = AveragePooling2D(pool_size=(7, 7))
        self.avg_pool2 = AveragePooling2D(pool_size=(5, 5))
        
        self.conv1 = Conv2D(filters=32, kernel_size=(7,7), padding='same', activation=activation)
        self.conv2 = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation=activation)
        self.conv3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=activation)
        
    def call(self, eye_patch):
        x = self.conv1(eye_patch)
        x = self.avg_pool1(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
        x = self.conv3(x)
        
        return x
    