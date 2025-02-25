import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import losses
from tensorflow.keras import Model
from tensorflow.keras.layers import *


"""
    ShuffleNet v2 Blocks
    https://arxiv.org/pdf/1807.11164v1.pdf
    
    TODO: batch normal training 일 때 true/false
"""
class ShuffleNetBasicUnitV2(tf.Module):
    def __init__(self, out_channel, **kwargs):
        super().__init__(name="ShuffleNetBasicUnitV2")
        
        def mn(m):
            if 'name' in kwargs:
                return '{}_basic_{}'.format(kwargs.get('name'), m)
            return 'basic_{}'.format( m)
        
        num_filters = out_channel // 2
        norm = tfa.layers.InstanceNormalization
        
        self.pw_conv1 = Conv2D(num_filters, kernel_size=(1,1), 
                               use_bias=False, name=mn('pw_conv1'))
        self.bn_norm1 = norm(name=mn('bn_norm1'))
        self.dw_conv1 = DepthwiseConv2D(kernel_size=(3,3), padding='same', 
                                        use_bias=False, name=mn('dw_conv1'))
        self.bn_norm2 = norm(name=mn('bn_norm2'))
        self.pw_conv2 = Conv2D(num_filters, kernel_size=(1,1), 
                               use_bias=False, name=mn('pw_conv2'))
        self.bn_norm3 = norm(name=mn('bn_norm3'))

    def __call__(self, x):
        xl, xr = tf.split(x, 2, axis=3)
        
        xr = self.pw_conv1(xr)
        xr = self.bn_norm1(xr)
        xr = tf.nn.relu6(xr)
        xr = self.dw_conv1(xr)
        xr = self.bn_norm2(xr)
        xr = self.pw_conv2(xr)
        xr = self.bn_norm3(xr)
        xr = tf.nn.relu6(xr)
        x = tf.concat((xl, xr), axis=3)
        
        return channel_shuffle(x)

    
class ShuffleNetDownSampleUnitV2(tf.Module):
    def __init__(self, out_channel, **kwargs):
        super().__init__(name="ShuffleNetDownSampleUnitV2")

        def mn(m):
            if 'name' in kwargs:
                return '{}_down_{}'.format(kwargs.get('name'), m)
            return 'down_{}'.format( m)
        
        num_filters = out_channel // 2
        norm = tfa.layers.InstanceNormalization
         
        # left branch
        self.dw_conv_l1 = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), 
                                          padding='same', use_bias=False,
                                          name=mn('dw_conv_l1'))
        self.bn_norm_l1 = norm(name=mn('bn_norm_l1'))
        self.pw_conv_l1 = Conv2D(num_filters, kernel_size=(1,1), 
                                 use_bias=False, name=mn('pw_conv_l1'))
        
        # right branch
        self.pw_conv_r1 = Conv2D(num_filters, kernel_size=(1,1), 
                                 use_bias=False, name=mn('pw_conv_r1'))
        self.bn_norm_r1 = norm(name=mn('bn_norm_r1'))
        self.dw_conv_r1 = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), 
                                          padding='same', use_bias=False,
                                          name=mn('dw_conv_r1'))
        self.bn_norm_r2 = norm(name=mn('bn_norm_r2'))
        self.pw_conv_r2 = Conv2D(num_filters, kernel_size=(1,1), 
                                 use_bias=False, name=mn('pw_conv_r2'))

        # final concat
        self.bn_norm_f1 = norm(name=mn('bn_norm_f1'))

    def __call__(self, x):
        # left branch
        xl = self.dw_conv_l1(x)
        xl = self.bn_norm_l1(xl)
        xl = self.pw_conv_l1(xl)
        
        # right branch
        xr = self.pw_conv_r1(x)
        xr = self.bn_norm_r1(xr)
        xr = tf.nn.relu6(xr)
        xr = self.dw_conv_r1(xr)
        xr = self.bn_norm_r2(xr)
        xr = self.pw_conv_r2(xr)
        
        # final concat
        x = tf.concat((xl, xr), axis=3)
        x = self.bn_norm_f1(x)
        x = tf.nn.relu6(x)
        
        return channel_shuffle(x)
    