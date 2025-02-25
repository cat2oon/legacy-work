import numpy as np
import tensorflow as tf

from nets.layers import *
from tensorflow.keras import Model
from tensorflow.keras.layers import *

BatchNorm = BatchNormalization
InstanceNorm = tfa.layers.InstanceNormalization


def channel_shuffle(x):
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, [-1, h, w, 2, c // 2])
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, h, w, c])
    return x

class EyeNet(tf.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(name="eyenet", *args, **kwargs)
        
        # stem conv
        self.stem_conv1 = Conv2D(filters=32, kernel_size=(3,3), padding='same', use_bias=False, activation='selu', name='stem_conv1')
        self.stem_pool1 = MaxPool2D(padding='same', strides=(2,2), name='stem_pool1')
        self.stem_bn1   = BatchNorm(name='stem_bn1')
        
        # shuffle-net v2
        self._make_shuffle_block(64)
        
        # dilated conv  (B, 64, 128, 1) -> (B, 4, 8, C)
        # fomoro.com/research/article/receptive-field-calculator
        self.dl_conv1 = Conv2D(32, kernel_size=(3,3), dilation_rate=(2, 2), use_bias=False, activation='selu', name='dl_conv1')
        self.dl_bn1 = BatchNorm(name='dl_bn1')
        self.dl_conv2 = Conv2D(32, kernel_size=(3,3), dilation_rate=(2, 4), use_bias=False, activation='selu', name='dl_conv2')
        self.dl_bn2 = BatchNorm(name='dl_bn2')
        self.dl_conv3 = Conv2D(64, kernel_size=(3,3), dilation_rate=(2, 6), use_bias=False, activation='selu', name='dl_conv3')
        self.dl_bn3 = BatchNorm(name='dl_bn3')
        self.gap = GlobalAveragePooling2D()
        
        # dense
        self.fc1 = Dense(64, use_bias=False, name='fc1')

    def _make_shuffle_block(self, filters):
        num_filters = filters // 2
        
        # ShuffleNetDownSampleV2 - left branch
        self.dw_conv_l1 = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, name='dw_conv_l1')
        self.in_norm_l1 = InstanceNorm(name='in_norm_l1')
        self.pw_conv_l1 = Conv2D(num_filters, kernel_size=(1,1), use_bias=False, name='pw_conv_l1')
        
        # ShuffleNetDownSampleV2 - right branch
        self.pw_conv_r1 = Conv2D(num_filters, kernel_size=(1,1), use_bias=False, name='pw_conv_r1')
        self.in_norm_r1 = InstanceNorm(name='in_norm_r1')
        self.dw_conv_r1 = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, name='dw_conv_r1')
        self.in_norm_r2 = InstanceNorm(name='in_norm_r2')
        self.pw_conv_r2 = Conv2D(num_filters, kernel_size=(1,1), use_bias=False, name='pw_conv_r2')
        
        # ShuffleNetDownSampleV2 - concat
        self.in_norm_f1 = InstanceNorm(name='in_norm_f1')
        
        # ShuffleNetBasicUnitV2 
        self.pw_conv1 = Conv2D(num_filters, kernel_size=(1,1), use_bias=False, name='pw_conv1')
        self.in_norm1 = InstanceNorm(name='in_norm1')
        self.dw_conv1 = DepthwiseConv2D(kernel_size=(3,3), padding='same', use_bias=False, name='dw_conv1')
        self.in_norm2 = InstanceNorm(name='in_norm2')
        self.pw_conv2 = Conv2D(num_filters, kernel_size=(1,1), use_bias=False, name='pw_conv2')
        self.in_norm3 = InstanceNorm(name='in_norm3')
        
    def _call_stem_conv(self, x):
        x = self.stem_conv1(x)
        x = self.stem_pool1(x)
        x = self.stem_bn1(x)
        return x
        
    def _call_shuffle_block(self, x):
        # left branch
        xl = self.dw_conv_l1(x)
        xl = self.in_norm_l1(xl)
        xl = self.pw_conv_l1(xl)
        
        # right branch
        xr = self.pw_conv_r1(x)
        xr = self.in_norm_r1(xr)
        xr = tf.nn.relu6(xr)
        xr = self.dw_conv_r1(xr)
        xr = self.in_norm_r2(xr)
        xr = self.pw_conv_r2(xr)
        
        # downsample concat
        x = tf.concat((xl, xr), axis=3)
        x = self.in_norm_f1(x)
        x = tf.nn.relu6(x)
        x = channel_shuffle(x)
        
        # basic unit 
        xl, xr = tf.split(x, 2, axis=3)
        
        xr = self.pw_conv1(xr)
        xr = self.in_norm1(xr)
        xr = tf.nn.relu6(xr)
        xr = self.dw_conv1(xr)
        xr = self.in_norm2(xr)
        xr = self.pw_conv2(xr)
        xr = self.in_norm3(xr)
        xr = tf.nn.relu6(xr)
        x = tf.concat((xl, xr), axis=3)
        return channel_shuffle(x)
    
    def _call_dialated(self, x):
        x = self.dl_conv1(x)
        x = self.dl_bn1(x)
        x = self.dl_conv2(x)
        x = self.dl_bn2(x)
        x = self.dl_conv3(x)
        x = self.dl_bn3(x)
        x = self.gap(x)
        return x
        
    def __call__(self, x):
        x = self._call_stem_conv(x)
        x = self._call_shuffle_block(x)
        x = self._call_dialated(x)
        x = self.fc1(x)
        x = tf.nn.dropout(x, 0.5)
        return x
        