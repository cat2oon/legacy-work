from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from nets.layers import *
from tensorflow.keras import Model
from tensorflow.keras.layers import *
    
    
"""
    SNet
"""
class SNet(tf.keras.Model):
    
    @classmethod
    def create(cls, act='selu'):
        Norm = BatchNormalization
        Basic = ShuffleNetBasicUnitV2
        SDown = ShuffleNetDownSampleUnitV2
        
        # inputs
        input_sp = Input(shape=(56), name="support_input")
        input_le = Input(shape=(128, 128, 3), name='left_eye_patch')
        input_re = Input(shape=(128, 128, 3), name='right_eye_patch')
        
        # support
        spx = Dense(64, use_bias=False, name='support_fc1')(input_sp)
        spx = Norm()(spx)
        spx = Activation(act)(spx)
        spx = Dense(32, activation=act, use_bias=False, name='support_fc2')(spx)
        spx = Dense(32, activation=act, use_bias=False, name='support_fc3')(spx)
        
        sf, nf, lf = 32, 64, 64

        # left
        lex = Conv2D(filters=sf, kernel_size=(3,3), strides=(2,2), use_bias=False,
                     activation='selu', name='l_stem_conv')(input_le)
        lex = MaxPool2D(padding='SAME', strides=(2,2))(lex)
        lex = tf.nn.relu6(lex)
        lex = tfa.layers.InstanceNormalization(name='l_stem_norm')(lex)

        lex = SDown(out_channel=nf, name='stage_l2')(lex)
        lex = Basic(out_channel=nf, name='stage_l2')(lex)
        lex = SDown(out_channel=nf, name='stage_l3')(lex)
        lex = Basic(out_channel=nf, name='stage_l3')(lex)
        lex = SDown(out_channel=lf, name='stage_l4')(lex)
        lex = Basic(out_channel=lf, name='stage_l4')(lex)
        lex = GlobalAveragePooling2D()(lex)

        # right
        rex = Conv2D(filters=sf, kernel_size=(3,3), strides=(2,2), use_bias=False,
                     activation='selu', name='r_stem_conv')(input_le)
        rex = MaxPool2D(padding='SAME', strides=(2,2))(rex)
        rex = tf.nn.relu6(rex)
        rex = tfa.layers.InstanceNormalization(name='r_stem_norm')(rex)

        rex = SDown(out_channel=nf, name='stage_r2')(rex)
        rex = Basic(out_channel=nf, name='stage_r2')(rex)
        rex = SDown(out_channel=nf, name='stage_r3')(rex)
        rex = Basic(out_channel=nf, name='stage_r3')(rex)
        rex = SDown(out_channel=lf, name='stage_r4')(rex)
        rex = Basic(out_channel=lf, name='stage_r4')(rex)
        rex = GlobalAveragePooling2D()(rex)
        
        # final
        z = Concatenate(name='both_eye')([lex, rex])
        z = Dense(64, activation=act, use_bias=False,  name='eye_fc1')(z)
        
        z = Concatenate(name='bottleneck')([z, spx])
        z = Dense(8, activation='linear', use_bias=True,  name='final_fc1')(z)
        z = Dense(2, activation='linear', use_bias=False, name='final_fc2')(z)
        
        model = Model(inputs=[input_le, input_re, input_sp], outputs=z)
        return model

    

