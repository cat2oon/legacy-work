from cheat.tfs import *


class HeadModule():
    
    @classmethod
    def create(cls):
        chs = [32, 32//2, 32, 32, 32, 32]
        input_frame = Input(shape=(224, 224, 3), name='input_frame')
        
        # stem conv
        x = conv3x3(chs[0], name='stem_conv1')(input_frame)
        x = BatchNorm(name='stem_bn1')(x)
        
        # ShuffleNet
        num_filters = chs[1]
        
        """ Shuffle Stage 1 """
        # ShuffleNetDownSampleV2 - left branch
        xl = dw_conv3x3(strides=(2,2), name='dw_conv_l1_s1')(x)
        xl = InstanceNorm(name='in_norm_l1_s1')(xl)
        xl = pw_conv1x1(num_filters,   name='pw_conv_l1_s1')(xl)
        
        # ShuffleNetDownSampleV2 - right branch
        xr = pw_conv1x1(num_filters,   name='pw_conv_r1_s1')(x)
        xr = InstanceNorm(name='in_norm_r1_s1')(xr)
        xr = dw_conv3x3(strides=(2,2), name='dw_conv_r1_s1')(xr)
        xr = InstanceNorm(name='in_norm_r2_s1')(xr)
        xr = pw_conv1x1(num_filters,   name='pw_conv_r2_s1')(xr)
        
        # ShuffleNetDownSampleV2 - concat
        x = concatenate([xl, xr], axis=3)
        x = InstanceNorm(name='in_norm_f1_s1')(x)
        x = tf.nn.relu6(x)
        x = channel_shuffle(x)
        
        # ShuffleNetBasicUnitV2 split
        xl, xr = tf.split(x, 2, axis=3)
        
        # ShuffleNetBasicUnitV2 
        xr = pw_conv1x1(num_filters, name='pw_conv1_s1')(xr)
        xr = InstanceNorm(name='in_norm1_s1')(xr)
        xr = tf.nn.relu6(xr)
        xr = dw_conv3x3(name='dw_conv1_s1')(xr)
        xr = InstanceNorm(name='in_norm2_s1')(xr)
        xr = pw_conv1x1(num_filters, name='pw_conv2_s1')(xr)
        xr = InstanceNorm(name='in_norm3_s1')(xr)
        xr = tf.nn.relu6(xr)
        x = concatenate([xl, xr], axis=3)
        x = channel_shuffle(x)
        
        """ Shuffle Stage 2 """
        num_filters = chs[1]

        # ShuffleNetDownSampleV2 - left branch
        xl = dw_conv3x3(strides=(2,2), name='dw_conv_l1_s2')(x)
        xl = InstanceNorm(name='in_norm_l1_s2')(xl)
        xl = pw_conv1x1(num_filters,   name='pw_conv_l1_s2')(xl)

        # ShuffleNetDownSampleV2 - right branch
        xr = pw_conv1x1(num_filters,   name='pw_conv_r1_s2')(x)
        xr = InstanceNorm(name='in_norm_r1_s2')(xr)
        xr = dw_conv3x3(strides=(2,2), name='dw_conv_r1_s2')(xr)
        xr = InstanceNorm(name='in_norm_r2_s2')(xr)
        xr = pw_conv1x1(num_filters,   name='pw_conv_r2_s2')(xr)

        # ShuffleNetDownSampleV2 - concat
        x = concatenate([xl, xr], axis=3)
        x = InstanceNorm(name='in_norm_f1_s2')(x)
        x = tf.nn.relu6(x)
        x = channel_shuffle(x)

        # ShuffleNetBasicUnitV2 split
        xl, xr = tf.split(x, 2, axis=3)

        # ShuffleNetBasicUnitV2 
        xr = pw_conv1x1(num_filters, name='pw_conv1_s2')(xr)
        xr = InstanceNorm(name='in_norm1_s2')(xr)
        xr = tf.nn.relu6(xr)
        xr = dw_conv3x3(name='dw_conv1_s2')(xr)
        xr = InstanceNorm(name='in_norm2_s2')(xr)
        xr = pw_conv1x1(num_filters, name='pw_conv2_s2')(xr)
        xr = InstanceNorm(name='in_norm3_s2')(xr)
        xr = tf.nn.relu6(xr)
        x = concatenate([xl, xr], axis=3)
        x = channel_shuffle(x)
            
        """ dilation """
        # dilated fomoro.com/research/article/receptive-field-calculator
        # 56 -> [2,4,8,12] RF-size: 53
        x = tf.keras.layers.Conv2D(chs[2], kernel_size=(3,3), dilation_rate=(2,2),  
                                   activation='selu', name='dl_conv1')(x)
        x = BatchNorm(name='dl_bn1')(x)
        x = tf.keras.layers.Conv2D(chs[3], kernel_size=(3,3), dilation_rate=(4,4),  
                                   activation='selu', name='dl_conv2')(x)
        x = BatchNorm(name='dl_bn2')(x)
        x = tf.keras.layers.Conv2D(chs[4], kernel_size=(3,3), dilation_rate=(8,8),
                                   activation='selu', name='dl_conv3')(x)
        x = BatchNorm(name='dl_bn3')(x)        
        x = tf.keras.layers.Conv2D(chs[5], kernel_size=(3,3), dilation_rate=(12,12),  
                                   activation='selu', name='dl_conv4')(x)
        x = BatchNorm(name='dl_bn4')(x) 
        fv = GlobalAveragePooling2D(name='gap')(x)
        
        """ regression net """ 
        x1 = Dense(256, activation='selu', kernel_regularizer=l2_reg(0.001))(fv)
        qx = Dense(1,  activation='linear',kernel_regularizer=l2_reg(0.001), name='qx')(x1)
    
        y1 = Dense(256, activation='selu', kernel_regularizer=l2_reg(0.001))(fv)
        qy = Dense(1,  activation='linear',kernel_regularizer=l2_reg(0.001), name='qy')(y1)
    
        z1 = Dense(256, activation='selu', kernel_regularizer=l2_reg(0.001))(fv)
        qz = Dense(1,  activation='linear',kernel_regularizer=l2_reg(0.001), name='qz')(z1)

        w1 = Dense(256, activation='selu', kernel_regularizer=l2_reg(0.001))(fv)
        qw = Dense(1,  activation='linear',kernel_regularizer=l2_reg(0.001), name='qw')(w1)   
        
        """ ranking nets [qx/qy/qz/ 6/6/19] 
            ORIGINAL>
            Kx {-60', -40', -20', 20', 40', 60'}
            Ky {-60', -40', -20', 20', 40', 60'}
            Kz {-81', -72', ..., 0, ..., 72', 81'}
        """ 
        qx_1 = Dense(2, activation='relu', name='qx-t1')(fv)
        qx_2 = Dense(2, activation='relu', name='qx-t2')(fv)
        qx_3 = Dense(2, activation='relu', name='qx-t3')(fv)
        qx_4 = Dense(2, activation='relu', name='qx-t4')(fv)
        qx_5 = Dense(2, activation='relu', name='qx-t5')(fv)
        qx_6 = Dense(2, activation='relu', name='qx-t6')(fv)
        
        qy_1 = Dense(2, activation='relu', name='qy-t1')(fv)
        qy_2 = Dense(2, activation='relu', name='qy-t2')(fv)
        qy_3 = Dense(2, activation='relu', name='qy-t3')(fv)
        qy_4 = Dense(2, activation='relu', name='qy-t4')(fv)
        qy_5 = Dense(2, activation='relu', name='qy-t5')(fv)
        qy_6 = Dense(2, activation='relu', name='qy-t6')(fv)        
        
        qz_1  = Dense(2, activation='relu', name='qz-t1')(fv)
        qz_2  = Dense(2, activation='relu', name='qz-t2')(fv)
        qz_3  = Dense(2, activation='relu', name='qz-t3')(fv)
        qz_4  = Dense(2, activation='relu', name='qz-t4')(fv)
        qz_5  = Dense(2, activation='relu', name='qz-t5')(fv)
        qz_6  = Dense(2, activation='relu', name='qz-t6')(fv)
        qz_7  = Dense(2, activation='relu', name='qz-t7')(fv)
        qz_8  = Dense(2, activation='relu', name='qz-t8')(fv)
        qz_9  = Dense(2, activation='relu', name='qz-t9')(fv)
        qz_10 = Dense(2, activation='relu', name='qz-t10')(fv)
        qz_11 = Dense(2, activation='relu', name='qz-t11')(fv)
        qz_12 = Dense(2, activation='relu', name='qz-t12')(fv)
        qz_13 = Dense(2, activation='relu', name='qz-t13')(fv)
        qz_14 = Dense(2, activation='relu', name='qz-t14')(fv)
        qz_15 = Dense(2, activation='relu', name='qz-t15')(fv)
        qz_16 = Dense(2, activation='relu', name='qz-t16')(fv)
        
        outs = [qx, qy, qz, qw, 
                qx_1, qx_2, qx_3, qx_4, qx_5, qx_6, qy_1, qy_2, qy_3, qy_4, qy_5, qy_6,  
                qz_1, qz_2, qz_3, qz_4, qz_5, qz_6, qz_7, qz_8, qz_9, qz_10, qz_11, qz_12, 
                qz_13, qz_14, qz_15, qz_16]
        
        return Model(inputs=[input_frame], outputs=outs, name='head_shot_net')

    
