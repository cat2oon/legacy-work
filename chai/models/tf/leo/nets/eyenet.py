from cheat.tfs import *


class EyeModule():
    
    """ gaze origin eye pos depth regressor """
    @classmethod
    def create_depth_regressor(cls):
        ins = Input(shape=(9), name='input_depth_regressor')
        
        x = Dense(8, kernel_regularizer=l2_reg(0.001))(ins)
        x = tf.nn.selu(x)
        z = Dense(2, kernel_regularizer=l2_reg(0.001))(x)
        
        return Model(inputs=[ins], outputs=z, name='depth_regressor') 
    
    @classmethod
    def create_bec(cls, out_dim):
        input_eye = Input(shape=(48, 96, 2), name='input_eye_bec')
        
        chs = [32, 32//2, 32, 32, 32, 32, 32, 16]
        fv = cls.backbone(chs, input_eye)
        
        # dense
        fv1 = Dense(chs[6], activation='selu', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc1')(fv)
        fv2 = Dense(chs[7], activation='selu', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc2')(fv1)
        z = Dense(out_dim, activation='linear', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc3')(fv2)
    
        return Model(inputs=[input_eye], outputs=[z, fv, fv2]) 
    
    @classmethod
    def create_mono(cls, out_dim):
        input_eye = Input(shape=(48, 96, 1), name='input_mono_eye')
        
        # chs = [16, 16//2, 16, 16, 16, 32, 64, 32]    # 더 좋아졌는데?
        chs = [16, 16//2, 16, 16, 16, 16, 32, 16]      # 이것도 괜찮네
        
        x = cls.backbone(chs, input_eye)
        
        # dense
        x = Dense(chs[6], activation='selu', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc1')(x)
        # x = tf.nn.dropout(x, 0.5)
        x = Dense(chs[7], activation='selu', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc2')(x)
        # x = tf.nn.dropout(x, 0.5)
        z = Dense(out_dim,  activation='linear', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc3')(x)
    
        return Model(inputs=[input_eye], outputs=z)
    
    @classmethod
    def backbone(cls, chs, inputs):
        # stem conv
        x = conv3x3(chs[0], name='stem_conv1')(inputs)
        x = BatchNorm(name='stem_bn1')(x)
        x = tf.nn.relu6(x)
        
        # ShuffleNet
        num_filters = chs[1]
        
        # ShuffleNetDownSampleV2 - left branch
        xl = dw_conv3x3(strides=(2,2), name='dw_conv_l1')(x)
        xl = InstanceNorm(name='in_norm_l1')(xl)
        xl = pw_conv1x1(num_filters,   name='pw_conv_l1')(xl)
        
        # ShuffleNetDownSampleV2 - right branch
        xr = pw_conv1x1(num_filters,   name='pw_conv_r1')(x)
        xr = InstanceNorm(name='in_norm_r1')(xr)
        xr = dw_conv3x3(strides=(2,2), name='dw_conv_r1')(xr)
        xr = InstanceNorm(name='in_norm_r2')(xr)
        xr = pw_conv1x1(num_filters,   name='pw_conv_r2')(xr)
        
        # ShuffleNetDownSampleV2 - concat
        x = concatenate([xl, xr], axis=3)
        x = InstanceNorm(name='in_norm_f1')(x)
        x = tf.nn.relu6(x)
        x = channel_shuffle(x)
        
        # ShuffleNetBasicUnitV2 split
        xl, xr = tf.split(x, 2, axis=3)
        
        # ShuffleNetBasicUnitV2 
        xr = pw_conv1x1(num_filters, name='pw_conv1')(xr)
        xr = InstanceNorm(name='in_norm1')(xr)
        xr = tf.nn.relu6(xr)
        xr = dw_conv3x3(name='dw_conv1')(xr)
        xr = InstanceNorm(name='in_norm2')(xr)
        xr = pw_conv1x1(num_filters, name='pw_conv2')(xr)
        xr = InstanceNorm(name='in_norm3')(xr)
        xr = tf.nn.relu6(xr)
        x = concatenate([xl, xr], axis=3)
        x = channel_shuffle(x)
        
        # dilated fomoro.com/research/article/receptive-field-calculator
        x = tf.keras.layers.Conv2D(chs[2], kernel_size=(3,3), dilation_rate=(2,2),  
                                   activation='selu', name='dl_conv1')(x)
        x = BatchNorm(name='dl_bn1')(x)
        x = tf.keras.layers.Conv2D(chs[3], kernel_size=(3,3), dilation_rate=(2,4),  
                                   activation='selu', name='dl_conv2')(x)
        x = BatchNorm(name='dl_bn2')(x)
        x = tf.keras.layers.Conv2D(chs[4], kernel_size=(3,3), dilation_rate=(3,6),  
                                   activation='selu', name='dl_conv3')(x)
        x = BatchNorm(name='dl_bn3')(x)
        x = tf.keras.layers.Conv2D(chs[5], kernel_size=(3,3), dilation_rate=(4,8), 
                                   activation='selu', name='dl_conv4')(x)
        x = BatchNorm(name='dl_bn4')(x)
        
        fv = GlobalAveragePooling2D(name='gap')(x)
        # fv = Flatten(name='feature')(x)
        
        return fv

    
    @classmethod
    def create_niqab(cls, ctx):
        input_frame = Input(shape=(128, 256, 1), name='niqab_frame')
        
        # stem conv
        x = conv3x3(32, name='stem_conv1')(input_frame)
        x = MaxPool2D(padding='same', strides=(2,2), name='stem_pool1')(x)
        x = BatchNorm(name='stem_bn1')(x)
        
        # ShuffleNet
        num_filters = 64 // 2
        
        # ShuffleNetDownSampleV2 - left branch
        xl = dw_conv3x3(strides=(2,2), name='dw_conv_l1')(x)
        xl = InstanceNorm(name='in_norm_l1')(xl)
        xl = pw_conv1x1(num_filters,   name='pw_conv_l1')(xl)
        
        # ShuffleNetDownSampleV2 - right branch
        xr = pw_conv1x1(num_filters,   name='pw_conv_r1')(x)
        xr = InstanceNorm(name='in_norm_r1')(xr)
        xr = dw_conv3x3(strides=(2,2), name='dw_conv_r1')(xr)
        xr = InstanceNorm(name='in_norm_r2')(xr)
        xr = pw_conv1x1(num_filters,   name='pw_conv_r2')(xr)
        
        # ShuffleNetDownSampleV2 - concat
        x = concatenate([xl, xr], axis=3)
        x = InstanceNorm(name='in_norm_f1')(x)
        x = tf.nn.relu6(x)
        x = channel_shuffle(x)
        
        # ShuffleNetBasicUnitV2 split
        xl, xr = tf.split(x, 2, axis=3)
        
        # ShuffleNetBasicUnitV2 
        xr = pw_conv1x1(num_filters, name='pw_conv1')(xr)
        xr = InstanceNorm(name='in_norm1')(xr)
        xr = tf.nn.relu6(xr)
        xr = dw_conv3x3(name='dw_conv1')(xr)
        xr = InstanceNorm(name='in_norm2')(xr)
        xr = pw_conv1x1(num_filters, name='pw_conv2')(xr)
        xr = InstanceNorm(name='in_norm3')(xr)
        xr = tf.nn.relu6(xr)
        x = concatenate([xl, xr], axis=3)
        x = channel_shuffle(x)
        
        # dilated fomoro.com/research/article/receptive-field-calculator
        x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), dilation_rate=(2,2),  
                                   activation='selu', name='dl_conv1')(x)
        x = BatchNorm(name='dl_bn1')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), dilation_rate=(3,4),  
                                   activation='selu', name='dl_conv2')(x)
        x = BatchNorm(name='dl_bn2')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), dilation_rate=(4,6),  
                                   activation='selu', name='dl_conv3')(x)
        x = BatchNorm(name='dl_bn3')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), dilation_rate=(5,12), 
                                   activation='selu', name='dl_conv4')(x)
        x = BatchNorm(name='dl_bn4')(x)
        
        # x = GlobalAveragePooling2D(name='gap')(x)
        x = Flatten(name='feature')(x)
        
        # dense
        x = Dense(64, activation='selu', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc1')(x)
        x = tf.nn.dropout(x, 0.5)
        x = Dense(32, activation='selu', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc2')(x)
        x = tf.nn.dropout(x, 0.5)
        x = Dense(4,  activation='selu', kernel_regularizer=l2_reg(0.001), 
                  use_bias=False, name='fc3')(x)
    
        return Model(inputs=[input_frame], outputs=x, name='niqab')
    