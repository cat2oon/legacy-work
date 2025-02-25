from keras.engine.input_layer import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


def make_simple_net(input_shape=(112, 112, 3)):
    inputs = Input(shape=input_shape)

    # 112 x 112 x 128
    g1 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv1')(inputs)
    g1 = Activation('relu', name='conv1_activation')(g1)
    g1 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2')(g1)
    g1 = Activation('relu', name='conv2_activation')(g1)
    g1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(g1)

    # 56 x 56 x 256
    g2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3')(g1)
    g2 = Activation('relu', name='conv3_activation')(g2)
    g2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv4')(g2)
    g2 = Activation('relu', name='conv4_activation')(g2)
    g2 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv5')(g2)
    g2 = Activation('relu', name='conv5_activation')(g2)
    g2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(g2)

    # 28 x 28 x 512
    g3 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv6')(g2)
    g3 = Activation('relu', name='conv6_activation')(g3)
    g3 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv7')(g3)
    g3 = Activation('relu', name='conv7_activation')(g3)
    g3 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv8')(g3)
    g3 = Activation('relu', name='conv8_activation')(g3)
    g3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(g3)

    # 14 x 14 x 512
    g4 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv9')(g3)
    g4 = Activation('relu', name='conv9_activation')(g4)
    g4 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv10')(g4)
    g4 = Activation('relu', name='conv10_activation')(g4)
    g4 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv11')(g4)
    g4 = Activation('relu', name='conv11_activation')(g4)
    g4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(g4)

    # 7 x 7 x 512
    x = GlobalAveragePooling2D()(g4)
    x = Dense(128)(x)
    x = Dense(2, name="pred-ellipse-param")(x)

    model = Model(inputs, x, name='iris-simple')

    return model
