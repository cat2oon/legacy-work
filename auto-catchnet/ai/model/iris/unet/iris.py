from keras.engine.input_layer import Input
from keras.layers import core
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model


def make_uiris_net(input_shape=(112, 112, 3)):
    h, w = input_shape[:2]
    inputs = Input(shape=input_shape)

    # 112 x 112 x 32
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=3)     # 56 x 56 x 192 (64 + 128)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=3)     # 112 x 112 x 96 (32 + 64)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same')(conv5)
    conv6 = core.Reshape((2, h * w))(conv6)
    conv6 = core.Permute((2, 1))(conv6)         # 112 x 112 x 2
    class_map = core.Activation('softmax', name="class-map-softmax")(conv6)

    model = Model(inputs=inputs, outputs=class_map)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
