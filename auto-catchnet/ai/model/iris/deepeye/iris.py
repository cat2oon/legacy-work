
from keras import backend as K
from keras.engine.input_layer import Input
from keras.layers import core
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model


def make_conv_unit(x, num_filters):
    cu = x
    cu = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(cu)
    cu = BatchNormalization(momentum=0.99)(cu)
    cu = core.Activation('relu')(cu)
    cu = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(cu)
    cu = BatchNormalization(momentum=0.99)(cu)
    cu = Concatenate(axis=3)([cu, x])
    cu = core.Activation('relu')(cu)

    return cu


def make_stride_unit(x, num_filters):
    su = x
    su = Conv2D(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(su)
    su = BatchNormalization(momentum=0.99)(su)
    su = core.Activation('relu')(su)
    su = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(su)
    su = BatchNormalization(momentum=0.99)(su)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(num_filters, [1, 1], strides=(1, 1), padding='same')(x)
    su = Concatenate(axis=3)([su, x])
    su = core.Activation('relu')(su)

    return su


def make_atrous_unit(x, num_filters, dilation):
    au = x
    au = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation, padding='same')(au)
    au = BatchNormalization(momentum=0.99)(au)
    au = core.Activation('relu')(au)
    au = Conv2D(num_filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=dilation, padding='same')(au)
    au = BatchNormalization(momentum=0.99)(au)
    au = Concatenate(axis=3)([au, x])
    au = core.Activation('relu')(au)

    return au


def padding_reshape(layers):
    import tensorflow as tf
    asu0, asu4, asu5 = layers
    shape_asu0 = tf.shape(asu0)
    shape_asu5 = tf.shape(asu5)
    pad_height = shape_asu0[1] - shape_asu5[1]
    pad_width = shape_asu0[2] - shape_asu5[2]
    asu5 = tf.pad(asu5, [[0, 0], [pad_height, 0], [pad_width, 0], [0, 0]])
    asu5 = tf.reshape(asu5, K.shape(asu4))

    return asu5


def make_aspp_unit(x, num_filters):
    asu0 = x
    asu1 = make_atrous_unit(asu0, num_filters, 4)
    asu2 = make_atrous_unit(asu1, num_filters, 8)
    asu3 = make_atrous_unit(asu2, num_filters, 16)
    asu4 = make_conv_unit(asu3, num_filters)
    asu5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(asu4)
    asu5 = Lambda(padding_reshape)([asu0, asu4, asu5])

    asu = Concatenate(axis=3)([asu1, asu2, asu3, asu4, asu5])
    asu = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(asu)
    asu = BatchNormalization(momentum=0.99)(asu)
    asu = core.Activation('relu')(asu)

    return asu


def make_aspp_layers(x, num_filters, num_deep):
    aspp = x
    for i in range(num_deep):
        aspp = make_aspp_unit(aspp, num_filters)
    return aspp


def make_resize_bilinear(resize_shape):
    def resize_bilinear(x):
        import tensorflow as tf
        return tf.image.resize_bilinear(x, resize_shape)
    return resize_bilinear


def make_deep_eye_net(input_shape=(64, 64, 3), num_filters=16, aspp_deep=2, num_classes=1):
    inputs = Input(shape=input_shape)

    x = inputs
    x = make_conv_unit(x, num_filters)
    x = make_stride_unit(x, 2 * num_filters)
    x = make_conv_unit(x, 2 * num_filters)
    x = make_stride_unit(x, 4 * num_filters)
    x = make_aspp_layers(x, 4 * num_filters, num_deep=aspp_deep)
    x = Conv2D(num_classes, kernel_size=(1, 1), padding='same')(x)
    x = Lambda(make_resize_bilinear(input_shape[:2]), name='upsample-bilinear')(x)

    class_map = core.Activation('softmax', name="class-map-softmax")(x)

    model = Model(inputs=inputs, outputs=class_map)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# class DeepEye:
#     def blob_location(self, prob_mask):
#         factor = prob_mask.size / (288.0 * 384.0)
#         params = cv2.SimpleBlobDetector_Params()
#         params.filterByArea = True
#         params.minArea = 100 * factor
#         params.maxArea = 25000 * factor
#         params.filterByConvexity = True
#         params.minConvexity = 0.1

#         detector = cv2.SimpleBlobDetector_create(params)
#
#         found_blob = False
#         prob = 0.5
#         raw_img = prob_mask.copy()
#         while found_blob == False:
#             image = raw_img.copy()
#             image[image < prob] = 0
#             image[image > prob] = 1
#             image = (image * 255).astype('uint8')
#             image = 255 - image
#             keypoints = detector.detect(image)
#
#             if len(keypoints) > 0:
#
#                 blob_sizes = []
#                 for k in keypoints:
#                     blob_sizes.append(k.size)
#                 detection = np.argmax(np.asarray(blob_sizes))
#                 out_coordenate = [int(keypoints[detection].pt[0]), int(keypoints[detection].pt[1])]
#                 found_blob = True
#             else:
#                 out_coordenate = [0, 0]
#                 found_blob = False
#                 prob += -0.05
#                 if prob < 0.05:
#                     found_blob = True
#
#         return out_coordenate

