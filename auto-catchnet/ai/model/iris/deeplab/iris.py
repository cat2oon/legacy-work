from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.layers import *
from keras.models import Model
from keras.utils import conv_utils
from keras.applications import imagenet_utils


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]), align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0], self.output_size[1]), align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inv_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

    x = inputs
    prefix = 'expanded_conv_'

    if block_id:
        prefix = 'expanded_conv_{}_'.format(block_id)
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False,
                        padding='same', dilation_rate=(rate, rate), name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same',
               use_bias=False, activation=None, name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    return x


def make_iris_deeplab_v3(input_tensor=None, input_shape=(112, 112, 3), classes=2, alpha=1.):
    """
    input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
    alpha: controls the width of the MobileNetV2 network. This is known as the width multiplier in the MobileNetV2 paper
        - If `alpha` < 1.0, proportionally decreases the number of filters in each layer.
        - If `alpha` > 1.0, proportionally increases the number of filters in each layer.
        - If `alpha` = 1, default number of filters from the paper are used at each layer.
    """

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    first_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_filters, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv')(img_input)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    x = _inv_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0, skip_connection=False)
    x = _inv_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1, skip_connection=False)
    x = _inv_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2, skip_connection=True)

    x = _inv_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3, skip_connection=False)
    x = _inv_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4, skip_connection=True)
    x = _inv_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inv_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=6, skip_connection=False)
    x = _inv_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=7, skip_connection=True)
    x = _inv_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=8, skip_connection=True)
    x = _inv_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=9, skip_connection=True)

    x = _inv_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=10, skip_connection=False)
    x = _inv_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=11, skip_connection=True)
    x = _inv_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=12, skip_connection=True)

    x = _inv_res_block(x, filters=160, alpha=alpha, stride=1, rate=2, expansion=6, block_id=13, skip_connection=False)
    x = _inv_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=14, skip_connection=True)
    x = _inv_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=15, skip_connection=True)

    x = _inv_res_block(x, filters=320, alpha=alpha, stride=1, rate=4, expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling (ASPP)
    # Image Feature branch
    output_stride = 8  # determines input_shape/feature_extractor_output ratio
    output_size = (int(np.ceil(input_shape[0] / output_stride)), int(np.ceil(input_shape[1] / output_stride)))

    b4 = AveragePooling2D(pool_size=output_size)(x)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling(output_size)(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    x = Concatenate()([b4, b0])
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    # x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # decoder
    x = Conv2D(classes, (1, 1), padding='same', name='iris_logits_semantic')(x)
    print(x.shape)
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='iris-deeplab-v3+')

    return model


def preprocess_input(x):
    """ TODO: Check this function is call from somewhere implicitly """
    return imagenet_utils.preprocess_input(x, mode='tf')
