import tensorflow as tf

from ai.nas.utils.tfs import get_channel
from ai.libs.tf.ops.regularizes import batch_norm
from ai.libs.tf.ops.weights import create_weight

"""
 TODO:
 - scope는 외부에서 정의하도록 변경할 것 
"""


def stem_conv(x, out_filters, is_training, ch=3, filter_size=3, stride=1, dilations=1, suffix=""):
    with tf.variable_scope("stem_conv_{}".format(suffix)):
        w = create_weight("w", [filter_size, filter_size, ch, out_filters * 3])  # 3x3 conv
        x = tf.nn.conv2d(x, w, [1, stride, stride, 1], "SAME", dilations=[1, dilations, dilations, 1])  # 1x1 stride
        x = batch_norm(x, is_training)
    return x


def conv_1(x, out_filters, is_training, bn=True, non_linear=True, pad="SAME", name="inp_conv_1"):
    # TODO : deprecated - no var scope in func
    inp_c = x.shape[3].value
    with tf.variable_scope(name):  # handle channel correlation
        w = create_weight("w", [1, 1, inp_c, out_filters])  # reduction effect
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], pad)  # 1x1 conv
        if bn:
            x = batch_norm(x, is_training)  # bn
        if non_linear:
            x = tf.nn.relu(x)  # non-linearity (xception 차이 지점)
    return x


# TODO : reduction 용 호출은 별도의 함수로 분리 할 것
def conv_1_f(x, out_filters, is_training, bn=True, pad="SAME",
             prev_non_linearity=False, last_non_linearity=True, var_name="w"):
    # 1x1 conv (dimension reduction effect, cross channel learning, more non-linearity)
    # Going deeper with convolution (arxiv.org/pdf/1409.4842v1.pdf)
    # https://iamaaditya.github.io/2016/03/one-by-one-convolution/ (git 블로그글)
    inp_c = get_channel(x)
    w = create_weight(var_name, [1, 1, inp_c, out_filters])  # reduction effect
    if prev_non_linearity:
        x = tf.nn.relu(x)  # non-linearity (xception 순서)
    x = tf.nn.conv2d(x, w, [1, 1, 1, 1], pad)  # 1x1 conv
    if bn:
        x = batch_norm(x, is_training)
    if last_non_linearity:
        x = tf.nn.relu(x)  # non-linearity (xception 역순 차이)
    return x


def conv_1_multi(x, num_blocks, block_id, out_filters, pad="SAME"):
    """
    멀티 block 으로 이루어진 복합 웨이트를 한번에 생성해놓고
    필요한 block_id로 접근하는 방식의 convolution op
    """
    inp_ch = x.shape[3].value
    w = create_weight("w", [num_blocks, inp_ch * out_filters])
    w = w[block_id]
    w = tf.reshape(w, [1, 1, inp_ch, out_filters])
    x = tf.nn.relu(x)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=pad)
    x = batch_norm(x, is_training=True)
    return x


def sep_conv(x, out_filters, filter_size, is_training, ch_mul=1, name='sc'):
    # TODO : filter_size : 숫자 하나라면 정방 사이즈, 2개면 tuple or []
    with tf.variable_scope("{}_out_conv_{}".format(name, filter_size)):  # handle spatial correlation
        x = conv_1(x, out_filters, is_training, name=name)
        w_depth = create_weight("w_depth", [filter_size, filter_size, out_filters, ch_mul])  # expand
        w_point = create_weight("w_point", [1, 1, out_filters * ch_mul, out_filters])  # reduction
        x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1], padding="SAME")
        x = batch_norm(x, is_training)
        x = tf.nn.relu(x)
    return x


def sep_conv_f(x, out_filters, filter_size, is_training, strides=None,
               use_reduce=True, prev_non_linearity=False, last_non_linearity=True, ch_mul=1):
    if strides is None:
        strides = [1, 1, 1, 1]
    if use_reduce:
        x = conv_1(x, out_filters, is_training)
        inp_c = out_filters
    else:
        inp_c = get_channel(x)

    w_depth = create_weight("w_depth", [filter_size, filter_size, inp_c, ch_mul])
    w_point = create_weight("w_point", [1, 1, inp_c * ch_mul, out_filters])
    if prev_non_linearity:
        x = tf.nn.relu(x)
    x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=strides, padding="SAME")
    x = batch_norm(x, is_training)
    if last_non_linearity:
        x = tf.nn.relu(x)

    return x


def sep_conv_multi(x, num_blocks, block_id, out_filters, filter_size):
    inp_ch = x.shape[3].value

    w_depth = create_weight("w_depth", [num_blocks, filter_size * filter_size * inp_ch])
    w_depth = w_depth[block_id, :]
    w_depth = tf.reshape(w_depth, [filter_size, filter_size, inp_ch, 1])

    w_point = create_weight("w_point", [num_blocks, inp_ch * out_filters])
    w_point = w_point[block_id, :]
    w_point = tf.reshape(w_point, [1, 1, inp_ch, out_filters])

    with tf.variable_scope("bn"):
        zero_init = tf.initializers.zeros(dtype=tf.float32)
        one_init = tf.initializers.ones(dtype=tf.float32)
        offset = create_weight("offset", [num_blocks, out_filters], initializer=zero_init)
        scale = create_weight("scale", [num_blocks, out_filters], initializer=one_init)
        offset = offset[block_id]
        scale = scale[block_id]

    x = tf.nn.relu(x)
    x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1], padding="SAME")
    x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, epsilon=1e-5, is_training=True)

    return x


def spatial_conv(x, out_filters, filter_size, is_training):
    x = conv_1(x, out_filters, is_training)
    x = conv(x, out_filters, filter_size, is_training)
    return x


def conv(x, out_filters, filter_size, is_training):
    inp_c = x.shape[3].value
    w = create_weight("w", [filter_size, filter_size, inp_c, out_filters])
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
    x = batch_norm(x, is_training)
    x = tf.nn.relu(x)
    return x


def apply_drop_path(x, layer_id, keep_prob, num_layers, global_step, num_train_steps):
    layer_ratio = float(layer_id + 1) / (num_layers + 2)
    keep_prob = 1.0 - layer_ratio * (1.0 - keep_prob)
    step_ratio = tf.to_float(global_step + 1) / tf.to_float(num_train_steps)
    step_ratio = tf.minimum(1.0, step_ratio)
    keep_prob = 1.0 - step_ratio * (1.0 - keep_prob)
    return drop_path(x, keep_prob)


def drop_path(x, keep_prob):
    batch_size = tf.shape(x)[0]
    noise_shape = [batch_size, 1, 1, 1]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
    binary_tensor = tf.floor(random_tensor)
    x = tf.div(x, keep_prob) * binary_tensor

    return x
