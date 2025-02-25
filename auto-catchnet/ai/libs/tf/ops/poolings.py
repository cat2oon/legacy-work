import tensorflow as tf

from ai.libs.tf.ops.convs import conv_1


def spatial_max_pool(x, kernel_size, stride, is_training, out_filters=None):
    inp_c = x.shape[3].value if out_filters is None else out_filters
    with tf.variable_scope("spatial_max_pool_{}_{}".format(kernel_size, stride)):
        x = conv_1(x, inp_c, is_training)
        x = tf.nn.max_pool(x, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding="SAME")
    return x


def spatial_avg_pool(x, kernel_size, stride, is_training, out_filters=None):
    inp_c = x.shape[3].value if out_filters is None else out_filters
    with tf.variable_scope("spatial_avg_pool_{}_{}".format(kernel_size, stride)):
        x = conv_1(x, inp_c, is_training)
        x = tf.nn.avg_pool(x, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding="SAME")
    return x


def avg_pooling(x, kernel_size, stride, padding="VALID"):
    return tf.layers.average_pooling2d(x, kernel_size, stride, padding)


def max_pooling(x, kernel_size, stride, padding="VALID"):
    return tf.layers.max_pooling2d(x, kernel_size, stride, padding)


def global_avg_pool(x):
    x = tf.reduce_mean(x, [1, 2])  # global avg for w x h
    return x
