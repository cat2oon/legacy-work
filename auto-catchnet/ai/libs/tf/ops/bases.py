from ai.libs.tf.ops.convs import *


def identity(x):
    with tf.variable_scope("id"):
        x = tf.identity(x)
    return x


def tf_print(x, data=None, msg="", num_sum=10):
    if data is None:
        data = [x]
    x = tf.Print(x, data, msg, summarize=num_sum)
    return x


def tf_int(number, name):
    x = tf.constant(number, dtype=tf.int32, name=name)
    return x


def tf_float(number, name):
    x = tf.constant(number, dtype=tf.float32, name=name)
    return x
