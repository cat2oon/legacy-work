import tensorflow as tf


def tf_round(x, round_precision=0):
    multiplier = tf.constant(10 ** round_precision, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier
