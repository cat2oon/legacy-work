import tensorflow as tf
from keras import initializers

xavier = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)


def constant_init(constant=0.0):
    return tf.constant_initializer(constant, dtype=tf.float32)


def weight_initializer(type_for=None, seed=None):
    if type_for == "lstm":
        return initializers.random_uniform(minval=-0.1, maxval=0.1)
    return initializers.he_normal(seed=seed)
