import tensorflow as tf


def wrap_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


def feature_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=wrap_to_list(value)))


def feature_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=wrap_to_list(value)))


def feature_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=wrap_to_list(value)))