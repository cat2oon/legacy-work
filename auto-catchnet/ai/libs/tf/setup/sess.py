import tensorflow as tf


def get_sess_config():
    tf_config = tf.ConfigProto()

    tf_config.allow_soft_placement = True
    tf_config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    return tf_config
