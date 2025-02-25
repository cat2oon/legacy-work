import tensorflow as tf


def get_init_ops(data_iterator):
    init_ops = [tf.global_variables_initializer()]

    if data_iterator is not None:
        init_ops.append(data_iterator.initializer)

    return init_ops


def get_or_make_global_step():
    # increment op은 tf.optimizer::apply_gradients 사용
    global_step = tf.train.get_or_create_global_step()
    return global_step
