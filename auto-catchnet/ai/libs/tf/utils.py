import numpy as np
import tensorflow as tf


def count_model_params(tf_variables):
    num_vars = 0
    for var in tf_variables:
        num_vars += np.prod([dim.value for dim in var.get_shape()])
    return num_vars


def print_tf_vars(tf_variables):
    print("")
    print("=" * 80)
    for var in tf_variables:
        print(var)
    print("=" * 80)


def count_num_batches(num_examples, batch_size):
    return (num_examples + batch_size - 1) // batch_size


def get_trainable_vars(name, exclude_aux=True):
    tf_vars = [var for var in tf.trainable_variables() if var.name.startswith(name)]
    if exclude_aux:
        tf_vars = [var for var in tf_vars if "aux_head" not in var.name]
    return tf_vars


def get_aux_head_vars(name):
    return [var for var in tf.trainable_variables() if var.name.startswith(name) and "aux_head" in var.name]
