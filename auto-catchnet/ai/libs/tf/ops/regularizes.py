import tensorflow as tf
from keras import regularizers
from tensorflow.python.training import moving_averages

from ai.libs.tf.ops.initializers import constant_init


def batch_norm(x, is_training, name="bn", decay=0.9, epsilon=1e-5):
    shape = [x.shape[3]]
    reuse = False if is_training else True

    with tf.variable_scope(name, reuse=reuse):
        offset = tf.get_variable("offset", shape, initializer=constant_init(0.0))
        scale = tf.get_variable("scale", shape, initializer=constant_init(1.0))
        moving_mean = tf.get_variable("moving_mean", shape, trainable=False, initializer=constant_init(0.0))
        moving_var = tf.get_variable("moving_variance", shape, trainable=False, initializer=constant_init(1.0))

        x, mean, variance = tf.nn.fused_batch_norm(x, scale, offset,
                                                   mean=None if is_training else moving_mean,
                                                   variance=None if is_training else moving_var,
                                                   epsilon=epsilon,
                                                   is_training=is_training)
        if is_training:
            update_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_variance = moving_averages.assign_moving_average(moving_var, variance, decay)
            with tf.control_dependencies([update_mean, update_variance]):
                x = tf.identity(x)
    return x


def batch_norm_with_mask(x, is_training, mask, num_channels, name="bn", decay=0.9, epsilon=1e-3):
    shape = [num_channels]
    indices = tf.where(mask)
    indices = tf.to_int32(indices)
    indices = tf.reshape(indices, [-1])

    with tf.variable_scope(name, reuse=None if is_training else True):
        offset = tf.get_variable("offset", shape, initializer=constant_init(0.0))
        scale = tf.get_variable("scale", shape, initializer=constant_init(1.0))
        offset = tf.boolean_mask(offset, mask)
        scale = tf.boolean_mask(scale, mask)

        moving_mean = tf.get_variable("moving_mean", shape, trainable=False, initializer=constant_init(0.0))
        moving_variance = tf.get_variable("moving_variance", shape, trainable=False, initializer=constant_init(1.0))

        if is_training:
            x, mean, variance = tf.nn.fused_batch_norm(x, scale, offset, epsilon=epsilon, is_training=True)
            mean = (1.0 - decay) * (tf.boolean_mask(moving_mean, mask) - mean)
            variance = (1.0 - decay) * (tf.boolean_mask(moving_variance, mask) - variance)
            update_mean = tf.scatter_sub(moving_mean, indices, mean, use_locking=True)
            update_variance = tf.scatter_sub(moving_variance, indices, variance, use_locking=True)
            with tf.control_dependencies([update_mean, update_variance]):
                x = tf.identity(x)
        else:
            masked_moving_mean = tf.boolean_mask(moving_mean, mask)
            masked_moving_variance = tf.boolean_mask(moving_variance, mask)
            x, _, _ = tf.nn.fused_batch_norm(x, scale, offset,
                                             mean=masked_moving_mean,
                                             variance=masked_moving_variance,
                                             epsilon=epsilon,
                                             is_training=False)
    return x


def l2_regularizer(trainable_vars=None, alpha=0.001):
    if trainable_vars is None:
        trainable_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * alpha
    return l2_loss


def weight_regularizer(rate=1e-4):
    return regularizers.l2(rate)
