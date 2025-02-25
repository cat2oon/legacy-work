import tensorflow as tf


def get_dist_loss(inputs, preds):
    pred_x, pred_y = preds
    cam_x, cam_y = inputs["cam_x"], inputs["cam_y"]

    labels = tf.transpose(tf.stack([cam_x, cam_y]))
    preds = tf.transpose(tf.stack([pred_x, pred_y]))
    mse = tf.losses.mean_squared_error(labels, preds)

    return mse


def get_orientation_cx_loss(inputs, preds):
    _, _, pred_o = preds
    labels = tf.one_hot(inputs["orientation"], 4)
    loss = tf.losses.softmax_cross_entropy(labels, pred_o)
    return loss
