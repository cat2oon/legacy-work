from ai.libs.tf.ops.cells import *
from ai.nas.layer.nas import factorized_reduction, maybe_calibrate_size
from ai.libs.tf.ops.custom.fcs import fully_connect
from ai.libs.tf.ops.poolings import global_avg_pool


def build_model(frames, eyes, cam_to_x, cam_to_y, is_training=True, keep_prob=0.7, name="exp01"):
    fx = frame_model(frames, is_training)
    ex = eye_model(eyes, is_training)

    fxx = global_avg_pool(fx)
    exx = global_avg_pool(ex)

    cam_to_x = tf.expand_dims(cam_to_x, 1)
    cam_to_y = tf.expand_dims(cam_to_y, 1)
    x = tf.concat([fxx, exx, cam_to_x, cam_to_y], axis=1)

    if is_training:
        x = tf.nn.dropout(x, keep_prob)

    pred_x = fully_connect(x, 1, name="fc_x")
    pred_y = fully_connect(x, 1, name="fc_y")

    pred_x = tf.squeeze(pred_x)
    pred_y = tf.squeeze(pred_y)

    return pred_x, pred_y


def frame_model(x, is_training, out_filters=32):
    with tf.variable_scope("frame_model"):
        x0 = stem_conv(x, out_filters, is_training, filter_size=5, dilations=2)
        x1 = factorized_reduction(x0, out_filters * 2, 2, is_training, idx=1)

        idx = 2
        a0, a1 = maybe_calibrate_size(x0, x1, out_filters, is_training, idx=idx)
        a2 = reduction_cell(a0, a1, out_filters, is_training, idx=idx)
        a3 = reduction_cell(a1, a2, out_filters, is_training, idx="{}_{}".format(idx, 1))

        a4 = convolution_cell(a2, a3, out_filters, is_training, idx=idx)
        ax = convolution_cell(a3, a4, out_filters, is_training, idx="{}_{}".format(idx, 1))

        idx = 3
        b0 = factorized_reduction(ax, out_filters * 2, 2, is_training, idx=idx)
        b1, b2 = maybe_calibrate_size(ax, b0, out_filters, is_training, idx=idx)
        b3 = reduction_cell(b1, b2, out_filters, is_training, idx=idx)

        b4 = convolution_cell(b2, b3, out_filters, is_training, idx=idx)
        bx = convolution_cell(b3, b4, out_filters, is_training, idx="{}_{}".format(idx, 1))

        idx = 4
        c0 = factorized_reduction(bx, out_filters * 2, 2, is_training, idx=idx)
        c1, c2 = maybe_calibrate_size(bx, c0, out_filters, is_training, idx=idx)
        c3 = reduction_cell(c1, c2, out_filters, is_training, idx=idx)
        cx = convolution_cell(c2, c3, out_filters, is_training, idx=idx)

    return cx


def eye_model(x, is_training, out_filters=32):
    with tf.variable_scope("eye_model"):
        x = stem_conv(x, out_filters, is_training, ch=1)

        idx = 1
        a0 = factorized_reduction(x, out_filters * 2, 2, is_training, idx=idx)
        a1, a2 = maybe_calibrate_size(x, a0, out_filters, is_training, idx=idx)
        a3 = reduction_cell(a1, a2, out_filters, is_training, idx=idx)
        ax = convolution_cell(a2, a3, out_filters, is_training, idx=idx)

        idx = 2
        b0 = factorized_reduction(ax, out_filters * 2, 2, is_training, idx=idx)
        b1, b2 = maybe_calibrate_size(ax, b0, out_filters, is_training, idx=idx)
        b3 = reduction_cell(b1, b2, out_filters, is_training, idx=idx)
        bx = convolution_cell(b2, b3, out_filters, is_training, idx=idx)

        idx = 3
        c0 = factorized_reduction(bx, out_filters * 2, 2, is_training, idx=idx)
        c1, c2 = maybe_calibrate_size(bx, c0, out_filters, is_training, idx=idx)
        c3 = reduction_cell(c1, c2, out_filters, is_training, idx=idx)
        cx = convolution_cell(c2, c3, out_filters, is_training, idx=idx)

    return cx


"""
def frame_model(x, is_training, out_filters=32):
    with tf.variable_scope("frame_model"):
        x0 = stem_conv(x, out_filters, is_training, filter_size=3)
        x1 = factorized_reduction(x0, out_filters * 2, 2, is_training, idx=1)

        idx = 2
        a0, a1 = maybe_calibrate_size(x0, x1, out_filters, is_training, idx=idx)
        a2 = reduction_cell(a0, a1, out_filters, is_training, idx=idx)
        a3 = reduction_cell(a1, a2, out_filters, is_training, idx="{}_{}".format(idx, 1))

        a4 = convolution_cell(a2, a3, out_filters, is_training, idx=idx)
        ax = convolution_cell(a3, a4, out_filters, is_training, idx="{}_{}".format(idx, 1))

        idx = 3
        b0 = factorized_reduction(ax, out_filters * 2, 2, is_training, idx=idx)
        b1, b2 = maybe_calibrate_size(ax, b0, out_filters, is_training, idx=idx)
        b3 = reduction_cell(b1, b2, out_filters, is_training, idx=idx)

        b4 = convolution_cell(b2, b3, out_filters, is_training, idx=idx)
        bx = convolution_cell(b3, b4, out_filters, is_training, idx="{}_{}".format(idx, 1))

        idx = 4
        c0 = factorized_reduction(bx, out_filters * 2, 2, is_training, idx=idx)
        c1, c2 = maybe_calibrate_size(bx, c0, out_filters, is_training, idx=idx)
        c3 = reduction_cell(c1, c2, out_filters, is_training, idx=idx)
        cx = convolution_cell(c2, c3, out_filters, is_training, idx=idx)

    return cx
    
    
    def frame_model(x, is_training, out_filters=32):
    with tf.variable_scope("frame_model"):
        x0 = stem_conv(x, out_filters, is_training, filter_size=3)
        x1 = factorized_reduction(x0, out_filters * 2, 2, is_training, idx=1)
        x2 = factorized_reduction(x1, out_filters * 2, 2, is_training, idx=1)

        idx = "2"
        a0, a1 = maybe_calibrate_size(x0, x1, out_filters, is_training, idx=idx)
        a2 = reduction_cell(a0, a1, out_filters, is_training, idx=idx)
        bx = convolution_cell(b1, b2, out_filters, is_training, idx=idx)

        idx = 3
        c0 = factorized_reduction(bx, out_filters * 2, 2, is_training, idx=idx)
        c1, c2 = maybe_calibrate_size(bx, c0, out_filters, is_training, idx=idx)
        c3 = reduction_cell(c1, c2, out_filters, is_training, idx=idx)
        cx = convolution_cell(c2, c3, out_filters, is_training, idx=idx)

        idx = 4
        d0 = factorized_reduction(cx, out_filters * 2, 2, is_training, idx=idx)
        d1, d2 = maybe_calibrate_size(cx, d0, out_filters, is_training, idx=idx)
        d3 = reduction_cell(d1, d2, out_filters, is_training, idx=idx)
        dx = convolution_cell(d2, d3, out_filters, is_training, idx=idx)

    return dx
"""

