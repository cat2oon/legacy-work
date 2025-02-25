from ai.libs.tf.ops.bases import *
from ai.libs.tf.ops.convs import *
from ai.libs.tf.ops.poolings import spatial_max_pool, spatial_avg_pool


def convolution_cell(prev_x, x, out_filters, is_training, idx=0):
    with tf.variable_scope("conv_cell_{}_{}".format(out_filters, idx)):
        prev_c = prev_x.shape[3].value

        a0 = spatial_avg_pool(x, 3, 1, is_training)
        a1 = spatial_max_pool(x, 3, 1, is_training)
        ax = tf.add_n([a0, a1])

        b0 = sep_conv(prev_x, prev_c, 5, is_training)
        b1 = identity(prev_x)
        bx = tf.add_n([b0, b1])

        c_filters = 16
        c0 = sep_conv(x, c_filters, 5, is_training, name="c0")
        c1 = sep_conv(x, c_filters, 5, is_training, name="c1")
        cx = tf.add_n([c0, c1])

        d_filters = 32
        d0 = sep_conv(prev_x, d_filters, 5, is_training, name="d0")
        d1 = sep_conv(prev_x, d_filters, 5, is_training, name="d1")
        dx = tf.add_n([d0, d1])

        e_filters = 32
        e0 = sep_conv(x, e_filters, 3, is_training, name="e0")
        e1 = sep_conv(prev_x, e_filters, 5, is_training, name="e1")
        ex = tf.add_n([e0, e1])

        # merge
        x = tf.concat([ax, bx, cx, dx, ex], 3)

        # reduction
        x = conv_1(x, out_filters, is_training, name="out_cell")
    return x


def reduction_cell(prev_x, x, out_filters, is_training, idx=0):
    with tf.variable_scope("red_cell_{}_{}".format(out_filters, idx)):
        prev_c = prev_x.shape[3].value

        a0 = sep_conv(prev_x, prev_c, 5, is_training, name="a0")
        a1 = identity(prev_x)
        ax = tf.add_n([a0, a1])

        b_filters = 16
        b0 = sep_conv(ax, b_filters, 3, is_training, name="b0")
        b1 = sep_conv(x, b_filters, 3, is_training, name="b1")
        bx = tf.add_n([b0, b1])

        c_filters = 16
        c0 = sep_conv(prev_x, c_filters, 3, is_training, name="c0")
        c1 = sep_conv(prev_x, c_filters, 3, is_training, name="c1")
        cx = tf.add_n([c0, c1])

        d_filters = prev_c
        d0 = spatial_max_pool(prev_x, 3, 1, is_training)
        d1 = sep_conv(x, d_filters, 5, is_training, name="d1")
        dx = tf.add_n([d0, d1])

        e_filters = 16
        e0 = sep_conv(dx, e_filters, 5, is_training, name="e0")
        e1 = sep_conv(x, e_filters, 5, is_training, name="e1")
        ex = tf.add_n([e0, e1])

        # merge
        x = tf.concat([bx, cx, ex], 3)

        # reduction
        x = conv_1(x, out_filters, is_training, name="out_cell")
    return x
