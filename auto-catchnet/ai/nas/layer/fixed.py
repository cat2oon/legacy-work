import numpy as np
import tensorflow as tf

from ai.nas.utils.arcs import get_xy_id_and_op_id
from ai.nas.layer.nas import factorized_reduction, maybe_calibrate_size
from ai.nas.utils.tfs import get_strides, get_width, get_channel
from ai.libs.tf.ops.convs import sep_conv_f, conv_1_f
from ai.libs.tf.ops.poolings import avg_pooling, max_pooling


def fixed_layer(layer_id, prev_layers, arc, out_filters, stride, is_training, num_cells, drop_path_apply):
    assert len(prev_layers) == 2
    assert drop_path_apply is not None
    layers = [prev_layers[0], prev_layers[1]]
    layers = maybe_calibrate_size(layers[-2], layers[-1], out_filters, is_training=is_training)

    with tf.variable_scope("layer_base"):
        layers[1] = conv_1_f(layers[1], out_filters, is_training, prev_non_linearity=True, last_non_linearity=False)

    id_used = np.zeros([num_cells + 2], dtype=np.int32)
    for cell_id in range(num_cells):
        x_id, x_op_id, y_id, y_op_id = get_xy_id_and_op_id(arc, cell_id)
        id_used[x_id] += 1
        id_used[y_id] += 1
        with tf.variable_scope("cell_{}".format(cell_id)):
            with tf.variable_scope("x_conv"):
                x_stride = stride if x_id in [0, 1] else 1
                x = fixed_cell(layers[x_id], x_op_id, out_filters, x_stride, is_training)
                x = drop_path_apply(x, layer_id, x_op_id, is_training)
            with tf.variable_scope("y_conv"):
                y_stride = stride if y_id in [0, 1] else 1
                y = fixed_cell(layers[y_id], y_op_id, out_filters, y_stride, is_training)
                y = drop_path_apply(y, layer_id, y_op_id, is_training)
            out = x + y
            layers.append(out)
    out = fixed_combine(layers, id_used, out_filters, is_training)
    return out


def fixed_cell(x, op_id, out_filters, stride, is_training, filter_sizes=[3, 5]):
    # NOTE: Order is matter. Follow NAS Layer Builder
    # TODO : 두 곳의 순서 및 판단을 추상화 할 것
    # MAPPING X_OP { 0:conv3x3, 1:conv5x5, 2:avg_pool, 3:max_pool, 4: identity w/ reduction}
    if op_id in [0, 1]:
        filter_size = filter_sizes[op_id]
        x = fixed_conv(x, out_filters, filter_size, stride, is_training)
    elif op_id in [2, 3]:
        if op_id == 2:
            x = avg_pooling(x, [3, 3], [stride, stride], "SAME")
        else:
            x = max_pooling(x, [3, 3], [stride, stride], "SAME")
        if get_channel(x) != out_filters:
            x = conv_1_f(x, out_filters, is_training, prev_non_linearity=True, last_non_linearity=False)
    else:
        if stride > 1:
            assert stride == 2
            x = factorized_reduction(x, out_filters, 2, is_training)
        if get_channel(x) != out_filters:
            x = conv_1_f(x, out_filters, is_training, prev_non_linearity=True, last_non_linearity=False)
    return x


def fixed_conv(x, out_filters, filter_size, stride, is_training, stack_convs=2):
    for conv_id in range(stack_convs):
        if conv_id == 0:
            strides = get_strides(stride)
        else:
            strides = [1, 1, 1, 1]
        with tf.variable_scope("sep_conv_{}".format(conv_id)):
            x = sep_conv_f(x, out_filters, filter_size, is_training, strides, use_reduce=False,
                           prev_non_linearity=True, last_non_linearity=False)
    return x


def fixed_combine(layers, used, out_filters, is_training):
    out_hw = min([get_width(layer) for i, layer in enumerate(layers) if used[i] == 0])
    out = []

    with tf.variable_scope("final_combine"):
        for i, layer in enumerate(layers):
            if used[i] == 0:
                hw = get_width(layer)
                if hw > out_hw:
                    assert hw == out_hw * 2, ("i_hw={0} != {1}=o_hw".format(hw, out_hw))
                    with tf.variable_scope("calibrate_{0}".format(i)):
                        x = factorized_reduction(layer, out_filters, 2, is_training)
                else:
                    x = layer
                out.append(x)
        out = tf.concat(out, axis=3)
    return out
