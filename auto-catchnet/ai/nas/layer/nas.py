from ai.nas.utils.arcs import get_xy_id_and_op_id
from ai.nas.utils.tfs import get_height
from ai.libs.tf.ops.convs import *
from ai.libs.tf.ops.convs import conv_1_multi, conv_1
from ai.libs.tf.ops.poolings import avg_pooling, max_pooling
from ai.libs.tf.ops.regularizes import batch_norm
from ai.libs.tf.ops.weights import create_weight


def nas_layer(prev_layers, arc, out_filters, num_cells):
    assert len(prev_layers) == 2, "need exactly 2 inputs"
    layers = [prev_layers[0], prev_layers[1]]
    layers = maybe_calibrate_size(layers[-2], layers[-1], out_filters, is_training=True)

    id_used = []
    for cell_id in range(num_cells):
        prev_layers = tf.stack(layers, axis=0)
        x_id, x_op, y_id, y_op = get_xy_id_and_op_id(arc, cell_id)
        with tf.variable_scope("cell_{0}".format(cell_id)):
            with tf.variable_scope("x"):
                x = prev_layers[x_id, :, :, :, :]
                x = nas_cell(x, cell_id, x_id, x_op, out_filters)
                x_id_used = tf.one_hot(x_id, depth=num_cells + 2, dtype=tf.int32)
            with tf.variable_scope("y"):
                y = prev_layers[y_id, :, :, :, :]
                y = nas_cell(y, cell_id, y_id, y_op, out_filters)
                y_id_used = tf.one_hot(y_id, depth=num_cells + 2, dtype=tf.int32)

            out = x + y
            id_used.extend([x_id_used, y_id_used])
            layers.append(out)

    id_used = tf.add_n(id_used)  # one-hots. ON/OFF 덧셈
    indices = tf.where(tf.equal(id_used, 0))  # TODO: tf.print
    indices = tf.to_int32(indices)
    indices = tf.reshape(indices, [-1])
    num_outs = tf.size(indices)
    out = tf.stack(layers, axis=0)
    out = tf.gather(out, indices, axis=0)

    inp = prev_layers[0]
    N = tf.shape(inp)[0]
    H = tf.shape(inp)[1]
    W = tf.shape(inp)[2]
    out = tf.transpose(out, [1, 2, 3, 0, 4])
    out = tf.reshape(out, [N, H, W, num_outs * out_filters])

    with tf.variable_scope("final_conv"):
        w = create_weight("w", [num_cells + 2, out_filters * out_filters])
        w = tf.gather(w, indices, axis=0)
        w = tf.reshape(w, [1, 1, num_outs * out_filters, out_filters])
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME")
        out = batch_norm(out, is_training=True)

    out = tf.reshape(out, tf.shape(prev_layers[0]))
    return out


def nas_cell(x, curr_cell_id, prev_cell_id, op_id, out_filters):
    num_possible_inputs = curr_cell_id + 1

    with tf.variable_scope("avg_pool"):
        avg_pool = avg_pooling(x, [3, 3], [1, 1], "SAME")
        if get_channel(avg_pool) != out_filters:
            with tf.variable_scope("conv"):
                avg_pool = conv_1_multi(avg_pool, num_possible_inputs, prev_cell_id, out_filters)

    with tf.variable_scope("max_pool"):
        max_pool = max_pooling(x, [3, 3], [1, 1], "SAME")
        if get_channel(max_pool) != out_filters:
            with tf.variable_scope("conv"):
                max_pool = conv_1_multi(max_pool, num_possible_inputs, prev_cell_id, out_filters)

    if get_channel(x) != out_filters:
        with tf.variable_scope("x_conv"):
            x = conv_1_multi(x, num_possible_inputs, prev_cell_id, out_filters)

    # NOTE: Order is matter when used in Fixed builder
    out = [
        nas_conv(x, curr_cell_id, prev_cell_id, 3, out_filters),
        nas_conv(x, curr_cell_id, prev_cell_id, 5, out_filters),
        avg_pool,
        max_pool,
        x,
    ]

    out = tf.stack(out, axis=0)
    out = out[op_id, :, :, :, :]
    return out


def nas_conv(x, curr_cell_id, prev_cell_id, filter_size, out_filters, stack_conv=2):
    with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
        num_possible_inputs = curr_cell_id + 2
        for conv_id in range(stack_conv):
            with tf.variable_scope("stack_{0}".format(conv_id)):
                x = sep_conv_multi(x, num_possible_inputs, prev_cell_id, out_filters, filter_size)
    return x


def factorized_reduction(x, out_filters, stride, is_training, idx=0):
    assert out_filters % 2 == 0, "Need even number of filters"

    if stride == 1:
        with tf.variable_scope("path_conv_{}".format(idx)):
            x = conv_1(x, out_filters, is_training, non_linear=False, name="path_conv")
            return x

    # Skip path 1
    path1 = tf.nn.avg_pool(x, [1, 1, 1, 1], [1, stride, stride, 1], "VALID")
    path1 = conv_1(path1, out_filters // 2, is_training, bn=False, non_linear=False,
                   pad="VALID", name="path1_conv_{}".format(idx))

    # Skip path 2
    # 우하단에 패딩 한줄 추가하고 좌상단 한줄 버림
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
    path2 = tf.nn.avg_pool(path2, [1, 1, 1, 1], [1, stride, stride, 1], "VALID")
    path2 = conv_1(path2, out_filters // 2, is_training, bn=False, non_linear=False,
                   pad="VALID", name="path2_conv_{}".format(idx))

    # Concat and apply BN
    final_path = tf.concat(values=[path1, path2], axis=3)
    final_path = batch_norm(final_path, is_training, name="bn_{}".format(idx))

    return final_path


def maybe_calibrate_size(x, y, out_filters, is_training, idx=0):
    """ order : [x, y] """
    x_h, y_h = get_height(x), get_height(y)
    with tf.variable_scope("calibrate_{}".format(idx)):
        if x_h != y_h:
            assert x_h == 2 * y_h
            with tf.variable_scope("pool_x"):
                x = tf.nn.relu(x)
                x = factorized_reduction(x, out_filters, 2, is_training)
        elif get_channel(x) != out_filters:
            with tf.variable_scope("pool_x"):
                x = conv_1_f(x, out_filters, is_training, prev_non_linearity=True,
                             last_non_linearity=False, var_name="w_{}".format(idx))
        if get_channel(y) != out_filters:
            with tf.variable_scope("pool_y"):
                y = conv_1_f(y, out_filters, is_training, prev_non_linearity=True,
                             last_non_linearity=False, var_name="w_{}".format(idx))
    return [x, y]
