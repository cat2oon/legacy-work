import glob
import os

from ai.dataset.everyone.tfreader import to_dataset
from ai.libs.tf.ops.convs import *
from ai.libs.tf.ops.weights import *
from ai.libs.tf.setup.sess import get_sess_config

"""
[1]                 # sep conv 3x3      v
[4 1]               # avg-pool          v
[2 1 1]             # conv 5x5          v
[2 1 0 1]           # conv 5x5
[4 0 1 1 1]         # avg-pool
[0 1 0 1 1 1]       # conv 3x3
[0 1 0 1 0 1 0]     # conv 3x3
[3 1 1 1 1 1 1 1]   # sep conv 5x5

+ 3층에 한번씩 factorized_reduction

"""

# TODO: 전체 과정 framework로 작성
# 참고 : https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/ops.py


def make_frame_model(frames, is_training, ch_mul, keep_prob):
    print(">>> input {}".format(frames.shape))

    with tf.variable_scope("frame"):
        # 480 x 480 x 3
        with tf.variable_scope("stem_conv"):
            w = create_weight("w", [3, 3, 3, 64])  # RGB for frame
            x = tf.nn.conv2d(frames, w, [1, 3, 3, 1], "SAME", dilations=[1, 2, 2, 1])  # stride: 3x3
            x = batch_norm(x, is_training)
            print(">>> stem conv {} >> dilated conv 3x3 [2,2]".format(x.shape))

        # 160 x 160 x 64
        " layer-0: [1] "
        inp_c = x.shape[3].value
        with tf.variable_scope("layer_0"):
            # TODO : 첫 레이어에서는 불필요 하므로 생략하고 비교
            # 1x1 conv (dimension reduction effect)
            # Going deeper with convolution (arxiv.org/pdf/1409.4842v1.pdf)
            with tf.variable_scope("inp_conv_1"):
                w = create_weight("w", [1, 1, inp_c, 64])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")  # [1, stride, stride, 1]
                x = batch_norm(x, is_training)
                x = tf.nn.relu(x)

            # depth-wise separable conv
            # Xception (1610.02357v3.pdf)
            with tf.variable_scope("out_conv"):
                w_depth = create_weight("w_depth", [3, 3, 64, ch_mul])  # 3x3 sep conv
                w_point = create_weight("w_point", [1, 1, 64 * ch_mul, 64])
                x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1], padding="SAME")
                x = batch_norm(x, is_training)
                l_0 = tf.identity(x, name='layer_0_out')
                print(">>> layer_0_out {}".format(x.shape))

        # 160 x 160 x 64
        # " layer-1: [4 1] "
        # inp_c = x.shape[3].value
        # with tf.variable_scope("layer_1"):
        #     with tf.variable_scope("conv_1"):
        #         w = create_weight("w", [1, 1, inp_c, 64])
        #         x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
        #         x = batch_norm(x, is_training)
        #         x = tf.nn.relu(x)
        #     with tf.variable_scope("pool"):
        #         x = tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 3, 3, 1], "SAME")   # 3x3 avg-pool
        #         l_1 = tf.identity(x, name='layer_1_out')
        #         print(">>> layer_1 {}".format(x.shape))
        #
        # # skip_connection
        # x = tf.concat([x, l_0], axis=3)
        # print(">>> layer_1_concat_out {}".format(x.shape))
        #
        # # TODO: require reshape?
        #
        # " layer-2: [2 1 1] conv 5x5"
        # inp_c = x.shape[3].value
        # with tf.variable_scope("layer_2"):
        #     # 1x1 conv (dimension reduction effect)
        #     # Going deeper with convolution (arxiv.org/pdf/1409.4842v1.pdf)
        #     with tf.variable_scope("inp_conv_1"):
        #         w = create_weight("w", [1, 1, inp_c, 64])
        #         x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
        #         x = batch_norm(x, is_training)
        #         x = tf.nn.relu(x)
        #
        #     # conv 5x5
        #     with tf.variable_scope("out_conv"):
        #         w = create_weight("w", [5, 5, inp_c, 1])      # 5x5 sep conv
        #         x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
        #         x = batch_norm(x, is_training)
        #         l_2 = tf.identity(x, name='layer_2_out')
        #
        # # skip_connection
        # x = tf.add_n(x, l_1, l_2)

        # GAP
        x = global_avg_pool(x)
        if is_training:
            x = tf.nn.dropout(x, keep_prob)

        with tf.variable_scope("frame_last_layer"):
            fx = tf.identity(x, name="frame_out")

    return fx


def make_eye_model(eyes, is_training, ch_mul, keep_prob):
    with tf.variable_scope("eye"):
        # 96 x 96 x 1
        with tf.variable_scope("stem_conv"):
            w = create_weight("w", [3, 3, 1, 64])  # Grey for eye
            x = tf.nn.conv2d(eyes, w, [1, 1, 1, 1], "SAME", dilations=[1, 1, 1, 1])
            x = batch_norm(x, is_training)

        " layer-0: [1] "
        inp_c = x.shape[3].value
        with tf.variable_scope("layer_0"):
            # 1x1 conv (dimension reduction effect)
            # Going deeper with convolution (arxiv.org/pdf/1409.4842v1.pdf)
            with tf.variable_scope("inp_conv_1"):
                w = create_weight("w", [1, 1, inp_c, 64])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
                x = batch_norm(x, is_training)
                x = tf.nn.relu(x)

            # depth-wise separable conv
            # Xception (1610.02357v3.pdf)
            with tf.variable_scope("out_conv"):
                w_depth = create_weight("w_depth", [3, 3, 64, ch_mul])  # 3x3 sep conv
                w_point = create_weight("w_point", [1, 1, 64 * ch_mul, 64])
                x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1], padding="SAME")
                x = batch_norm(x, is_training)
                l_0 = tf.identity(x, name='layer_0_out')

        # 160 x 160 x 64
        " layer-1: [1] "
        inp_c = x.shape[3].value
        with tf.variable_scope("layer_1"):
            # 1x1 conv (dimension reduction effect)
            # Going deeper with convolution (arxiv.org/pdf/1409.4842v1.pdf)
            with tf.variable_scope("inp_conv_1"):
                w = create_weight("w", [1, 1, inp_c, 64])
                x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")  # [1, stride, stride, 1]
                x = batch_norm(x, is_training)
                x = tf.nn.relu(x)

            # depth-wise separable conv
            # Xception (1610.02357v3.pdf)
            with tf.variable_scope("out_conv"):
                w_depth = create_weight("w_depth", [3, 3, 64, ch_mul])  # 3x3 sep conv
                w_point = create_weight("w_point", [1, 1, 64 * ch_mul, 64])
                x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1], padding="SAME")
                x = batch_norm(x, is_training)
                print(">>> layer_1_out {}".format(x.shape))

        # GAP
        x = global_avg_pool(x)
        if is_training:
            x = tf.nn.dropout(x, keep_prob)

        with tf.variable_scope("eye_last_layer"):
            ex = tf.identity(x, name="eye_out")

    return ex


def make_model(frames, eyes, is_training, ch_mul=1, keep_prob=0.7):
    fx = make_frame_model(frames, is_training, ch_mul, keep_prob)
    ex = make_eye_model(eyes, is_training, ch_mul, keep_prob)

    print(">>> fx {}".format(fx.shape))
    print(">>> ex {}".format(ex.shape))

    # concat 128 dim
    x = tf.concat([fx, ex], axis=1)
    print(">>> concat {}".format(x.shape))

    # FC Layer
    with tf.variable_scope("fc"):
        inp_c = x.shape[1].value
        w = create_weight("w", [inp_c, 2])
        preds = tf.matmul(x, w)

    return preds


"""
Entry
"""

with tf.Graph().as_default() as g:
    tf_src_path = "/home/chy/archive-data/processed/everyone-tfr"
    tfrecord_files = glob.glob(os.path.join(tf_src_path, '*.tfr'))

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    iterator = to_dataset(tfrecord_files, num_parallel=8, batch_size=32, num_epochs=40)
    frames, eyes, _, cam_x, cam_y = iterator.get_next()
    is_training = True

    with tf.variable_scope("Exp_v1"):
        preds = make_model(frames, eyes, is_training, ch_mul=1, keep_prob=0.7)

    # loss
    labels = tf.transpose(tf.stack([cam_x, cam_y]))
    print(labels.shape)
    print(preds.shape)
    loss = tf.losses.mean_squared_error(labels, preds)

    # optimizer + decay
    lr_init = 0.001
    # lr_dec = tf.train.exponential_decay(lr_init, tf.maximum())
    opt = tf.train.AdamOptimizer(0.0001, beta1=0.0, epsilon=1e-3, use_locking=True).minimize(loss)

    init_op = tf.global_variables_initializer()

    # train
    with tf.Session(config=get_sess_config()) as sess:
        sess.run(iterator.initializer)
        sess.run(init_op)

        while True:
            try:
                sess.run(opt)
                loss_val = sess.run(loss)
                print(loss_val)
            except Exception as e:
                print(e)
