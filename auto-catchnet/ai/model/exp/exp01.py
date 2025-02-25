import glob
import os

from ai.libs.tf.ops.weights import *
from ai.libs.tf.setup.inits import get_init_ops
from ai.libs.tf.ops.regularizes import l2_regularizer
from ai.libs.tf.setup.sess import get_sess_config
from ai.model.exp.blocks import build_model
from ds.core.tf.tfrecords import to_dataset

info = tf.logging.info


if __name__ == '__main__':

    # load dataset
    tf_src_path = "/home/chy/archive-data/processed/everyone-tfr"
    tfrecord_files = glob.glob(os.path.join(tf_src_path, '*.tfr'))

    iterator = to_dataset(tfrecord_files, None, batch_size=16)
    frames, lefts, rights, cam_x, cam_y, cam_to_x, cam_to_y = iterator.get_next()

    # build models
    pred_x, pred_y = build_model(frames, lefts, cam_to_x, cam_to_y, True)

    # loss
    labels = tf.transpose(tf.stack([cam_x, cam_y]))
    preds = tf.transpose(tf.stack([pred_x, pred_y]))
    mse = tf.losses.mean_squared_error(labels, preds)

    # loss_x = tf.losses.mean_squared_error(cam_x, pred_x)
    # loss_y = tf.losses.mean_squared_error(cam_y, pred_y)
    loss_l2 = l2_regularizer()
    loss = mse + loss_l2

    # optimizer
    lr_init = 0.001
    decay_steps = 2500
    lr_decay_factor = 0.7

    lr = tf.train.exponential_decay(
        learning_rate=lr_init,
        global_step=tf.train.get_or_create_global_step(),
        decay_steps=int(decay_steps),
        decay_rate=lr_decay_factor,
        staircase=True)

    train_op = tf.train.AdamOptimizer(lr, beta1=0.0, epsilon=1e-3, use_locking=True).minimize(loss)

    global_step = tf.train.get_or_create_global_step()
    init_ops = get_init_ops(iterator)

    with tf.Session(config=get_sess_config()) as sess:
        sess.run(init_ops)

        for i in range(10 ** 5):
            _, v_mse, v_x, v_y, _ = sess.run([global_step, mse, train_op])

            if i % 50 == 0:
                print("[{:06d}] mse:{:.2f}  ({:.2f},{:.2f})".format(i, v_mse, v_x, v_y))

            if i % 1000 == 0:
                v_lr = sess.run(lr)
                print("[{:06d}] lr: {:.5f}".format(i, v_lr))
