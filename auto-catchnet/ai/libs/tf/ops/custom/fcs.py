import tensorflow as tf

from ai.libs.tf.ops.activations import ranged_sigmoid
from ai.libs.tf.ops.bases import tf_int
from ai.libs.tf.ops.weights import create_weight, create_bias


def fully_connect(x, num_class, name="fc", initializer=None, activation=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()

    inp_c = x.shape[1].value
    with tf.variable_scope(name):
        w = create_weight("w", [inp_c, num_class], initializer=initializer)
        b = create_bias("b", [num_class], initializer=tf.zeros_initializer())
        x = tf.matmul(x, w) + b
    if activation is not None:
        x = activation(x)
    return x


def fc(x, num_class, use_bias=True):
    inp_c = x.shape[1].value
    w = create_weight("w", [inp_c, num_class])
    if use_bias:
        b = create_bias("b", [num_class], initializer=tf.zeros_initializer())
        x = tf.matmul(x, w) + b
    else:
        x = tf.matmul(x, w)
    return x


def uni_fc(fv, name="uni"):
    act = ranged_sigmoid
    fx = fully_connect(fv, 1024, name="{}x1".format(name), activation=act)
    fx = fully_connect(fx, 256, name="{}x2".format(name), activation=act)
    fx = fully_connect(fx, 32, name="{}x3".format(name), activation=act)
    return fx


def dynamic_branch(fv, inputs):
    orient = tf.cast(inputs["orientation"], tf.int32)
    orient = tf.squeeze(orient)

    def build_last_fc(orientation):
        def last_fc(name="fc_o{}_".format(orientation), act=ranged_sigmoid):
            fx = fully_connect(fv, 1024, name="{}x1".format(name), activation=act)
            fx = fully_connect(fx, 256, name="{}x2".format(name), activation=act)
            fx = fully_connect(fx, 16, name="{}x3".format(name), activation=act)
            return fx

        return last_fc

    fc_branches = []
    for o in range(1, 4):
        fc_branches.append(build_last_fc(o))

    # fx, fy = tf.case({
    fx = tf.case({
        tf.equal(tf_int(1, "case_orientation_1"), orient): fc_branches[0],
        tf.equal(tf_int(3, "case_orientation_3"), orient): fc_branches[1],
        tf.equal(tf_int(4, "case_orientation_4"), orient): fc_branches[2],
    }, exclusive=True)

    return fx #, fy