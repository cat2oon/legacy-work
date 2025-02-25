import sys

import tensorflow as tf

from ac.common.prints import pretty_print_pairs

FLAG_LIST = []
FLAGS = tf.app.flags.FLAGS


def run_tf_flags_compatibility():
    """ tensorflow 1.5 버전 부터 FLAG 방식 변경됨 stackoverflow.com/a/48227272 """
    remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
    assert (remaining_args == [sys.argv[0]])


run_tf_flags_compatibility()


def DEFINE_string(name, default_value, doc_string):
    tf.app.flags.DEFINE_string(name, default_value, doc_string)
    FLAG_LIST.append(name)


def DEFINE_integer(name, default_value, doc_string):
    tf.app.flags.DEFINE_integer(name, default_value, doc_string)
    FLAG_LIST.append(name)


def DEFINE_float(name, default_value, doc_string):
    tf.app.flags.DEFINE_float(name, default_value, doc_string)
    FLAG_LIST.append(name)


def DEFINE_boolean(name, default_value, doc_string):
    tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
    FLAG_LIST.append(name)


def is_debug_mode():
    return FLAGS.debug_mode


def get_flag_list():
    return FLAG_LIST


def get_FLAGS():
    return FLAGS


def print_flags():
    print("=" * 80)
    pairs = [(name, getattr(FLAGS, name)) for name in sorted(FLAG_LIST)]
    pretty_print_pairs(pairs)
