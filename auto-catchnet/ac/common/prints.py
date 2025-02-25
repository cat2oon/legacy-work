import datetime

from ai.nas.params.nas import NASParams


def print_experiment(params: NASParams):
    console_print(">>> EXPERIMENT: {}".format(params.exp_name))


def console_print(msg, line=80):
    print("")
    print("=" * line)
    print(msg)
    print("=" * line)


def pretty_print_pairs(name_value_pairs, line_limit=80):
    for name, value in name_value_pairs:
        value = "{}".format(value)
        space = "." * (line_limit - len(name) - len(value))
        print("{}{}{}".format(name, space, value))
    print("=" * 80)


def print_user_flags():
    import tensorflow as tf
    from ac.common.flags import FLAG_LIST

    print("=" * 80)
    pairs = [(flag_name, getattr(tf.app.flags.FLAGS, flag_name)) for flag_name in sorted(FLAG_LIST)]
    pretty_print_pairs(pairs)


def print_time_stamp():
    console_print("current date time: {}".format(datetime.datetime.now()))
