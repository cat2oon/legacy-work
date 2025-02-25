from ac.common.flags import DEFINE_string, DEFINE_integer, DEFINE_float, DEFINE_boolean, get_FLAGS


def define_configs():
    """"""

    DEFINE_string("child_fixed_arc", None, "")
    # DEFINE_string("child_fixed_arc", "0 0 1 0 2 0 2 1 1 0 0 0 0 0 0 1", "None or arc seq")
    # DEFINE_string("child_fixed_arc", "0 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1", "None or arc seq")
    DEFINE_boolean("debug_mode", True, "tf debug mode")

    """ Top priority flags """
    DEFINE_string("optimizer_name", "adam", "for child models. adam / sgd / momentum")
    DEFINE_string("exp_name", "UNI-^10^8^5D Batch8 filter48", "experiment code")
    DEFINE_float("monitor_thresholds", 4.5, "")
    DEFINE_string("dataset_mode", "fast", "full / slow / mid / fast / single / None")
    DEFINE_float("child_l2_reg", 1e-4, "")
    DEFINE_float("child_lr", 1e-3, "")

    DEFINE_string("pool_layers", "3 5 8", "")
    DEFINE_integer("child_num_layers", 7, "")
    DEFINE_integer("child_out_filters", 48, "")
    DEFINE_integer("child_num_cells", 2, "")

    DEFINE_integer("batch_size", 8, "")
    DEFINE_integer("eval_batch_size", 32, "")

    DEFINE_integer("child_lr_dec_every", 1, "")
    DEFINE_float("child_lr_dec_rate", 0.7, "")
    DEFINE_boolean("reset_output_dir", True, "Delete output_dir if exists.")
    DEFINE_string("data_path", "/home/chy/archive-data/processed/everyone-tfr", "tfrecord path")
    DEFINE_string("clip_mode", None, "None / norm / global")
    DEFINE_integer("log_every", 1, "How many steps to log")
    DEFINE_integer("controller_log_every", 1, "How many steps to controller log")
    DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")
    DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")

    """ Env """
    DEFINE_string("output_dir", "./out", "")
    DEFINE_string("data_format", "NHWC", "Only NHWC")
    DEFINE_string("search_for", "micro", "Only search micro mode.")

    """ Child Train Ops params """
    DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
    DEFINE_float("child_lr_max", None, "for lr schedule")
    DEFINE_float("child_lr_min", None, "for lr schedule")
    DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
    DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")
    DEFINE_integer("child_num_aggregate", None, "")
    DEFINE_integer("child_lr_dec_start", 0, "")
    DEFINE_integer("child_lr_T_0", None, "for lr schedule")
    DEFINE_integer("child_lr_T_mul", None, "for lr schedule")

    """ Model & Child Params """
    DEFINE_integer("num_epochs", 300, "")
    # DEFINE_integer("child_filter_size", 5, "사용되는 곳 없음")
    DEFINE_integer("child_out_filters_scale", 1, "")
    DEFINE_integer("child_num_branches", 5, "기본 micro search 경우 5")
    DEFINE_integer("child_num_replicas", 1, "")
    DEFINE_integer("child_block_size", 3, "RNN child를 위한 설정")
    DEFINE_integer("child_cutout_size", None, "CutOut size")
    DEFINE_float("child_keep_prob", 0.5, "")
    DEFINE_float("child_drop_path_keep_prob", 1.0, "minimum drop_path_keep_prob")
    DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")

    """ Controller Params """
    DEFINE_integer("controller_train_steps", 200, "")
    DEFINE_float("controller_lr", 1e-3, "")
    DEFINE_float("controller_lr_dec_rate", 1.0, "")
    DEFINE_integer("controller_lr_dec_every_epochs", 2, "")
    DEFINE_float("controller_keep_prob", 0.5, "")
    DEFINE_float("controller_l2_reg", 0.0, "")
    DEFINE_float("controller_bl_dec", 0.99, "")
    DEFINE_float("controller_tanh_constant", None, "")
    DEFINE_float("controller_op_tanh_reduce", 1.0, "")
    DEFINE_float("controller_temperature", None, "")
    DEFINE_float("controller_entropy_weight", None, "")
    DEFINE_float("controller_skip_target", 0.8, "")
    DEFINE_float("controller_skip_weight", 0.0, "")
    DEFINE_integer("controller_num_aggregate", 1, "")
    DEFINE_integer("controller_num_replicas", 1, "")
    DEFINE_integer("controller_forwards_limit", 2, "")
    DEFINE_integer("controller_train_every", 1, "train the controller after this number of epochs")
    DEFINE_boolean("controller_search_whole_channels", True, "")
    DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
    DEFINE_boolean("controller_training", True, "")
    DEFINE_boolean("controller_use_critic", False, "")

    """
    NAS params
    """
    # DEFINE_integer("log_every", 1, "How many steps to log")
    # DEFINE_integer("controller_log_every", 1, "How many steps to controller log")
    # DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

    return get_FLAGS()
