import tensorflow as tf


def make_hooks(output_dir):
    hooks = []

    # checkpoint
    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(output_dir, save_steps=30000, saver=saver)
    hooks.append(checkpoint_saver_hook)

    # replicas
    # if FLAGS.child_sync_replicas:
    #         sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
    #         hooks.append(sync_replicas_hook)
    # if FLAGS.controller_training and FLAGS.controller_sync_replicas:
    #         sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
    #         hooks.append(sync_replicas_hook)

    return hooks

