import tensorflow as tf
from ai.libs.tf.ops.bases import tf_float, tf_print


def make_loss_monitor(loss, preds, inputs, arc, threshold=100.0):
    pred_x, pred_y = preds
    uid = inputs["uid"]
    cam_x = inputs["cam_x"]
    cam_y = inputs["cam_y"]
    ori = inputs["orientation"]

    # TODO: fixed 모드 일 때에는 arc 출력 불필요
    monitor_tensors = [uid, ori, pred_x, pred_y, cam_x, cam_y]
    bypass = lambda: loss
    monitor = lambda: tf_print(loss, monitor_tensors, "")

    branch = {
        # tf.greater(loss, tf_float(threshold, "threshold")): monitor,
        tf.greater(loss, tf_float(threshold, "threshold")): monitor,
        # tf.logical_and(
        #     tf.greater(loss, tf_float(10, "min_monitor")),
        #     tf.less(loss, tf_float(60, "max_monitor"))
        # ): monitor
    }

    return tf.case(branch, default=bypass, exclusive=True)


