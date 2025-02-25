import tensorflow as tf
from keras import backend as K

"""
Metrics for keras
- Pixel Accuracy
- Mean Accuracy
- Mean IoU
"""


def mean_iou(y_true, y_pred):
    iou = []
    nb_classes = K.int_shape(y_pred)[-1]
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes):
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i)

    iou = tf.stack(iou)
    return K.mean(iou)

def pixel_accuracy(y_true, y_pred):
    pass
