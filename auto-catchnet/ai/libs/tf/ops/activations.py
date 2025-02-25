import tensorflow as tf


def relu(x, leaky=0.0):
    return tf.where(tf.greater(x, 0), x, x * leaky)


def ranged_sigmoid(x, scale=100, curvature=-0.04):
    """
    f(x) = (100 / 1 + e^(-0.04x)) - 50
    (-20, 20) 구간에 대해 선형성을 띄며 그 밖의
     범위에서 축소 성향을 가지도록 되어 있다
    """
    x = scale / (1 + tf.math.exp(curvature * x))
    return x - (scale / 2)
