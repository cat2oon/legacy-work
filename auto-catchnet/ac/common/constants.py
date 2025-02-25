import numpy as np


def max_value(dtype='int'):
    return np.finfo(dtype).max


def min_value(dtype='int'):
    return np.finfo(dtype).min
