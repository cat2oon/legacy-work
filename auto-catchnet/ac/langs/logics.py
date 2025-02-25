import collections

import numpy as np


def no_none(*args):
    return not any([x is None for x in args])


def is_empty(x):
    if isinstance(x, np.ndarray):
        return not x.size or not x.any()
    if isinstance(x, collections.Sized):
        return not x
    return not x


def is_not_empty(x):
    if isinstance(x, np.ndarray):
        return x.size != 0
    if isinstance(x, collections.Sized):
        return x
    return x is not None
