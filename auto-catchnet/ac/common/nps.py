import numpy as np


def set_np_print_precision(precision=6):
    np.set_printoptions(precision=precision)


def set_np_print_non_science():
    np.set_printoptions(suppress=True)


def clone_npz(npz):
    npz_data = {}
    for key in npz.keys():
        npz_data[key] = np.array(npz[key])
    return npz_data


def merge_npzs(npz_x, npz_y):
    if len(npz_x) is 0:
        return npz_y
    if len(npz_y) is 0:
        return npz_x
    npz_merged = {}
    for key in npz_x.keys():
        npz_merged[key] = np.concatenate([npz_x[key], npz_y[key]])
    return npz_merged


def shuffle_npz(npz):
    randomize = np.arange(count_item_in_npz(npz))
    np.random.shuffle(randomize)
    for key in npz.keys():
        npz[key] = npz[key][randomize]
    return npz


def count_item_in_npz(npz):
    return npz[list(npz.keys())[0]].shape[0]


"""
- elements operators
"""


def contains_nan(nd_arr):
    return np.isnan(nd_arr).any()


"""
- vector operators
"""


def normalize(v, order=2):
    norm = get_norm(v, order)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def get_norm(v, order=2):
    return np.linalg.norm(v, ord=order)


"""
- change dimension NHWC <--> NCHW 
"""


def from_NHWC_to_NCHW(nd_arr):
    return np.moveaxis(nd_arr, 3, 1)


def from_NCHW_to_NHWC(nd_arr):
    return np.moveaxis(nd_arr, 1, 3)
