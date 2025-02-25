import math
import numpy as np


def set_attrs_from_seq(obj, arrs, keys):
    for i, key in enumerate(keys):
        obj[key] = arrs[i]


def intersect_list(*args):
    return list(set.intersection(*map(set, args)))


def shuffle_partition(x, purpose_ratios):
    if purpose_ratios is None:
        purpose_ratios = {"full": 100}

    assert type(purpose_ratios) is dict
    assert sum(purpose_ratios.values()) == 100

    num_x = len(x)
    np.random.shuffle(x)
    p, current = {}, 0
    for purpose, ratio in purpose_ratios.items():
        num_part = math.ceil(num_x * ratio)
        p[purpose] = (current, current + num_part)
        current = num_part
    return p


def have_same_length(*arrs):
    lens = [len(arr) for arr in arrs]
    return all_identical_in(lens)


def all_identical_in(arr):
    return arr.count(arr[0]) == len(arr)


def shuffle_multiple_arrs(*arrs):
    assert len(arrs) > 1, 'at least two arr required'
    assert have_same_length(arrs), 'all arr must have same length'

    seed_arr = list(range(len(arrs[0])))
    np.random.shuffle(seed_arr)
    return [arr[seed_arr] for arr in arrs]


def truth_percentage(arr):
    return arr.count(True) / len(arr)


def falsy_percentage(arr):
    return arr.count(False) / len(arr)


def abs_seq(seq):
    s = [max(0, t) for t in seq]

    seq_type = type(seq)
    if seq_type is 'tuple':
        return tuple(s)
    return s


def chunks(seq, chunk_size):
    return [seq[i:i + chunk_size] for i in (range(0, len(seq), chunk_size))]


def chunks_range(list_to_chunk, chunk_size):
    return range(0, len(list_to_chunk), chunk_size)


def flatten(two_folded_seq):
    return [item for sub_seq in two_folded_seq for item in sub_seq]
