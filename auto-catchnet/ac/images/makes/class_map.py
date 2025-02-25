import numpy as np

from al.maths.geos.ellipse import is_inside, centric_score

"""
 Makers class map for label like segmentation
"""


def make_softmax_map_from_ellipse(shape, ellipse_param):
    softmax_map = []
    max_y, max_x = shape
    for py in range(max_y):
        for px in range(max_x):
            score = centric_score(px, py, ellipse_param)
            if score:
                softmax_map.append([0, 1])
            else:
                softmax_map.append([1, 0])
    return np.array(softmax_map)


def make_class_map_from_ellipse(shape, ellipse_param, is_one_hot=False):
    max_y, max_x = shape
    cx, cy, width, height, phi = ellipse_param

    binary_mask = []
    for py in range(max_y):
        for px in range(max_x):
            is_in = is_inside(px, py, cx, cy, width, height, phi)
            if is_in:
                label = [0, 1]
            else:
                label = [1, 0]
            binary_mask.append(label)

    mask = np.array(binary_mask)
    if is_one_hot:
        out_shape = (max_y, max_x, 2)
        mask = mask.reshape(out_shape)  # e.g 11x x 112 x 2

    return mask


def make_class_map_for_ones(shape):
    return np.ones(shape)[:, :, np.newaxis]
