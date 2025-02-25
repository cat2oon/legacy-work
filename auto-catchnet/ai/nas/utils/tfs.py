def get_channel(x):
    shape = x.get_shape()
    if len(shape) == 2:
        return shape[1].value
    return shape[3].value


def get_width(x):
    return x.get_shape()[2].value


def get_height(x):
    return x.get_shape()[1].value


def get_strides(stride):
    return [1, stride, stride, 1]
