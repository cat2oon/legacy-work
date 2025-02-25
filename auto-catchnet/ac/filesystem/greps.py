import glob
from itertools import chain

from ac.filesystem.paths import *


def grep_recur(base_path, pattern="*.*"):
    sub_greps = list(chain(*[grep_recur(dp, pattern) for dp in grep_dirs(base_path)]))
    return grep_files(base_path, pattern) + sub_greps


def grep_files(base_path, pattern="*.*"):
    return glob.glob("{}/{}".format(base_path, pattern))


def grep_pairs(base_path, ext_x, ext_y, check_missing=False):
    xs = grep_files(base_path, "*.{}".format(ext_x))
    ys = grep_files(base_path, "*.{}".format(ext_y))
    xs.sort()
    ys.sort()
    return list(zip(xs, ys))


def grep_dirs(base_path):
    file_paths = [path_join(base_path, name) for name in os.listdir(base_path)]
    return [p for p in file_paths if os.path.isdir(p)]
