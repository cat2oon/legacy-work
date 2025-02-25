import os
from pathlib import Path
import shutil


def path_join(*paths):
    return os.path.join(*paths)


def exists(path):
    return os.path.exists(path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def mk_path_join(*paths):
    path = path_join(*paths)
    parent = path_join(*paths[:-1])
    if not os.path.exists(parent):
        os.makedirs(path, exist_ok=True)
    return path


def init_directory(output_dir, reset=False):
    if not os.path.isdir(output_dir):
        print("Path {} does not exist. Creating.".format(output_dir))
        os.makedirs(output_dir)
    elif reset:
        print("Path {} exists. Remove and remake.".format(output_dir))
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)


def file_paths_to_dir_paths(paths):
    paths = [Path(p).parent for p in paths]
    paths = list(set(paths))
    return paths


def make_dirs(dir_paths):
    for p in dir_paths:
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)


def basename_in_path(path):
    return os.path.basename(path)


def stem_in_path(path):
    return Path(path).stem


def directory_path(path):
    return Path(path).parent


def twin_path(path, extension="none"):
    return path_join(directory_path(path), "{}.{}".format(stem_in_path(path), extension))
