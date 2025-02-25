import cv2

from ac.common.images import read_img_to_byte_arr, byte_arr_to_img
from ac.filesystem.greps import grep_files
from ac.images.filters.filters import bgr_to_rgb
from ac.perf.measure_time import MeasureTime


@MeasureTime
def load_images(src_dir_path, extension="jpg"):
    return __load_images_cv(src_dir_path, extension)


def load_image(img_path, as_rgb=True):
    img = cv2.imread(img_path)
    if as_rgb:
        return bgr_to_rgb(img)
    return img


def write_image(img, save_path):
    cv2.imwrite(img, save_path)


def __load_images_vectorized(src_dir_path, extension):
    paths = grep_files(src_dir_path, "*.{}".format(extension))
    encoded_imgs = [read_img_to_byte_arr(path) for path in paths]
    imgs = [byte_arr_to_img(encoded) for encoded in encoded_imgs]
    return imgs


def __load_images_cv(src_dir_path, extension):
    paths = grep_files(src_dir_path, "*.{}".format(extension))
    imgs = [cv2.imread(path) for path in paths]
    return imgs
