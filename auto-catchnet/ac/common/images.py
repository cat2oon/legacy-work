import cv2
import math
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

from ac.images.resize_box import ResizeBox


def read_img_to_byte_arr(path):
    return open(path, 'rb').read()


def byte_arr_to_img(byte_arr):
    dt = np.dtype(np.int8)  # 255
    dt = dt.newbyteorder('>')  # depend on mach arch
    np_arr = np.frombuffer(byte_arr, dt)
    return cv2.imdecode(np_arr, 1)


def read_encoded_img(path):
    return read_img_to_byte_arr(path)


def decode_img(encoded_img):
    return byte_arr_to_img(encoded_img)


def byte_arrs_to_imgs(arr_imgs):
    return list(map(decode_img, arr_imgs))


def normalize_img(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def scale_abs(img):
    return cv2.convertScaleAbs(img)  # to 8bit


def normalize_per_channel(img):
    img = (img - img.mean()) / img.std()
    return img


def normalize_per_images(imgs):
    imgs -= imgs.mean(axis=(1, 2), keepdims=True)
    imgs /= imgs.std(axis=(1, 2), keepdims=True)
    return imgs


def is_readable_images(*path_args):
    try:
        for path in path_args:
            imread(path)
        return True
    except Exception as ex:
        print(ex)
    return False


def is_shrinking(origin_hw, resize_hw):
    return origin_hw[0] > resize_hw[0] or origin_hw[1] > resize_hw[1]


def select_interpolation_method(origin_hw, resize_hw):
    if is_shrinking(origin_hw, resize_hw):
        return cv2.INTER_AREA
    return cv2.INTER_CUBIC


def is_horizontal(img_shape):
    h, w = img_shape[:2]
    return w / h > 1


def is_vertical(img_shape):
    h, w = img_shape[:2]
    return w / h < 1


def shape_to_hw(shape):
    return shape[1], shape[0]


def aspect_wh_ratio(shape):
    h, w = shape_to_hw(shape)
    return w / h


def get_box_for_over_resize(img_shape, resize_hw):
    rh, rw = resize_hw
    aspect_wh = aspect_wh_ratio(img_shape)
    box = ResizeBox(rh, rw, 0, 0, 0, 0)

    if is_horizontal(img_shape):
        new_h = np.round(rw / aspect_wh).astype(int)
        pad_vertical = (rh - new_h) / 2
        box.pad_top = np.floor(pad_vertical).astype(int)
        box.pad_bottom = np.ceil(pad_vertical).astype(int)
        box.resize_h = new_h
    elif is_vertical(img_shape):
        new_w = np.round(rh * aspect_wh).astype(int)
        pad_horizontal = (rw - new_w) / 2
        box.pad_left = np.floor(pad_horizontal).astype(int)
        box.pad_right = np.ceil(pad_horizontal).astype(int)
        box.resize_w = new_w
    return box


def get_box_for_fit_resize(img_shape, new_hw):
    new_h, new_w = new_hw
    origin_h, origin_w = img_shape[0], img_shape[1]  # 왜 인지 순서가 다른 듯

    kw, kh = (new_w / origin_w), (new_h / origin_h)
    resize_ratio = min(kw, kh)  # 기준 축으로 축소 ratio 선정

    box = ResizeBox(new_h, new_w, 0, 0, 0, 0)
    if kw < kh:  # width 기준 축소 (h pad 발생)
        cover_h = round(origin_h * resize_ratio)
        pad_length = (new_h - cover_h) / 2
        box.pad_top = math.floor(pad_length)
        box.pad_bottom = math.ceil(pad_length)
        box.resize_h = cover_h  # border 덫칠하는 방식이므로
    elif kh < kw:
        cover_w = round(origin_w * resize_ratio)
        pad_length = (new_w - cover_w) / 2
        box.pad_left = math.floor(pad_length)
        box.pad_right = math.ceil(pad_length)
        box.resize_w = cover_w
    return box


def resize_and_pad(img, resize_hw, over_mode=True, pad_color=0):
    img_shape = img.shape
    if over_mode:
        box = get_box_for_over_resize(img_shape, resize_hw)
    else:
        box = get_box_for_fit_resize(img_shape, resize_hw)

    if len(img_shape) is 3:
        pad_color = [pad_color] * 3

    im = select_interpolation_method(img_shape[:2], resize_hw)
    scaled_img = resize(img, (box.resize_w, box.resize_h), im)
    scaled_img = cv2.copyMakeBorder(scaled_img, box.pad_top, box.pad_bottom, box.pad_left, box.pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return scaled_img


def resize(img, shape, interpolation=cv2.INTER_AREA):
    return cv2.resize(img, shape, interpolation=interpolation)


def grey_to_color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def is_gray_shape(shape):
    return len(shape) == 3 and shape[2] == 1


def mirror_image(image):
    return image[:, ::-1]


def reverse_color(image):
    return image[:, :, ::-1]


def area(shape):
    return shape[0] * shape[1]


def resize_to_smaller(image_x, image_y):
    if area(image_x.shape) > area(image_y.shape):
        image_x = resize_and_pad(image_x, image_y.shape)
    else:
        image_x = resize_and_pad(image_y, image_x.shape)
    return image_x, image_y


def make_dual_channel_image(image_x, image_y, shape=None):
    if len(image_x.shape) == 3 and image_x.shape[2] == 3:
        image_x = rgb2gray(image_x)
    if len(image_y.shape) == 3 and image_y.shape[2] == 3:
        image_y = rgb2gray(image_y)
    if shape is not None:
        image_x = resize_and_pad(image_x, shape)
        image_y = resize_and_pad(image_y, shape)
    elif image_x.shape != image_y.shape:
        image_x, image_y = resize_to_smaller(image_x, image_y)
    image = np.dstack((image_x, image_y))

    return image


def is_screen_out(w, h, x, y):
    return x > w or y > h
