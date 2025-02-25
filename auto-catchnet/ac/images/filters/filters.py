import cv2
import numpy as np


# TODO: IDE 다시 키면 file rename edges

def gaussian_blur(img, kernel=(3, 3)):
    return cv2.GaussianBlur(img, kernel, 0)


def invert_color(img):
    return cv2.bitwise_not(img)


def color_to_grey(img, from_bgr=False):
    if from_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.squeeze(img)


def grey_to_color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def apply_sobel_x(img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)


def apply_sobel_y(img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)


def apply_sobel_xy(img):
    img = apply_sobel_x(img)
    img = apply_sobel_y(img)
    return img


def apply_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F)


def apply_canny(img, threshold1=30, threshold2=70):
    return cv2.Canny(img, threshold1, threshold2)


def to_grey_and_gaussian_blur(img, kernel=(3, 3), sigma_x=0, sigma_y=0):
    img = color_to_grey(img, from_bgr=True)
    img = cv2.GaussianBlur(img, kernel, sigma_x, sigma_y)
    return img
