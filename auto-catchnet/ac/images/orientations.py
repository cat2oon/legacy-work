import cv2
import numpy as np


def rotate_image(src_img, ccw_degree, scale=1.0):
    """
    bound rotation
    ref : www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """

    h, w = src_img.shape[:2]  # (rows, cols)
    cx, cy = w // 2, h // 2

    rot_mat = cv2.getRotationMatrix2D((cx, cy), ccw_degree, scale)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])

    new_h = int((h * cos) + (w * sin))
    new_w = int((h * sin) + (w * cos))

    rot_mat[0, 2] += (new_w / 2) - cx
    rot_mat[1, 2] += (new_h / 2) - cy
    dst_img = cv2.warpAffine(src_img, rot_mat, (new_w, new_h))

    return dst_img


def rotate_image_right_angle(src_img, ccw_degree):
    rot_times = {90: 1, 180: 2, 270: -1}.get(ccw_degree)
    return np.rot90(src_img, k=rot_times)


def apply_rotate_to_images(img_paths, ccw_degree):
    if ccw_degree % 90 == 0:
        rotate_fun = rotate_image_right_angle
    else:
        rotate_fun = rotate_image

    for img_path in img_paths:
        src_img = cv2.imread(img_path)
        dst_img = rotate_fun(src_img, ccw_degree)
        cv2.imwrite(img_path, dst_img)
