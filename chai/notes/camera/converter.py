import os, re, glob, math, cv2
import numpy as np

def get_length_diff(image_length, fov_image, fov_after, is_rad=True):
    tan = math.tan
    atan = math.atan
    sqrt = math.sqrt
    rad = math.radians
    deg = math.degrees
    to, fo, fa = image_length, fov_image, fov_after
    if not is_rad:
        fo, fa = rad(fo), rad(fa)
    far = to / tan(fo/2)
    tn_minus_to = far * (tan(fa/2)-tan(fo/2))
    return tn_minus_to
    
def is_portrait(img):
    h, w = img.shape[:2]
    return h > w
    
def convert_fov(img, fov_img_h, fov_img_w, fov_new_h, fov_new_w, is_rad=True):
    fov_new_h = math.radians(55) # 55
    fov_new_w = math.radians(42.65) # 42.65
    fov_img_w = 2 * np.arctan(np.tan(fov_img_h / 2) * 720 / 1280)
    is_port = is_portrait(img)
    if not is_port:   # TODO: 핸드폰 정dml에 따라 API가 알아서 변경해줬는지 체크
        fov_img_h, fov_img_w = fov_img_w, fov_img_h
        fov_new_h, fov_new_w = fov_new_w, fov_new_h
    img_h, img_w = img.shape[:2]
    h_diff = get_length_diff(img_h, fov_img_h, fov_new_h, is_rad)
    w_diff = get_length_diff(img_w, fov_img_w, fov_new_w, is_rad)
    if h_diff > 0 or w_diff > 0:
        h_pad, w_pad = max(0, int(h_diff/2)), max(0, int(w_diff/2))
        padding = [(h_pad, h_pad), (w_pad, w_pad), (0, 0)]
        img = np.pad(img, padding, 'edge')
        # img = cv2.copyMakeBorder(img, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT)
    if h_diff < 0:
        h_margin = -min(0, int(h_diff/2))
        img = img[h_margin:-h_margin, :, :]
    if w_diff < 0:
        w_margin = -min(0, int(w_diff/2))
        img = img[:, w_margin:-w_margin, :]
    if is_port:
        resize_to = (480, 640)
    else:
        resize_to = (640, 480)
    img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
    return img