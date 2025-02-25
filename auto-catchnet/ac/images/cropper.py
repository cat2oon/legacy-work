def crop_image(img, crop_rect):
    up_left_xy = crop_rect[0]
    down_right_xy = crop_rect[1]
    x0, x1 = up_left_xy[0], down_right_xy[0]
    y0, y1 = up_left_xy[1], down_right_xy[1]
    x0, x1 = int(x0), int(x1)
    y0, y1 = int(y0), int(y1)
    return img[y0:y1, x0:x1, :]

