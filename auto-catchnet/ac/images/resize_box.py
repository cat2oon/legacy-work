

class ResizeBox:
    def __init__(self,
                 resize_h,
                 resize_w,
                 pad_top,
                 pad_left,
                 pad_right,
                 pad_bottom):
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.pad_top = pad_top
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_bottom = pad_bottom

    def set_new_h(self, height):
        self.resize_h = height

    def set_new_w(self, width):
        self.resize_w = width
