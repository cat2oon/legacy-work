from enum import Enum, unique


@unique
class MeasureKeys(Enum):

    # internal
    UID = 0x1               # global unique id (each image)
    IID = 0x2               # item id in perf (from original dataset)
    PID = 0x3               # perf id (each person)
    DEVICE_NAME = 0x4       # device name

    POINT_X = 0x40          # x coordinate of point
    POINT_Y = 0x41          # y coordinate of point
    CAMERA_X = 0x50         # camera x distance from point
    CAMERA_Y = 0x51         # camera y distance from point

    CAMERA_TO_SCREEN_X = 0x52   # from camera to screen x distance
    CAMERA_TO_SCREEN_Y = 0x53   # from camera to screen y distance

    SCREEN_H = 0x60         # screen height size (ios unit)
    SCREEN_W = 0x61         # screen width size (ios unit)
    ORIENTATION = 0x62      # screen orientation

    FACE_GRID_X = 0x63      # face grid x
    FACE_GRID_Y = 0x64      # face grid y
    FACE_GRID_W = 0x65      # face grid width  (x,w)
    FACE_GRID_H = 0x66      # face grid height (y,h)
    FACE_VALID = 0x67       # face is_valid

    # 크롭이 필요한 경우 사용
    FACE_X = 0x70           # face x pos
    FACE_Y = 0x71           # face y pos
    FACE_W = 0x72           # face width
    FACE_H = 0x73           # face height
    EYE_LEFT_X = 0x74       # left eye x pos
    EYE_LEFT_Y = 0x75       # left eye y pos
    EYE_LEFT_W = 0x76       # left eye width
    EYE_LEFT_H = 0x77       # left eye height
    EYE_RIGHT_X = 0x78      # right eye x pos
    EYE_RIGHT_Y = 0x79      # right eye y pos
    EYE_RIGHT_W = 0x7A      # right eye width
    EYE_RIGHT_H = 0x7B      # right eye height

    IMAGE_FRAME = 0x80      # frame image binary
    IMAGE_EYE_LEFT = 0x81   # left eye image binary
    IMAGE_EYE_RIGHT = 0x82  # right eye image binary
    IMAGE_FACE = 0x83       # face image binary

    FRAME_PATH = 0x90       # frame image path
    EYE_LEFT_PATH = 0x91    # left eye image path
    EYE_RIGHT_PATH = 0x92   # right eye image path
    FACE_PATH = 0x93        # face image path

    @staticmethod
    def from_origin_key(str):
        # TODO : mapping table
        return MeasureKeys.CAMERA_FOV_H

    @staticmethod
    def make_empty_measures():
        return {key: None for key in MeasureKeys}

    @staticmethod
    def writable_meta_keys():
        return [
            MeasureKeys.UID,
            MeasureKeys.IID,
            MeasureKeys.PID,
            MeasureKeys.DEVICE_NAME,
            MeasureKeys.POINT_X,
            MeasureKeys.POINT_Y,
            MeasureKeys.CAMERA_X,
            MeasureKeys.CAMERA_Y,
            MeasureKeys.CAMERA_TO_SCREEN_X,
            MeasureKeys.CAMERA_TO_SCREEN_Y,
            MeasureKeys.SCREEN_H,
            MeasureKeys.SCREEN_W,
            MeasureKeys.ORIENTATION,
            MeasureKeys.FACE_GRID_X,
            MeasureKeys.FACE_GRID_Y,
            MeasureKeys.FACE_GRID_W,
            MeasureKeys.FACE_GRID_H,
            MeasureKeys.FACE_VALID,
            MeasureKeys.FRAME_PATH,
            MeasureKeys.EYE_LEFT_PATH,
            MeasureKeys.EYE_RIGHT_PATH,
            MeasureKeys.FACE_PATH,
        ]

    @staticmethod
    def writable_image_keys():
        return [
            MeasureKeys.IMAGE_FRAME,
            MeasureKeys.IMAGE_EYE_LEFT,
            MeasureKeys.IMAGE_EYE_RIGHT,
            MeasureKeys.IMAGE_FACE,
        ]


ATTR_TO_KEY = {
    "uid": MeasureKeys.UID,
    "iid": MeasureKeys.IID,
    "pid": MeasureKeys.PID,
    "device": MeasureKeys.DEVICE_NAME,
    "point_x": MeasureKeys.POINT_X,
    "point_y": MeasureKeys.POINT_Y,
    "camera_x": MeasureKeys.CAMERA_X,
    "camera_y": MeasureKeys.CAMERA_Y,
    "camera_to_screen_x": MeasureKeys.CAMERA_TO_SCREEN_X,
    "camera_to_screen_y": MeasureKeys.CAMERA_TO_SCREEN_Y,
    "screen_h": MeasureKeys.SCREEN_H,
    "screen_w": MeasureKeys.SCREEN_W,
    "orientation": MeasureKeys.ORIENTATION,
    "face_grid_x": MeasureKeys.FACE_GRID_X,
    "face_grid_y": MeasureKeys.FACE_GRID_Y,
    "face_grid_w": MeasureKeys.FACE_GRID_W,
    "face_grid_h": MeasureKeys.FACE_GRID_H,
    "face_grid_valid": MeasureKeys.FACE_VALID,
    "path_frame": MeasureKeys.FRAME_PATH,
    "path_left_eye": MeasureKeys.EYE_LEFT_PATH,
    "path_right_eye": MeasureKeys.EYE_RIGHT_PATH,
    "path_face": MeasureKeys.FACE_PATH,
    "image_frame": MeasureKeys.IMAGE_FRAME,
    "image_left_eye": MeasureKeys.IMAGE_EYE_LEFT,
    "image_right_eye": MeasureKeys.IMAGE_EYE_RIGHT,
    "image_face": MeasureKeys.IMAGE_FACE,
}

KEY_TO_ATTR = {v: k for k, v in ATTR_TO_KEY.items()}

