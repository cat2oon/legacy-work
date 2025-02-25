from enum import Enum, unique


@unique
class MetricsKeys(Enum):

    UID = 0x1               # global unique id (each image)
    DEVICE_NAME = 0x10      # device name

    POINT_SX = 0x110        # x scale coordinate of point
    POINT_SY = 0x111        # y scale coordinate of point

    HEAD_YAW = 0x120        # head pose angle yaw
    HEAD_PITCH = 0x121      # head pose angle pitch
    HEAD_ROLL = 0x122       # head pose angle roll
    HEAD_DISTANCE = 0x123   # rough distance between camera and subject

    GAZE_YAW = 0x130        # gaze vector yaw
    GAZE_PITCH = 0x131      # gaze vector pitch
    GAZE_ROLL = 0x132       # gaze vector roll

    EYE_GRID_R = 0x140      # right eye grid index
    EYE_GRID_L = 0x141      # left eye grid index

    CAMERA_FOV_V = 0x150    # camera vertical fov
    CAMERA_FOV_H = 0x151    # camera horizontal fov

