import numpy as np

from ac.common.nps import get_norm


class GazeItem:
    camera = None

    def __init__(self, tx, ty, px, py, ex=0, ey=0, ez=0):
        self.image = None
        self.truth_gaze_point = np.array([tx, ty])
        self.iris_center_pixel = np.array([px, py])
        self.eyeball_center_offset = np.array([ex, ey, ez])

    @staticmethod
    def set_camera(camera):
        GazeItem.camera = camera

    def get_oc_vec(self):
        px, py = self.iris_center_pixel
        i, j, k = GazeItem.camera.get_normalized_uv_vec_from_px_position(px, py)
        return np.array([i, j, k])

    def get_gaze_point(self):
        return self.truth_gaze_point

    def get_dist_error(self, pred_x, pred_y):
        tx, ty = self.truth_gaze_point
        return (pred_x - tx) ** 2 + (pred_y - ty) ** 2

    def evaluate(self, eye):
        tx, ty = self.truth_gaze_point
        pred_x, pred_y = eye.estimate_gaze_point(self.get_oc_vec())
        print("*** NORM: {} ***".format(get_norm([pred_x-tx, pred_y-ty])))
