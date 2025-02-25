import cv2
import glob
import numpy as np


def calibration(img_dir, grid_x=7, grid_y=6, draw_image=False):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    object_point_samples = np.zeros((grid_y * grid_x, 3), np.float32)
    object_point_samples[:, :2] = np.mgrid[0:grid_x, 0:grid_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    object_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane.
    image_matched = []

    images = glob.glob('{}/*.jpg'.format(img_dir))

    img = None
    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (grid_x, grid_y), None)

        if ret:
            image_matched.append(img)
            object_points.append(object_point_samples)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

            if draw_image:
                img = cv2.drawChessboardCorners(img, (grid_x, grid_y), corners2, ret)

    img_wh = img.shape[:2]
    ret, camera_mat, distort_coefs, rot_vecs, trans_vecs = cv2.calibrateCamera(object_points,
                                                                               image_points,
                                                                               img_wh,
                                                                               None, None)

    CV

    return camera_mat, distort_coefs, rot_vecs, trans_vecs, image_matched
