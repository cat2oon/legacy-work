import cv2
import numpy as np

from scipy import constants
from al.optics.rays.plane import PlaneRay
from ac.images.filters.filters import to_grey_and_gaussian_blur

"""
@ Starburst customized for iris disk in RGB CAM
- 3차원 공간에 정의된 eyeball 중심 좌표값의 이미지 대응 2d 좌표값
- 이미지 좌표에서 정의된 iris 중심 guess 좌표값

TODO: 
- 실제로 안구 중심에서 홍채까지의 거리는 안구 반지름을 넘어갈 수 없다. 
  따라서, 탐색 범위를 반지름 최대값 (또는 프로파일 측정 값) 픽셀 단위로 
  한정할 수 있다. 
- eyelid contour가 주어진 경우 iris contour에서 제외할 수 있다. 
"""


def extract_gradient(img):
    pass


def detect_iris_contour(img,
                        iris_center,
                        eyeball_center,
                        num_rays=250,
                        edge_thredshold=35):
    ic, ec = iris_center, eyeball_center

    # 1. remove noise
    img = to_grey_and_gaussian_blur(img, kernel=(5, 5), sigma_x=2, sigma_y=2)

    # 2. calculate gradient magnitude and direction
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = grad_x ** 2 + grad_y ** 2
    grad_atan = np.arctan2(grad_x, grad_y)

    # 3. get H vector with iris_center_pos and eyeball center 2d pos (transformed)
    heading = np.array([ic[0] - ec[0], ic[1] - ec[1]])
    H = PlaneRay(eyeball_center, heading)

    # 4. shoot rays
    rays = []
    max_angle = num_rays // 2
    for angle in range(-max_angle, max_angle):
        ray = H.make_branch_ray(angle)
        rays.append(ray)

    # 5. compute edge scores
    table = {}
    for ray in rays:
        x, y = ray.move_next()
        while is_inside(img, x, y):
            m = magnitude[x, y]
            gx = grad_x[x, y]
            gy = grad_y[x, y]
            d = np.array([gx / m, gy / m])
            active = abs(np.dot(H.vec, d))
            active = active * (active > 0.5)
            score = m * active
            table["{}:{}".format(x, y)] = (m, gx, gy, active, score)
            x, y = ray.move_next()

    return img, grad_x, grad_y, table


def is_inside(img, x, y):
    h, w = img.shape[:2]
    if x < 0 or y < 0:
        return False
    if x >= h or y >= w:
        return False
    return True
