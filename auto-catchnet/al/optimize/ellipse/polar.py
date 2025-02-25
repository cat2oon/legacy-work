import numpy as np

from scipy import optimize
from ac.langs.decorator.bases import Reference

Reference("www.desmos.com/calculator/wqptzjyuvn")
Reference("www.desmos.com/calculator/holmeyhjmk")
Reference("scipython.com/book/chapter-8-scipy/examples/non-linear-fitting-to-an-ellipse/")


# TODO: translation 적용해야 해서 보류
# mean 좌표를 빼고 계산한 다음 다시 복원하는 방식이 필요할 것 같은데
# 극방정식도 shifted 되어 있어서 애매...
def fit_ellipse_by_polar(contours):
    # parameters
    # k : 회전각
    # a : 주축 길이
    # e : 이심률
    def f(p, theta):  # ellipse polar eq
        k, a, e = p
        return a * (1 - e ** 2) / (1 - e * np.cos(theta - k * np.pi))

    def residuals(p, theta, r):
        return r - f(p, theta)

    def jacobian(p, theta, r):
        k, a, e = p
        kp = k * np.pi
        rt = theta - kp

        da = (1 - p ** 2) / (1 - p * np.cos(rt))
        dp = a * (p ** 2 * np.cos(rt) - 2 * p + np.cos(rt)) / (1 - p * np.cos(rt))
        dk = a * p * np.pi * np.sin(theta - kp) * (1 - p ** 2) / (1 - p * np.cos(theta - kp))

        return np.array([-da, -dp, -dk]).T

    def points_to_polar(points):
        return 0, 0

    p0 = (0, 1, 0.5)
    a_r, a_theta = points_to_polar(contours)
    res = optimize.least_squares(residuals, p0, jac=jacobian, args=(a_r, a_theta))

    return res.x
