import numpy as np

from scipy import optimize

from ac.langs.decorator.bases import Reference

Reference("https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html")


def fit_circle(contours, center_estimate=None):
    x, y = contours[:, 0], contours[:, 1]

    def f(xc, yc):
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def object_f(c):
        Ri = f(*c)
        return Ri - Ri.mean()

    def jacobian(c):
        xc, yc = c
        df_dc = np.empty((2, x.size))   # 기본값

        Ri = f(xc, yc)
        df_dc[0] = (xc - x) / Ri        # df_dxc
        df_dc[1] = (yc - y) / Ri        # df_dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]

        return df_dc

    if center_estimate is None:
        center_estimate = (x.mean(), y.mean())

    # res = optimize.least_squares(object_f, center_estimate, jac=jacobian)    # (col_derive 매개변수 없는 듯)
    center, _ = optimize.leastsq(object_f, center_estimate, Dfun=jacobian, col_deriv=True)
    dists = f(*center)
    radius = dists.mean()

    return center, radius
