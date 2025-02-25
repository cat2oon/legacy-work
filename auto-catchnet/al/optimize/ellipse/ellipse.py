from al.optimize.ellipse.lsq import LeastSquareEllipse


def fit_ellipse(contours):
    lsqe = LeastSquareEllipse()
    lsqe.fit(contours)
    center, width, height, phi = lsqe.parameters()
    center = (center[0].real, center[1].real)
    width, height, phi = width.real, height.real, phi.real

    return center, width, height, phi
