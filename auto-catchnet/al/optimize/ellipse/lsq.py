import numpy as np

from ac.langs.decorator.bases import Reference

Reference("https://github.com/bdhammel/least-squares-ellipse-fitting")


class LeastSquareEllipse:
    def __init__(self):
        self.coef = None

    def fit(self, contours):
        # returns [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g

        x, y = np.asarray(contours, dtype=np.float32)

        # Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.mat(np.vstack([x ** 2, x * y, y ** 2])).T
        # Linear part of design matrix [eqn. 16] from (*)
        D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T

        # forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T * D1
        S2 = D1.T * D2
        S3 = D2.T * D2

        # Constraint matrix [eqn. 18]
        C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

        # Reduced scatter matrix [eqn. 29]
        M = C1.I * (S1 - S2 * S3.I * S2.T)

        # M*|a b c >=l|a b c >. Find eigenvalues and eigen-vectors from this equation [eqn. 28]
        eig_val, eig_vec = np.linalg.eig(M)

        # eigen-vector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * np.multiply(eig_vec[0, :], eig_vec[2, :]) - np.power(eig_vec[1, :], 2)
        a1 = eig_vec[:, np.nonzero(cond.A > 0)[1]]

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -S3.I * S2.T * a1

        # eigen-vectors |a b c d f g>
        self.coef = np.vstack([a1, a2])
        self._save_parameters()

    def _save_parameters(self):
        """
        return
            center (List): of the form [x0, y0]
            width (float): major axis
            height (float): minor axis
            phi (float): rotation of major axis form the x-axis in radians
        """

        # eigen-vectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
        a = self.coef[0, 0]
        b = self.coef[1, 0] / 2.
        c = self.coef[2, 0]
        d = self.coef[3, 0] / 2.
        f = self.coef[4, 0] / 2.
        g = self.coef[5, 0]

        # finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c * d - b * f) / (b ** 2. - a * c)
        y0 = (a * f - b * d) / (b ** 2. - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        denominator1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        denominator2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        width = np.sqrt(numerator / denominator1)
        height = np.sqrt(numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        # or [eqn. 26] from (***).
        phi = .5 * np.arctan((2. * b) / (a - c))

        self._center = [x0, y0]
        self._width = width
        self._height = height
        self._phi = phi

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def phi(self):
        # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        return self._phi

    def parameters(self):
        return self.center, self.width, self.height, self.phi
