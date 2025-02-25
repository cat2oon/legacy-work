import b2ac.conversion
import b2ac.fit
import b2ac.preprocess

from ac.langs.decorator.bases import Reference

Reference("https://github.com/hbldh/b2ac")


# TODO: conic 계수에서 변환할 때 numerical unstable 버그 있음
def fit_ellipse_b2ac(contours):
    points, x_mean, y_mean = b2ac.preprocess.remove_mean_values(contours)
    conic_coefficient = b2ac.fit.fit_improved_B2AC_numpy(points)
    general_form = b2ac.conversion.conic_to_general_reference(conic_coefficient)
    general_form[0][0] += x_mean
    general_form[0][1] += y_mean

    return general_form
