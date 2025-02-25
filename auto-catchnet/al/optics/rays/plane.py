import numpy as np

from ac.common.nps import normalize
from al.maths.angles import degree_to_rad

"""
 - Reference coordinates : pixel coordinate
"""


class PlaneRay:
    def __init__(self, start, vec_heading):
        self.start = np.array(start)
        self.current = self.start
        self.vec = normalize(vec_heading)

    def make_branch_ray(self, theta_degree):
        theta = degree_to_rad(theta_degree)

        rot_mat = np.matrix([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        heading = np.dot(rot_mat, self.vec)
        r = PlaneRay(self.start, heading.A1)
        return r

    def move_next(self, dis_ratio=2):
        next_curr = self.current + self.vec
        self.current = next_curr
        return int(next_curr[0]), int(next_curr[1])

    def __str__(self):
        return "heading {} from {}".format(self.vec, self.start)
