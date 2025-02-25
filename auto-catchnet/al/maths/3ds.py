import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def RotX(phi):
    return np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])


def RotY(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def RotZ(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """ pair x, y, z"""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self.vertexes_3d = xs, ys, zs

    def draw(self, renderer):
        xs, ys, zs = self.vertexes_3d
        prj_xs, prj_ys, prj_zs = proj3d.proj_transform(xs, ys, zs, renderer.M)
        self.set_positions((prj_xs[0], prj_ys[0]), (prj_xs[1], prj_ys[1]))
        super().draw(renderer)


def draw_vector(fig, vec_from, vec_to, **kwargs):
    """ get ops """
    proj_on = kwargs.get('proj', True)
    ms = kwargs.get('mutation_scale', 20)
    ars = kwargs.get('arrowstyle', '-|>')
    lc = kwargs.get('line_color', 'k')
    pc = kwargs.get('proj_color', 'k')
    point_enable = kwargs.get('point_enable', True)

    if vec_from.size == 3:  # [x, y, z]
        xs = [vec_from[0], vec_to[0]]
        ys = [vec_from[1], vec_to[1]]
        zs = [vec_from[2], vec_to[2]]
    else:  # 동차변환행렬
        xs = [vec_from[0, 3], vec_to[0, 3]]
        ys = [vec_from[1, 3], vec_to[1, 3]]
        zs = [vec_from[2, 3], vec_to[2, 3]]

    out = Arrow3D(xs, ys, zs, mutation_scale=ms, arrowstyle=ars, color=lc)
    fig.add_artist(out)

    if point_enable:
        fig.scatter(xs[1], ys[1], zs[1], color='k', s=50)

    if proj_on:
        fig.plot(xs, ys, [0, 0], color=pc, linestyle='--')
        fig.plot([xs[0], xs[0]], [ys[0], ys[0]], [0, zs[0]], color=pc, linestyle='--')
        fig.plot([xs[1], xs[1]], [ys[1], ys[1]], [0, zs[1]], color=pc, linestyle='--')


def draw_point_with_axis(fig, *args, **kwargs):
    ms = kwargs.get('mutation_scale', 20)
    ars = kwargs.get('arrowstyle', '->')
    point_enable = kwargs.get('point_enable', True)
    axis_enable = kwargs.get('axis_enable', True)

    if len(args) == 4:
        ORG = args[0]
        hat_X = args[1]
        hat_Y = args[2]
        hat_Z = args[3]
        xs_n = [ORG[0], ORG[0] + hat_X[0]]
        ys_n = [ORG[1], ORG[1] + hat_X[1]]
        zs_n = [ORG[2], ORG[2] + hat_X[2]]
        xs_o = [ORG[0], ORG[0] + hat_Y[0]]
        ys_o = [ORG[1], ORG[1] + hat_Y[1]]
        zs_o = [ORG[2], ORG[2] + hat_Y[2]]
        xs_a = [ORG[0], ORG[0] + hat_Z[0]]
        ys_a = [ORG[1], ORG[1] + hat_Z[1]]
        zs_a = [ORG[2], ORG[2] + hat_Z[2]]
    else:
        tmp = args[0]
        ORG = tmp[:3, 3:]
        hat_X = tmp[:3, 0:1]
        hat_Y = tmp[:3, 1:2]
        hat_Z = tmp[:3, 2:3]
        xs_n = [ORG[0, 0], ORG[0, 0] + hat_X[0, 0]]
        ys_n = [ORG[1, 0], ORG[1, 0] + hat_X[1, 0]]
        zs_n = [ORG[2, 0], ORG[2, 0] + hat_X[2, 0]]
        xs_o = [ORG[0, 0], ORG[0, 0] + hat_Y[0, 0]]
        ys_o = [ORG[1, 0], ORG[1, 0] + hat_Y[1, 0]]
        zs_o = [ORG[2, 0], ORG[2, 0] + hat_Y[2, 0]]
        xs_a = [ORG[0, 0], ORG[0, 0] + hat_Z[0, 0]]
        ys_a = [ORG[1, 0], ORG[1, 0] + hat_Z[1, 0]]
        zs_a = [ORG[2, 0], ORG[2, 0] + hat_Z[2, 0]]

    if point_enable:
        fig.scatter(xs_n[0], ys_n[0], zs_n[0], color='k', s=50)

    if axis_enable:
        n = Arrow3D(xs_n, ys_n, zs_n, mutation_scale=ms, arrowstyle=ars, color='r')
        o = Arrow3D(xs_o, ys_o, zs_o, mutation_scale=ms, arrowstyle=ars, color='g')
        a = Arrow3D(xs_a, ys_a, zs_a, mutation_scale=ms, arrowstyle=ars, color='b')
        fig.add_artist(n)
        fig.add_artist(o)
        fig.add_artist(a)
