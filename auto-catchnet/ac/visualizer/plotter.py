import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.patches import Ellipse

from mpl_toolkits.axes_grid1 import ImageGrid
from ac.common.images import shape_to_hw
from ac.langs.sequences import chunks
from al.maths.angles import degree_to_rad
from al.optics.rays.plane import PlaneRay


def select_colormap_by_shape(shape):
    if len(shape) is 3 and shape[2] == 1:
        return 'gray'
    elif len(shape) is 2:
        return 'gray'
    return 'viridis'


def squeeze_if_gray(img):
    if len(img.shape) == 3 and img.shape[2] == 1:
        return np.squeeze(img)
    return img


def get_image_extent(img):
    h, w = shape_to_hw(img.shape)
    return 0, h, 0, w


def show_image(img, cmap=None, extent=None, title=None, fig_size=(10, 10)):
    if cmap is None:
        cmap = select_colormap_by_shape(img.shape)
    if extent is None:
        extent = get_image_extent(img)

    img = squeeze_if_gray(img)
    fig = plt.figure(figsize=fig_size)

    plt.axis('off')
    if title is not None:
        plt.title(title)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap, interpolation='none', extent=extent)
    plt.axis('image')

    return ax


def show_images(images, cmap='gray', fig_size=(10, 10), extent=(0, 1200, 0, 1600)):
    num_row = len(images)
    fig = plt.figure(figsize=fig_size)
    for i, img in enumerate(images):
        ax = fig.add_subplot(1, num_row, i + 1)
        ax.imshow(img, cmap=cmap, interpolation='none', extent=extent)


def show_pair_images(images, cmap='gray', fig_size=(10, 10)):
    num_row = len(images)
    fig = plt.figure(figsize=fig_size)
    for i, img_pair in enumerate(images):
        l_img, r_img = img_pair
        ax = fig.add_subplot(num_row, 2, 2 * i + 1)
        ax.imshow(l_img, cmap=cmap, interpolation='none')
        ax = fig.add_subplot(num_row, 2, 2 * i + 2)
        ax.imshow(r_img, cmap=cmap, interpolation='none')


def show_triple_image(images, cmap='gray', fig_size=(10, 10), extent=(0, 1200, 0, 1600)):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(images[0], cmap=cmap, interpolation='none', extent=extent)
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(images[1], cmap=cmap, interpolation='none', extent=extent)
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(images[2], cmap=cmap, interpolation='none', extent=extent)


def show_grid_images(images, fig_size=(2, 2), cmap=None, extent=None):
    sample = images[0]
    if cmap is None:
        cmap = select_colormap_by_shape(sample.shape)
    if extent is None:
        extent = get_image_extent(sample)

    img_rows = chunks(images, 4)
    num_rows = len(img_rows)
    fig = plt.figure(figsize=fig_size)

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(num_rows, 4),
                     axes_pad=0.0,
                     share_all=True,
                     label_mode="L",
                     cbar_mode="single")

    for i, img in enumerate(images):
        im = grid[i].imshow(img, cmap=cmap, extent=extent, interpolation="nearest")
    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    grid.axes_llc.set_xticks([-2, 0, 2])
    grid.axes_llc.set_yticks([-2, 0, 2])


def draw_point(ax, x, y, color='red', marker='D', size=5):
    ax.scatter(x, y, c=color, marker=marker, s=size)


def draw_line(ax, start, to, tick=2, color='blue', linestyle='-'):
    line = lines.Line2D((start[0], to[0]), (start[1], to[1]), color=color, linewidth=tick, linestyle=linestyle)
    ax.add_line(line)


def draw_ray(ax, ray: PlaneRay, tick=2, color='blue', linestyle='-'):
    start = (ray.start[0], ray.start[1])
    to = (start[0] + 10 * ray.vec[0], start[1] + 10 * ray.vec[1])
    draw_line(ax, start, to, tick=tick, color=color, linestyle=linestyle)


def render_head_pose(img, pitch, yaw, roll, tdx=None, tdy=None, size=100):
    # 좌표 + 땅/좌/시계 기준
    pitch = degree_to_rad(pitch)
    roll = degree_to_rad(roll)
    yaw = degree_to_rad(-yaw)

    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx, tdy = width / 2, height / 2

    cos, sin = np.cos, np.sin

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


"""
 plotter
"""


def plot_points(data, fmt="ro", label="", fig_size=(6, 6), z_order=1):
    plt.close('all')

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.plot(data[0], data[1], fmt, label=label, zorder=z_order)

    return ax


def make_ellipse(center, width, height, phi, edge_color='b', line_weight=2, z_order=2):
    e = Ellipse(xy=center,
                width=width,
                height=height,
                angle=np.rad2deg(phi),
                edgecolor=edge_color,
                fc='None',
                lw=line_weight,
                zorder=z_order)
    return e


def draw_ellipse(center, width, height, phi, xlim, ylim,
                 edge_color='b', line_weight=2, fig_size=(6, 6), z_order=2):
    plt.close('all')
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, aspect='auto')
    ax.axis('equal')

    e = make_ellipse(center, width, height, phi, edge_color, line_weight, z_order)
    ax.add_patch(e)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    plt.show()


def draw_ellipse_on(ax, center, width, height, phi, edge_color='b', line_weight=2, z_order=2):
    e = make_ellipse(center, width, height, phi, edge_color, line_weight, z_order)
    ax.add_patch(e)
