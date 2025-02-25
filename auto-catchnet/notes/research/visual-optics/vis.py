import sys
import itertools
import matplotlib.pyplot as plt

sys.path.append("../../../")

from ai.feature.face.candide import *
from ai.feature.face.facial import *

mean_3d, blend_shape, mesh, index_3d, index_2d = load_3D_face_model("/home/chy/archive-models/candide/candide.npz")

face = FacialModel()

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
plt.axis('equal', adjustable='datalim')

colors = itertools.cycle(["b", "g", "r", "c", "k"])
# targets = range(mean_3d.shape[1])
targets = [
    53, 56, 23, 20
]

for i in targets:
    vec = mean_3d[:, i]
    xs = vec[0] * 100
    ys = vec[1] * 100
    zs = vec[2] * 100
    ax.scatter(xs, ys, zs, c=next(colors), marker='D', s=50)
    ax.text(xs, ys, zs, "{}".format(i), size=10)

    # ax.scatter(xs, ys, c=next(colors), marker='s', s=50)
    # ax.text(xs, ys, "{}".format(i), size=10)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# for h_angle in range(0, int(360 / 4)):
#     for v_angle in range(0, int(360 / 4)):
#         ax.view_init(v_angle, h_angle * 4)
#         plt.draw()
#         plt.pause(.0001)

ax.view_init(-110, 270)
plt.show()
