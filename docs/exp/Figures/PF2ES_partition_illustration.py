"""
Multi-Panel Plot of Illustration for PF2ES partition
"""

from matplotlib import pyplot as plt
# import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.partition import ExactPartition2dNonDominated

# plt.rcParams['text.usetex'] = True

import matplotlib.path as mpath
import matplotlib.patches as mpatches

label_fontsize = 15
frame_linewidth = 1.5
tx_ftz = 15
legend_fontsize=15
# fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5))
fig = plt.figure(figsize=(12, 4))
axis = []
legend_plot_usage = 5
axis.append(fig.add_subplot(1, legend_plot_usage, 1))
axis.append(fig.add_subplot(1, legend_plot_usage, 2))
axis.append(fig.add_subplot(1, legend_plot_usage, 3, projection='3d'))

# first plot


axis[0].annotate('', xy=(1, 0.0),
                 xycoords='data',
                 xytext=(0, 0),
                 arrowprops=dict(lw=1, width=1, headwidth=10, color='k')
                 )

axis[0].annotate('', xy=(0.0, 1),
                 xycoords='data',
                 xytext=(0, 0),
                 arrowprops=dict(lw=1, width=1, headwidth=10, color='k')
                 )

# def segment_draw_function(x):
#     if x < 0.2:
#         return 0.7
#     elif 0.2 <= x <= 0.7:
#         return np.sqrt(0.5 ** 2 - (x - 0.2) ** 2) + 0.2  # 0.7 - (x - 0.2)
#     else: # x > 0.8
#         return 0.1


axis[0].plot(0.1, 0.1, marker='s', markersize=10 - 2, markerfacecolor='tab:blue', color='k')
axis[0].plot(0.9, 0.9, marker='^', markersize=10, markerfacecolor='tab:blue', color='k')

axis[0].plot([0.1, 0.1], [0.1, 0.7], color='k', linewidth=frame_linewidth, linestyle='--', zorder=0)
axis[0].plot([0.1, 0.9], [0.1, 0.1], color='k', linewidth=frame_linewidth, linestyle='--', zorder=0)
axis[0].plot([0.1, 0.9], [0.9, 0.9], color='k', linewidth=frame_linewidth, zorder=0)
axis[0].plot([0.1, 0.2], [0.7, 0.7], color='k', linewidth=frame_linewidth, zorder=0)
axis[0].plot([0.9, 0.9], [0.1, 0.9], color='k', linewidth=frame_linewidth, zorder=0)
axis[0].plot([0.8, 0.8], [0.25, 0.1], color='k', linewidth=frame_linewidth, zorder=0)
axis[0].plot([0.1, 0.1], [0.7, 0.9], color='k', linewidth=frame_linewidth, zorder=0)
axis[0].plot([0.8, 0.9], [0.1, 0.1], color='k', linewidth=frame_linewidth, zorder=0)
# Path = mpath.Path
# pp1 = mpatches.PathPatch(
#     Path([(0.2, 0.8), (0.5, 0), (0.8, 0.2), (0, 0)],
#          [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
#     fc="none", transform=axis[0].transData)
#
# axis[0].add_patch(pp1)
# [0.7, 0.7, 0.64375, 0.5875, 0.53125, 0.475, 0.41875, 0.3625, 0.30625, 0.25, 0.1, 0.1]
# np.array([0.7, 0.7, 0.64375, 0.5875, 0.53125, 0.475, 0.41875, 0.3625, 0.30625, 0.25, 0.1, 0.1])
xs = [0.2, 0.281, 0.395, 0.473, 0.539, 0.606, 0.688, 0.756, 0.80]
ys = [0.7, 0.68, 0.6312, 0.567, 0.501, 0.438, 0.385, 0.323, 0.25]

from scipy.interpolate.interpolate import interp1d

f = interp1d(xs, ys, kind='quadratic')
aug_xs = np.linspace(0.2, 0.8, 100)
aug_ys = f(aug_xs)
# np.linspace(0.2, 0.8, 9)
axis[0].fill_between(np.hstack([0.1, 0.2, aug_xs, 0.8, 0.9]),
                     np.hstack([0.7, 0.7, aug_ys, 0.1, 0.1]),
                     0.9 * np.ones(shape=104),
                     color='#ffe699', alpha=1.0, linewidth=0.0)
axis[0].plot(aug_xs, aug_ys, linewidth=3, color='tab:red', label='Realization of $\mathcal{F}$')

axis[0].text(0.581, 0.722, r'$A$', fontsize=tx_ftz)
axis[0].text(0.311, 0.340, r'$\overline{A}$', fontsize=tx_ftz)
## set stuff to invisible
axis[0].set_xlim([0, 1])
axis[0].set_ylim([0, 1])
axis[0].spines["top"].set_visible(False)
axis[0].spines["right"].set_visible(False)
axis[0].spines["left"].set_visible(False)
axis[0].spines["bottom"].set_visible(False)
axis[0].set_xticks([])
axis[0].set_yticks([])

axis[0].set_xlabel('Objective 1', fontsize=label_fontsize)
axis[0].set_ylabel('Objective 2', fontsize=label_fontsize)

box = axis[0].get_position()
axis[0].set_position([box.x0, box.y0 + box.height * 0.15,
                 box.width, box.height * 0.85])

# ---------------------------------------------------------
# Sub-Figure 2
import matplotlib.patches as patches
from trieste.acquisition.multi_objective.pareto import Pareto

axis[1].annotate('', xy=(1, 0.0),
                 xycoords='data',
                 xytext=(0, 0),
                 arrowprops=dict(lw=1, width=1, headwidth=10, color='k')
                 )

axis[1].annotate('', xy=(0.0, 1),
                 xycoords='data',
                 xytext=(0, 0),
                 arrowprops=dict(lw=1, width=1, headwidth=10, color='k')
                 )


axis[1].plot([0.1, 0.1], [0.1, f(0.3)], color='k', linewidth=frame_linewidth, linestyle='--', zorder=0)
axis[1].plot([0.1, 0.7], [0.1, 0.1], color='k', linewidth=frame_linewidth, linestyle='--', zorder=0)
# axis[1].plot([0.1, 0.9], [0.9, 0.9], color='k', linewidth=frame_linewidth, zorder=0)
# axis[1].plot([0.9, 0.9], [0.1, 0.9], color='k', linewidth=frame_linewidth, zorder=0)
# axis[1].plot([0.1, 0.1], [f(0.3), 0.9], color='k', linewidth=frame_linewidth, zorder=0)
# axis[1].plot([0.7, 0.9], [0.1, 0.1], color='k', linewidth=frame_linewidth, zorder=0)


obs = [[0.3, f(0.3)], [0.5, f(0.5)], [0.7, f(0.7)]]
lb = np.array([[0.1, f(0.3)], [0.3, f(0.5)], [0.5, f(0.7)], [0.7, 0.1]])
ub = np.array([[0.3, 0.9], [0.5, 0.9], [0.7, 0.9], [0.9, 0.9]])
for i in range(lb.shape[0]):
    rect = patches.Rectangle(lb[i], (ub - lb)[i, 0], (ub - lb)[i, 1],
                             linewidth=1, edgecolor='k', facecolor='#ffe699', alpha=1.0)
    axis[1].add_patch(rect)
axis[1].scatter(np.asarray(obs)[:, 0], np.asarray(obs)[:, 1], s=100, marker='*', c='tab:red',
                edgecolors='k', zorder=100)

axis[1].plot(0.1, 0.1, marker='s', markersize=10 - 2, markerfacecolor='tab:blue', color='k')
axis[1].plot(0.9, 0.9, marker='^', markersize=10, markerfacecolor='tab:blue', color='k')

axis[1].text(0.157, 0.722, r'$\tilde{A}_1$', fontsize=tx_ftz)
axis[1].text(0.358, 0.722, r'$\tilde{A}_2$', fontsize=tx_ftz)
axis[1].text(0.559, 0.722, r'$\tilde{A}_3$', fontsize=tx_ftz)
axis[1].text(0.756, 0.722, r'$\tilde{A}_4$', fontsize=tx_ftz)
axis[1].text(0.25, 0.25, r'$\tilde{\overline{A}}$', fontsize=tx_ftz)
# axis[1].set_title('2D Demo of partition the dominated region')

axis[1].set_xlim([0, 1])
axis[1].set_ylim([0, 1])
axis[1].spines["top"].set_visible(False)
axis[1].spines["right"].set_visible(False)
axis[1].spines["left"].set_visible(False)
axis[1].spines["bottom"].set_visible(False)
axis[1].set_xticks([])
axis[1].set_yticks([])

axis[1].set_xlabel('Objective 1', fontsize=label_fontsize)
axis[1].set_ylabel('Objective 2', fontsize=label_fontsize)

box = axis[1].get_position()
axis[1].set_position([box.x0, box.y0 + box.height * 0.15,
                 box.width, box.height * 0.85])
# --------------------------------------------------------
# Sub-Figure 3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

import tensorflow as tf
from trieste.acquisition.multi_objective.partition import HypervolumeBoxDecompositionIncrementalDominated
objs = tf.constant(
      [[0.9460075 , 0.        ],
       [0.9460075, 0.22119921],
       [0.83212055, 0.43212055],
       [0.22119921, 1.0],
       [0.        , 1.5 ]], dtype=tf.float32)
# true_obsrv = tf.concat([objs, tf.constant([[0.3], [0.4], [0.5], [0.0], [0.8]])], 1)
true_obsrv = tf.concat([objs, tf.constant([[0.4], [0.4], [0.5], [0.0], [0.0]])], 1)
projected_feasible_obsrv = tf.concat([objs, tf.zeros(shape=(5, 1))], 1)

Ideal_Point = tf.constant([3.0, 3.0, 1.0])
lb, ub = HypervolumeBoxDecompositionIncrementalDominated(projected_feasible_obsrv, Ideal_Point).partition_bounds()

def cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(positions, sizes=None, colors=None, alpha=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(r'#ffe699', 6), alpha=alpha, **kwargs)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# produce figure
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)

# ax.text(1, 1, 1.8, r'Area:$A$', fontsize=20)
# add feasible surface
x = np.linspace(-1, 3, 10)
y = np.linspace(-1, 3, 10)
X, Y = np.meshgrid(x, y)
Z = 0.0 * X + 0.0 * Y
surf = axis[2].plot_surface(X, Y, Z, alpha=0.1)

for pos, size in zip(lb.numpy().tolist(), (ub - lb).numpy().tolist()):
    pc = plotCubeAt2([pos], [size], edgecolor='k', alpha=0.2, colors=r'#ffe699', zorder=0)  # #1f77b4
    axis[2].add_collection3d(pc)

true_obsrv = tf.constant(
    [[0.9460075, 0.43212055, 0.4],
     [0.83212055, 1.0, 0.5],
     [0.22119921, 1.5, 0.0]], dtype=tf.float32)
projected_feasible_obsrv = tf.concat([true_obsrv[..., :-1], tf.zeros(shape=(3, 1))], 1)
axis[2].scatter(true_obsrv[:, 0], true_obsrv[:, 1], true_obsrv[:, 2], s=80, color='g', label='Observations',
           zorder=200)
axis[2].scatter(projected_feasible_obsrv[:, 0], projected_feasible_obsrv[:, 1],
           projected_feasible_obsrv[:, 2], s=100, color='tab:red', label='(Feasible) Pareto Frontier', marker="*",
           edgecolors='k', linewidths=1, zorder=20)
axis[2].scatter(Ideal_Point[0], Ideal_Point[1], Ideal_Point[2], s=100, color='tab:blue', label='Ideal Point', marker="^",
           edgecolors='k', linewidths=1, zorder=20)
axis[2].scatter(0, 0, -1, s=80, color='tab:blue', label='Anti Ideal Point', marker="s",
           edgecolors='k', linewidths=1)

## constraint axes
a = Arrow3D([-0.5, -0.5], [-0.5, -0.5], [0.0, 3], **arrow_prop_dict, zorder=20)
axis[2].add_artist(a)
b = Arrow3D([-0.5, 2.0], [-0.5, -0.5], [0.0, 0], **arrow_prop_dict, zorder=20)
axis[2].add_artist(b)
c = Arrow3D([-0.5, -0.5], [-0.5, 2.0], [0.0, 0], **arrow_prop_dict, zorder=20)
axis[2].add_artist(c)
# text_options = {'horizontalalignment': 'center',
#                 'verticalalignment': 'center',
#                 'fontsize': 14}
axis[2].text(1.5, -1, -0.9, r'Objective 2', (1, 0, 0), fontsize=15)
axis[2].text(-1, 0.0, 0.0, r'Objective 1', (0, 1, 0), fontsize=15)
axis[2].text(-0.5, -0.3, 0.5, r'Constraint', (0, 0, 1), fontsize=15, zorder=20)
axis[2].text(2.6, -1.2, 1.8, r'$\tilde{A}_1$', (0, 1, 0), fontsize=15)
axis[2].text(2.6, -0.6, 1.8, r'$\tilde{A}_2$', (0, 1, 0), fontsize=15)
axis[2].text(2.6, 0.1, 1.8, r'$\tilde{A}_3$', (0, 1, 0), fontsize=15)
axis[2].text(2.6, 1.0, 1.8, r'$\tilde{A}_4$', (0, 1, 0), fontsize=15)

line1_xs = [0.9460075, 0.0]
line1_ys = [0.0, 0.0]
line1_zs = [0.0, 0.0]
axis[2].plot(line1_xs, line1_ys, line1_zs, color='k', linestyle="--")

line2_xs = [0.0, 0.0]
line2_ys = [0.0, 1.5]
line2_zs = [0.0, 0.0]
axis[2].plot(line2_xs, line2_ys, line2_zs, color='k', linestyle="--")

line3_xs = [0.0, 0.0]
line3_ys = [0.0, 0.0]
line3_zs = [0.0, -1.0]
axis[2].plot(line3_xs, line3_ys, line3_zs, color='k', linestyle="--")
# make the panes transparent
axis[2].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
axis[2].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
axis[2].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
axis[2].xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
axis[2].yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
axis[2].zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
axis[2].set_box_aspect([1, 1, 1])
#
# ax.scatter(*worst_point, s=20, color='g', label='Worst Point')
plt.axis('off')
axis[2].set_zlim([-2, 4])
# ax.view_init(elev=-27, azim=44)
axis[2].view_init(elev=-23, azim=52)
axis[2].dist = 6.5
box = axis[2].get_position()
axis[2].set_position([box.x0, box.y0 - box.height * 0.1,
                 box.width * 1.3, box.height * 1.3])
# plt.tight_layout()
handles, labels = axis[2].get_legend_handles_labels()
handles2, labels2 = axis[0].get_legend_handles_labels()
# fig.legend([*handles, *handles2], [*labels, *labels2], loc='lower center', bbox_to_anchor=(0.5, 0.0),
#                fancybox=False, ncol=5, fontsize=legend_fontsize)
fig.legend([*handles, *handles2], [*labels, *labels2], loc='center right', bbox_to_anchor=(1, 0.5),
               fancybox=False, ncol=1, fontsize=legend_fontsize)

plt.savefig('Partition Illustration Legend.png', dpi=1000)
# plt.show()
