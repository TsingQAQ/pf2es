"""
Multi-Panel Plot of Illustration for PF2ES partition
"""

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from trieste.objectives.multi_objectives import VLMOP2
import trieste
from trieste.space import Box, SearchSpace
from trieste.data import Dataset
from trieste.models.interfaces import TrainableHasTrajectoryAndPredictJointReparamModelStack
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.models.gpflow.builders import build_gpr
from trieste.acquisition.function.multi_objective import PF2ES
from trieste.acquisition.multi_objective.partition import ExactPartition2dNonDominated

plt.rcParams['text.usetex'] = True

import matplotlib.path as mpath
import matplotlib.patches as mpatches


label_fontsize = 15
frame_linewidth = 1.5
tx_ftz = 15
legend_fontsize=15
# fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.5))
# fig = plt.figure(figsize=(12, 4))
fig = plt.figure(figsize=(15, 4)) # LEGEND usage
axis = []
axis.append(fig.add_subplot(1, 3, 1))
axis.append(fig.add_subplot(1, 3, 2))
axis.append(fig.add_subplot(1, 3, 3))

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

import matplotlib.patches as patches
from trieste.acquisition.multi_objective.pareto import Pareto


def lower_bound(_xs):
    if _xs <= 0.3:
        return f(0.3)
    elif 0.3<_xs<=0.5:
        return f(0.5)
    elif 0.5<_xs<=0.7:
        return f(0.7)
    else:
        return 0.1
f = interp1d(xs, ys, kind='quadratic')
aug_xs = np.linspace(0.2, 0.8, 100)
aug_ys = f(aug_xs)
aug_ys_lb = np.asarray([lower_bound(x) for x in aug_xs])
# np.linspace(0.2, 0.8, 9)
# axis[0].fill_between(np.hstack([0.1, 0.2, aug_xs, 0.8, 0.9]),
#                      np.hstack([0.7, 0.7, aug_ys, 0.1, 0.1]),
#                      0.9 * np.ones(shape=104),
#                      color='#ffe699', alpha=1.0, linewidth=0.0)
axis[0].plot(aug_xs, aug_ys, linewidth=3, color='tab:red', label='Realization of $\mathcal{F}^*$')

# axis[0].text(0.581, 0.722, r'$A$', fontsize=tx_ftz)
# axis[0].text(0.311, 0.340, r'$\overline{A}$', fontsize=tx_ftz)
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
axis[0].set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.85])

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


axis[0].plot([0.1, 0.1], [0.1, f(0.3)], color='k', linewidth=frame_linewidth, linestyle='--', zorder=0)
axis[0].plot([0.1, 0.7], [0.1, 0.1], color='k', linewidth=frame_linewidth, linestyle='--', zorder=0)
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
    axis[0].add_patch(rect)
axis[0].scatter(np.asarray(obs)[:, 0], np.asarray(obs)[:, 1], s=100, marker='*', c='tab:red',
                edgecolors='k', zorder=100, label='Pareto Frontier')

axis[0].fill_between(np.hstack([0.1, 0.2, aug_xs, 0.8, 0.9]),
                     np.hstack([f(0.3), f(0.3), aug_ys_lb, 0.1, 0.1]),
                     np.hstack([0.7, 0.7, aug_ys, 0.1, 0.1]),
                     color='olivedrab', alpha=1.0, linewidth=0.0, zorder=10,
                     label='Miss Classified \n Non-dominated Region')

axis[0].plot(0.1, 0.1, marker='s', markersize=10 - 2, markerfacecolor='tab:blue', color='k')
axis[0].plot(0.9, 0.9, marker='^', markersize=10, markerfacecolor='tab:blue', color='k')

axis[0].text(0.157, 0.722, r'$\tilde{A}_1$', fontsize=tx_ftz)
axis[0].text(0.358, 0.722, r'$\tilde{A}_2$', fontsize=tx_ftz)
axis[0].text(0.559, 0.722, r'$\tilde{A}_3$', fontsize=tx_ftz)
axis[0].text(0.756, 0.722, r'$\tilde{A}_4$', fontsize=tx_ftz)
axis[0].text(0.25, 0.25, r'$\tilde{\overline{A}}$', fontsize=tx_ftz)
# axis[1].set_title('2D Demo of partition the dominated region')


# ---------------------------------------------------------
# Sub-Figure 2  # Plot PF2ES Sample

def plot_pf_dominated_region_boarder_2d(pf, lower_bound, ax, color=None, linewidth=None):
    """
    Plot the Pareto front border
    This is not a clean implementation and only supports minimization
    """

    sorted_idx = tf.argsort(pf, axis=0)[:, 0]
    sorted_pf = tf.gather(pf, sorted_idx)
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(sorted_pf.shape[0]):
        if color is not None:
            default_color = color
        if i == 0:
            ax.plot((lower_bound[0], sorted_pf[i][0]), (sorted_pf[i][1], sorted_pf[i][1]),
                    color=color, linewidth=linewidth)
            if i != sorted_pf.shape[0] - 1:
                ax.plot((sorted_pf[i][0], sorted_pf[i][0]), (sorted_pf[i][1], sorted_pf[i+1][1]),
                        color=color, linewidth=linewidth)
            else:
                ax.plot((sorted_pf[i][0], sorted_pf[i][0]), (sorted_pf[i][1], lower_bound[1]),
                        color=color, linewidth=linewidth)
        elif i == sorted_pf.shape[0] - 1:
            ax.plot((sorted_pf[i][0], sorted_pf[i-1][0]), (sorted_pf[i][1], sorted_pf[i][1]),
                    color=color, linewidth=linewidth)
            ax.plot((sorted_pf[i][0], sorted_pf[i][0]), (sorted_pf[i][1], lower_bound[1]),
                    color=color, linewidth=linewidth)
        else:
            ax.plot((sorted_pf[i][0], sorted_pf[i-1][0]), (sorted_pf[i][1], sorted_pf[i][1]),
                    color=color, linewidth=linewidth)
            ax.plot((sorted_pf[i][0], sorted_pf[i][0]), (sorted_pf[i][1], sorted_pf[i+1][1]),
                    color=color, linewidth=linewidth)
    return ax


np.random.seed(1817)
tf.random.set_seed(1817)
OBJECTIVE = "OBJECTIVE"

# %%
num_initial_points = 10

vlmop2 = VLMOP2().objective()
vlmop2_observer = trieste.objectives.utils.mk_observer(vlmop2, key=OBJECTIVE)
vlmop2_mins = [-2, -2]
vlmop2_maxs = [2, 2]
vlmop2_search_space = Box(vlmop2_mins, vlmop2_maxs)
vlmop2_num_objective = 2


def build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainableHasTrajectoryAndPredictJointReparamModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(
            data.query_points, tf.gather(data.observations, [idx], axis=1)
        )
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7)
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=False), 1))

    return TrainableHasTrajectoryAndPredictJointReparamModelStack(*gprs)


initial_query_points = vlmop2_search_space.sample(num_initial_points)
initial_query_points = tf.concat([initial_query_points,
                                  tf.constant([[0.5, 0.45], [0.49, 0.48]], dtype=initial_query_points.dtype)], axis=0)
initial_data = vlmop2_observer(initial_query_points)[OBJECTIVE]
model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    initial_data, vlmop2_num_objective, vlmop2_search_space
)

builder = PF2ES(vlmop2_search_space, moo_solver='nsga2', sample_pf_num=5, remove_augmenting_region=False,
                population_size_for_approx_pf_searching=5, pareto_epsilon=0.00, remove_log=False,
                batch_mc_sample_size=32, discretize_input_sample_size=100000)
acq = builder.prepare_acquisition_function({OBJECTIVE: model}, {OBJECTIVE: initial_data})

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
for pf, idx in zip(acq._pf_samples, range(len(acq._pf_samples))):
    axis[1].scatter(- pf[:, 0], - pf[:, 1], color=colors[idx], s=5, marker='*')
    plot_pf_dominated_region_boarder_2d(-pf, [-1.2, -1.2], axis[1], color=colors[idx], linewidth=0.5)
box = axis[1].get_position()
axis[1].set_position([box.x0 - box.width * 0.1, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.85])
axis[1].scatter(-initial_data.observations[:, 0], -initial_data.observations[:, 1], marker='x', color='r')

axis[1].set_xticks([])
axis[1].set_yticks([])
# axis[1].set_title('- VLMOP2 Objective Space')

def acq_f(at):
    """
    Acquisition Function for plotting (expand the batch dim)
    """
    return acq(
        tf.expand_dims(at, axis=-2)
    )
from scipy.optimize import minimize
res = minimize(lambda x: -np.squeeze(acq_f(np.atleast_2d(x)).numpy()), np.array([0.5, 0.42]), bounds=[(-2, 2), (-2, 2)])
# axis[1].scatter(- vlmop2(np.atleast_2d(res.x))[:, 0], - vlmop2(np.atleast_2d(res.x))[:, 1],
#                 label='BO Added Point', s=30, color='darkorange')
axis[1].scatter(- vlmop2(np.atleast_2d(res.x))[:, 0], - vlmop2(np.atleast_2d(res.x))[:, 1],
                label=None, s=30, color='darkorange')




# plot the zoomed portion


sub_axes = plt.axes([.53, .65, .1, .25])
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
for pf, idx in zip(acq._pf_samples, range(len(acq._pf_samples))):
    sub_axes.scatter(- pf[:, 0], - pf[:, 1], color=colors[idx], s=5, marker='*')
    plot_pf_dominated_region_boarder_2d(-pf, [-1.2, -1.2], sub_axes, color=colors[idx], linewidth=0.5)
sub_axes.scatter(-initial_data.observations[:, 0], -initial_data.observations[:, 1], marker='x', color='r')
sub_axes.scatter(- vlmop2(np.atleast_2d(res.x))[:, 0], - vlmop2(np.atleast_2d(res.x))[:, 1], s=30, color='darkorange')
# # sub region of the original image
x1, x2, y1, y2 = -0.2, -0.03, -0.97, -0.92
sub_axes.set_xlim(x1, x2)
sub_axes.set_ylim(y1, y2)
sub_axes.set_xticklabels([])
sub_axes.set_yticklabels([])


from matplotlib.patches import Ellipse
pred_mean, pred_var = model.predict(np.atleast_2d(res.x))
pred_obj1, pred_obj2 = pred_mean[0, 0], pred_mean[0, 1]
var1, var2 = pred_var[0, 0], pred_var[0, 1]
ellipse = Ellipse(
    (pred_obj1, pred_obj2),
    2 * tf.sqrt(var1),
    2 * tf.sqrt(var2),
    angle=0,
    alpha=0.2,
    edgecolor="k",
)
sub_axes.add_artist(ellipse)



# Add the patch to the Axes

# TODO: Add zoom in frame
rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.3, edgecolor='k', facecolor='none')
axis[1].add_patch(rect)
axis[1].plot([x1, -0.345], [y2, -0.601], color='k', linestyle='-', linewidth=0.2)
# axis[1].plot([x2, y2], [-0.345, -0.601], color='k', linestyle='-', linewidth=2)
# sub_axes.patch.set_alpha(0.5)
# --------------------------------------------------------
# Sub-Figure 3  # Plot Acq Contour



var_range = [[-2, 2], [-2, 2]]
plot_fidelity = 100
func = acq_f

x_1 = np.linspace(*var_range[0], plot_fidelity)
x_2 = np.linspace(*var_range[-1], plot_fidelity)

# _z_data = np.array([func(np.array([[_x, _y]])) for _x in x for _y in y]).reshape(len(x), len(y))  # 这个写法很好
x_grid = np.array([[_x_1, _x_2] for _x_2 in x_2 for _x_1 in x_1])
try:
    _z_data = np.array(func(x_grid)).reshape(len(x_1), len(x_2))
except Exception:
    print(
        "{} do not support vectorize input, change to serial input automatically, process may be slow\n".format(
            func
        )
    )
    _z_data = np.array([func(serial_x) for serial_x in x_grid]).reshape(len(x_1), len(x_2))

CS = axis[2].contourf(
    x_1,
    x_2,
    _z_data,
)

cax = fig.add_axes([0.92, axis[2].get_position().y0 + axis[2].get_position().height * 0.2, 0.015, 0.7])
cbar = fig.colorbar(CS, cax=cax, pad = 0.06, shrink=1.0)
cbar.ax.set_ylabel('Acquisition Function Value')

# axis[2].set_xticks([-2, 0, 2])
axis[2].set_xticks([])
# axis[2].set_yticks([-2, 0, 2])
axis[2].set_yticks([])
# axis[2].set_xlim(var_range[0])
# axis[2].set_ylim(var_range[1])
axis[2].scatter(res.x[0], res.x[1], label='BO Query Point', s=30, color='darkorange', zorder=30)
axis[2].scatter(initial_data.query_points[:, 0],
                initial_data.query_points[:, 1],
                marker='x', color='r', label='Training data')
box = axis[2].get_position()
# axis[2].set_position([box.x0, box.y0 + box.height * 0.2,
#                  box.width, box.height * 0.85])
axis[2].set_position([box.x0 -  box.width * 1 , box.y0 + box.height * 0.2,
                 box.width, box.height * 0.85])
# axis[2].set_title('- VLMOP2 Input Space $\mathcal{X}$')
# plt.tight_layout()
handles, labels = axis[2].get_legend_handles_labels()
handles1, labels1 = axis[1].get_legend_handles_labels()
handles2, labels2 = axis[0].get_legend_handles_labels()
# fig.legend([*handles, *handles1, *handles2], [*labels, *labels1, *labels2], loc='lower center', bbox_to_anchor=(0.5, 0.0),
#                fancybox=False, ncol=3, fontsize=legend_fontsize)
fig.legend([*handles, *handles1, *handles2], [*labels, *labels1, *labels2], loc='lower center', bbox_to_anchor=(0.82, 0.4),
               fancybox=False, ncol=1, fontsize=legend_fontsize)

plt.savefig('Greedy_Issue1.png', dpi=1000)
# plt.savefig('Greedy_Issue_lgd.png', dpi=1000)
# plt.show()
