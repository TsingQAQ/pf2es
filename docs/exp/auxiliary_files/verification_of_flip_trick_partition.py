# from trieste.objectives.multi_objectives import DTLZ2
# import  tensorflow as tf
# import matplotlib.patches as patches
#
# pf = DTLZ2(3, 2).gen_pareto_optimal_points(5)
#
# from trieste.acquisition.multi_objective.partition import prepare_default_non_dominated_partition_bounds
# lbs, ubs = prepare_default_non_dominated_partition_bounds(
#     tf.constant([1.5, 2.0]), pf, anti_reference=tf.constant([-0.5, -0.8]))
#
# from matplotlib import pyplot as plt
# fig, ax = plt.subplots()
# plt.scatter(pf[:, 0], pf[:, 1], label='Pareto Frontier')
# plt.scatter(1.5, 2.0, s=30, marker='s', label='Reference Point')
# plt.scatter(-0.5, -0.8, s=30, marker='s', label='Anti Reference Point')
# # PLOT
# for lb, ub in zip(lbs, ubs):
#     rect = patches.Rectangle(lb, (ub - lb)[0], (ub - lb)[1],
#                              linewidth=1, edgecolor='r', facecolor='tab:blue', alpha=0.4)
#     # Add the patch to the Axes
#     ax.add_patch(rect)
# plt.legend()
# plt.title('2D Partition without augmentation region illustration')
# plt.show(block=True)


import tensorflow as tf
from trieste.acquisition.multi_objective.partition import FlipTrickPartitionNonDominated, \
    prepare_default_non_dominated_partition_bounds
from trieste.acquisition.multi_objective.pareto import Pareto
# objs = tf.constant([[1.0, 0.0, 2.0], [-1.0, 1.0, 3.0], [-0.5, -1.0, 2.5]]) # [1.5, -1.0, 2.5]
objs = tf.constant([[3.0, 1.0, 5.0], [2.0, 4.0, 4.0], [4, 3.0, 3.0]]) # [1.5, -1.0, 2.5]
# objs = tf.random.normal(shape=(5, 3))
# ideal_point = tf.constant([-2.0, -2.0, -2.0])
ideal_point = tf.constant([0.0, 0.0, 0.0])
worst_point = tf.constant([6.0, 6.0, 6.0])
# lb, ub = FlipTrickPartitionNonDominated(objs, ideal_point, worst_point).partition_bounds()
lb, ub = prepare_default_non_dominated_partition_bounds(
    worst_point, objs, anti_reference=ideal_point)

# %%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt


def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(positions,sizes=None,colors=None, alpha=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors,6), alpha=alpha, **kwargs)


fig = plt.figure()
ax = fig.gca(projection='3d')
for pos, size in zip(lb.numpy().tolist(), (ub-lb).numpy().tolist()):
    pc = plotCubeAt2([pos],[size], edgecolor="k", alpha=0.2, colors='#1f77b4')
    ax.add_collection3d(pc)
ax.scatter(Pareto(objs).front[:, 0],
           Pareto(objs).front[:, 1],
           Pareto(objs).front[:, 2], s=20, color='r', label='Pareto Front')

ax.scatter(tf.constant([3.5]), tf.constant([3.5]), tf.constant([4.5]), s=20, color='y', label='TEST POINT')
ax.scatter(tf.constant([3]), tf.constant([3]), tf.constant([4]), s=20, color='purple', label='loccal upper bound POINT')
ax.scatter(tf.constant([2.5]), tf.constant([1.5]), tf.constant([3.5]), s=20, color='k', label='SHIFT TEST POINT')
ax.scatter(tf.constant([2.0]), tf.constant([1.0]), tf.constant([3.0]), s=20, color='k', label='BEST POINT')
ax.scatter(*worst_point, s=20, color='g', label='Worst Point')
plt.title('3D Demo of partition the dominated region')
plt.legend()
plt.show()
