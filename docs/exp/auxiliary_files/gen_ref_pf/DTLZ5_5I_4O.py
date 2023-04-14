from trieste.objectives.multi_objectives import DTLZ5
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

print(DTLZ5(input_dim=5, num_objective=4).objective()(tf.constant([[0.15, 0.2, 0.3, 0.4, 0.5]])))
print(DTLZ5(input_dim=5, num_objective=4).objective()(tf.constant([[9.94777809e-01, 9.96358570e-01, 4.88908128e-01, 5.01394680e-01, 4.71606567e-01]])))
raise ValueError
res, resx = moo_nsga2_pymoo(DTLZ5(input_dim=5, num_objective=4).objective(), input_dim=5, obj_num=4,
                      bounds= tf.convert_to_tensor(DTLZ5(input_dim=5, num_objective=4).bounds),
                            popsize= 100, return_pf_x=True,
                            num_generation=2000)
# from matplotlib import pyplot as plt
# # res = DTLZ2(input_dim=4, num_objective=3).gen_pareto_optimal_points(100)
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
#
# ax.scatter(res[:, 0], res[:, 1], res[:, 2])
# plt.show()
print(tf.reduce_min(res, axis=0))
print(tf.reduce_max(res, axis=0))
np.savetxt('DTLZ5_5I_4O_PF_F.txt', res)
np.savetxt('DTLZ5_5I_4O_PF_X.txt', resx)