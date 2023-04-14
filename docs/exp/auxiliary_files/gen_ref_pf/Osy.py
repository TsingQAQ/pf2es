from trieste.objectives.multi_objectives import Osy
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

# res = moo_nsga2_pymoo(Osy().objective(), input_dim= 6, obj_num= 2, cons_num=6, cons=Osy().constraint(),
#                       bounds= tf.convert_to_tensor(Osy.bounds), popsize= 50, num_generation=1000)
# resx = res.inputs
# resf = res.fronts
# from matplotlib import pyplot as plt
# _, axis = plt.subplots(ncols=1)
# axis.scatter(resf[:, 0], resf[:, 1])
# # axis[0].scatter(cbc_res[:, 0], cbc_res[:, 1])
# #
# # axis[1].scatter(bc_resx[:, 0], bc_resx[:, 1])
# # axis[1].scatter(cbc_resx[:, 0], cbc_resx[:, 1])
# plt.show()


# print(Pareto(res).hypervolume_indicator(tf.constant([18.0, 6.0], dtype=tf.float64))) # 59.161270390382754
# np.savetxt('Osy_PF_F.txt', resf)
# np.savetxt('Osy_PF_X.txt', resx)
from trieste.acquisition.multi_objective.pareto import Pareto
pf = np.loadtxt(R'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\constraint_exp\cfg\ref_opts\Osy_PF_F.txt')
print(Pareto(pf).hypervolume_indicator(tf.constant([50.0, 100.0], dtype=tf.float64)))