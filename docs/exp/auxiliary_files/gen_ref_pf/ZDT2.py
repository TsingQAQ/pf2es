from trieste.objectives.multi_objectives import ZDT2
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

res, resx = moo_nsga2_pymoo(ZDT2(input_dim=5).objective(), input_dim= 5, obj_num= 2,
                      bounds= tf.convert_to_tensor(ZDT2(input_dim=5).bounds), popsize= 50, return_pf_x=True,
                            num_generation=1000)
from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1])
plt.show()

# print(Pareto(res).hypervolume_indicator(tf.constant([18.0, 6.0], dtype=tf.float64))) # 59.161270390382754
np.savetxt('ZDT2_5I_2O_PF_F.txt', res)
np.savetxt('ZDT2_5I_2O_PF_X.txt', resx)