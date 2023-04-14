from trieste.objectives.multi_objectives import BraninCurrin
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo
# print(BraninCurrin().objective()(tf.constant([[0.5, 0.3]])))
# raise ValueError
res = moo_nsga2_pymoo(BraninCurrin().objective(), input_dim= 2, obj_num= 2,
                      bounds= tf.convert_to_tensor(BraninCurrin.bounds), popsize= 10,
                            num_generation=1)

from matplotlib import pyplot as plt
plt.scatter(res.fronts[:, 0], res.fronts[:, 1])
plt.show()

# print(Pareto(res).hypervolume_indicator(tf.constant([18.0, 6.0], dtype=tf.float64))) # 59.161270390382754
np.savetxt('BraninCurrin_PF_NLL_F.txt', res)
np.savetxt('BraninCurrin_PF_NLL_X.txt', resx)