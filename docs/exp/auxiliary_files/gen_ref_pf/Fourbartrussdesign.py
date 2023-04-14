from trieste.objectives.multi_objectives import FourBarTruss
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

pf = np.loadtxt('FourBarTruss_PF_F.txt')
ideal_hv = Pareto(observations=pf).hypervolume_indicator(np.array([3400, 0.05]))
print(ideal_hv)
# print(FourBarTruss().objective()(tf.constant([[0.44, 0.33, 0.22, 0.11]])))
raise ValueError
res = moo_nsga2_pymoo(FourBarTruss().objective(), input_dim= 4, obj_num= 2,
                      bounds= tf.convert_to_tensor(FourBarTruss.bounds), popsize= 100,
                      num_generation=2000)

from matplotlib import pyplot as plt
plt.scatter(res.fronts[:, 0], res.fronts[:, 1])
plt.show()

# print(Pareto(res).hypervolume_indicator(tf.constant([18.0, 6.0], dtype=tf.float64))) # 59.161270390382754
np.savetxt('FourBarTruss_PF_F.txt', res.fronts)
np.savetxt('FourBarTruss_PF_X.txt', res.inputs)