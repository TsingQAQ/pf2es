from trieste.objectives.multi_objectives import DiscBrakeDesign
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo
from trieste.acquisition.multi_objective.pareto import Pareto
pf = np.loadtxt('DiscBrakeDesign_PF_F.txt')
ideal_hv = Pareto(observations=pf).hypervolume_indicator(np.array([8.0, 4.0]))
print(ideal_hv)
raise ValueError
# class CRE23():
#     def __init__(self):
#         self.problem_name = 'CRE23'
#         self.n_objectives = 2
#         self.n_variables = 4
#         self.n_constraints = 4
#
#         self.ubound = np.zeros(self.n_variables)
#         self.lbound = np.zeros(self.n_variables)
#         self.lbound[0] = 55
#         self.lbound[1] = 75
#         self.lbound[2] = 1000
#         self.lbound[3] = 11
#         self.ubound[0] = 80
#         self.ubound[1] = 110
#         self.ubound[2] = 3000
#         self.ubound[3] = 20
#
#     def evaluate(self, x):
#         x = x * (self.ubound - self.lbound) + self.lbound
#         f = np.zeros(self.n_objectives)
#         g = np.zeros(self.n_constraints)
#
#         x1 = x[0]
#         x2 = x[1]
#         x3 = x[2]
#         x4 = x[3]
#
#         # First original objective function
#         f[0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
#         # Second original objective function
#         f[1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
#
#         # Reformulated objective functions
#         g[0] = (x2 - x1) - 20.0
#         g[1] = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
#         g[2] = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
#         g[3] = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
#         # g = np.where(g < 0, -g, 0)
#
#         return f, g

# import numpy as np
# print(CRE23().evaluate(np.array([0.0, 0.3, 0.2, 0.4])))
print(DiscBrakeDesign().objective()(tf.constant([[0.0, 0.3, 0.2, 0.4]])))
print(DiscBrakeDesign().constraint()(tf.constant([[0.0, 0.3, 0.2, 0.4]])))
raise ValueError
res = moo_nsga2_pymoo(DiscBrakeDesign().objective(), input_dim= 4, obj_num= 2,
                      bounds= tf.convert_to_tensor(DiscBrakeDesign.bounds), popsize= 100,
                      cons=DiscBrakeDesign().constraint(), cons_num=4, num_generation=1000)
from matplotlib import pyplot as plt
plt.scatter(res.fronts[:, 0], res.fronts[:, 1])
plt.show()

np.savetxt('DiscBrakeDesign_PF_F.txt', res.fronts)
np.savetxt('DiscBrakeDesign_PF_X.txt', res.inputs)