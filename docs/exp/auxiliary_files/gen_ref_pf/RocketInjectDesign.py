from trieste.objectives.multi_objectives import RocketInjectorDesign
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

class RE37():
    def __init__(self):
        self.problem_name = 'RE37'
        self.n_objectives = 3
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 0

        self.lbound = np.full(self.n_variables, 0)
        self.ubound = np.full(self.n_variables, 1)

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)

        xAlpha = x[0]
        xHA = x[1]
        xOA = x[2]
        xOPTT = x[3]

        # f1 (TF_max)
        f[0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (
                    0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (
                           0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (
                           0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (
                           0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f[1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (
                    0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (
                           0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (
                           0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (
                           0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f[2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (
                    0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (
                           0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (
                           0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (
                           0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                           0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                           0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

        return f


# print(RE37().evaluate(np.array([0.5, 0.3, 0.4, 0.7])))
#
# print(Rocket_Injector_Design().objective()(tf.constant([[0.5, 0.3, 0.4, 0.7]])))
# raise ValueError
res = moo_nsga2_pymoo(RocketInjectorDesign().objective(), input_dim= 4, obj_num= 3,
                      bounds= tf.convert_to_tensor(RocketInjectorDesign().bounds),
                      popsize= 100,
                      num_generation=2000)
from matplotlib import pyplot as plt
# res = DTLZ2(input_dim=4, num_objective=3).gen_pareto_optimal_points(100)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(res.fronts[:, 0], res.fronts[:, 1], res.fronts[:, 2])
plt.show()
np.savetxt('RocketInjectorDesign_PF_F.txt', res.fronts)
np.savetxt('RocketInjectorDesign_PF_X.txt', res.inputs)