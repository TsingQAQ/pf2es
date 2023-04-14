from trieste.objectives.multi_objectives import C2DTLZ2
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

b =  C2DTLZ2(input_dim=4, num_objective=2).objective()
a = C2DTLZ2(input_dim=4, num_objective=2).constraint()
# print(b(tf.constant([[0.1, 0.2, 0.3, 0.4]])))
# print(a(tf.constant([[0.1, 0.2, 0.3, 0.4],
#                      [0.15, 0.25, 0.35, 0.45]])))
# raise ValueError
tf.random.set_seed(1817)
np.random.seed(1817)

xs = tf.convert_to_tensor(np.loadtxt('C2DTLZ2_4D_PF_X.txt'))
res = moo_nsga2_pymoo(C2DTLZ2(input_dim=4, num_objective=2).objective(), input_dim= 4, obj_num= 2,
                      bounds= tf.convert_to_tensor(C2DTLZ2(input_dim=4, num_objective=2).bounds), popsize= 50,
                      cons=C2DTLZ2(input_dim=4, num_objective=2).constraint(), cons_num=1, num_generation=500,
                      initial_candidates=None)
from matplotlib import pyplot as plt
plt.scatter(res.fronts[:, 0], res.fronts[:, 1])
plt.show()
# np.savetxt('C2DTLZ2_4D_PF_F.txt', res.fronts)
# np.savetxt('C2DTLZ2_4D_PF_X.txt', res.inputs)