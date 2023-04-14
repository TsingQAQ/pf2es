from trieste.objectives.multi_objectives import CVLMOP2
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

res = moo_nsga2_pymoo(CVLMOP2().objective(), input_dim= 2, obj_num= 2,
                      bounds= tf.convert_to_tensor(CVLMOP2.bounds), popsize= 100,
                      cons=CVLMOP2().constraint(), cons_num=1)
from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1])
plt.show()
np.savetxt('CVLMOP2_PF.txt', res)