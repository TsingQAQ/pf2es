from trieste.objectives.multi_objectives import DTLZ2
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

res, resx = moo_nsga2_pymoo(DTLZ2(input_dim=3, num_objective=2).objective(), input_dim= 3, obj_num= 2,
                      bounds= tf.convert_to_tensor(DTLZ2(input_dim=3, num_objective=2).bounds),
                            popsize= 50, return_pf_x=True,
                            num_generation=2000)
from matplotlib import pyplot as plt
plt.scatter(res[:, 0], res[:, 1])
plt.show()
np.savetxt('DTLZ2_3I_2O_PF_F.txt', res)
np.savetxt('DTLZ2_3I_2O_PF_X.txt', resx)