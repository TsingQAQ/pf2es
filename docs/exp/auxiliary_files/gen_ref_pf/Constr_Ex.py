from trieste.objectives.multi_objectives import Constr_Ex
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

res = moo_nsga2_pymoo(Constr_Ex().objective(), input_dim= 2, obj_num= 2,
                      bounds= tf.convert_to_tensor(Constr_Ex.bounds), popsize= 100,
                      cons=Constr_Ex().constraint(), cons_num=1)
from matplotlib import pyplot as plt
plt.scatter(res.fronts[:, 0], res.fronts[:, 1])
plt.show()
np.savetxt('Constr_Ex_PF_F.txt', res.fronts)
np.savetxt('Constr_Ex_PF_X.txt', res.inputs)