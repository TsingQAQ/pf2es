from trieste.objectives.multi_objectives import DTLZ5
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo
from matplotlib import pyplot as plt

res, resx = moo_nsga2_pymoo(DTLZ5(input_dim=4, num_objective=3).objective(), input_dim=4, obj_num=3,
                      bounds= tf.convert_to_tensor(DTLZ5(input_dim=4, num_objective=3).bounds),
                            popsize= 50, return_pf_x=True,
                            num_generation=2000)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:, 0], res[:, 1], res[:, 2])
plt.show()

np.savetxt('DTLZ5_4I_3O_PF_F.txt', res)
np.savetxt('DTLZ5_4I_3O_PF_X.txt', resx)