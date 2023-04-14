from trieste.objectives.multi_objectives import SRN
import tensorflow as tf
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

print(SRN().objective()(tf.constant([[0.55, 0.31]], dtype=tf.float64)))
print(SRN().constraint()(tf.constant([[0.55, 0.31]], dtype=tf.float64)))
raise ValueError

res = moo_nsga2_pymoo(SRN().objective(), input_dim= 2, obj_num= 2,
                      bounds= tf.convert_to_tensor(SRN.bounds), cons_num=2, cons=SRN().constraint(),
                      popsize= 100, num_generation=500)

from matplotlib import pyplot as plt
plt.scatter(res.fronts[:, 0], res.fronts[:, 1])
plt.show()

import numpy as np
np.savetxt('SRN_PF_F.txt', res.fronts)
np.savetxt('SRN_PF_X.txt', res.inputs)