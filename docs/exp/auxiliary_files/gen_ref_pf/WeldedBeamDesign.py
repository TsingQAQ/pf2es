from trieste.objectives.multi_objectives import WeldedBeamDesign
import tensorflow as tf
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo
res = moo_nsga2_pymoo(WeldedBeamDesign().objective(), input_dim= 4, obj_num= 2,
                      bounds= tf.convert_to_tensor(WeldedBeamDesign.bounds),
                      cons_num=4, cons=WeldedBeamDesign().constraint(),
                      popsize= 50, num_generation=2000)
from matplotlib import pyplot as plt
plt.scatter(res.fronts[:, 0], res.fronts[:, 1])
plt.show()


import numpy as np
np.savetxt('WeldedBeamDesign_PF_F.txt', res.fronts)
np.savetxt('WeldedBeamDesign_PF_X.txt', res.inputs)




