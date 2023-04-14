from trieste.objectives.multi_objectives import TNK
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

print(TNK().objective()(tf.constant([[0.5, 0.3]])))
print(TNK().constraint()(tf.constant([[0.5, 0.3]])))
raise ValueError
tnk_res = moo_nsga2_pymoo(TNK().objective(), cons=TNK().constraint(),
                                    input_dim= 2, obj_num= 2, cons_num=2,
                      bounds= tf.convert_to_tensor(TNK.bounds), popsize= 50, num_generation=500)


from matplotlib import pyplot as plt
_, axis = plt.subplots(ncols=1)
axis.scatter(tnk_res.fronts[:, 0], tnk_res.fronts[:, 1])

# axis[1].scatter(bc_resx[:, 0], bc_resx[:, 1])
# axis[1].scatter(cbc_resx[:, 0], cbc_resx[:, 1])
plt.show()

# print(Pareto(res).hypervolume_indicator(tf.constant([18.0, 6.0], dtype=tf.float64))) # 59.161270390382754
np.savetxt('TNK_PF_F.txt', tnk_res.fronts)
np.savetxt('TNK_PF_X.txt', tnk_res.inputs)