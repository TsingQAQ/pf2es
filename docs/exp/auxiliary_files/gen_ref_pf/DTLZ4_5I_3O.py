from trieste.objectives.multi_objectives import DTLZ4
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

print(DTLZ4(input_dim=5, num_objective=3).objective()(
    tf.constant([[0.15, 0.2, 0.3, 0.4, 0.5],
                 [9.94777809e-01, 9.96358570e-01, 4.88908128e-01, 5.01394680e-01, 4.71606567e-01]])))
raise ValueError
res = moo_nsga2_pymoo(DTLZ4(input_dim=5, num_objective=3).objective(), input_dim= 5, obj_num= 3,
                      bounds= tf.convert_to_tensor(DTLZ4(input_dim=5, num_objective=3).bounds),
                            popsize= 100, num_generation=2000)

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(res.fronts[:, 0], res.fronts[:, 1], res.fronts[:, 2])
plt.show()
# np.savetxt('DTLZ4_5I_3O_PF_F.txt', res.fronts)
# np.savetxt('DTLZ4_5I_3O_PF_X.txt', res.inputs)