from trieste.objectives.multi_objectives import VehicleCrashSafety
import tensorflow as tf
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

res, resx = moo_nsga2_pymoo(VehicleCrashSafety().objective(), input_dim= 5, obj_num= 3,
                      bounds= tf.convert_to_tensor(VehicleCrashSafety().bounds),
                            popsize= 100, return_pf_x=True,
                            num_generation=2000)
from matplotlib import pyplot as plt
# res = DTLZ2(input_dim=4, num_objective=3).gen_pareto_optimal_points(100)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(res[:, 0], res[:, 1], res[:, 2])
plt.show()
np.savetxt('VehicleCrashSafety_PF_F.txt', res)
np.savetxt('VehicleCrashSafety_PF_X.txt', resx)