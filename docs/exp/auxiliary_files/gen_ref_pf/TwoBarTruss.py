from trieste.objectives.multi_objectives import TwoBarTruss
import tensorflow as tf
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo
res = moo_nsga2_pymoo(TwoBarTruss().objective(), input_dim= 3, obj_num= 2,
                      bounds= tf.convert_to_tensor(TwoBarTruss.bounds),
                      cons_num=1, cons=TwoBarTruss().constraint(),
                      popsize= 50, num_generation=500)

from matplotlib import pyplot as plt
plt.scatter(res.fronts[:, 0], res.fronts[:, 1])
plt.show()
