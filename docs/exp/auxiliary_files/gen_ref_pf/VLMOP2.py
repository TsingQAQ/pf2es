from trieste.objectives.multi_objectives import VLMOP2
import tensorflow as tf


_x = tf.linspace([-1 / tf.sqrt(2.0)], [1 / tf.sqrt(2.0)], 50)
xs = tf.concat([_x, _x], axis=1)
_pf = VLMOP2().objective()(xs)

from matplotlib import pyplot as plt
plt.scatter(_pf[:, 0], _pf[:, 1])
plt.show()

import numpy as np
np.savetxt('VLMOP2_PF_X.txt', xs)
np.savetxt('VLMOP2_PF_F.txt', _pf)

