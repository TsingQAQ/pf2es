from trieste.objectives.multi_objectives import BraninCurrin, CBraninCurrin
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto
import numpy as np
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo
from matplotlib import pyplot as plt


os_input_cbranincurrin_ = np.loadtxt('good_os_input_for_check.txt')
plt.scatter(os_input_cbranincurrin_[:, 0], os_input_cbranincurrin_[:, 1])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()


test_input = tf.convert_to_tensor(os_input_cbranincurrin_)
obj = CBraninCurrin().objective()
con = CBraninCurrin().constraint()

print(obj(test_input))
print(con(test_input))

# bc_res= moo_nsga2_pymoo(BraninCurrin().objective(), input_dim= 2, obj_num= 2,
#                       bounds= tf.convert_to_tensor(BraninCurrin.bounds), popsize= 50,
#                             num_generation=500)

# cbc_res= moo_nsga2_pymoo(CBraninCurrin().objective(), cons=CBraninCurrin().constraint(),
#                                     input_dim= 2, obj_num= 2, cons_num=1,
#                       bounds= tf.convert_to_tensor(CBraninCurrin.bounds), popsize= 50,
#                             num_generation=500)
#
#
# from matplotlib import pyplot as plt
# _, axis = plt.subplots(ncols=2)
# # axis[0].scatter(bc_res[:, 0], bc_res[:, 1])
# # axis[0].scatter(cbc_res.fronts[:, 0], cbc_res.fronts[:, 1])
# axis[0].scatter(cbc_res.fronts[:, 0], cbc_res.fronts[:, 1])
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.show()
#
# # axis[1].scatter(bc_resx[:, 0], bc_resx[:, 1])
# # axis[1].scatter(cbc_resx[:, 0], cbc_resx[:, 1])
# plt.show()
#
# # print(Pareto(res).hypervolume_indicator(tf.constant([18.0, 6.0], dtype=tf.float64))) # 59.161270390382754
# np.savetxt('C_BraninCurrin_200pop_PF_F.txt', cbc_res.fronts)
# np.savetxt('C_BraninCurrin_200pop_PF_X.txt', cbc_res.inputs)