from trieste.objectives.multi_objectives import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5
from trieste.types import TensorType
import tensorflow as tf
from trieste.acquisition.optimizer import perform_parallel_continuous_multi_objective_optimization
from trieste.space import Box
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo


def sim_constraint(input_data):
    if tf.rank(input_data) == 1:
        input_data = tf.expand_dims(input_data, axis=-2)
    con = tf.reduce_sum(input_data, axis=-1, keepdims=True)
    return 0.45 - con


class StackedDTLZFunction:
    def __init__(self, input_dim: int, obj_num: int):
        self._dtlz1_obj = DTLZ1(input_dim, obj_num).objective()
        self._dtlz2_obj = DTLZ2(input_dim, obj_num).objective()
        self._dtlz3_obj = DTLZ3(input_dim, obj_num).objective()
        self._dtlz4_obj = DTLZ4(input_dim, obj_num).objective()
        self._dtlz5_obj = DTLZ5(input_dim, obj_num).objective()

    def objective(self):
        def stacked_obj(inputs: TensorType) -> TensorType:  # [N, D] -> [N, 2, D]
            obj1_inputs, obj2_inputs, obj3_inputs, obj4_inputs, obj5_inputs = tf.split(inputs, axis=-2, num_or_size_splits=5)
            obj1_inputs = tf.squeeze(obj1_inputs, axis=-2) # [N, 1, D] -> [N, D]
            obj2_inputs = tf.squeeze(obj2_inputs, axis=-2) # [N, 1, D] -> [N, D]
            obj3_inputs = tf.squeeze(obj3_inputs, axis=-2) # [N, 1, D] -> [N, D]
            obj4_inputs = tf.squeeze(obj4_inputs, axis=-2) # [N, 1, D] -> [N, D]
            obj5_inputs = tf.squeeze(obj5_inputs, axis=-2) # [N, 1, D] -> [N, D]
            obj1 = tf.expand_dims(self._dtlz1_obj(obj1_inputs), -2) # [N, 1, M]
            obj2 = tf.expand_dims(self._dtlz2_obj(obj2_inputs), -2) # [N, 1, M]
            obj3 = tf.expand_dims(self._dtlz3_obj(obj3_inputs), -2) # [N, 1, M]
            obj4 = tf.expand_dims(self._dtlz4_obj(obj4_inputs), -2) # [N, 1, M]
            obj5 = tf.expand_dims(self._dtlz5_obj(obj5_inputs), -2) # [N, 1, M]
            return tf.concat([obj1, obj2, obj3, obj4, obj5], axis=-2) # [N, 5, M]
        return stacked_obj


class StackedConstraint:
    def __init__(self):
        self._con_1 = sim_constraint
        self._con_2 = sim_constraint
        self._con_3 = sim_constraint
        self._con_4 = sim_constraint
        self._con_5 = sim_constraint

    def constraint(self):
        def stacked_cons(inputs: TensorType) -> TensorType:  # [N, D] -> [N, 2, D]
            con1_inputs, con2_inputs, con3_inputs, con4_inputs, con5_inputs = \
                tf.split(inputs, axis=-2, num_or_size_splits=5)
            con1_inputs = tf.squeeze(con1_inputs, axis=-2) # [N, 1, D] -> [N, D]
            con2_inputs = tf.squeeze(con2_inputs, axis=-2) # [N, 1, D] -> [N, D]
            con3_inputs = tf.squeeze(con3_inputs, axis=-2) # [N, 1, D] -> [N, D]
            con4_inputs = tf.squeeze(con4_inputs, axis=-2) # [N, 1, D] -> [N, D]
            con5_inputs = tf.squeeze(con5_inputs, axis=-2) # [N, 1, D] -> [N, D]
            con1 = tf.expand_dims(self._con_1(con1_inputs), -2) # [N, 1, M]
            con2 = tf.expand_dims(self._con_2(con2_inputs), -2) # [N, 1, M]
            con3 = tf.expand_dims(self._con_3(con3_inputs), -2) # [N, 1, M]
            con4 = tf.expand_dims(self._con_4(con4_inputs), -2) # [N, 1, M]
            con5 = tf.expand_dims(self._con_5(con5_inputs), -2) # [N, 1, M]
            return tf.concat([con1, con2, con3, con4, con5], axis=-2) # [N, 5, M]
        return stacked_cons


# res, res_x = moo_nsga2_pymoo(
#     DTLZ1(3, 2).objective(), 3, obj_num=2,
#     bounds=tf.convert_to_tensor([DTLZ1(3, 2).bounds[0], DTLZ1(3, 2).bounds[1]]), popsize=50,
#     return_pf_x=True, cons_num=1, cons=sim_constraint)
#
# from matplotlib import pyplot as plt
# plt.scatter(res[:, 0], res[:, 1])
# plt.show()
res, res_x = perform_parallel_continuous_multi_objective_optimization(
                StackedDTLZFunction(3, 2).objective(), 5, Box(*DTLZ1(3, 2).bounds), 2, num_generation=500,
                cons_num=1, vectorized_con_func=StackedConstraint().constraint(), return_pf_input=True)


from matplotlib import pyplot as plt
fig, ax = plt.subplots(ncols=5)
for i, pf in zip(range(res.shape[0]), res):
    ax[i].scatter(pf.to_tensor()[:, 0], pf.to_tensor()[:, 1])
    ax[i].set_title(f'DTLZ{i+1}')
plt.tight_layout()
plt.show(block=True)


# if __name__ == '__main__':
#     failure = 0
#     for i in range(100000):
#         # res = perform_parallel_continuous_multi_objective_optimization(
#         #     StackedDTLZFunction(3, 2).objective(), 5, Box(*DTLZ1(3, 2).bounds), 2, num_generation=100)
#         if i % 10 == 0:
#             print(f'Now tried {i} times')
#         try:
#             res = perform_parallel_continuous_multi_objective_optimization(
#                 StackedDTLZFunction(3, 2).objective(), 5, Box(*DTLZ1(3, 2).bounds), 2, num_generation=100)
#         except Exception as e:
#             print(e)
#             failure +=1
#             print(f'Failed {failure} times out of {i} trials')