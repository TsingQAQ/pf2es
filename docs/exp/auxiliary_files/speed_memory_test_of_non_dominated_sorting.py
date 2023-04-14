from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from trieste.acquisition.multi_objective.dominance import non_dominated
import tensorflow as tf
import time
import numpy as np


def non_dominated_sortting(observations):
    """
    from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python#:~:text=def%20is_pareto_efficient(,%3A%0A%20%20%20%20%20%20%20%20return%20is_efficient
    Find the pareto-efficient points
    :param observations: An (n_points, n_costs) array
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    input_dtype = observations.dtype
    is_efficient = np.arange(observations.shape[0])
    n_points = observations.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(observations):
        nondominated_point_mask = np.any(observations < observations[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        observations = observations[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    dominated_mask = np.ones(n_points, dtype = bool)
    dominated_mask[is_efficient] = 0
    return tf.convert_to_tensor(observations, dtype = input_dtype), \
           tf.convert_to_tensor(dominated_mask, dtype = input_dtype)
    # return observations
    # if return_mask:
    #     is_efficient_mask = np.zeros(n_points, dtype = bool)
    #     is_efficient_mask[is_efficient] = True
    #     return is_efficient_mask
    # else:
    #     return is_efficient

xs = tf.random.normal(shape=(1000000, 2))
a, b = non_dominated_sortting(xs)
# NonDominatedSorting().do(xs.numpy())
