from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import trieste
from trieste.acquisition.multi_objective.utils import (
    sample_pareto_fronts_from_parametric_gp_posterior,
)
from trieste.data import Dataset
from trieste.models import (
    TrainableHasTrajectoryAndPredictJointReparamModelStack,
    TrainableHasTrajectorySamplerModelStack,
    TrainableModelStack,
)
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import multi_objectives
from trieste.space import Box, SearchSpace
from trieste.types import TensorType

# %%


np.random.seed(1793)
tf.random.set_seed(1793)

# --------------
ftz = 15
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"

# --------------

# def build_stacked_independent_objectives_has_traj_model(
#         data: Dataset, num_output: int, search_space: SearchSpace, use_decoupled_sampler: bool = False
# ) -> TrainableHasTrajectorySamplerModelStack:
#     gprs = []
#     for idx in range(num_output):
#         single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
#         gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7)
#         gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=use_decoupled_sampler), 1))
#
#     return TrainableHasTrajectorySamplerModelStack(*gprs)


def build_stacked_independent_objectives_model(
    data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainableModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7)
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=True), 1))

    return TrainableModelStack(*gprs)


def build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    data: Dataset, num_output: int, search_space: SearchSpace, use_decoupled_sampler: bool = False
) -> TrainableHasTrajectoryAndPredictJointReparamModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7)
        gprs.append(
            (GaussianProcessRegression(gpr, use_decoupled_sampler=use_decoupled_sampler), 1)
        )

    return TrainableHasTrajectoryAndPredictJointReparamModelStack(*gprs)


pb = multi_objectives.VLMOP2()
mins, maxs = pb.bounds[0], pb.bounds[1]
search_space = Box(mins, maxs)
num_objective = 2

# %% [markdown]
# Let's randomly sample some initial data from the observer ...
observer = trieste.objectives.utils.mk_observer(pb.objective(), key=OBJECTIVE)

# %%
num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %%
# Objective Model
obj_models_has_traj_for_decoupled_sample = (
    build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        initial_data[OBJECTIVE], num_objective, search_space, use_decoupled_sampler=True
    )
)

obj_models_has_traj_for_decoupled_sample.optimize(initial_data[OBJECTIVE])

obj_models_for_monte_carlo = build_stacked_independent_objectives_model(
    initial_data[OBJECTIVE], num_objective, search_space
)

obj_models_has_traj_for_rff_sample = (
    build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        initial_data[OBJECTIVE], num_objective, search_space, use_decoupled_sampler=False
    )
)

test = obj_models_has_traj_for_rff_sample.get_trajectories()
test[0](tf.ones(shape=(2, 1, 1)))

test2 = obj_models_has_traj_for_rff_sample.get_trajectories()
test2[0](tf.ones(shape=(5, 3, 1)))
