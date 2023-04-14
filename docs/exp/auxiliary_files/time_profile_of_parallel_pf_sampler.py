from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

import trieste
from trieste.acquisition.multi_objective.utils import (
    sample_pareto_fronts_from_parametric_gp_posterior,
sample_pareto_fronts_from_parametric_gp_posterior_using_parallel_nsga2
)
from trieste.data import Dataset
from trieste.models import TrainableHasTrajectorySamplerModelStack, TrainableModelStack
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import multi_objectives
from trieste.space import Box, SearchSpace
from trieste.types import TensorType
import time

# %%


np.random.seed(1793)
tf.random.set_seed(1793)

# --------------
ftz = 15
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"

# --------------


def build_stacked_independent_objectives_has_traj_model(
    data: Dataset, num_output: int, search_space: SearchSpace, use_decoupled_sampler: bool = False
) -> TrainableHasTrajectorySamplerModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7)
        gprs.append(
            (GaussianProcessRegression(gpr, use_decoupled_sampler=use_decoupled_sampler), 1)
        )

    return TrainableHasTrajectorySamplerModelStack(*gprs)


def show_pareto_frontier_samples_on_problem(
    pb_name, num_doe: int, num_obj: int = 2, num_con: int = 0, pf_sample_num: int = 5
):
    # %%
    pb = getattr(multi_objectives, pb_name)
    pb_obj = pb().objective()

    if isinstance(pb(), multi_objectives.ConstraintMultiObjectiveTestProblem):
        pb_con = pb().constraint()

        # observe both objective and constraint data
        def observer(query_points: TensorType) -> dict[str, Dataset]:
            return {
                OBJECTIVE: Dataset(query_points, pb_obj(query_points)),
                CONSTRAINT: Dataset(query_points, pb_con(query_points)),
            }

    else:
        pb_con = None
        observer = trieste.objectives.utils.mk_observer(pb_obj, key=OBJECTIVE)

    # %%
    mins, maxs = pb.bounds[0], pb.bounds[1]
    search_space = Box(mins, maxs)
    num_objective = num_obj

    # %% [markdown]
    # Let's randomly sample some initial data from the observer ...

    # %%
    num_initial_points = num_doe
    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)

    obj_models_has_traj_for_rff_sample = build_stacked_independent_objectives_has_traj_model(
        initial_data[OBJECTIVE], num_objective, search_space, use_decoupled_sampler=False
    )
    obj_models_has_traj_for_rff_sample.optimize(initial_data[OBJECTIVE])

    if isinstance(pb(), multi_objectives.ConstraintMultiObjectiveTestProblem):
        # Objective Model
        con_models_has_traj_for_decoupled_sample = (
            build_stacked_independent_objectives_has_traj_model(
                initial_data[CONSTRAINT], num_con, search_space, use_decoupled_sampler=True
            )
        )

        con_models_has_traj_for_rff_sample = build_stacked_independent_objectives_has_traj_model(
            initial_data[CONSTRAINT], num_con, search_space, use_decoupled_sampler=False
        )
        con_models_has_traj_for_rff_sample.optimize(initial_data[CONSTRAINT])
    else:
        con_models_has_traj_for_rff_sample = None

    start_time = time.time()
    pfs_traj_approx_parallel_rff = sample_pareto_fronts_from_parametric_gp_posterior_using_parallel_nsga2(
        obj_models_has_traj_for_rff_sample,
        obj_num=num_obj,
        sample_pf_num=pf_sample_num,
        search_space=search_space,
        cons_num=num_con,
        constraint_models=con_models_has_traj_for_rff_sample,
        moo_solver="nsga2",
    )
    end_time = time.time()
    parallel_time = end_time - start_time


    obj_models_has_traj_for_rff_sample.get_trajectories(regenerate=True)
    start_time = time.time()
    pfs_traj_approx_rff = sample_pareto_fronts_from_parametric_gp_posterior(
        obj_models_has_traj_for_rff_sample,
        obj_num=num_obj,
        sample_pf_num=pf_sample_num,
        search_space=search_space,
        cons_num=num_con,
        constraint_models=con_models_has_traj_for_rff_sample,
        moo_solver="nsga2",
    )
    end_time = time.time()
    sequential_time = end_time - start_time
    return sequential_time, parallel_time


if __name__ == "__main__":
    sequential_time = []
    parallel_time = []
    for pf_sample_size in np.arange(5, 100, 10):
        print(f'Now test pf_size: {pf_sample_size}')
        seq_time, para_time = show_pareto_frontier_samples_on_problem('VLMOP2', num_doe=20, pf_sample_num=pf_sample_size)
        sequential_time.append(seq_time)
        parallel_time.append(para_time)
    from matplotlib import pyplot as plt
    plt.plot(np.arange(5, 100, 10), np.asarray(sequential_time), label='Seq for loop PF sample')
    plt.plot(np.arange(5, 100, 10), np.asarray(parallel_time), label='Parallel PF sample')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # for doe in np.arange(20, 120, 20):
    #     show_pareto_frontier_samples_on_problem('BraninCurrin', doe)
    # for doe in np.arange(30, 70, 10):
    #     show_pareto_frontier_samples_on_problem('CVLMOP2', doe, num_con=1)
