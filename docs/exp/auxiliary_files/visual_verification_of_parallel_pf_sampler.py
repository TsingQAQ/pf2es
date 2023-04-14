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


def build_stacked_independent_objectives_model(
    data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainableModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7)
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=True), 1))

    return TrainableModelStack(*gprs)


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

    # %%
    # Objective Model
    obj_models_has_traj_for_decoupled_sample = build_stacked_independent_objectives_has_traj_model(
        initial_data[OBJECTIVE], num_objective, search_space, use_decoupled_sampler=True
    )

    obj_models_has_traj_for_decoupled_sample.optimize(initial_data[OBJECTIVE])

    obj_models_for_monte_carlo = build_stacked_independent_objectives_model(
        initial_data[OBJECTIVE], num_objective, search_space
    )

    obj_models_has_traj_for_rff_sample = build_stacked_independent_objectives_has_traj_model(
        initial_data[OBJECTIVE], num_objective, search_space, use_decoupled_sampler=False
    )

    # Set hyperparam of second model stack the same as the first one
    for model_has_traj_decoupled, model, model_has_traj_rff in zip(
        obj_models_has_traj_for_decoupled_sample._models,
        obj_models_for_monte_carlo._models,
        obj_models_has_traj_for_rff_sample._models,
    ):
        model._models.kernel.lengthscales = model_has_traj_decoupled._models.kernel.lengthscales
        model._models.mean_function.c = model_has_traj_decoupled._models.mean_function.c
        model._models.kernel.variance = model_has_traj_decoupled._models.kernel.variance
        model._models.likelihood.variance = model_has_traj_decoupled._models.likelihood.variance

        model_has_traj_rff._models.kernel.lengthscales = (
            model_has_traj_decoupled._models.kernel.lengthscales
        )
        model_has_traj_rff._models.mean_function.c = model_has_traj_decoupled._models.mean_function.c
        model_has_traj_rff._models.kernel.variance = model_has_traj_decoupled._models.kernel.variance
        model_has_traj_rff._models.likelihood.variance = (
            model_has_traj_decoupled._models.likelihood.variance
        )

    if isinstance(pb(), multi_objectives.ConstraintMultiObjectiveTestProblem):
        # Objective Model
        con_models_has_traj_for_decoupled_sample = (
            build_stacked_independent_objectives_has_traj_model(
                initial_data[CONSTRAINT], num_con, search_space, use_decoupled_sampler=True
            )
        )

        con_models_has_traj_for_decoupled_sample.optimize(initial_data[CONSTRAINT])

        con_models_for_monte_carlo = build_stacked_independent_objectives_model(
            initial_data[CONSTRAINT], num_con, search_space
        )

        con_models_has_traj_for_rff_sample = build_stacked_independent_objectives_has_traj_model(
            initial_data[CONSTRAINT], num_con, search_space, use_decoupled_sampler=False
        )

        # Set hyperparam of second model stack the same as the first one
        for model_has_traj_decoupled, model, model_has_traj_rff in zip(
            con_models_has_traj_for_decoupled_sample._models,
            con_models_for_monte_carlo._models,
            con_models_has_traj_for_rff_sample._models,
        ):
            model._models.kernel.lengthscales = model_has_traj_decoupled._models.kernel.lengthscales
            model._models.mean_function.c = model_has_traj_decoupled._models.mean_function.c
            model._models.kernel.variance = model_has_traj_decoupled._models.kernel.variance
            model._models.likelihood.variance = model_has_traj_decoupled._models.likelihood.variance

            model_has_traj_rff._models.kernel.lengthscales = (
                model_has_traj_decoupled._models.kernel.lengthscales
            )
            model_has_traj_rff._models.mean_function.c = (
                model_has_traj_decoupled._models.mean_function.c
            )
            model_has_traj_rff._models.kernel.variance = (
                model_has_traj_decoupled._models.kernel.variance
            )
            model_has_traj_rff._models.likelihood.variance = (
                model_has_traj_decoupled._models.likelihood.variance
            )
    else:
        con_models_for_monte_carlo = None
        con_models_has_traj_for_rff_sample = None


    pfs_traj_approx_parallel_rff = sample_pareto_fronts_from_parametric_gp_posterior_using_parallel_nsga2(
        obj_models_has_traj_for_rff_sample,
        obj_num=num_obj,
        sample_pf_num=pf_sample_num,
        search_space=search_space,
        cons_num=num_con,
        constraint_models=con_models_has_traj_for_rff_sample,
        moo_solver="nsga2",
    )

    obj_models_has_traj_for_rff_sample.get_trajectories(regenerate=True)
    if con_models_has_traj_for_rff_sample is not None:
        con_models_has_traj_for_rff_sample.get_trajectories(regenerate=True)
    pfs_traj_approx_rff = sample_pareto_fronts_from_parametric_gp_posterior(
        obj_models_has_traj_for_rff_sample,
        obj_num=num_obj,
        sample_pf_num=pf_sample_num,
        search_space=search_space,
        cons_num=num_con,
        constraint_models=con_models_has_traj_for_rff_sample,
        moo_solver="nsga2",
    )

    pfs_discrete_approx = sample_pareto_fronts_from_parametric_gp_posterior(
        obj_models_for_monte_carlo,
        obj_num=num_obj,
        sample_pf_num=pf_sample_num,
        search_space=search_space,
        cons_num=num_con,
        constraint_models=con_models_for_monte_carlo,
        moo_solver="monte_carlo",
    )

    from matplotlib import pyplot as plt

    from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo

    plt.rcParams["xtick.labelsize"] = ftz
    plt.rcParams["ytick.labelsize"] = ftz
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 5), sharex=True, sharey=True)

    pf = moo_nsga2_pymoo(
        f=pb_obj,
        input_dim=len(mins),
        obj_num=num_obj,
        cons=pb_con,
        cons_num=num_con,
        bounds=(tf.convert_to_tensor(mins), tf.convert_to_tensor(maxs)),
        popsize=50,
    )
    axes[0].scatter(pf[:, 0], pf[:, 1])
    axes[0].set_title("Groun Truth PF", fontsize=ftz)

    for pf, pf_idx in zip(pfs_discrete_approx, range(len(pfs_discrete_approx))):
        axes[1].scatter(pf[:, 0], pf[:, 1], label=f"PF Sample {pf_idx}")
    axes[1].set_title("PF samples using Monte Carlo", fontsize=ftz)

    for pf, pf_idx in zip(pfs_traj_approx_parallel_rff, range(pfs_traj_approx_parallel_rff.shape[0])):
        axes[2].scatter(pf.to_tensor()[:, 0], pf.to_tensor()[:, 1], label=f"PF Sample {pf_idx}")
    axes[2].set_title("RFF Parallel Sampler ", fontsize=ftz)

    for pf, pf_idx in zip(pfs_traj_approx_rff, range(len(pfs_traj_approx_rff))):
        axes[3].scatter(pf[:, 0], pf[:, 1], label=f"PF Sample {pf_idx}")
    axes[3].set_title("PF samples (RFF)", fontsize=ftz)

    fig.suptitle(
        f"{pb_name}: Pareto Frontier Samples From GP with {num_doe} Training Samples",
        fontsize=ftz + 5,
    )
    plt.tight_layout()
    plt.savefig(os.path.join("pf_sample_inspect", "parallel_pf_sample_inspect", f"{pb_name}_{num_doe}DOE.png"))


if __name__ == "__main__":
    for doe in np.arange(10, 100, 10):
        show_pareto_frontier_samples_on_problem('VLMOP2', doe, pf_sample_num=20)
    for doe in np.arange(20, 120, 10):
        show_pareto_frontier_samples_on_problem('BraninCurrin', doe, pf_sample_num=20)
    for doe in np.arange(10, 70, 10):
        show_pareto_frontier_samples_on_problem('CVLMOP2', doe, num_con=1, pf_sample_num=20)
