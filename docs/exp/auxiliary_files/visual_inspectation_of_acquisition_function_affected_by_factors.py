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
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.models.interfaces import (
    TrainableHasTrajectoryAndPredictJointReparamModelStack,
    TrainableHasTrajectorySamplerModelStack,
    TrainableModelStack,
    TrainablePredictJointReparamModelStack,
)
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


def build_stacked_independent_objectives_model(
    data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainablePredictJointReparamModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7)
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=True), 1))

    return TrainablePredictJointReparamModelStack(*gprs)


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


def plot_acquisition_function_affected_by_factors(
    pb_name,
    num_doe: int,
    num_obj: int = 2,
    num_con: int = 0,
    pf_sample_num: list = [5],
    pf_element_size: list = [50],
    acq: str = "PF2ES",
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
            build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
                initial_data[CONSTRAINT], num_con, search_space, use_decoupled_sampler=True
            )
        )

        con_models_has_traj_for_decoupled_sample.optimize(initial_data[CONSTRAINT])

        con_models_for_monte_carlo = build_stacked_independent_objectives_model(
            initial_data[CONSTRAINT], num_con, search_space
        )

        con_models_has_traj_for_rff_sample = (
            build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
                initial_data[CONSTRAINT], num_con, search_space, use_decoupled_sampler=False
            )
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
        con_models_has_traj_for_decoupled_sample = None
        con_models_for_monte_carlo = None
        con_models_has_traj_for_rff_sample = None

    if acq == "PF2ES":
        from matplotlib import pyplot as plt

        from trieste.acquisition.function.multi_objective import PF2ES

        plt.rcParams["xtick.labelsize"] = ftz
        plt.rcParams["ytick.labelsize"] = ftz
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
        if len(mins) == 1:  # 1 Dimensional Problem
            test_xs = tf.cast(
                tf.linspace([0.0], [1.0], 5000), dtype=initial_data[OBJECTIVE].observations.dtype
            )
            # Monte Carlo
            # for pf_sample in pf_sample_num:
            #     acq = PF2ES(search_space, sample_pf_num=pf_sample, moo_solver='monte_carlo', remove_log=True).prepare_acquisition_function(
            #         {OBJECTIVE: obj_models_for_monte_carlo}, {OBJECTIVE: initial_data[OBJECTIVE]},
            #     )
            #     axes[0, 0].plot(tf.squeeze(test_xs), tf.squeeze(acq(tf.expand_dims(test_xs, -2))),
            #                     label=f'PF2ES PF:{pf_sample}, PFe: 50')
            # axes[0, 0].legend()
            # axes[0, 0].set_title("Acq var by PF samples size (MC)", fontsize=ftz)
            #
            # for pf_element in [500, 1000, 1500, 2000]:
            #     acq = PF2ES(search_space, discretize_input_sample_size=pf_element,
            #                 moo_solver='monte_carlo', remove_log=True).prepare_acquisition_function(
            #         {OBJECTIVE: obj_models_for_monte_carlo}, {OBJECTIVE: initial_data[OBJECTIVE]})
            #     axes[1, 0].plot(tf.squeeze(test_xs), tf.squeeze(acq(tf.expand_dims(test_xs, -2))),
            #                     label=f'PF2ES PF: 5, PFe MC: {pf_element}')
            # axes[1, 0].legend()
            # axes[1, 0].set_title("Acq var by PF element size (MC)", fontsize=ftz)

            # RFF
            # for pf_sample in pf_sample_num:
            #     acq = PF2ES(search_space, sample_pf_num=pf_sample, remove_log=True).prepare_acquisition_function(
            #         {OBJECTIVE: obj_models_has_traj_for_rff_sample}, {OBJECTIVE: initial_data[OBJECTIVE]})
            #     axes[0, 1].plot(tf.squeeze(test_xs), tf.squeeze(acq(tf.expand_dims(test_xs, -2))),
            #                     label=f'PF2ES PF:{pf_sample}, PFe: 50')
            # axes[0, 1].legend()
            # axes[0, 1].set_title("Acq var by PF samples size (RFF)", fontsize=ftz)
            # # plt.show(block=True)
            # # FIXME: PF_sample num is actually 1
            # for pf_element in pf_element_size:
            #     acq = PF2ES(search_space, population_size_for_approx_pf_searching=pf_element, remove_log=True).prepare_acquisition_function(
            #         {OBJECTIVE: obj_models_has_traj_for_rff_sample}, {OBJECTIVE: initial_data[OBJECTIVE]})
            #     axes[1, 1].plot(tf.squeeze(test_xs), tf.squeeze(acq(tf.expand_dims(test_xs, -2))),
            #                     label=f'PF2ES PF: 5, PFe: {pf_element}')
            # axes[1, 1].legend()
            # axes[1, 1].set_title("Acq var by PF element size (RFF)", fontsize=ftz)

            # Decoupled
            for pf_sample in pf_sample_num:
                acq = PF2ES(
                    search_space, sample_pf_num=pf_sample, remove_log=True
                ).prepare_acquisition_function(
                    {OBJECTIVE: obj_models_has_traj_for_decoupled_sample},
                    {OBJECTIVE: initial_data[OBJECTIVE]},
                )
                axes[0, 2].plot(
                    tf.squeeze(test_xs),
                    tf.squeeze(acq(tf.expand_dims(test_xs, -2))),
                    label=f"PF2ES PF:{pf_sample}, PFe: 50",
                )
            axes[0, 2].legend()
            axes[0, 2].set_title("Acq var by PF samples size (Decoupled)", fontsize=ftz)
            # plt.show(block=True)
            # FIXME: PF_sample num is actually 1
            for pf_element in pf_element_size:
                acq = PF2ES(
                    search_space,
                    population_size_for_approx_pf_searching=pf_element,
                    remove_log=True,
                ).prepare_acquisition_function(
                    {OBJECTIVE: obj_models_has_traj_for_decoupled_sample},
                    {OBJECTIVE: initial_data[OBJECTIVE]},
                )
                axes[1, 2].plot(
                    tf.squeeze(test_xs),
                    tf.squeeze(acq(tf.expand_dims(test_xs, -2))),
                    label=f"PF2ES PF: 5, PFe: {pf_element}",
                )
            axes[1, 2].legend()
            axes[1, 2].set_title("Acq var by PF element size (Decoupled)", fontsize=ftz)

            fig.suptitle(
                f"{pb_name}: Pareto Frontier Samples From GP with {num_doe} Training Samples",
                fontsize=ftz + 5,
            )
            plt.tight_layout()
            # plt.show(block=True)
            # plt.savefig(os.path.join('acq_sensitivity_analysis', f'{pb_name}_Acq_{num_doe}DOE.png'))
            plt.savefig(
                os.path.join(
                    "acq_sensitivity_analysis", f"{pb_name}_Acq_{num_doe}DOE_without_log.png"
                )
            )
        else:
            assert len(mins) == 2
            raise NotImplementedError


if __name__ == "__main__":
    for doe in np.arange(20, 25, 5):
        plot_acquisition_function_affected_by_factors(
            "Quadratic", doe, pf_sample_num=[1, 5, 10, 20], pf_element_size=[20, 50, 100, 200]
        )
    # for doe in np.arange(30, 35, 5):
    #     plot_acquisition_function_affected_by_factors(
    #         'SinLinearForrester', doe, pf_sample_num=[1, 5, 10, 20], pf_element_size=[20, 50, 100, 200])
    # for doe in np.arange(25, 40, 5):
    #
    # for doe in np.arange(10, 60, 10):
    #     show_pareto_frontier_samples_on_problem('VLMOP2', doe)
    # for doe in np.arange(60, 70, 10):
    #     show_pareto_frontier_samples_on_problem('CVLMOP2', doe, num_con=1)
    # plot_acquisition_function_affected_by_factors('BraninCurrin', 20)
