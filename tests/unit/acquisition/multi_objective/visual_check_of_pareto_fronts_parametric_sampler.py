import math

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# %%
import trieste
from trieste.acquisition.multi_objective.utils import (
    sample_pareto_fronts_from_parametric_gp_posterior,
)
from trieste.data import Dataset
from trieste.models import TrainableHasTrajectorySamplerModelStack, TrainableModelStack
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives.multi_objectives import VLMOP2
from trieste.space import Box, SearchSpace

np.random.seed(1793)
tf.random.set_seed(1793)

# --------------
pf_sample_num = 5
ftz = 15
# --------------
# %%
vlmop2 = VLMOP2().objective()
observer = trieste.objectives.utils.mk_observer(vlmop2)

# %%
mins = [-2, -2]
maxs = [2, 2]
search_space = Box(mins, maxs)
num_objective = 2

# %% [markdown]
# Let's randomly sample some initial data from the observer ...

# %%
num_initial_points = 50
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)
#
#
# # %%
def build_stacked_independent_objectives_has_traj_model(
    data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainableHasTrajectorySamplerModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7)
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=True), 1))

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


# %%
models_has_traj = build_stacked_independent_objectives_has_traj_model(
    initial_data, num_objective, search_space
)

models_has_traj.optimize(initial_data)

models = build_stacked_independent_objectives_model(initial_data, num_objective, search_space)

# Set hyperparam of second model stack the same as the first one
for model_has_traj, model in zip(models_has_traj._models, models._models):
    model._models.kernel.lengthscales = model_has_traj._models.kernel.lengthscales
    model._models.mean_function.c = model_has_traj._models.mean_function.c
    model._models.kernel.variance = model_has_traj._models.kernel.variance
    model._models.likelihood.variance = model_has_traj._models.likelihood.variance


pfs_traj_approx = sample_pareto_fronts_from_parametric_gp_posterior(
    models_has_traj,
    obj_num=2,
    sample_pf_num=pf_sample_num,
    search_space=search_space,
    moo_solver="nsga2",
)


pfs_discrete_approx = sample_pareto_fronts_from_parametric_gp_posterior(
    models,
    obj_num=2,
    sample_pf_num=pf_sample_num,
    search_space=search_space,
    moo_solver="monte_carlo",
)

from matplotlib import pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)
for pf, pf_idx in zip(pfs_traj_approx, range(len(pfs_traj_approx))):
    axes[0].scatter(pf[:, 0], pf[:, 1], label=f"PF Sample {pf_idx}")
axes[0].set_title("PF samples with continuous \nMOO optimizer on trajectories", fontsize=ftz)

for pf, pf_idx in zip(pfs_discrete_approx, range(len(pfs_discrete_approx))):
    axes[1].scatter(pf[:, 0], pf[:, 1], label=f"PF Sample {pf_idx}")
axes[1].set_title("PF samples with discrete \nMOO optimizer on trajectories", fontsize=ftz)
# plt.legend()
# plt.title('PF samples comparison: decoupled sampling vs Monte Carlo')
plt.tight_layout()
plt.show(block=True)


# Constraint PF samples check
class Sim:
    threshold = 0.75

    @staticmethod
    def objective(input_data):
        return vlmop2(input_data)

    @staticmethod
    def constraint(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return Sim.threshold - z[:, None]  # - 10


OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"

# from PyOptimize.utils.visualization import view_2D_function_in_contour

# view_2D_function_in_contour(Sim.constraint, [[-2, 2]] * 2, colorbar=True)


def observer_cst(query_points):
    return {
        OBJECTIVE: Dataset(query_points, Sim.objective(query_points)),
        CONSTRAINT: Dataset(query_points, Sim.constraint(query_points)),
    }


initial_query_points = search_space.sample(num_initial_points)
initial_data = observer_cst(initial_query_points)

num_constraint = 1

models_obj_has_traj = build_stacked_independent_objectives_has_traj_model(
    initial_data[OBJECTIVE], num_objective, search_space
)


models_con_has_traj = build_stacked_independent_objectives_has_traj_model(
    initial_data[CONSTRAINT], num_constraint, search_space
)

models_obj_has_traj.optimize(initial_data[OBJECTIVE])
models_con_has_traj.optimize(initial_data[CONSTRAINT])

models_obj = build_stacked_independent_objectives_model(
    initial_data[OBJECTIVE], num_objective, search_space
)

models_con = build_stacked_independent_objectives_model(
    initial_data[CONSTRAINT], num_constraint, search_space
)

for model_has_traj, model in zip(models_obj_has_traj._models, models_obj._models):
    model._models.kernel.lengthscales = model_has_traj._models.kernel.lengthscales
    model._models.mean_function.c = model_has_traj._models.mean_function.c
    model._models.kernel.variance = model_has_traj._models.kernel.variance
    model._models.likelihood.variance = model_has_traj._models.likelihood.variance


for cmodel_has_traj, cmodel in zip(models_con_has_traj._models, models_con._models):
    cmodel._models.kernel.lengthscales = cmodel_has_traj._models.kernel.lengthscales
    cmodel._models.mean_function.c = cmodel_has_traj._models.mean_function.c
    cmodel._models.kernel.variance = cmodel_has_traj._models.kernel.variance
    cmodel._models.likelihood.variance = cmodel_has_traj._models.likelihood.variance

cpfs_traj_approx = sample_pareto_fronts_from_parametric_gp_posterior(
    models_obj_has_traj,
    obj_num=2,
    cons_num=1,
    constraint_models=models_con_has_traj,
    sample_pf_num=pf_sample_num,
    search_space=search_space,
    moo_solver="nsga2",
)


cpfs_discrete_approx = sample_pareto_fronts_from_parametric_gp_posterior(
    models_obj,
    obj_num=2,
    cons_num=1,
    constraint_models=models_con,
    sample_pf_num=pf_sample_num,
    search_space=search_space,
    moo_solver="monte_carlo",
)


from matplotlib import pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)
for cpf, cpf_idx in zip(cpfs_traj_approx, range(len(cpfs_traj_approx))):
    axes[0].scatter(cpf[:, 0], cpf[:, 1], label=f"PF Sample {cpf_idx}")
axes[0].set_title("cPF samples with continuous \nMOO optimizer on trajectories", fontsize=ftz)

for cpf, cpf_idx in zip(cpfs_discrete_approx, range(len(cpfs_discrete_approx))):
    axes[1].scatter(cpf[:, 0], cpf[:, 1], label=f"PF Sample {cpf_idx}")
axes[1].set_title("cPF samples with discrete \nMOO optimizer on trajectories", fontsize=ftz)
# plt.legend()
# plt.title('PF samples comparison: decoupled sampling vs Monte Carlo')
plt.tight_layout()
plt.show(block=True)
