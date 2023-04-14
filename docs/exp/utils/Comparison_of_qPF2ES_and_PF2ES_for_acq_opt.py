import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from trieste.experimental.plotting import (
    plot_bo_points,
    plot_function_2d,
    plot_mobo_history,
    plot_mobo_points_in_obj_space,
)

# %%
import trieste
from trieste.acquisition.function import MESMO, PF2ES, ExpectedHypervolumeImprovement, PFES, MESMOC, CEHVI, \
    BatchMonteCarloConstrainedExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset
from trieste.models import TrainableModelStack, TrainableHasTrajectorySamplerModelStack, \
    TrainableHasTrajectoryAndPredictJointReparamModelStack
from trieste.models.interfaces import TrainablePredictJointModelStack, TrainablePredictJointReparamModelStack
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.models.gpflow.wrapper_model.transformed_models import StandardizedGaussianProcessRegression
from trieste.space import Box, SearchSpace
from trieste.objectives.multi_objectives import VLMOP2, ZDT1, ZDT2, ZDT3, Osy, TNK, TwoBarTruss, SRN, C2DTLZ2, \
    DiscBrakeDesign, EE6, FourBarTruss
from trieste.objectives.multi_objectives import SinLinearForrester, BraninCurrin, GMMForrester, CBraninCurrin
from trieste.acquisition.function import Fantasizer
from os.path import dirname, join
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.acquisition.utils import split_acquisition_function_calls

np.random.seed(1813)
tf.random.set_seed(1813)
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"

# good_num_initial_points = 5
# good_num_initial_points = 10
num_initial_points = 6


fig, axes = plt.subplots(1, 4, figsize=(15, 3))


srn = SRN
srn_obj = SRN().objective()
srn_cons = SRN().constraint()
srn_search_space = Box(*SRN.bounds)
srn_num_objective = 2
srn_num_con = 2


def srn_observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, srn_obj(query_points)),
        CONSTRAINT: Dataset(query_points, srn_cons(query_points)),
    }

# tnk = TNK
# tnk_obj = TNK().objective()
# tnk_cons = TNK().constraint()
# tnk_search_space = Box(*TNK.bounds)
# tnk_num_objective = 2
# tnk_num_con = 2


# def tnk_observer(query_points):
#     return {
#         OBJECTIVE: Dataset(query_points, tnk_obj(query_points)),
#         CONSTRAINT: Dataset(query_points, tnk_cons(query_points)),
#     }


def build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainableHasTrajectoryAndPredictJointReparamModelStack:
    gprs = []
    # likelihood_variance_list = [0.5, 1e-7]
    likelihood_variance_list = [1e-7, 1e-7]
    for idx in range(num_output):
        single_obj_data = Dataset(
            data.query_points, tf.gather(data.observations, [idx], axis=1)
        )
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=likelihood_variance_list[idx],
                        trainable_likelihood=False)
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=False), 1))

    return TrainableHasTrajectoryAndPredictJointReparamModelStack(*gprs)


# initial_query_points = srn_search_space.sample(num_initial_points)
initial_query_points = tf.convert_to_tensor(
    np.loadtxt(R'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\constraint_exp\exp_res\SRN\PF2ES\SRN_0_q1_queried_X.txt'), dtype=tf.float64)
initial_data = srn_observer(initial_query_points[:num_initial_points])
obj_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    initial_data[OBJECTIVE], srn_num_objective, srn_search_space
)
con_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
    initial_data[CONSTRAINT], srn_num_con, srn_search_space
)
models = {OBJECTIVE: obj_model, CONSTRAINT: con_model}
# aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\cbranincurrin'
aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'SRN')
search_space = srn_search_space
observer = srn_observer

# Acquisition Function Builder
np.random.seed(1813)
tf.random.set_seed(1813)
builder = PF2ES(search_space, moo_solver='nsga2', objective_tag=OBJECTIVE, constraint_tag=CONSTRAINT,
                sample_pf_num=5, remove_augmenting_region=False,
                population_size_for_approx_pf_searching=50, pareto_epsilon=0.04, remove_log=False,
                batch_mc_sample_size=64, use_dbscan_for_conservative_epsilon=False, parallel_sampling=False,
                temperature_tau=1E-3)

acq = builder.prepare_acquisition_function(models, initial_data)


def acq_f(at):
    """
    Acquisition Function for plotting (expand the batch dim)
    """
    return acq(
        tf.expand_dims(at, axis=-2)
    )


var_range = [[0, 1], [0, 1]]
plot_fidelity = 20
func = acq_f

x_1 = np.linspace(*var_range[0], plot_fidelity)
x_2 = np.linspace(*var_range[-1], plot_fidelity)



x_grid = np.array([[_x_1, _x_2] for _x_2 in x_2 for _x_1 in x_1])
_z_data = np.array(func(x_grid)).reshape(len(x_1), len(x_2))
CS = axes[0].contourf(
    x_1,
    x_2,
    _z_data,
)

# cax = fig.add_axes([0.92, ax.get_position().y0 + ax.get_position().height * 0.2, 0.015, 0.7])
# cbar = fig.colorbar(CS, cax=cax, pad = 0.06, shrink=1.0)
# cbar.ax.set_ylabel('Acquisition Function Value')
# ax.set_xticks([-2, 0, 2])
axes[0].set_xticks([])
# ax.set_yticks([-2, 0, 2])
axes[0].set_yticks([])
# ax.set_xlim(var_range[0])
# ax.set_ylim(var_range[1])
# ax.scatter(res.x[0], res.x[1], label='BO Added Input', s=30, color='darkorange', zorder=30)
initial_data_dataset = initial_data[OBJECTIVE]
axes[0].scatter(initial_data_dataset.query_points[:, 0],
                initial_data_dataset.query_points[:, 1],
                marker='x', color='r', label='Training data')
box = axes[0].get_position()
axes[0].set_position([box.x0 - box.width * 0.7, box.y0 + box.height * 0.1 ,
                 box.width, box.height * 0.85])
axes[0].set_title('{PF}$^2$ES Acq Contour', fontsize=12)
# plt.tight_layout()
handles, labels = axes[0].get_legend_handles_labels()

np.random.seed(1813)
tf.random.set_seed(1813)
mc32_builder = PF2ES(search_space, moo_solver='nsga2', objective_tag=OBJECTIVE, constraint_tag=CONSTRAINT,
                sample_pf_num=5, remove_augmenting_region=False,
                population_size_for_approx_pf_searching=50, pareto_epsilon=0.04, remove_log=False,
                batch_mc_sample_size=32, use_dbscan_for_conservative_epsilon=False, parallel_sampling=True,
                temperature_tau=1E-3, qMC = True)

mc32_acq = mc32_builder.prepare_acquisition_function(models, initial_data)


def mc32_acq_f(at):
    """
    Acquisition Function for plotting (expand the batch dim)
    """
    # max_batch_element = 500
    # splited_test_data = tf.split(
    #     at,
    #     at.shape[0] // max_batch_element,
    # )
    # splited_test_data = tf.concat([data[None] for data in splited_test_data], axis=0)
    # target_func_values = []
    # for splited_data in splited_test_data:
    #     target_func_values.append(mc32_acq(
    #         tf.expand_dims(splited_data, axis=-2)
    #         ))
    # target_func_values = tf.concat(target_func_values, 0)
    # return target_func_values
    return mc32_acq(
        tf.expand_dims(at, axis=-2)
    )


mc32_func = mc32_acq_f

_z_data = np.array(mc32_func(x_grid)).reshape(len(x_1), len(x_2))
CS = axes[1].contourf(
    x_1,
    x_2,
    _z_data,
)

axes[1].set_xticks([])
axes[1].set_yticks([])
initial_data_dataset = initial_data[OBJECTIVE]
axes[1].scatter(initial_data_dataset.query_points[:, 0],
                initial_data_dataset.query_points[:, 1],
                marker='x', color='r', label='Training data')
box = axes[1].get_position()
axes[1].set_position([box.x0 - box.width * 0.75, box.y0  + box.height * 0.1,
                 box.width, box.height * 0.85])
axes[1].set_title('q-{PF}$^2$ES Acq Contour \n (qMC = 32)', fontsize=12)

np.random.seed(1813)
tf.random.set_seed(1813)
mc512_builder = PF2ES(search_space, moo_solver='nsga2', objective_tag=OBJECTIVE, constraint_tag=CONSTRAINT,
                sample_pf_num=5, remove_augmenting_region=False,
                population_size_for_approx_pf_searching=50, pareto_epsilon=0.04, remove_log=False,
                batch_mc_sample_size=128, use_dbscan_for_conservative_epsilon=False, parallel_sampling=True,
                temperature_tau=1E-3, qMC = True)

mc512_acq = mc512_builder.prepare_acquisition_function(models, initial_data)


def mc512_acq_f(at):
    """
    Acquisition Function for plotting (expand the batch dim)
    """
    # max_batch_element = 500
    # splited_test_data = tf.split(
    #     at,
    #     at.shape[0] // max_batch_element,
    # )
    # splited_test_data = tf.concat([data[None] for data in splited_test_data], axis=0)
    # target_func_values = []
    # for splited_data in splited_test_data:
    #     target_func_values.append(mc512_acq(
    #         tf.expand_dims(splited_data, axis=-2)
    #         ))
    # target_func_values = tf.concat(target_func_values, 0)
    # return target_func_values
    return mc512_acq(
        tf.expand_dims(at, axis=-2)
    )


mc512_func = mc512_acq_f

_z_data = np.array(mc512_func(x_grid)).reshape(len(x_1), len(x_2))

CS = axes[2].contourf(
    x_1,
    x_2,
    _z_data,
)

# cax = fig.add_axes([0.92, ax.get_position().y0 + ax.get_position().height * 0.2, 0.015, 0.7])
# cbar = fig.colorbar(CS, cax=cax, pad = 0.06, shrink=1.0)
# cbar.ax.set_ylabel('Acquisition Function Value')
# ax.set_xticks([-2, 0, 2])
axes[2].set_xticks([])
# ax.set_yticks([-2, 0, 2])
axes[2].set_yticks([])
# ax.set_xlim(var_range[0])
# ax.set_ylim(var_range[1])
# ax.scatter(res.x[0], res.x[1], label='BO Added Input', s=30, color='darkorange', zorder=30)
initial_data_dataset = initial_data[OBJECTIVE]
axes[2].scatter(initial_data_dataset.query_points[:, 0],
                initial_data_dataset.query_points[:, 1],
                marker='x', color='r', label='Training data')
box = axes[2].get_position()
axes[2].set_position([box.x0 - box.width * 0.8, box.y0 + box.height * 0.1 ,
                 box.width, box.height * 0.85])
axes[2].set_title('q-{PF}$^2$ES Acq Contour \n (qMC = 128)', fontsize=12)

#
# norm_candx_100iter_1024 = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\auxiliary_files\acq_opt_difficulty\norm_candx_100iter_1024.txt')
# norm_candx_MC_32 = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\auxiliary_files\acq_opt_difficulty\norm_candx_MC_32.txt')
norm_candx_exact = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\auxiliary_files\acq_opt_difficulty\norm_candx_exact.txt')
norm_candx_mc_128_q1 = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\auxiliary_files\acq_opt_difficulty\norm_candx_qMC_fixed_q1.txt')
norm_candx_mc_128_q2 = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\auxiliary_files\acq_opt_difficulty\norm_candx_qMC_fixed_q2.txt')
# norm_candx_512_qMC_fixed = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\auxiliary_files\acq_opt_difficulty\norm_candx_512_qMC_fixed.txt')
# norm_candx_512_MC_fixed = np.loadtxt(r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\auxiliary_files\acq_opt_difficulty\norm_candx_512_MC_fixed.txt')

# for norm_cand_100iter_1024 in norm_candx_100iter_1024.T:
#     axes[3].plot(np.arange(norm_cand_100iter_1024.shape[0]), np.log(norm_cand_100iter_1024), alpha=0.4, color='red')
# axes[3].plot(np.arange(norm_cand_100iter_1024.shape[0]), np.log(norm_candx_100iter_1024.T[0]), alpha=0.4,
#          color='red', label='qPF2ES (q=2);  MC size: 1024')
#
# for norm_cand_MC_32 in norm_candx_MC_32.T:
#     axes[3].plot(np.arange(norm_cand_MC_32.shape[0]), np.log(norm_cand_MC_32), alpha=0.4, color='green')
# axes[3].plot(np.arange(norm_cand_MC_32.shape[0]), np.log(norm_candx_MC_32.T[0]), alpha=0.4,
#          color='green', label='qPF2ES (q=2);  MC size: 32')

for norm_cand_mc_128_q1 in norm_candx_mc_128_q1.T:
    axes[3].plot(np.arange(norm_cand_mc_128_q1.shape[0]), np.log(norm_cand_mc_128_q1), alpha=0.4, color='green')
axes[3].plot(np.arange(norm_cand_mc_128_q1.shape[0]), np.log(norm_candx_mc_128_q1.T[0]), alpha=0.4,
         color='green', label='q-{PF}$^2$ES (q=1, qMC=128)')

for norm_cand_mc_128_q2 in norm_candx_mc_128_q2.T:
    axes[3].plot(np.arange(norm_cand_mc_128_q2.shape[0]), np.log(norm_cand_mc_128_q2), alpha=0.4, color='darkorange')
axes[3].plot(np.arange(norm_cand_mc_128_q2.shape[0]), np.log(norm_candx_mc_128_q2.T[0]), alpha=0.4,
         color='darkorange', label='q-{PF}$^2$ES (q=2, qMC=128)')


for norm_cand_exact in norm_candx_exact.T:
    axes[3].plot(np.arange(norm_cand_exact.shape[0]), np.log(norm_cand_exact), alpha=0.4, color='blue')
axes[3].plot(np.arange(norm_cand_exact.shape[0]), np.log(norm_candx_exact.T[0]), alpha=0.4,
         color='blue', label='{PF}$^2$ES')

# for norm_cand_512_qMC_fixed in norm_candx_512_qMC_fixed.T:
#     axes[3].plot(np.arange(norm_cand_512_qMC_fixed.shape[0]), np.log(norm_cand_512_qMC_fixed), alpha=0.4, color='purple')
# axes[3].plot(np.arange(norm_cand_512_qMC_fixed.shape[0]), np.log(norm_candx_512_qMC_fixed.T[0]), alpha=0.4,
#          color='purple', label='qPF2ES (q=2);  qMC size: 512')
#
#
# for norm_cand_512_MC_fixed in norm_candx_512_MC_fixed.T:
#     axes[3].plot(np.arange(norm_cand_512_MC_fixed.shape[0]), np.log(norm_cand_512_MC_fixed), alpha=0.4, color='darkorange')
# axes[3].plot(np.arange(norm_cand_512_MC_fixed.shape[0]), np.log(norm_candx_512_MC_fixed.T[0]), alpha=0.4,
#          color='darkorange', label='qPF2ES (q=2);  MC size: 512')

box = axes[3].get_position()
axes[3].set_position([box.x0 - box.width * 0.7, box.y0 + box.height * 0.1 ,
                 box.width, box.height * 0.85])

axes[3].set_xlabel('L-BFGS-B Evaluations', fontsize=12)
axes[3].set_ylabel('Log Euclidean Derivatives Norms ', fontsize=12)
axes[3].set_ylim([-25, 5])
# axes[3].legend(bbox_to_anchor=[1.02, 0.5])

handles2, labels2 = axes[3].get_legend_handles_labels()
fig.legend([*handles, *handles2], [*labels, *labels2],
           bbox_to_anchor=(1.0, 0.8), fancybox=False, fontsize=12)
plt.title('Convergence History of L-BFGS-B \n on Different Acquisition Functions', fontsize=12)
plt.savefig('Acq Opt Comparison', dpi=1000)
# plt.show()