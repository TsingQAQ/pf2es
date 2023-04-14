# %% [markdown]
# # MOO Test

# %% [markdown]
# Here we benchmark several well-known unconstraint problem using the unified interface `serial_benchmarker.py`, this is useful for making sure 1) the `serial_benchmarker.py` is working correctly. 

# %% [markdown]
# Make sure we have imported stuff

# %%
from docs.exp.multi_objective_benchmarker import serial_benchmarker
from docs.notebooks.util.plotting import plot_mobo_points_in_obj_space, plot_bo_points, plot_function_2d
from trieste.acquisition.function.multi_objective import (
    MESMO,
    MESMOC,
    BatchFeasibleParetoFrontierEntropySearch,
    BatchFeasibleParetoFrontierEntropySearchInformationLowerBound,
    BatchMonteCarloExpectedHypervolumeImprovement,
    ExpectedHypervolumeImprovement,
    CEHVI
)
from matplotlib import pyplot as plt
import tensorflow as tf

# %% [markdown]
# ## Unconstraint MOO Problem

# %% [markdown]
# ### VLMOP2-2IN-2OUT

# %% [markdown]
# Optimization Code

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES'
file_info_prefix = ""
total_iter = 20
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "pf_mc_sample_num": 100, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}}
kwargs_for_optimize={"inspect_input_contour": True, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="VLMOP2", acq=acq, num_obj=2, q=1,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize,
                               acq_name='PFES-ILB')
                               # CustomBayesianOptimizer=True, kwargs_for_optimize={}, acq_name='PFES-ILB')
                              # 
                               # CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# Plot of the result

# %%
dataset = bo_result.try_get_final_dataset()
data_observations = dataset.observations

plot_mobo_points_in_obj_space(
    data_observations, num_init=100
)
plt.title('PFES DOE100')
plt.show()

# %%
dataset = bo_result.try_get_final_dataset()
data_observations = dataset.observations

plot_mobo_points_in_obj_space(
    data_observations, num_init=100
)
plt.title('PFES-ILB DOE100')
plt.show()

# %% [markdown]
# ### BraninCurrin

# %% [markdown]
# #### Calculate PF and HV
#

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import branincurrin, evaluate_slack_true
import tensorflow as tf

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(branincurrin, 2, 2, bounds = (tf.constant([0.0, 0]), tf.constant([1, 1])),
                   popsize=100, return_pf_x=True, num_generation=500)


# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %% [markdown]
# Optimization Code

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound #  BatchFeasibleParetoFrontierEntropySearch # 
file_info_prefix = ""
total_iter = 30
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "pf_mc_sample_num": 100, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}}
kwargs_for_optimize={"inspect_input_contour": True, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# %%
acq_name = 'PFES_IBO'
PFES_IBO_bo_result = serial_benchmarker(process_identifier=0, benchmark_name="BraninCurrin", acq=acq, num_obj=2,
                                        kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10, q=1,
                                        CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# Plot of the result

# %%
# %matplotlib notebook

# %%
dataset = PFES_IBO_bo_result.try_get_final_dataset()
data_observations = dataset.observations

plot_mobo_points_in_obj_space(
    data_observations, num_init=10
)
plt.title('FPFES-ILB Monte Carlo')
plt.show()

# %%
dataset = PFES_IBO_bo_result.try_get_final_dataset()
data_observations = dataset.observations

plot_mobo_points_in_obj_space(
    data_observations, num_init=10
)
plt.title('FPFES-ILB Analytical')
plt.show()

# %% [markdown]
# ### DTLZ2-3IN-2OUT

# %% [markdown]
# #### Calculate PF and HV
#

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ2, evaluate_slack_true
import tensorflow as tf

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(DTLZ2(3, 2).objective(), 3, 2, bounds = (tf.constant([0.0, 0.0, 0.0]), tf.constant([1, 1, 1])),
                   popsize=100, return_pf_x=True, num_generation=500)


# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
from trieste.acquisition.function.multi_objective import Pareto
Pareto(uc_res).hypervolume_indicator(tf.constant([2.5, 2.5], dtype=tf.float64))

# %% [markdown]
# Optimization Code

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 10
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "pf_mc_sample_num": 1, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}}
kwargs_for_benchmark = {"input_dim": 4, "num_objective":3}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="DTLZ2", acq=acq, num_obj=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               kwargs_for_benchmark=kwargs_for_benchmark, q=5)

# %% [markdown]
# Plot of the result

# %%
# %matplotlib notebook

# %%
dataset = bo_result.try_get_final_dataset()
data_observations = dataset.observations

plot_mobo_points_in_obj_space(
    data_observations, num_init=10
)
plt.show()

# %% [markdown]
# ### DTLZ3-3IN-2OUT

# %% [markdown]
# #### Calculate PF and HV
#

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ3, evaluate_slack_true
import tensorflow as tf

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(DTLZ3(3, 2).objective(), 3, 2, bounds = (tf.constant([0.0, 0.0, 0.0]), tf.constant([1, 1, 1])),
                   popsize=100, return_pf_x=True, num_generation=2000)


# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
np.savetxt('DTLZ3_PF.txt', uc_res)

# %%
from trieste.acquisition.function.multi_objective import Pareto
Pareto(uc_res).hypervolume_indicator(tf.constant([2.5, 2.5], dtype=tf.float64))

# %% [markdown]
# Optimization Code

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES'
file_info_prefix = ""
total_iter = 40
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "pf_mc_sample_num": 5, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}
kwargs_for_benchmark = {"input_dim": 3, "num_objective":2}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="DTLZ3", acq=acq, num_obj=2, q=2,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=100,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize,
                               acq_name='PFES-ILB', kwargs_for_benchmark=kwargs_for_benchmark)
                               # CustomBayesianOptimizer=True, kwargs_for_optimize={}, acq_name='PFES-ILB')
                              # 
                               # CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# ### DTLZ4-3IN-2OUT

# %% [markdown]
# #### Calculate PF and HV
#

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ4, evaluate_slack_true
import tensorflow as tf

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(DTLZ4(3, 2).objective(), 3, 2, bounds = (tf.constant([0.0, 0.0, 0.0]), tf.constant([1, 1, 1])),
                   popsize=200, return_pf_x=True, num_generation=2000)


# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
from trieste.acquisition.function.multi_objective import Pareto
Pareto(uc_res).hypervolume_indicator(tf.constant([2.5, 2.5], dtype=tf.float64))

# %%
np.savetxt('DTLZ4_3I2O_PF.txt', uc_res)

# %% [markdown]
# Optimization Code

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES'
file_info_prefix = ""
total_iter = 20
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "pf_mc_sample_num": 5, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="DTLZ4", acq=acq, num_obj=3, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize,
                               acq_name='PFES-ILB')
                               # CustomBayesianOptimizer=True, kwargs_for_optimize={}, acq_name='PFES-ILB')
                              # 
                               # CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# Plot of the result

# %%
dataset = bo_result.try_get_final_dataset()
data_observations = dataset.observations

plot_mobo_points_in_obj_space(
    data_observations, num_init=100
)
plt.title('PFES DOE100')
plt.show()

# %% [markdown]
# ### DTLZ4-4IN-3OUT

# %% [markdown]
# #### Calculate PF and HV
#

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ4, evaluate_slack_true
import tensorflow as tf

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(DTLZ4(4, 3).objective(), 4, 3, bounds = (tf.constant([0.0, 0.0, 0.0, 0.0]), tf.constant([1, 1, 1, 1])),
                   popsize=200, return_pf_x=True, num_generation=2000)


# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(uc_res[:, 0], uc_res[:, 1], uc_res[:, 2], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
from trieste.acquisition.function.multi_objective import Pareto
Pareto(uc_res).hypervolume_indicator(tf.constant([2.5, 2.5, 2.5], dtype=tf.float64))

# %% [markdown]
# Optimization Code

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES'
file_info_prefix = ""
total_iter = 20
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "pf_mc_sample_num": 5, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="DTLZ4", acq=acq, num_obj=3, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize,
                               acq_name='PFES-ILB')
                               # CustomBayesianOptimizer=True, kwargs_for_optimize={}, acq_name='PFES-ILB')
                              # 
                               # CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# Plot of the result

# %%
dataset = bo_result.try_get_final_dataset()
data_observations = dataset.observations

plot_mobo_points_in_obj_space(
    data_observations, num_init=100
)
plt.title('PFES DOE100')
plt.show()

# %% [markdown]
# ### DTLZ5-4IN-3OUT

# %% [markdown]
# Correctness check: 
# since DTLZ5 is implemented ourselve, i am doing a check of whether this implementation is numerically correct

# %%
import tensorflow as tf
from pymoo.factory import get_problem

f_pymoo = lambda x: tf.convert_to_tensor(get_problem("dtlz5", n_var = 4, n_obj = 3).four_bar_truss_obj(tf.convert_to_tensor(x)))
f_imple = DTLZ5(4, 3).objective

# %%
print(f_pymoo(tf.constant([0.2, 0.3, 0.4, 0.5])))
print(f_imple(tf.constant([[0.2, 0.3, 0.4, 0.5]])))

# %% [markdown]
# #### Calculate PF and HV
#

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import DTLZ5
import tensorflow as tf

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(DTLZ5(4, 3).objective(), 4, 3, bounds = (tf.constant([0.0, 0.0, 0.0, 0.0]), tf.constant([1, 1, 1, 1])),
                   popsize=200, return_pf_x=True, num_generation=2000)


# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(uc_res[:, 0], uc_res[:, 1], uc_res[:, 2], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
from trieste.acquisition.function.multi_objective import Pareto
Pareto(uc_res).hypervolume_indicator(tf.constant([2.5, 2.5, 2.5], dtype=tf.float64))

# %% [markdown]
# ### Vehicle Safety

# %% [markdown]
# #### Calculate PF and HV
#

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import vehicle_safety
import tensorflow as tf

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(vehicle_safety, 5, 3, bounds = (tf.constant([0.0, 0.0, 0.0, 0.0, 0.0]), tf.constant([1, 1, 1, 1, 1])),
                   popsize=200, return_pf_x=True, num_generation=2000)


# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(uc_res[:, 0], uc_res[:, 1], uc_res[:, 2], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
np.savetxt('VehicleCrash_PF.txt', uc_res)

# %%
from trieste.acquisition.function.multi_objective import Pareto
Pareto(uc_res).hypervolume_indicator(tf.constant([1695, 11, 0.30], dtype=tf.float64))

# %% [markdown]
# ## Constraint MOO Problem

# %% [markdown]
# ### Constraint-VLMOP2

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import CVLMOP2
import tensorflow as tf
# constraint version
res, x = moo_optimize_pymoo(CVLMOP2().objective(), 2, 2, bounds = (tf.constant([-2, -2]), tf.constant([2, 2])),
                            popsize=100, cons = CVLMOP2().constraint(), cons_num=1,
                            return_pf_x=True, num_generation=1000)

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(CVLMOP2().objective(), 2, 2, bounds = (tf.constant([0.0, 0]), tf.constant([2, 2])),
                                  popsize=50, return_pf_x=True, num_generation=500)


# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
# plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
np.savetxt('CVLMOP2_PF.txt', res)

# %%
from trieste.acquisition.function.multi_objective import Pareto, get_reference_point

# %%
get_reference_point(res)

# %%
Pareto(res).hypervolume_indicator(tf.constant([1.20, 1.20], dtype=tf.float64))

# %% [markdown]
# ---------------

# %%
import numpy as np
import tensorflow as tf
res = np.loadtxt('Y:\codes\Spearmint-master\examples\cvlmop2\CPF_49.txt')
from trieste.acquisition.function.multi_objective import Pareto, get_reference_point
Pareto(res).hypervolume_indicator(tf.constant([1.20, 1.20], dtype=tf.float64))

# %% [markdown]
# Testing if `pareto_front_on_mean_based_on_model` recommender works properlly

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 1
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

bo_result = serial_benchmarker(process_identifier=0, benchmark_name="Sim", acq=acq, num_obj=2, q=1,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=100,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

bo_models = bo_result.try_get_final_models()

# %%
from trieste.utils.post_recommender import pareto_front_on_mean_based_on_model

# %%
from trieste.space import Box
recx = pareto_front_on_mean_based_on_model(Box([-2.0] * 2, [2.0] * 2), bo_models, min_feasibility_probability=0.95, data=None)

# %%
from trieste.objectives.multi_objectives import CVLMOP2
pb = CVLMOP2().objective()
resf = pb(recx)

from matplotlib import pyplot as plt
plt.scatter(resf[:, 0], resf[:, 1])

# %% [markdown]
# ---------

# %% [markdown]
# BFPFES-ILB

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 50
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="Sim", acq=acq, num_obj=2, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# MESMOC
#

# %%
acq = MESMOC # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'MESMOC'
file_info_prefix = ""
total_iter = 20
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}}
kwargs_for_optimize={"inspect_input_contour": True, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="CVLMOP2", acq=acq, num_obj=2, q=1,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='MESMOC')

# %% [markdown]
# Plot of the result

# %%
# %matplotlib notebook

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=100
)
plt.title('VLMOP2 FPFES-ILB MC q=3 PF_MC=3, DOE=100')
plt.show()

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=100
)
plt.title('VLMOP2 FPFES-ILB MC q=1 DOE=100')
plt.show()

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=100
)
plt.title('VLMOP2 FPFES q=1 DOE=100')
plt.show()

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=100
)
plt.title('VLMOP2 FPFES-IBO q=1 DOE=100')
plt.show()

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=100
)
plt.title('VLMOP2 FPFES-ILB q=1')
plt.show()

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=10
)
plt.title('VLMOP2 MC FPFES-ILB q=1')
plt.show()

# %% [markdown]
# ### Constraint BraninCurrin

# %% [markdown]
# This has some Cholesky decomposition issue

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import branincurrin, evaluate_slack_true
import tensorflow as tf
# constraint version
res, x = moo_optimize_pymoo(branincurrin, 2, 2, bounds = (tf.constant([0.0, 0]), tf.constant([1, 1])),
                   popsize=100, cons = evaluate_slack_true, cons_num=1,
                         return_pf_x=True, num_generation=1000)

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(branincurrin, 2, 2, bounds = (tf.constant([0.0, 0]), tf.constant([1, 1])),
                   popsize=50, return_pf_x=True, num_generation=500)


# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
np.savetxt('ConsBraninCurrin_PF.txt', res)

# %%
from trieste.acquisition.function.multi_objective import Pareto, get_reference_point

# %%
get_reference_point(res)

# %%
Pareto(res).hypervolume_indicator(tf.constant([72.0, 12.0], dtype=tf.float64))

# %% [markdown]
# Optimization Code

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 20
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}
# kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
#     {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3, kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="ConstraintBraninCurrin", acq=acq, num_obj=2, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=False, kwargs_for_optimize={}, acq_name='PFES-ILB')
                              # CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# Plot of the result

# %%
# %matplotlib notebook

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=10
)
plt.show()

# %% [markdown]
# ### Constr-Ex

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import Constr_Ex
import tensorflow as tf
# constraint version
res, x = moo_optimize_pymoo(Constr_Ex().objective, 2, 2, bounds = (tf.constant([0, 0]), tf.constant([1, 1])),
                            popsize=100, cons = Constr_Ex().constraint(), cons_num=1,
                            return_pf_x=True, num_generation=1000)

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(Constr_Ex().objective, 2, 2, bounds = (tf.constant([0.0, 0]), tf.constant([1, 1])),
                                  popsize=50, return_pf_x=True, num_generation=500)


# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
# plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
from trieste.acquisition.function.multi_objective import Pareto, get_reference_point

# %%
get_reference_point(res)

# %%
Pareto(res).hypervolume_indicator(tf.constant([1.1, 10.0], dtype=tf.float64))

# %% [markdown]
# Optimization Code

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 50
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}
# kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
#     {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3, kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="Constr_Ex", acq=acq, num_obj=2, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# Plot of the result

# %%
# %matplotlib notebook

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=10
)
plt.show()

# %% [markdown]
# ### Osy

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 50
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 1, "kwargs_for_pf_sampler":
#     {"popsize": 50, "num_moo_iter": 500}, "epsilon": 1e-3}

# %% [markdown]
# There is a Cholesky error on gradient of Cholesky

# %%
tnk_bo_result = serial_benchmarker(process_identifier=0, benchmark_name="Osy", acq=acq, num_obj=2, q=3,
                                   kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                                   CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# ### TNK 

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 50
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 1, "kwargs_for_pf_sampler":
#     {"popsize": 50, "num_moo_iter": 500}, "epsilon": 1e-3}

# %%
tnk_bo_result = serial_benchmarker(process_identifier=0, benchmark_name="TNK", acq=acq, num_obj=2, q=3,
                                   kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                                   CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# Plot of the result

# %%
# %matplotlib notebook

# %%
datasets = tnk_bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=10
)
plt.title('TNK 10 DOE FPFES q=1')
plt.show()

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=100
)
plt.title('TNK 100 DOE FPFES-IBO q=1')
plt.show()

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=100
)
plt.title('TNK 100 DOE MC FPFES-IBO q=1')
plt.show()

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=100
)
plt.title('TNK 100 DOE FPFES q=1')
plt.show()

# %% [markdown]
# ### SRN 

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 30
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 3, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-1}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="SRN", acq=acq, num_obj=2, q=1,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10)

# %% [markdown]
# Plot of the result

# %%
# %matplotlib notebook

# %%
datasets = bo_result.try_get_final_datasets()
data_observations = datasets['OBJECTIVE'].observations
mask_fail = tf.reduce_any(datasets['CONSTRAINT'].observations < 0, -1) 
plot_mobo_points_in_obj_space(
    data_observations, mask_fail=mask_fail, num_init=10
)
plt.show()

# %% [markdown]
# ### TwoBarTruss

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import two_bar_truss_obj, two_bar_truss_con
import tensorflow as tf
# constraint version
res, x = moo_optimize_pymoo(two_bar_truss_obj, 3, 2, bounds = (tf.constant([0.0, 0.0, 0.0]), 
                                                                    tf.constant([1, 1, 1])),
                   popsize=100, cons = two_bar_truss_con, cons_num=1,
                         return_pf_x=True, num_generation=1000)

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(two_bar_truss_obj, 3, 2, bounds = (tf.constant([0.0, 0.0, 0.0]), 
                                                                          tf.constant([1, 1, 1])),
                   popsize=50, return_pf_x=True, num_generation=500)

# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %% [markdown]
# -----------

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 20
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 5, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 1, "kwargs_for_pf_sampler":
#     {"popsize": 50, "num_moo_iter": 500}, "epsilon": 1e-3}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="TwoBarTruss", acq=acq, num_obj=2, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# ### Welded Beam

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import welded_beam_design_obj, welded_beam_design_con
import tensorflow as tf
# constraint version
res, x = moo_optimize_pymoo(welded_beam_design_obj, 4, 2, bounds = (tf.constant([0.0, 0.0, 0.0, 0.0]), 
                                                                    tf.constant([1, 1, 1, 1])),
                   popsize=100, cons = welded_beam_design_con, cons_num=1,
                         return_pf_x=True, num_generation=1000)

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(welded_beam_design_obj, 4, 2, bounds = (tf.constant([0.0, 0.0, 0.0, 0.0]), 
                                                                          tf.constant([1, 1, 1, 1])),
                   popsize=50, return_pf_x=True, num_generation=500)


# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
# plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %% [markdown]
# -----

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 20
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 5, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 1, "kwargs_for_pf_sampler":
#     {"popsize": 50, "num_moo_iter": 500}, "epsilon": 1e-3}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="WeldedBeamDesign", acq=acq, num_obj=2, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %% [markdown]
# ### Disc Brake Design

# %%
from trieste.acquisition.multi_objective.mo_utils import moo_optimize_pymoo
from trieste.objectives.multi_objectives import disc_brake_objective, disc_brake_constraint
import tensorflow as tf
# constraint version
res, x = moo_optimize_pymoo(disc_brake_objective, 4, 2, bounds = (tf.constant([0.0, 0.0, 0.0, 0.0]), 
                                                                    tf.constant([1, 1, 1, 1])),
                   popsize=100, cons = disc_brake_constraint, cons_num=1,
                         return_pf_x=True, num_generation=1000)

# None constraint version
uc_res, uc_x = moo_optimize_pymoo(disc_brake_objective, 4, 2, bounds = (tf.constant([0.0, 0.0, 0.0, 0.0]), 
                                                                          tf.constant([1, 1, 1, 1])),
                   popsize=50, return_pf_x=True, num_generation=500)


# %%
# %matplotlib notebook

# %%
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(res[:, 0], res[:, 1], label='constraint Pareto Frontier')
plt.scatter(uc_res[:, 0], uc_res[:, 1], label='unconstraint Pareto Frontier')
plt.legend()
plt.show()

# %%
np.savetxt('DiscBrake_PF.txt', res)

# %%
from trieste.acquisition.function.multi_objective import Pareto
Pareto(res).hypervolume_indicator(tf.constant([3, 20], dtype=tf.float64))

# %% [markdown]
# ----------------

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 20
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 5, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 1, "kwargs_for_pf_sampler":
#     {"popsize": 50, "num_moo_iter": 500}, "epsilon": 1e-3}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="DiscBrake", acq=acq, num_obj=2, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=10,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %%
import tensorflow as tf
tf.__version__


# %% [markdown]
# ### EE 6

# %%
acq = BatchFeasibleParetoFrontierEntropySearchInformationLowerBound # BatchFeasibleParetoFrontierEntropySearchInformationLowerBound
acq_name = 'PFES_IBO'
file_info_prefix = ""
total_iter = 50
kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 5, "kwargs_for_pf_sampler":
    {"popsize": 50, "num_moo_iter": 300}, "epsilon": 1e-3}
kwargs_for_optimize={"inspect_input_contour": False, "inspect_inferred_pareto": True, "inspect_in_obj_space": True}

# kwargs_for_acq = {"objective_tag": "OBJECTIVE", "constraint_tag": "CONSTRAINT", "pf_mc_sample_num": 1, "kwargs_for_pf_sampler":
#     {"popsize": 50, "num_moo_iter": 500}, "epsilon": 1e-3}

# %%
bo_result = serial_benchmarker(process_identifier=0, benchmark_name="EE6", acq=acq, num_obj=2, q=3,
                               kwargs_for_acq=kwargs_for_acq, is_return=True, total_iter=total_iter, doe_num=9,
                               CustomBayesianOptimizer=True, kwargs_for_optimize=kwargs_for_optimize, acq_name='PFES-ILB')

# %%
data = bo_result.try_get_final_datasets()

# %%
# %matplotlib notebook

# %%
mask_fail = tf.squeeze(data['CONSTRAINT'].observations < 0)
plot_mobo_points_in_obj_space(data['OBJECTIVE'].observations, mask_fail=mask_fail)

# %%
failure_xs = data['OBJECTIVE'].query_points[data['OBJECTIVE'].observations[:, 0] == 2]

# %%
ee_6_bounds = [[6.5, 2.0, 0.2, 8.0], [8.0, 3.5, 0.4, 11]]

# %%
failure_unscaled_xs = failure_xs * (tf.constant(ee_6_bounds[-1], dtype=failure_xs.dtype) - tf.constant(ee_6_bounds[0], dtype=failure_xs.dtype)) + \
                 tf.constant(ee_6_bounds[0], dtype=failure_xs.dtype)

# %%
failure_unscaled_xs

# %%
tf.reduce_sum(tf.cast(data['OBJECTIVE'].observations[:, 0] == 2, dtype=tf.float64))/158
