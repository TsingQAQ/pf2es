# -*- coding: utf-8 -*-
# %% [markdown]
# # Multi-objective optimization with Expected HyperVolume Improvement

# %%
import math

import gpflow
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
    DiscBrakeDesign, EE6, FourBarTruss, WeldedBeamDesign, DTLZ2
from trieste.objectives.multi_objectives import SinLinearForrester, BraninCurrin, GMMForrester, CBraninCurrin
from trieste.acquisition.function import Fantasizer
from os.path import dirname, join
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.acquisition.utils import split_acquisition_function_calls
import tensorflow_probability as tfp

np.random.seed(1727)
tf.random.set_seed(1727)
OBJECTIVE = "OBJECTIVE"

# %%
vlmop2 = VLMOP2().objective()
vlmop2_observer = trieste.objectives.utils.mk_observer(vlmop2, key=OBJECTIVE)
vlmop2_mins = [-2, -2]
vlmop2_maxs = [2, 2]
vlmop2_search_space = Box(vlmop2_mins, vlmop2_maxs)
vlmop2_num_objective = 2

sinlinforrester = SinLinearForrester().objective()
sinlinforrester_observer = trieste.objectives.utils.mk_observer(sinlinforrester, key=OBJECTIVE)
sinlinforrester_mins = [0.0]
sinlinforrester_maxs = [1.0]
sinlinforrester_search_space = Box(sinlinforrester_mins, sinlinforrester_maxs)
sinlinforrester_num_objective = 2

gmmforrester = GMMForrester().objective()
gmmforrester_observer = trieste.objectives.utils.mk_observer(gmmforrester, key=OBJECTIVE)
gmmforrester_mins = [0.0]
gmmforrester_maxs = [1.0]
gmmforrester_search_space = Box(gmmforrester_mins, gmmforrester_maxs)
gmmforrester_num_objective = 2

branincurrin = BraninCurrin().objective()
branincurrin_cons = CBraninCurrin().constraint()
branincurrin_observer = trieste.objectives.utils.mk_observer(branincurrin, key=OBJECTIVE)
branincurrin_mins = [0.0, 0.0]
branincurrin_maxs = [1.0, 1.0]
branincurrin_search_space = Box(branincurrin_mins, branincurrin_maxs)
branincurrin_num_objective = 2

fourbartruss = FourBarTruss().objective()
fourbartruss_observer = trieste.objectives.utils.mk_observer(fourbartruss, key=OBJECTIVE)
fourbartruss_mins = [0.0] * 4
fourbartruss_maxs = [1.0] * 4
fourbartruss_search_space = Box(fourbartruss_mins, fourbartruss_maxs)
fourbartruss_num_objective = 2

dtlz2 = DTLZ2(input_dim=8, num_objective=2).objective()
dtlz2_observer = trieste.objectives.utils.mk_observer(dtlz2, key=OBJECTIVE)
dtlz2_mins = [0.0] * 8
dtlz2_maxs = [1.0] * 8
dtlz2_search_space = Box(dtlz2_mins, dtlz2_maxs)
dtlz2_num_objective = 2


def cbranincurrin_observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, branincurrin(query_points)),
        CONSTRAINT: Dataset(query_points, branincurrin_cons(query_points)),
    }


cbranincurrin_num_constraint = 1

osy = Osy
osy_obj = Osy().objective()
osy_cons = Osy().constraint()
osy_search_space = Box(*Osy.bounds)
osy_num_objective = 2
osy_num_con = 6


def osy_observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, osy_obj(query_points)),
        CONSTRAINT: Dataset(query_points, osy_cons(query_points)),
    }


tnk = TNK
tnk_obj = TNK().objective()
tnk_cons = TNK().constraint()
tnk_search_space = Box(*TNK.bounds)
tnk_num_objective = 2
tnk_num_con = 2


def tnk_observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, tnk_obj(query_points)),
        CONSTRAINT: Dataset(query_points, tnk_cons(query_points)),
    }


twobartruss = TwoBarTruss
twobartruss_obj = TwoBarTruss().objective()
twobartruss_cons = TwoBarTruss().constraint()
twobartruss_search_space = Box(*TwoBarTruss.bounds)
twobartruss_num_objective = 2
twobartruss_num_con = 1


def twobartruss_observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, twobartruss_obj(query_points)),
        CONSTRAINT: Dataset(query_points, twobartruss_cons(query_points)),
    }


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


diskbrakedesign = DiscBrakeDesign
diskbrakedesign_obj = DiscBrakeDesign().objective()
diskbrakedesign_cons = DiscBrakeDesign().constraint()
diskbrakedesign_search_space = Box(*DiscBrakeDesign.bounds)
diskbrakedesign_num_objective = 2
diskbrakedesign_num_con = 4

ee6 = EE6
ee6_joint_obj_con = EE6(input_dim=4).joint_objective_con()
ee6_search_space = Box(*EE6(input_dim=4).restricted_bounds)
ee6_num_objective = 2
ee6_num_con = 1


def ee6_observer(query_points):
    objs, cons = ee6_joint_obj_con(query_points)
    return {
        OBJECTIVE: Dataset(query_points, objs),
        CONSTRAINT: Dataset(query_points, cons),
    }


def diskbrakedesign_observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, diskbrakedesign_obj(query_points)),
        CONSTRAINT: Dataset(query_points, diskbrakedesign_cons(query_points)),
    }


c2dtlz2 = C2DTLZ2
c2dtlz2_obj = C2DTLZ2(input_dim=6, num_objective=2).objective()
c2dtlz2_cons = C2DTLZ2(input_dim=6, num_objective=2).constraint()
c2dtlz2_search_space = Box(*C2DTLZ2(input_dim=6, num_objective=2).bounds)
c2dtlz2_num_objective = 2
c2dtlz2_num_con = 1


def c2dtlz2_observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, c2dtlz2_obj(query_points)),
        CONSTRAINT: Dataset(query_points, c2dtlz2_cons(query_points)),
    }


zdt1_5i_2o = ZDT1(input_dim=5).objective()
zdt1_5i_2o_observer = trieste.objectives.utils.mk_observer(zdt1_5i_2o, key=OBJECTIVE)
zdt1_5i_2o_mins = [0.0] * 5
zdt1_5i_2o_maxs = [1.0] * 5
zdt1_5i_2o_search_space = Box(zdt1_5i_2o_mins, zdt1_5i_2o_maxs)
zdt1_5i_2o_num_objective = 2

zdt2_5i_2o = ZDT2(input_dim=5).objective()
zdt2_5i_2o_observer = trieste.objectives.utils.mk_observer(zdt2_5i_2o, key=OBJECTIVE)
zdt2_5i_2o_mins = [0.0] * 5
zdt2_5i_2o_maxs = [1.0] * 5
zdt2_5i_2o_search_space = Box(zdt2_5i_2o_mins, zdt2_5i_2o_maxs)
zdt2_5i_2o_num_objective = 2

zdt3_5i_2o = ZDT3(input_dim=5).objective()
zdt3_5i_2o_observer = trieste.objectives.utils.mk_observer(zdt3_5i_2o, key=OBJECTIVE)
zdt3_5i_2o_mins = [0.0] * 5
zdt3_5i_2o_maxs = [1.0] * 5
zdt3_5i_2o_search_space = Box(zdt3_5i_2o_mins, zdt3_5i_2o_maxs)
zdt3_5i_2o_num_objective = 2


class cvlmop2:
    threshold = 0.75

    @staticmethod
    def objective(input_data):
        return vlmop2(input_data)

    @staticmethod
    def constraint(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return cvlmop2.threshold - z[:, None]


OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"


def cvlmop2_observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, cvlmop2.objective(query_points)),
        CONSTRAINT: Dataset(query_points, cvlmop2.constraint(query_points)),
    }


cvlmop2_search_space = Box(vlmop2_mins, vlmop2_maxs)


def build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
        data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainableHasTrajectoryAndPredictJointReparamModelStack:
    gprs = []
    # likelihood_variance_list = [0.5, 1e-7]
    # likelihood_variance_list = [1e-7, 1e-7]
    for idx in range(num_output):
        single_obj_data = Dataset(
            data.query_points, tf.gather(data.observations, [idx], axis=1)
        )
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-7,
                        trainable_likelihood=False)
        # add prior
        # gpr.likelihood.variance.prior = tfp.distributions.LogNormal(
        #     tf.math.log(tf.constant(1e-2, dtype=tf.float64)), tf.constant(100.0, dtype=tf.float64))
        # gpr.likelihood.variance.prior = tfp.distributions.Gamma(tf.constant(1., dtype=tf.float64),
        #                                                         tf.constant(1., dtype=tf.float64))
        # TODO: NOTE WE CHANGED HERE!!!
        # gprs.append((StandardizedGaussianProcessRegression(gpr, use_decoupled_sampler=False), 1))
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=False), 1))

    return TrainableHasTrajectoryAndPredictJointReparamModelStack(*gprs)
    # return TrainablePredictJointReparamModelStack(*gprs)


acq = "PF2ES"  # PF2ES-Fantasize
exp = 'UNCONSTRAINED'  # 'PF2ES_UNCONSTRAINED'
# pb = 'sinlinforrester'
pb = 'sinlinforrester'  # 'TNK' # 'branincurrin'# 'vlmop2'# 'c-branincurrin' # 'TNK' # 'Osy'#   # '' 'zdt2_5i_2o'
num_initial_points = 3
q = 2
num_steps = 100
if exp == 'UNCONSTRAINED':
    # %%
    if pb == 'vlmop2':
        initial_query_points = vlmop2_search_space.sample(num_initial_points)
        initial_data = vlmop2_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, vlmop2_num_objective, vlmop2_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\vlmop2'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'vlmop2')
        search_space = vlmop2_search_space
        observer = vlmop2_observer
    elif pb == 'gmmforrester':
        initial_query_points = gmmforrester_search_space.sample(num_initial_points)
        initial_data = gmmforrester_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, gmmforrester_num_objective, gmmforrester_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\gmmforrester'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'gmmforrester')
        search_space = gmmforrester_search_space
        observer = gmmforrester_observer
    elif pb == 'sinlinforrester':
        initial_query_points = sinlinforrester_search_space.sample_halton(num_initial_points)
        initial_data = sinlinforrester_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, sinlinforrester_num_objective, sinlinforrester_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\sinlinearforrester'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'sinlinearforrester')
        search_space = sinlinforrester_search_space
        observer = sinlinforrester_observer
    elif pb == 'branincurrin':
        initial_query_points = branincurrin_search_space.sample(num_initial_points)
        initial_data = branincurrin_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, branincurrin_num_objective, branincurrin_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\branincurrin'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'branincurrin')
        search_space = branincurrin_search_space
        observer = branincurrin_observer
    elif pb == 'zdt1_5i_2o':
        initial_query_points = zdt1_5i_2o_search_space.sample(num_initial_points)
        initial_data = zdt1_5i_2o_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, zdt1_5i_2o_num_objective, zdt1_5i_2o_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\zdt1_5i_2o'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'zdt1_5i_2o')
        search_space = zdt1_5i_2o_search_space
        observer = zdt1_5i_2o_observer
    elif pb == 'zdt2_5i_2o':
        initial_query_points = zdt2_5i_2o_search_space.sample(num_initial_points)
        initial_data = zdt2_5i_2o_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, zdt2_5i_2o_num_objective, zdt2_5i_2o_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\zdt2_5i_2o'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'zdt2_5i_2o')
        search_space = zdt2_5i_2o_search_space
        observer = zdt2_5i_2o_observer
    elif pb == 'zdt3_5i_2o':
        initial_query_points = zdt3_5i_2o_search_space.sample(num_initial_points)
        initial_data = zdt3_5i_2o_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, zdt3_5i_2o_num_objective, zdt3_5i_2o_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\zdt3_5i_2o'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'zdt3_5i_2o')
        search_space = zdt3_5i_2o_search_space
        observer = zdt3_5i_2o_observer
    elif pb == 'fourbartruss':
        initial_query_points = fourbartruss_search_space.sample(num_initial_points)
        initial_data = fourbartruss_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, fourbartruss_num_objective, fourbartruss_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\zdt3_5i_2o'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'FourBarTruss')
        search_space = fourbartruss_search_space
        observer = fourbartruss_observer
    elif pb == 'dtlz2':
        initial_query_points = dtlz2_search_space.sample(num_initial_points)
        initial_data = dtlz2_observer(initial_query_points)[OBJECTIVE]
        model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data, dtlz2_num_objective, dtlz2_search_space
        )
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\zdt3_5i_2o'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'DTLZ2')
        search_space = dtlz2_search_space
        observer = dtlz2_observer
    else:
        raise NotImplementedError

    # %%
    if acq == 'PF2ES':
        builder = PF2ES(search_space, moo_solver='nsga2', sample_pf_num=5, remove_augmenting_region=False,
                        population_size_for_approx_pf_searching=50, pareto_epsilon=0.04, remove_log=False,
                        batch_mc_sample_size=128, discretize_input_sample_size=100000, averaging_partition=False,
                        mean_field_pf_approx=False, use_dbscan_for_conservative_epsilon=False, parallel_sampling=True,
                        qMC=True)
    elif acq == 'PF2ES-Fantasize':
        builder = Fantasizer(PF2ES(search_space, moo_solver='nsga2', sample_pf_num=5, remove_augmenting_region=False,
                                   population_size_for_approx_pf_searching=50, pareto_epsilon=0.05, remove_log=False))
    elif acq == 'MESMO':
        builder = MESMO(vlmop2_search_space, moo_solver='monte_carlo', sample_pf_num=5)
    elif acq == 'EHVI':
        builder = ExpectedHypervolumeImprovement()
    elif acq == 'PFES':
        builder = PFES(search_space, moo_solver='nsga2', sample_pf_num=5,
                       population_size_for_approx_pf_searching=50)
    else:
        raise NotImplementedError
    rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=builder, num_query_points=q)

    # %%
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule,
                         inspect_acq_contour=True, inspect_inferred_pareto=True, inspect_in_obj_space=True,
                         auxiliary_file_output_path=aux_file_path,
                         kwargs_for_input_space_plot={'plot_acq_pf_sample': True,
                                                      'kwargs_for_pf_sample': {'obj_num': 2,
                                                                               'sample_pf_num': 5,
                                                                               'search_space': search_space,
                                                                               'cons_num': 0}},
                         kwargs_for_obj_space_plot={'plot_acq_pf_samples': True,
                                                    'kwargs_for_pf_sample': {'obj_num': 2,
                                                                             'sample_pf_num': 5,
                                                                             'search_space': search_space,
                                                                             'cons_num': 0},
                                                    },
                         kwargs_for_inferred_pareto={'plot_rec_pf_actual_obx': True},
                         generate_recommendation=True)

    # %%
    dataset = result.try_get_final_dataset()
    data_query_points = dataset.query_points
    data_observations = dataset.observations

    # %%
    plot_mobo_points_in_obj_space(data_observations, num_init=num_initial_points)
    plt.show()

elif exp == 'CONSTRAINED':
    if pb == 'c-branincurrin':
        initial_query_points = branincurrin_search_space.sample(num_initial_points)
        initial_data = cbranincurrin_observer(initial_query_points)
        obj_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[OBJECTIVE], branincurrin_num_objective, branincurrin_search_space
        )
        con_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[CONSTRAINT], cbranincurrin_num_constraint, branincurrin_search_space
        )
        models = {OBJECTIVE: obj_model, CONSTRAINT: con_model}
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\cbranincurrin'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'cbranincurrin')
        search_space = branincurrin_search_space
        observer = cbranincurrin_observer
    elif pb == 'Osy':
        initial_query_points = osy_search_space.sample(num_initial_points)
        initial_data = osy_observer(initial_query_points)
        obj_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[OBJECTIVE], osy_num_objective, osy_search_space
        )
        con_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[CONSTRAINT], osy_num_con, osy_search_space
        )
        models = {OBJECTIVE: obj_model, CONSTRAINT: con_model}
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\cbranincurrin'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'Osy')
        search_space = osy_search_space
        observer = osy_observer
    elif pb == 'TNK':
        initial_query_points = tnk_search_space.sample(num_initial_points)
        initial_data = tnk_observer(initial_query_points)
        obj_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[OBJECTIVE], tnk_num_objective, tnk_search_space
        )
        con_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[CONSTRAINT], tnk_num_con, tnk_search_space
        )
        models = {OBJECTIVE: obj_model, CONSTRAINT: con_model}
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\cbranincurrin'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'TNK')
        search_space = tnk_search_space
        observer = tnk_observer
    elif pb == 'TwoBarTruss':
        initial_query_points = twobartruss_search_space.sample(num_initial_points)
        initial_data = twobartruss_observer(initial_query_points)
        obj_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[OBJECTIVE], twobartruss_num_objective, twobartruss_search_space
        )
        con_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[CONSTRAINT], twobartruss_num_con, twobartruss_search_space
        )
        models = {OBJECTIVE: obj_model, CONSTRAINT: con_model}
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\cbranincurrin'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'TwoBarTruss')
        search_space = twobartruss_search_space
        observer = twobartruss_observer
    elif pb == 'SRN':
        initial_query_points = srn_search_space.sample(num_initial_points)
        initial_data = srn_observer(initial_query_points)
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
    elif pb == 'DiscBrakeDesign':
        initial_query_points = diskbrakedesign_search_space.sample(num_initial_points)
        initial_data = diskbrakedesign_observer(initial_query_points)
        obj_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[OBJECTIVE], diskbrakedesign_num_objective, diskbrakedesign_search_space
        )
        con_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[CONSTRAINT], diskbrakedesign_num_con, diskbrakedesign_search_space
        )
        models = {OBJECTIVE: obj_model, CONSTRAINT: con_model}
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\cbranincurrin'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'DiscBrakeDesign')
        search_space = diskbrakedesign_search_space
        observer = diskbrakedesign_observer
    elif pb == 'C2DTLZ2':
        initial_query_points = c2dtlz2_search_space.sample(num_initial_points)
        initial_data = c2dtlz2_observer(initial_query_points)
        obj_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[OBJECTIVE], c2dtlz2_num_objective, c2dtlz2_search_space
        )
        con_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[CONSTRAINT], c2dtlz2_num_con, c2dtlz2_search_space
        )
        models = {OBJECTIVE: obj_model, CONSTRAINT: con_model}
        # aux_file_path = r'C:\Users\Administrator\Desktop\trieste_pfes\docs\exp\acq_iter_check\cbranincurrin'
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'C2DTLZ2')
        search_space = c2dtlz2_search_space
        observer = c2dtlz2_observer
    elif pb == 'ee6':
        initial_query_points = ee6_search_space.sample(num_initial_points)
        initial_data = ee6_observer(initial_query_points)
        obj_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[OBJECTIVE], ee6_num_objective, ee6_search_space
        )
        con_model = build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[CONSTRAINT], ee6_num_con, ee6_search_space
        )
        models = {OBJECTIVE: obj_model, CONSTRAINT: con_model}
        aux_file_path = join(dirname(dirname(__file__)), 'exp', 'acq_iter_check', 'EE6')
        search_space = ee6_search_space
        observer = ee6_observer
    else:
        raise NotImplementedError
    if acq == 'PF2ES':
        builder = PF2ES(search_space, moo_solver='nsga2', objective_tag=OBJECTIVE, constraint_tag=CONSTRAINT,
                        sample_pf_num=5, remove_augmenting_region=False,
                        population_size_for_approx_pf_searching=50, pareto_epsilon=0.04, remove_log=False,
                        batch_mc_sample_size=128, use_dbscan_for_conservative_epsilon=False, parallel_sampling=True,
                        temperature_tau=1E-3, qMC=False)
    elif acq == 'PF2ES-Fantasize':
        builder = Fantasizer(PF2ES(search_space, moo_solver='nsga2', sample_pf_num=5, remove_augmenting_region=False,
                                   population_size_for_approx_pf_searching=50, pareto_epsilon=0.04, remove_log=False))
    elif acq == 'MESMOC':
        builder = MESMOC(search_space, moo_solver='nsga2', objective_tag=OBJECTIVE, constraint_tag=CONSTRAINT,
                         sample_pf_num=5, population_size_for_approx_pf_searching=50, )
    elif acq == 'CEHVI':
        builder = CEHVI(objective_tag=OBJECTIVE, constraint_tag=CONSTRAINT)
    elif acq == 'qECHVI':
        builder = BatchMonteCarloConstrainedExpectedHypervolumeImprovement(
            objective_tag=OBJECTIVE, constraint_tag=CONSTRAINT, sample_size=32)
    else:
        raise NotImplementedError
    rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=builder, num_query_points=q,
                                                                    optimizer=split_acquisition_function_calls(
                                                                        automatic_optimizer_selector, split_size=5000))

    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule,
                         inspect_acq_contour=False, inspect_inferred_pareto=True, inspect_in_obj_space=True,
                         auxiliary_file_output_path=aux_file_path,
                         kwargs_for_input_space_plot={'plot_acq_pf_sample': True,
                                                      'plot_constraint_contour': False,
                                                      'kwargs_for_pf_sample': {'obj_num': 2,
                                                                               'sample_pf_num': 5,
                                                                               'search_space': search_space,
                                                                               'cons_num': 6}},
                         kwargs_for_obj_space_plot={'plot_acq_pf_samples': True,
                                                    'kwargs_for_pf_sample': {'obj_num': 2,
                                                                             'sample_pf_num': 5,
                                                                             'search_space': search_space,
                                                                             'cons_num': 6}
                                                    },

                         kwargs_for_inferred_pareto={'plot_rec_pf_actual_obx': True},
                         generate_recommendation=True)

    # %%
    dataset = result.try_get_final_dataset()
    data_query_points = dataset.query_points
    data_observations = dataset.observations

    # %%
    plot_mobo_points_in_obj_space(data_observations, num_init=num_initial_points)
    plt.show()

    # %%
    objective_dataset = result.final_result.unwrap().datasets[OBJECTIVE]
    constraint_dataset = result.final_result.unwrap().datasets[CONSTRAINT]
    data_query_points = objective_dataset.query_points
    data_observations = objective_dataset.observations

    mask_fail = constraint_dataset.observations.numpy() < 0
    plot_mobo_points_in_obj_space(
        data_observations, num_init=num_initial_points, mask_fail=mask_fail[:, 0]
    )
    plt.show()


else:
    raise NotImplementedError
