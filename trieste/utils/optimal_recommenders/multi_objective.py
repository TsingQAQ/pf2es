"""
Different post-processing method for recommending optimal candidate points in input
"""
from __future__ import annotations

from typing import Mapping, Optional

import tensorflow as tf

from trieste.acquisition.multi_objective.dominance import non_dominated
from trieste.acquisition.multi_objective.utils import (
    inference_pareto_fronts_from_gp_mean,
    sample_pareto_fronts_from_parametric_gp_posterior,
)
from trieste.data import Dataset
from trieste.models.interfaces import HasTrajectorySamplerModelStack, ModelStack
from trieste.observer import CONSTRAINT, OBJECTIVE
from trieste.space import Box
from trieste.types import TensorType

IN_SAMPLE = "In_Sample"
OUT_OF_SAMPLE = "Out_Of_Sample"
MODEL_BELIEVE = "Model_Believe"


def recommend_pareto_front_from_existing_data(
    models: Mapping[str, ModelStack],
    datas: Mapping[str, Dataset],
    min_feasibility_probability: float = 0.5,
    return_direct_obs=False,
) -> [TensorType, Optional[TensorType]]:
    """
    get pareto front from the provided data, this implementation assumes noise free scenario
    for constraint multi-objective optimization, it assumes >0 is feasible, also referred to as 'In Sample' strategy.
    :param models
    :param datas
    :param min_feasibility_probability
    :param constraint_tag
    :param return_direct_obs: whether to return observations
    """
    if CONSTRAINT not in models.keys():
        _, dominance = non_dominated(datas[OBJECTIVE].observations)
        return tf.gather_nd(datas[OBJECTIVE].query_points, tf.where(tf.equal(dominance, 0)))
    # In case there are constraints
    else:
        cons = datas[CONSTRAINT].observations
        feasible_index = tf.where(tf.reduce_all(cons >= 0.0, axis=-1))
        # calculate the dominance through feasible query points
        feasible_obs = tf.gather_nd(datas[OBJECTIVE].observations, feasible_index)
        nd_feasible_obs, _ = non_dominated(feasible_obs)
        # get index of non-dominated feasible obs
        # [N, D] & [M, D] -> [N, M, D] -> [M, D]
        fea_nd_idx = tf.reduce_any(
            tf.reduce_all(
                tf.equal(datas[OBJECTIVE].observations, tf.expand_dims(nd_feasible_obs, -2)), -1
            ),
            0,
        )

        pf_x = tf.gather_nd(  # 1. Return Pareto Optimal Input 2. Return Pareto Optimal Output
            datas[OBJECTIVE].query_points, tf.where(fea_nd_idx)
        )
        pf_x = tf.expand_dims(pf_x, axis=-2) if tf.rank(pf_x) == 1 else pf_x
        if return_direct_obs is True:
            pf_y = tf.gather_nd(datas[OBJECTIVE].observations, tf.where(fea_nd_idx))
            pf_y = tf.expand_dims(pf_y, axis=-2) if tf.rank(pf_y) == 1 else pf_y
            return pf_x, pf_y
        else:  # Only return Pareto Optimal Input
            return pf_x


def recommend_pareto_front_from_model_prediction(
    models: Mapping[str, ModelStack],
    data: TensorType,
    search_space: Box,
    kwargs_for_inferred_pareto={},
    min_feasibility_probability: float = 0.5,
    discrete_input: bool = False,
    hard_constraint_threshold_perc = 1e-3,
) -> Optional[TensorType]:
    """
    extract Pareto front from posterior mean of GP model.
    for constraint multi-objective optimization, it assumes >0 is feasible,
    also referred to as 'In Sample' strategy.

    :param models
    :param data
    :param search_space
    :param kwargs_for_inferred_pareto
    :param min_feasibility_probability
    :param hard_constraint_threshold_perc: force C > epsilon，
        where epsilon = (constraint range) * hard_constraint_threshold_pec

    The following is from PESMOC paper:
    If constraint has present: we consider that a constraint is satisfied at an input
    location x if the probability that the constraint is larger than zero is above 1 − δ
    where δ is 0.05. That is, p(c j (x ≥ 0) ≥ 1 − δ. When no feasible solution is found,
    we simply return the points that are most likely to be feasible by iteratively
    increasing δ in 0.05 units.
    :param discrete_input:
    """
    # perform MOO on GP posterior mean
    # for CMOO, need PoF > min_feasibility_probability
    while min_feasibility_probability >= 0:
        res, resx = inference_pareto_fronts_from_gp_mean(
            models[OBJECTIVE],
            search_space,
            popsize=kwargs_for_inferred_pareto.get("popsize", 50),
            cons_models=models[CONSTRAINT] if CONSTRAINT in models.keys() else None,
            num_moo_iter=kwargs_for_inferred_pareto.get("num_moo_iter", 500),
            min_feasibility_probability=min_feasibility_probability,
            monte_carlo_input=discrete_input,
            constraint_enforce_percentage = hard_constraint_threshold_perc,
        )
        if resx is None:
            min_feasibility_probability -= 0.05
            print(
                f"no satisfied constraint PF found, "
                f"decrease min_feasible_prob to: {min_feasibility_probability}"
            )
            if min_feasibility_probability < 0:
                print(
                    f"no satisfied constraint PF found even decrease "
                    f"min_feasible_prob: {min_feasibility_probability} to be negative!"
                )
                return tf.zeros(shape=(0, search_space.lower.shape[0]), dtype=search_space.lower.dtype)
        else:
            resx = tf.expand_dims(resx, axis=-2) if tf.rank(resx) == 1 else resx
            return resx


def inspecting_pareto_front_distributions_from_model(
    models: Mapping[str, ModelStack],
    obj_num: int,
    sample_pf_num: int,
    search_space: Box,
    cons_num: Optional[int] = 0,
) -> list[TensorType]:
    """
    This is just a wrapper of sample_pareto_fronts_from_parametric_gp_posterior,
    used for inspecting the uncertainty of model believed pareto frontier
    """
    if CONSTRAINT in models.keys():
        assert cons_num != 0
        constraint_model = models[CONSTRAINT]
        assert isinstance(constraint_model, HasTrajectorySamplerModelStack)
        constraint_model.get_trajectories(regenerate=True)
    else:
        assert cons_num == 0
        constraint_model = None
    obj_model = models[OBJECTIVE]
    assert isinstance(obj_model, HasTrajectorySamplerModelStack)
    obj_model.get_trajectories(regenerate=True)
    return sample_pareto_fronts_from_parametric_gp_posterior(
        objective_models=models[OBJECTIVE],
        obj_num=obj_num,
        sample_pf_num=sample_pf_num,
        search_space=search_space,
        cons_num=cons_num,
        constraint_models=constraint_model,
        moo_solver="nsga2",
        return_pf_input=False,
    )
