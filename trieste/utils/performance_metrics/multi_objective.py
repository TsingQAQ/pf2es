from typing import Mapping, Optional

import tensorflow as tf
from gpflow.utilities.model_utils import add_noise_cov
from tensorflow_probability import distributions as tfd

from ...acquisition.multi_objective.pareto import Pareto
from ...models.interfaces import TrainableHasTrajectoryAndPredictJointReparamModelStack
from ...objectives.multi_objectives import MultiObjectiveTestProblem
from ...observer import CONSTRAINT, OBJECTIVE
from ...types import TensorType
from ...types import Callable

"""
Performance metric for multi-objective Bayesian Optimization
In case the recommendation contains infeasible point, we set that feasible point the same as reference point for 
calculating the performance metric
"""

HypervolumeIndicator = "HypervolumeIndicator"
LogHypervolumeDifference = "LogHypervolumeDifference"
AdditiveEpsilonIndicator = "AdditiveEpsilonIndicator"
AverageHausdauffDistance = "AverageHausdauffDistance"
NegLogMarginalParetoFrontier = "NegLogMarginalParetoFrontier"


def strong_infeasible_penalizer(observations: TensorType, constraints: TensorType) -> TensorType:
    """
    This functionality is used if the recommendations in fact contains infeasible observations.
    If the constraint has been violated actually, return empty observations as a strong penalizer:
    used in PESMOC/MESMOC
    """
    infeasible_mask = tf.reduce_any(constraints < tf.zeros(1, dtype=observations.dtype))
    if infeasible_mask:  # return empty observation
        return tf.zeros(shape=(0, observations.shape[-1]), dtype=observations.dtype)
    else:
        return observations


def hypervolume_indicator(
    recommendation_input: TensorType,
    ref_point: TensorType,
    true_func_inst: Callable,
    recommendation_ouput: Optional[TensorType] = None,
    **kwargs,
) -> TensorType:
    """
    Hypervolume Indicator, only used for expensive multi-objective optimization
    Used in noise free setting, using the recommendation_output and ref_point to calculate the indicator

    :param recommendation_input
    :param ref_point
    """
    if tf.equal(tf.size(recommendation_input), 0):
        return tf.zeros(1, dtype=ref_point.dtype)

    if recommendation_ouput is None:
        obj_obs, con_obs = true_func_inst(recommendation_input)

        if tf.reduce_any(con_obs < 0):
            return tf.zeros(1, dtype=ref_point.dtype)
    else:
        obj_obs = recommendation_ouput # This assumes recommendation output is all feasible

    if tf.reduce_any(tf.reduce_all(obj_obs < ref_point, -1)):
        below_mask = tf.reduce_all(obj_obs < ref_point, -1)
        print(f"blow_mask: {below_mask}")
        obs_hv = Pareto(obj_obs[below_mask]).hypervolume_indicator(ref_point)
    else:
        obs_hv = tf.zeros(1, dtype=ref_point.dtype)
    print(f"obs hv: {obs_hv}")
    return obs_hv


def log_hypervolume_difference(
    ref_point: TensorType,
    true_func_inst: Optional[MultiObjectiveTestProblem] = None,
    recommendation_input: Optional[TensorType] = None,
    test_pf: Optional[TensorType] = None,
    reference_pf: Optional[TensorType] = None,
    reference_hv: Optional[TensorType] = None,
) -> TensorType:
    """
    Compute log HV difference: log(reference_hv - observation_hv)
    log HV difference calculate different input situation:

    For observation_hv:
        1. if a test_pf is specified, it will calculate an observation_hv from test_pf.
        2. else, it will try to calculate an observation_hv from recommendation_input, if recommendation_input is empty
        (i.e., no satisfied constraint solution, or no solution dominates ref_point), observation_hv will just be 0.

    For reference_hv:
        1. if specified, use reference_hv (and will ignore reference_pf even it has been specified as well).
        1. if not specified, calculate reference_hv based on ref_point and reference_pf

    if test_pf has been specified, it will calculate absolute value of hv difference

    :param recommendation_input
    :param true_func_inst
    :param ref_point
    :param test_pf
    :param reference_pf
    :param reference_hv if ideal_hv is provided, use it as the ground truth
    """

    def calculate_obs_hv_from_recommendation_input(
        rec_input: TensorType, ref_hv: TensorType
    ) -> TensorType:
        if tf.equal(tf.size(rec_input), 0): # no recommedation input (e.g., no feasible obs)
            return tf.zeros(1, dtype=ref_hv.dtype)

        obj_obs = true_func_inst.objective()(rec_input)
        # obj_obs = tf.expand_dims(obj_obs, axis=-2) if tf.rank(obj_obs) == 1 else obj_obs
        if hasattr(true_func_inst, "constraint"):
            # print("constraint pb detected")
            obj_obs = strong_infeasible_penalizer(obj_obs, true_func_inst.constraint()(rec_input))
        if tf.rank(obj_obs) == 1:
            obj_obs = tf.expand_dims(obj_obs, axis=-2)
        # screen and remain observations below ref point
        if tf.reduce_any(tf.reduce_all(obj_obs < ref_point, axis=-1)):
            below_mask = tf.reduce_all(obj_obs < ref_point, axis=-1)
            if tf.rank(below_mask) == 0:
                below_mask = tf.expand_dims(below_mask, -1)
            _obs_hv = Pareto(obj_obs[below_mask]).hypervolume_indicator(ref_point)
        else:  # no observations below reference point
            _obs_hv = tf.zeros(1, dtype=ref_hv.dtype)
        return _obs_hv

    def calculate_obs_hv_from_test_pf(obj_obs: TensorType) -> TensorType:
        # screen and remain observations below ref point
        if tf.reduce_any(tf.reduce_all(obj_obs < ref_point, axis=-1)):
            below_mask = tf.reduce_all(obj_obs < ref_point, axis=-1)
            if tf.rank(below_mask) == 0:
                below_mask = tf.expand_dims(below_mask, -1)
            _obs_hv = Pareto(obj_obs[below_mask]).hypervolume_indicator(ref_point)
        else:  # no observations below reference point
            _obs_hv = tf.zeros(1, dtype=reference_hv.dtype)
        return _obs_hv

    assert reference_hv is not None or reference_pf is not None, ValueError(
        "At least one of the ideal_hv or reference_pf need to be specified to calculate ref_hv"
    )
    if reference_hv is not None and reference_pf is not None:
        print("Both reference_hv and reference_pf specified, will only use reference_hv!")
    if reference_hv is None:
        reference_hv = Pareto(reference_pf).hypervolume_indicator(ref_point)

    assert (
        tf.logical_and(recommendation_input is not None, test_pf is not None) == False
    ), ValueError("recommendation_input and test_pf cannot be specified simultaneously!")

    if test_pf is not None:
        obs_hv = calculate_obs_hv_from_test_pf(obj_obs=test_pf)
        return tf.abs(reference_hv - obs_hv)
    else:
        obs_hv = calculate_obs_hv_from_recommendation_input(
            rec_input=recommendation_input, ref_hv=reference_hv
        )
        if reference_hv - obs_hv <= 0:
            print("Negative Hypervolume Difference detected!!!")
        return tf.math.log(reference_hv - obs_hv)


# TODO: Not In Use
def relative_hypervolume(
    recommendation_input: TensorType,
    true_func_inst: MultiObjectiveTestProblem,
    ref_point: TensorType,
    reference_pf: [TensorType, None] = None,
    reference_hv: [TensorType, None] = None,
) -> TensorType:
    """
    Relative Hypervolume, calculated as obs_hv/ideal_hv
    """
    raise NotImplementedError
    if tf.equal(tf.size(recommendation_input), 0):
        return tf.zeros(1, dtype=ref_point.dtype)
    assert reference_hv is not None or reference_pf is not None, ValueError(
        "At least one of the ideal_hv or reference_pf need to be specified to calculate ref_hv"
    )
    if reference_hv is not None and reference_pf is not None:
        print("Both reference_hv and reference_pf specified, will only use reference_hv!")
    if reference_hv is None:
        reference_hv = Pareto(reference_pf).hypervolume_indicator(ref_point)

    obj_obs = true_func_inst.objective()(recommendation_input)

    if hasattr(true_func_inst, "constraint"):
        con = true_func_inst.constraint()
        obj_obs = strong_infeasible_penalizer(obj_obs, con(recommendation_input))

    if tf.reduce_any(tf.reduce_all(obj_obs < ref_point, axis=-1)):
        below_mask = tf.reduce_all(obj_obs < ref_point, axis=-1)
        obs_hv = Pareto(obj_obs[below_mask]).hypervolume_indicator(ref_point)
    else:
        obs_hv = tf.zeros(1, dtype=recommendation_input.dtype)
    return obs_hv / reference_hv


def additive_epsilon_indicator(
    reference_pf: TensorType,
    worst_eps_value: Optional = 1e5,
    true_func_inst: Optional[MultiObjectiveTestProblem] = None,
    recommendation_input: Optional[TensorType] = None,
    test_pf: Optional[TensorType] = None,
    **kwargs,
) -> TensorType:
    """
    additive epsilon indicator, see Definition 2.12 (p21) at :cite: gaudrie2019high, which
    calculate different input situation:

    1. if a test_pf is specified, it will calculate the additive epsilon indicator between test_pf and reference_pf
    2. else if a recommendation_input is specified, it will calculate the test_pf based on recommendation_input first

    :param recommendation_input [N, D]
    :param true_func_inst
    :param test_pf
    :param reference_pf [M, D]
    :param worst_eps_value if no recommendation_input, or recommendation_input has violated
           some constraints that cause zero obj_obs, use the worst_eps_value as output of this function
    """

    assert (
        tf.logical_and(recommendation_input is not None, test_pf is not None) == False
    ), ValueError("recommendation_input and test_pf cannot be specified simultaneously!")
    if test_pf is not None:
        distance2true_pf = test_pf - tf.expand_dims(reference_pf, axis=-2)  # [M, N, D]
    else:  # recommendation_input is not None
        if tf.equal(tf.size(recommendation_input), 0):
            return tf.convert_to_tensor(worst_eps_value, dtype=reference_pf.dtype)
        obj = true_func_inst.objective()
        obj_obs = obj(recommendation_input)

        if hasattr(true_func_inst, "constraint"):
            con = true_func_inst.constraint()
            obj_obs = strong_infeasible_penalizer(obj_obs, con(recommendation_input))
            if tf.equal(tf.size(obj_obs), 0):  # empty obs after penalize
                return tf.convert_to_tensor(worst_eps_value, dtype=reference_pf.dtype)
        # if for any m in M, there is point in N dominate it, then [m, N, D] shall contain points that < 0,
        # otherwise, we calculate the minimum shift it needed to be dominated:
        # [m, N, D] -> dim wise max dist -> [m, N] (remove <0 as not interested) -> find min dist -> [m] ->
        # find least shift work for all: max (m) -> distance
        distance2true_pf = obj_obs - tf.expand_dims(reference_pf, axis=-2)  # [M, N, D]
    return tf.reduce_max(
        tf.reduce_min(tf.maximum(tf.reduce_max(distance2true_pf, axis=-1), y=0), axis=-1)
    )


def _GDp(observations: TensorType, true_pf: TensorType, p=2):
    """
    :cite: schutze2012using Eq. 11
    """
    N = observations.shape[0]
    gd_plus = 0.0
    if tf.rank(observations) == 1:
        observations = tf.expand_dims(observations, axis=-2)
    for popts in observations:
        nearest_idx = tf.argmin(tf.norm(popts - true_pf, ord="euclidean", axis=1))
        # pairwise_max = tf.maximum(popts, true_pf[nearest_idx])  # element wise max
        d_plus = (tf.norm(popts - true_pf[nearest_idx], ord="euclidean") ** p) / N
        gd_plus += d_plus
    return gd_plus ** (1 / p)


def _IGDp(observations: TensorType, true_pf: TensorType, p=2):
    """
    :cite: schutze2012using Eq. 35
    """
    N = true_pf.shape[0]
    igd_plus = 0.0
    if tf.rank(observations) == 1:
        observations = tf.expand_dims(observations, axis=-2)
    for popts in true_pf:
        nearest_idx = tf.argmin(tf.norm(observations - popts, ord="euclidean", axis=1))
        d_plus = (tf.norm(observations[nearest_idx] - popts, ord="euclidean") ** p) / N
        igd_plus += d_plus
    return igd_plus ** (1 / p)


def average_hausdoff_distance(
    reference_pf: TensorType,
    true_func_inst: Optional[MultiObjectiveTestProblem] = None,
    recommendation_input: Optional[TensorType] = None,
    test_pf: Optional[TensorType] = None,
    p=2,
    scaler: [tuple, None] = None,
    worst_avd_value=1e5,
) -> TensorType:
    """
    :cite: schutze2012using Eq. 45, which calculate different input situation:

    1. if a test_pf is specified, it will calculate the additive epsilon indicator between test_pf and reference_pf
    2. else if a recommendation_input is specified, it will calculate the test_pf based on recommendation_input first

    The scaler is used for scaling each obj to roughly the same range [0, 1]^K, so that
    euclidean based metric won't bias on certain obj(e.g., if y1 ranges [0, 1] and y2 ranges [0, 1000],
    then if not scale there might be a bias)
    """
    assert (
        tf.logical_and(recommendation_input is not None, test_pf is not None) == False
    ), ValueError("recommendation_input and test_pf cannot be specified simultaneously!")
    if test_pf is not None:
        obj_obs = test_pf
    else:
        if tf.equal(tf.size(recommendation_input), 0):
            return tf.convert_to_tensor(worst_avd_value, dtype=reference_pf)

        obj_obs = true_func_inst.objective()(recommendation_input)

        if hasattr(true_func_inst, "constraint"):
            con = true_func_inst.constraint()
            obj_obs = strong_infeasible_penalizer(obj_obs, con(recommendation_input))
            if tf.equal(tf.size(obj_obs), 0):  # empty obs after penalize
                return tf.convert_to_tensor(worst_avd_value, dtype=reference_pf)
    if scaler is not None:
        reference_pf = (reference_pf - scaler[0]) / (scaler[1] - scaler[0])
        obj_obs = (obj_obs - scaler[0]) / (scaler[1] - scaler[0])
    return tf.maximum(_GDp(obj_obs, reference_pf, p), _IGDp(obj_obs, reference_pf, p))


def negative_log_marginal_likelihood_of_target_pareto_frontier(
    models: Mapping[str, TrainableHasTrajectoryAndPredictJointReparamModelStack],
    reference_pf: TensorType,
    reference_pf_input: TensorType,
    reference_con: Optional[TensorType] = None,
    numerical_stability_term: float = 0.0,
) -> TensorType:
    """
    Negative Marginal Likelihood of target Pareto frontier, used to quantify how model
        is confident about this Pareto frontier
    :param models
    :param reference_pf
    :param reference_pf_input
    :param reference_con
    :param numerical_stability_term add a diagonal term to increase numerical stability, the observation likelihood
        is recommended to be used here
    """
    obj_mean, obj_cov = models[OBJECTIVE].predict_joint(reference_pf_input)

    try:
        # sum log-likelihoods for each independent dimension of output
        obj_likelihood = 0.0
        for i in range(obj_cov.shape[0]):
            obj_likelihood += tf.math.log(
                tfd.MultivariateNormalFullCovariance(
                    obj_mean[:, i],
                    add_noise_cov(
                        obj_cov[i, :, :],
                        tf.convert_to_tensor(numerical_stability_term, dtype=obj_mean.dtype),
                    ),
                ).prob(reference_pf[:, i])
            )

        if CONSTRAINT in models.keys():
            con_mean, con_cov = models[CONSTRAINT].predict_joint(reference_pf_input)
            for j in range(con_cov.shape[0]):
                obj_likelihood += tf.math.log(
                    tfd.MultivariateNormalFullCovariance(
                        con_mean[:, j],
                        add_noise_cov(
                            con_cov[j, :, :],
                            tf.convert_to_tensor(numerical_stability_term, dtype=con_mean.dtype),
                        ),
                    ).prob(reference_con[:, j])
                )
        print(f"NLL_PF: {- tf.convert_to_tensor(obj_likelihood, dtype=reference_pf.dtype)}")
        return -tf.convert_to_tensor(obj_likelihood, dtype=reference_pf.dtype)
    except:
        import math

        return math.inf
