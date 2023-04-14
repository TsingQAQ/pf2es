# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A module searching for (feasible) Pareto frontiers
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union, cast, Mapping

import greenlet as gr
import numpy as np
import tensorflow as tf
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from tensorflow_probability import distributions as tfd
from functools import partial
from pymoo.core.result import Result as PyMOOResult

from ...models.interfaces import HasTrajectorySamplerModelStack, ModelStack
from ...space import Box, DiscreteSearchSpace, SearchSpace, TaggedProductSearchSpace
from ...types import Callable, TensorType
from .dominance import non_dominated, non_dominated_sortting
from ...observer import OBJECTIVE
from ...data import Dataset


def extract_pf_from_data(
        dataset: Mapping[str, Dataset],
        objective_tag: str = OBJECTIVE,
        constraint_tag: Optional[str] = None,
) -> [TensorType, TensorType]:
    """
    Extract (feasible) Pareto Frontier Input and Output from Given dataset

    This assumes the objective and constraints are at the same location!
    return PF_X, PF_Y
    """
    obj_obs = dataset[objective_tag].observations
    pf_obs, pf_boolean_mask = non_dominated(obj_obs)
    if constraint_tag is not None:  # extract feasible pf data
        assert tf.reduce_all(
            tf.equal(dataset[objective_tag].query_points, dataset[constraint_tag].query_points)
        )

        feasible_mask = tf.reduce_all(
            dataset[constraint_tag].observations >= tf.zeros(shape=1, dtype=obj_obs.dtype), axis=-1
        )
        _, un_constraint_dominance_rank = non_dominated(dataset[OBJECTIVE].observations)
        un_constraint_dominance_mask = tf.squeeze(un_constraint_dominance_rank == True)
        feasible_pf_obs = dataset[OBJECTIVE].observations[
            tf.logical_and(un_constraint_dominance_mask, feasible_mask)
        ]
        feasible_pf_x = dataset[OBJECTIVE].query_points[
            tf.logical_and(un_constraint_dominance_mask, feasible_mask)
        ]
        return feasible_pf_x, feasible_pf_obs
    else:
        return tf.boolean_mask(dataset[objective_tag].query_points, pf_boolean_mask == True), pf_obs


class MOOResult:
    """
    Wrapper for pymoo result, the main difference the constraint, if have, is >0 is feasible
    """

    def __init__(self, result: PyMOOResult):
        self._res = result

    @property
    def inputs(self):
        return self._res.X

    @property
    def fronts(self):
        """
        return the (feasible) Pareto Front from Pymoo, if no constraint has been satisfied
        """
        return self._res.F

    @property
    def constraint(self):  # Note in Pymoo, <0 is feasible, so now we need to inverse
        return - self._res.G

    def _initialize_existing_result_as_empty(self, inputs: TensorType, observations: TensorType,
                                             constraints: TensorType = None):
        self._res.X = tf.zeros(shape=(0, inputs.shape[-1]), dtype=inputs.dtype)
        self._res.F = tf.zeros(shape=(0, observations.shape[-1]), dtype=observations.dtype)
        if constraints is not None:
            self._res.G = tf.zeros(shape=(0, constraints.shape[-1]), dtype=constraints.dtype)

    def _check_if_existing_result_is_empty(self):
        if self._res.X is None:
            return True

    def concatenate_with(self, inputs: TensorType, observations: TensorType, constraints: TensorType = None):
        """
        Add result with some other input & observations that possibly is also Pareto optimal
        """

        if self._check_if_existing_result_is_empty():
            self._initialize_existing_result_as_empty(inputs, observations, constraints)
        aug_inputs = tf.concat([self._res.X, inputs], axis=0)
        aug_observations = tf.concat([self._res.F, observations], axis=0)
        if constraints is not None:
            aug_constraints = tf.concat([self.constraint, constraints], axis=0)
            feasible_mask = tf.reduce_all(aug_constraints >= 0, axis=-1)
        else:  # no constrain, all feasible
            aug_constraints = None
            feasible_mask = tf.ones(aug_observations.shape[0], dtype=tf.bool)
        _, dominance_mask_on_feasible_candidate = non_dominated(aug_observations[feasible_mask])
        self._res.X = tf.boolean_mask(aug_inputs[feasible_mask], dominance_mask_on_feasible_candidate == True)
        self._res.F = tf.boolean_mask(aug_observations[feasible_mask], dominance_mask_on_feasible_candidate == True)
        if constraints is not None:
            assert aug_constraints is not None
            self._res.G = - tf.boolean_mask(aug_constraints[feasible_mask], dominance_mask_on_feasible_candidate == True)


def moo_nsga2_pymoo(
        f: Callable[[TensorType], TensorType],
        input_dim: int,
        obj_num: int,
        bounds: tuple,
        popsize: int,
        num_generation: int = 1000,
        cons: Optional[Callable] = None,
        cons_num: float = 0,
        initial_candidates: Optional[TensorType] = None,
        verbose: bool = False
) -> MOOResult:
    """
    Multi-Objective Optimizer using NSGA2 algorithm by pymoo

    When there is no optimal result, the return
    :param f
    :param obj_num
    :param input_dim
    :param bounds: [[lb_0, lb_1， ..., lb_D], [ub_0, ub_1， ..., ub_D]]
    :param popsize: population size for NSGA2
    :param num_generation: number of generations used for NSGA2
    :param cons: Callable function representing the constraints of the problem
    :param cons_num: constraint number
    :return if no feasible pareto frontier has been located, return None or [None, None]
    """

    if cons is not None:
        assert cons_num > 0

    def func(x):
        "wrapper objective function for Pymoo written in numpy"
        return f(tf.convert_to_tensor(x)).numpy()

    def cfunc(x):
        "wrapper constraint function for Pymoo written in numpy"
        return cons(tf.convert_to_tensor(x)).numpy()

    class MyProblem(Problem):
        def __init__(self, n_var: int, n_obj: int, n_constr: int = cons_num):
            """
            :param n_var input variables
            :param n_obj number of objective functions
            :param n_constr number of constraint numbers
            """
            super().__init__(
                n_var=n_var,
                n_obj=n_obj,
                n_constr=n_constr,
                xl=bounds[0].numpy(),
                xu=bounds[1].numpy(),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = func(x)
            if cons_num > 0:
                out["G"] = -cfunc(x)  # in pymoo, by default <0 is feasible

    problem = MyProblem(n_var=input_dim, n_obj=obj_num)
    if initial_candidates is None:  # random start sampling
        pop = get_sampling("real_lhs")
    else:
        # https://pymoo.org/customization/initialization.html
        if initial_candidates.shape[0] >= popsize:  #
            pop = initial_candidates[: popsize].numpy()  # we only use the first pop size
        else:  # we need to fill a bit more
            pop = tf.concat(
                [initial_candidates, Box(bounds[0], bounds[1]).sample_halton(
                    popsize - initial_candidates.shape[0])], axis=0).numpy()

    algorithm = NSGA2(  # we keep the hyperparameter for NSGA2 fixed here
        pop_size=popsize,
        sampling=pop,
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", num_generation), save_history=False, verbose=verbose)

    return MOOResult(res)


def sample_pareto_fronts_from_parametric_gp_posterior(
        objective_models: [HasTrajectorySamplerModelStack, ModelStack],
        obj_num: int,
        sample_pf_num: int,
        search_space: [Box, DiscreteSearchSpace, TaggedProductSearchSpace],
        cons_num: int = 0,
        constraint_models: Optional[HasTrajectorySamplerModelStack, ModelStack] = None,
        moo_solver="nsga2",
        return_pf_input: bool = False,
        return_pf_constraints: Optional[bool] = False,
        *,
        popsize: Optional[int] = 50,
        num_moo_iter: Optional[int] = 500,
        reference_pf_inputs: Optional[TensorType] = None,
        discretize_input_sample_size: Optional[int] = 5000,
        mean_field_approx: Optional[bool] = False
) -> [tuple[list[TensorType, None], list[list[TensorType, None]]], list[TensorType, None]]:
    """
    Sample (feasible) Pareto frontier from Gaussian Process posteriors

    There are two methods implemented in this function:
    1. approximate the Pareto frontier by using a discrete Thompson sampling approach
       as used in cite:fernandez2020improved, see:
       - https://github.com/fernandezdaniel/Spearmint/blob/3c9e0a4be6108c3d652606bd957f0c9ae1bfaf84/spearmint/choosers/default_chooser.py#L994
       - https://github.com/fernandezdaniel/Spearmint/blob/3c9e0a4be6108c3d652606bd957f0c9ae1bfaf84/spearmint/utils/moop.py#L99
       this provides a fast, while coarse approximation of the Pareto frontier while do
       not scale well with input dimensionality (in the above code it uses 20000 sobol grid).

    2. approximate the Pareto frontier by using multi-objective based evolutionary algorithm (MOEA)
       based on a parametric GP posterior, this is the most common approach in existing literatures.

    Note which parametric trajectory sampler to used (i.e., RFF or decoupled sampler) is specified within model
    level

    :param objective_models
    :param obj_num: objective number
    :param sample_pf_num:
    :param search_space:
    :param constraint_models
    :param cons_num
    :param popsize: MOEA pop size
    :param num_moo_iter: MOEA number of iterations
    :cons_num: constraint number
    :param moo_solver: [nsga2, monte_carlo]
    :param return_pf_input return Pareto front corresponding input
    :param return_pf_constraints return constraint of Pareto Frontier
    :param reference_pf_inputs the reference pareto frontier used to
        extract potential candidate point and sampled PF
    :param discretize_input_sample_size: used to generate input samples for discrete Pareto frontier sampling

    """
    con_func = None
    pf_samples: list = []
    pf_samples_x: list = []
    pf_sample_cons: list = []
    mc_potential_candidates: Optional[TensorType] = None
    # pre-processing
    if moo_solver == "monte_carlo":
        # hack: concat existing promising points, that for instance,
        #       can be extracted from discrete observations
        assert discretize_input_sample_size is not None
        # assert return_pf_input is False, NotImplementedError(
        #     "Discrete Pareto Frontier Sample " "Not Support Extracting Input yet"
        # )
        candidate_points = search_space.sample(discretize_input_sample_size)
        mc_potential_candidates = (
            tf.concat([candidate_points, reference_pf_inputs], axis=-2)
            if reference_pf_inputs is not None
            else candidate_points
        )

    if moo_solver == "nsga2":
        assert isinstance(objective_models, HasTrajectorySamplerModelStack)
        if constraint_models is not None:
            assert isinstance(constraint_models, HasTrajectorySamplerModelStack)
        for trajectory_idx in range(sample_pf_num):  # for each sample of trajectories, calculate pf
            # construct objective and constraint function
            objective_models.resample_trajectories()
            obj_func = lambda x: tf.squeeze(
                objective_models.eval_on_trajectory(tf.expand_dims(x, -2)),
                axis=-2
            )

            if constraint_models is not None:
                constraint_models.resample_trajectories()
                con_func = lambda x: tf.squeeze(
                    constraint_models.eval_on_trajectory(tf.expand_dims(x, -2)),
                    axis=-2
                )
            moo_res = moo_nsga2_pymoo(
                obj_func,
                obj_num=obj_num,
                input_dim=len(search_space.lower),
                bounds=(search_space.lower, search_space.upper),
                popsize=popsize,
                num_generation=num_moo_iter,
                cons=con_func,
                cons_num=cons_num
                # here assume each model only have 1 output
            )
            if reference_pf_inputs is not None and tf.size(reference_pf_inputs) != 0:
                reference_obj = obj_func(reference_pf_inputs)
                if constraint_models is not None:
                    reference_con = con_func(reference_pf_inputs)
                else:
                    reference_con = None
                moo_res.concatenate_with(reference_pf_inputs, reference_obj, reference_con)
            pf_samples.append(moo_res.fronts)
            if return_pf_input:
                pf_samples_x.append(moo_res.inputs)
            if return_pf_constraints:
                pf_sample_cons.append(moo_res.constraint)
    elif moo_solver == "monte_carlo":
        if mean_field_approx:
            obj_samples = objective_models.independent_sample(mc_potential_candidates, num_samples=sample_pf_num)
        else:
            obj_samples = objective_models.sample(mc_potential_candidates, num_samples=sample_pf_num)
        if constraint_models is not None:  # extract feasible obj samples
            if mean_field_approx:
                con_samples = constraint_models.independent_sample(
                    mc_potential_candidates, num_samples=sample_pf_num
                )
            else:
                con_samples = constraint_models.sample(
                    mc_potential_candidates, num_samples=sample_pf_num
                )
            con_feasible_mask_samples = tf.reduce_all(con_samples > 0, axis=-1)
        else:
            con_feasible_mask_samples = tf.ones_like(obj_samples)  # all idx is feasible
            con_feasible_mask_samples = tf.reduce_all(con_feasible_mask_samples > 0, axis=-1)
        for obj_sample, con_feasible_mask in zip(obj_samples, con_feasible_mask_samples):
            if tf.reduce_any(con_feasible_mask):
                _, un_constraint_dominance_rank = non_dominated_sortting(obj_sample)
                un_constraint_dominance_mask = tf.squeeze(un_constraint_dominance_rank == 0)
                pf_samples.append(
                    obj_sample[tf.logical_and(un_constraint_dominance_mask, con_feasible_mask)]
                )
                pf_samples_x.append(
                    mc_potential_candidates[
                        tf.logical_and(un_constraint_dominance_mask, con_feasible_mask)
                    ]
                )
            else:  # no feasible obs in current sample
                pf_samples.append(None)
                pf_samples_x.append(None)
    else:
        raise NotImplementedError(
            f"moo_solver: {moo_solver} do not supported yet! "
            f"only support [nsga2, monte_carlo] at the moment"
        )
    if return_pf_input and return_pf_constraints:
        return pf_samples, pf_samples_x, pf_sample_cons
    elif return_pf_input and not return_pf_constraints:
        return pf_samples, pf_samples_x
    elif not return_pf_input and return_pf_constraints:
        return pf_samples, pf_sample_cons
    return pf_samples


def inference_pareto_fronts_from_gp_mean(
        models: ModelStack,
        search_space: Box,
        popsize: int = 20,
        num_moo_iter: int = 500,
        cons_models: [ModelStack, None] = None,
        min_feasibility_probability=0.5,
        monte_carlo_input: bool = False,
        monte_carlo_input_size: int = None,
        constraint_enforce_percentage: float = 0.0,
        use_model_data_as_initialization: bool = True
) -> (TensorType, TensorType):
    """
    Get the (feasible) pareto frontier from GP posterior mean optionally subject to
    a probability of being feasible constraint
    """
    assert isinstance(models, ModelStack)

    def obj_post_mean(at):
        return models.predict(at)[0]

    def con_prob_feasible(at, enforcement=0.0):
        """
        Calculate the probability of being feasible
        """
        mean, var = cons_models.predict(at)
        prob_fea = 1 - tfd.Normal(mean, tf.sqrt(var)).cdf(0.0 + enforcement)
        return tf.reduce_prod(prob_fea, axis=-1, keepdims=True) - min_feasibility_probability

    if cons_models is not None and constraint_enforce_percentage != 0.0:
        assert constraint_enforce_percentage >= 0
        stacked_constraint_obs = tf.concat([_model.get_internal_data().observations for _model in cons_models._models],
                                           1)
        constraint_range = tf.reduce_max(stacked_constraint_obs, -2) - tf.reduce_min(stacked_constraint_obs, -2)
        constraint_enforcement = constraint_range * constraint_enforce_percentage
    else:
        constraint_enforcement = 0.0

    if use_model_data_as_initialization is True:
        if cons_models is not None:
            constraint_tag = 'CONSTRAINT'
            _dataset = {OBJECTIVE: models.get_internal_data(), constraint_tag: cons_models.get_internal_data()}
        else:
            constraint_tag = None
            _dataset = {OBJECTIVE: models.get_internal_data()}
        initial_candidates, _ = extract_pf_from_data(
            _dataset, objective_tag=OBJECTIVE, constraint_tag=constraint_tag
        )
    else:
        initial_candidates = None

    if not monte_carlo_input:
        moo_res = moo_nsga2_pymoo(
            obj_post_mean,
            input_dim=len(search_space.lower),
            obj_num=len(models._models),
            bounds=(search_space.lower, search_space.upper),
            popsize=popsize,
            num_generation=num_moo_iter,
            cons=partial(con_prob_feasible, enforcement=constraint_enforcement) if cons_models is not None else None,
            cons_num=len(cons_models._models) if cons_models is not None else 0,
            initial_candidates=initial_candidates
        )
        return moo_res.fronts, moo_res.inputs

    else:
        if monte_carlo_input_size is None:
            monte_carlo_input_size = len(search_space.lower) * 10000  # we use 10k * D
        test_xs = search_space.sample(monte_carlo_input_size)
        obj = obj_post_mean(test_xs)
        cons = con_prob_feasible(test_xs, enforcement=constraint_enforcement)
        if tf.reduce_any(cons > 0):
            feasible_idx = tf.squeeze(cons > 0)
            front, ranking = non_dominated(obj[feasible_idx])
            return front, test_xs[feasible_idx][ranking == 0]

        raise NotImplementedError("The current approach only support continuous opt")


# TODO: In the future we hope this can replace sample_pareto_fronts_from_parametric_gp_posterior
def sample_pareto_fronts_from_parametric_gp_posterior_using_parallel_nsga2(
        objective_models: [HasTrajectorySamplerModelStack, ModelStack],
        obj_num: int,
        sample_pf_num: int,
        search_space: [Box, DiscreteSearchSpace, TaggedProductSearchSpace],
        cons_num: int = 0,
        constraint_models: Optional[HasTrajectorySamplerModelStack, ModelStack] = None,
        moo_solver="nsga2",
        return_pf_input: bool = False,
        *,
        popsize: Optional[int] = 50,
        num_moo_iter: Optional[int] = 500,
) -> [
    tuple[list[tf.RaggedTensor, None], list[list[tf.RaggedTensor, None]]],
    list[tf.RaggedTensor, None],
]:
    """
    Sample (feasible) Pareto frontier from Gaussian Process posteriors
    This function uses the greenlet implementation, aiming for samping multiple pf_sample size at the same time instead
    of the standard for loop

    Note which parametric trajectory sampler to used (i.e., RFF or decoupled sampler) is specified within model
    level

    :param objective_models
    :param obj_num: objective number
    :param sample_pf_num:
    :param search_space:
    :param constraint_models
    :param cons_num
    :param popsize: MOEA pop size
    :param num_moo_iter: MOEA number of iterations
    :cons_num: constraint number
    :param moo_solver: [nsga2, monte_carlo]
    :param return_pf_input return Pareto front corresponding input

    """
    con_func = None
    if moo_solver == "nsga2":
        assert isinstance(objective_models, HasTrajectorySamplerModelStack)
        if constraint_models is not None:
            assert isinstance(constraint_models, HasTrajectorySamplerModelStack)
            # construct objective and constraint function
        objective_models.resample_trajectories()
        obj_func = objective_models.eval_on_trajectory

        if constraint_models is not None:
            constraint_models.resample_trajectories()
            con_func = constraint_models.eval_on_trajectory

        moo_res = perform_parallel_continuous_multi_objective_optimization(
            vectorized_obj_func=obj_func,
            vectorization_size=sample_pf_num,
            obj_num=obj_num,
            space=search_space,
            population_size=popsize,
            num_generation=num_moo_iter,
            vectorized_con_func=con_func,
            cons_num=cons_num,
            return_pf_input=return_pf_input,
        )
        return moo_res
    else:
        raise NotImplementedError(
            f"moo_solver: {moo_solver} do not supported yet! "
            f"only support [nsga2, monte_carlo] at the moment"
        )


# TODO
# TODO: for now, we assume each problem have exactly the same number of constraint
def perform_parallel_continuous_multi_objective_optimization(
        vectorized_obj_func: Callable,
        vectorization_size: int,
        space: SearchSpace,
        obj_num: int,
        population_size: Optional[int] = 50,
        num_generation: Optional[int] = 500,
        return_pf_input: bool = False,
        vectorized_con_func: Optional[Callable] = None,
        cons_num: Optional[int] = 0,
) -> Tuple[TensorType, TensorType]:
    """
    A function to perform parallel optimization of our acquisition functions
    using PyMOO. We perform NSGA2 starting from each of the locations contained
    in `starting_points`, i.e. the number of individual optimization runs is
    given by the leading dimension of `starting_points`.

    To provide a parallel implementation of Scipy's L-BFGS-B that can leverage
    batch calculations with TensorFlow, this function uses the Greenlet package
    to run each individual optimization on micro-threads.

    nsga2 updates for each individual optimization are performed by
    independent greenlets working with Numpy arrays, however, the evaluation
    of our acquisition function (and its gradients) is calculated in parallel
    (for each optimization step) using Tensorflow.

    For :class:'TaggedProductSearchSpace' we only apply gradient updates to
    its :class:'Box' subspaces, fixing the discrete elements to the best values
    found across the initial random search. To fix these discrete elements, we
    optimize over a continuous :class:'Box' relaxation of the discrete subspaces
    which has equal upper and lower bounds, i.e. we specify an equality constraint
    for this dimension in the scipy optimizer.

    This function also support the maximization of vectorized target functions (with
    vectorization V).

    :param vectorized_obj_func: The function(s) to maximise, with input shape [..., V, D] and
        output shape [..., V].
    :param vectorization_size
    :param space: The original search space.
    :param obj_num
    :param population_size
    :param num_generation
    :param return_pf_input
    :param vectorized_con_func: The constraint function(s) to consider to satisfy, with input shape
        [..., V, D] and output shape [..., V].
    :param cons_num
    :return: the Pareto Frontier [V, ...]
    """
    tf_dtype = space.lower.dtype
    V = vectorization_size  # vectorized batch size
    D = len(space.lower)  # search space dimension # TODO: Only support Box atm
    num_moo_runs = V  # need to run in total V times
    if vectorized_con_func is not None:
        constraint_moo = True
    else:
        constraint_moo = False

    optimizer_args = (D, obj_num, [space.lower, space.upper], population_size)
    optimizer_kwargs = {
        "num_generation": num_generation,
        "return_pf_x": return_pf_input,
        "is_constraint_moo": constraint_moo,
        "cons_num": cons_num,
    }

    if isinstance(
            space, TaggedProductSearchSpace
    ):  # build continuous relaxation of discrete subspaces
        raise NotImplementedError

    # Initialize the numpy arrays to be passed to the greenlets
    np_batch_x = np.zeros((population_size, num_moo_runs, D), dtype=np.float64)
    np_batch_y = np.zeros((population_size, num_moo_runs, obj_num), dtype=np.float64)
    if constraint_moo:
        np_batch_c = np.zeros((population_size, num_moo_runs, cons_num), dtype=np.float64)

    child_greenlets = [PymooNSGA2Greenlet() for _ in range(num_moo_runs)]
    vectorized_child_results: List[Union[MOOResult, "np.ndarray[Any, Any]"]] = [
        gr.switch(*optimizer_args, **optimizer_kwargs) for i, gr in enumerate(child_greenlets)
    ]

    while True:
        all_done = True
        for i, result in enumerate(vectorized_child_results):  # Process results from children.
            if isinstance(result, MOOResult):
                continue  # children return a `MOOResult` if they are finished
            all_done = False
            assert isinstance(result, np.ndarray)  # or an `np.ndarray` with the query `x` otherwise
            np_batch_x[:, i, :] = result  # assign the query x to batch to wait for query

        if all_done:
            break

        # Batch evaluate query `x`s from all children.
        batch_x = tf.constant(np_batch_x, dtype=tf_dtype)  # [population_size, V, d]
        batch_y = vectorized_obj_func(batch_x)
        np_batch_y = batch_y.numpy().astype("float64")
        if constraint_moo:
            batch_c = vectorized_con_func(batch_x)
            np_batch_c = batch_c.numpy().astype("float64")

        for i, greenlet in enumerate(child_greenlets):  # Feed `y` and `dy_dx` back to children.
            if greenlet.dead:  # Allow for crashed greenlets
                continue
            if constraint_moo:
                vectorized_child_results[i] = greenlet.switch(
                    np_batch_y[:, i, :], np_batch_c[:, i, :]
                )
            else:
                vectorized_child_results[i] = greenlet.switch(np_batch_y[:, i, :])

    final_vectorized_child_results: List[MOOResult] = vectorized_child_results
    vectorized_fun_values = (
        tf.stack(  # Use Ragged Tensor here because final pop size may not always be the same
            [
                tf.RaggedTensor.from_tensor(result.fronts)
                for result in final_vectorized_child_results
            ],
            axis=0,
        )
    )  # [num_optimization_runs]

    if return_pf_input:
        vectorized_chosen_x = tf.stack(
            [
                tf.RaggedTensor.from_tensor(result.inputs)
                for result in final_vectorized_child_results
            ],
            axis=0,
        )  # [num_optimization_runs, D]
        return vectorized_fun_values, vectorized_chosen_x
    else:
        return vectorized_fun_values


class PymooNSGA2Greenlet(gr.greenlet):
    def run(
            self,
            input_dim: int,
            obj_num: int,
            bounds: tuple,
            popsize: int,
            num_generation: Optional[int] = 1000,
            return_pf_x: Optional[bool] = False,
            is_constraint_moo: Optional[bool] = False,
            cons_num: Optional[int] = 0,
    ) -> MOOResult:
        """
        Multi-Objective Optimizer using NSGA2 algorithm by pymoo
        :param obj_num
        :param input_dim
        :param bounds: [[lb_0, lb_1， ..., lb_D], [ub_0, ub_1， ..., ub_D]]
        :param popsize: population size for NSGA2
        :param num_generation: number of generations used for NSGA2
        :param return_pf_x: whether to return the Pareto set
        :param is_constraint_moo
        :param cons_num
        :return if no feasible pareto frontier has been located, return None or [None, None]
        """
        cache_x = None  # Any value different from `start`. # [pop_size, input_dim]
        cache_y: Optional["np.ndarray[Any, Any]"] = None
        cache_c: Optional["np.ndarray[Any, Any]"] = None

        def func(x):
            "wrapper objective function for Pymoo written in numpy"
            nonlocal cache_x
            nonlocal cache_y
            nonlocal cache_c

            if not is_constraint_moo:
                if not (cache_x == x).all():
                    cache_x = x  # Copy the value of `x`. DO NOT copy the reference. # TODO: 注意这里不一样
                    # Send `x` to parent greenlet, which will evaluate all `x`s in a batch.
                    cache_y = self.parent.switch(cache_x)

                return cast("np.ndarray[Any, Any]", cache_y)
            else:  # constraint moo
                if not (cache_x == x).all():
                    cache_x = x  # Copy the value of `x`. DO NOT copy the reference. # TODO: 注意这里不一样
                    # Send `x` to parent greenlet, which will evaluate all `x`s in a batch.
                    cache_y, cache_c = self.parent.switch(cache_x)

                return cast("np.ndarray[Any, Any]", cache_y), cast("np.ndarray[Any, Any]", cache_c)

        class MyProblem(Problem):
            def __init__(self, n_var: int, n_obj: int, n_constr: int = cons_num):
                """
                :param n_var input variables
                :param n_obj number of objective functions
                :param n_constr number of constraint numbers
                """
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_constr=n_constr,
                    xl=bounds[0].numpy().copy(),
                    xu=bounds[1].numpy().copy(),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                if not is_constraint_moo:
                    out["F"] = func(x)
                else:
                    f, c = func(x)
                    out["F"] = f
                    out["G"] = -c  # in pymoo, by default <0 is feasible

        problem = MyProblem(n_var=input_dim, n_obj=obj_num)
        algorithm = NSGA2(  # we keep the hyperparameter for NSGA2 fixed here
            pop_size=popsize,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True,
        )

        res = minimize(
            problem, algorithm, ("n_gen", num_generation), save_history=False, verbose=False
        )
        return MOOResult(res)
