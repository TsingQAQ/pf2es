# Copyright 2021 The Trieste Contributors
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
This module contains multi-objective acquisition function builders.
"""
from __future__ import annotations

import math
from itertools import combinations, product
from typing import Callable, Mapping, Optional, Sequence, Tuple, cast

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel, ReparametrizationSampler
from ...models.gpflow.sampler import BatchReparametrizationSampler
from ...models.interfaces import (
    HasReparamSampler,
    HasTrajectorySamplerModelStack,
    ModelStack,
    TrainableHasTrajectoryAndPredictJointReparamModelStack,
    TrainableHasTrajectorySamplerModelStack,
)
from ...observer import OBJECTIVE
from ...space import SearchSpace
from ...types import Tag, TensorType
from ...utils import DEFAULTS
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    AcquisitionFunctionClass,
    GreedyAcquisitionFunctionBuilder,
    PenalizationFunction,
    ProbabilisticModelType,
    SingleModelAcquisitionBuilder,
)
from ..multi_objective.pareto import Pareto, get_reference_point, non_dominated, get_constraint_moo_reference_point
from ..multi_objective.partition import (
    prepare_default_dominated_partition_bounds,
    prepare_default_non_dominated_partition_bounds,
)
from ..multi_objective.utils import sample_pareto_fronts_from_parametric_gp_posterior, extract_pf_from_data
from .entropy import min_value_entropy_search, max_value_entropy_search
from .function import ExpectedConstrainedImprovement



class ExpectedHypervolumeImprovement(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the expected hypervolume improvement acquisition function.
    The implementation of the acquisition function largely
    follows :cite:`yang2019efficient`
    """

    def __init__(
            self,
            reference_point_spec: Sequence[float]
                                  | TensorType
                                  | Callable[..., TensorType] = get_reference_point,
    ):
        """
        :param reference_point_spec: this method is used to determine how the reference point is
            calculated. If a Callable function specified, it is expected to take existing
            posterior mean-based observations (to screen out the observation noise) and return
            a reference point with shape [D] (D represents number of objectives). If the Pareto
            front location is known, this arg can be used to specify a fixed reference point
            in each bo iteration. A dynamic reference point updating strategy is used by
            default to set a reference point according to the datasets.
        """
        if callable(reference_point_spec):
            self._ref_point_spec: tf.Tensor | Callable[..., TensorType] = reference_point_spec
        else:
            self._ref_point_spec = tf.convert_to_tensor(reference_point_spec)
        self._ref_point = None

    def __repr__(self) -> str:
        """"""
        if callable(self._ref_point_spec):
            return f"ExpectedHypervolumeImprovement(" f"{self._ref_point_spec.__name__})"
        else:
            return f"ExpectedHypervolumeImprovement({self._ref_point_spec!r})"

    def prepare_acquisition_function(
            self,
            model: ProbabilisticModel,
            dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The expected hypervolume improvement acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)

        if callable(self._ref_point_spec):
            self._ref_point = tf.cast(self._ref_point_spec(mean), dtype=mean.dtype)
        else:
            self._ref_point = tf.cast(self._ref_point_spec, dtype=mean.dtype)

        _pf = Pareto(mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point, screened_front
        )
        return expected_hv_improvement(model, _partition_bounds)

    def update_acquisition_function(
            self,
            function: AcquisitionFunction,
            model: ProbabilisticModel,
            dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, expected_hv_improvement), [tf.constant([])])
        mean, _ = model.predict(dataset.query_points)

        if callable(self._ref_point_spec):
            self._ref_point = self._ref_point_spec(mean)
        else:
            assert isinstance(self._ref_point_spec, tf.Tensor)  # specified a fixed ref point
            self._ref_point = tf.cast(self._ref_point_spec, dtype=mean.dtype)

        _pf = Pareto(mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point, screened_front
        )
        function.update(_partition_bounds)  # type: ignore
        return function


class expected_hv_improvement(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, partition_bounds: tuple[TensorType, TensorType]):
        r"""
        expected Hyper-volume (HV) calculating using Eq. 44 of :cite:`yang2019efficient` paper.
        The expected hypervolume improvement calculation in the non-dominated region
        can be decomposed into sub-calculations based on each partitioned cell.
        For easier calculation, this sub-calculation can be reformulated as a combination
        of two generalized expected improvements, corresponding to Psi (Eq. 44) and Nu (Eq. 45)
        function calculations, respectively.

        Note:
        1. Since in Trieste we do not assume the use of a certain non-dominated region partition
        algorithm, we do not assume the last dimension of the partitioned cell has only one
        (lower) bound (i.e., minus infinity, which is used in the :cite:`yang2019efficient` paper).
        This is not as efficient as the original paper, but is applicable to different non-dominated
        partition algorithm.
        2. As the Psi and nu function in the original paper are defined for maximization problems,
        we inverse our minimisation problem (to also be a maximisation), allowing use of the
        original notation and equations.

        :param model: The model of the objective function.
        :param partition_bounds: with shape ([N, D], [N, D]), partitioned non-dominated hypercell
            bounds for hypervolume improvement calculation
        :return: The expected_hv_improvement acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        """
        self._model = model
        self._lb_points = tf.Variable(
            partition_bounds[0], trainable=False, shape=[None, partition_bounds[0].shape[-1]]
        )
        self._ub_points = tf.Variable(
            partition_bounds[1], trainable=False, shape=[None, partition_bounds[1].shape[-1]]
        )
        self._cross_index = tf.constant(
            list(product(*[[0, 1]] * self._lb_points.shape[-1]))
        )  # [2^d, indices_at_dim]

    def update(self, partition_bounds: tuple[TensorType, TensorType]) -> None:
        """Update the acquisition function with new partition bounds."""
        self._lb_points.assign(partition_bounds[0])
        self._ub_points.assign(partition_bounds[1])

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        normal = tfp.distributions.Normal(
            loc=tf.zeros(shape=1, dtype=x.dtype), scale=tf.ones(shape=1, dtype=x.dtype)
        )

        def Psi(a: TensorType, b: TensorType, mean: TensorType, std: TensorType) -> TensorType:
            return std * normal.prob((b - mean) / std) + (mean - a) * (
                    1 - normal.cdf((b - mean) / std)
            )

        def nu(lb: TensorType, ub: TensorType, mean: TensorType, std: TensorType) -> TensorType:
            return (ub - lb) * (1 - normal.cdf((ub - mean) / std))

        def ehvi_based_on_partitioned_cell(
                neg_pred_mean: TensorType, pred_std: TensorType
        ) -> TensorType:
            r"""
            Calculate the ehvi based on cell i.
            """

            neg_lb_points, neg_ub_points = -self._ub_points, -self._lb_points

            neg_ub_points = tf.minimum(neg_ub_points, 1e10)  # clip to improve numerical stability

            psi_ub = Psi(
                neg_lb_points, neg_ub_points, neg_pred_mean, pred_std
            )  # [..., num_cells, out_dim]
            psi_lb = Psi(
                neg_lb_points, neg_lb_points, neg_pred_mean, pred_std
            )  # [..., num_cells, out_dim]

            psi_lb2ub = tf.maximum(psi_lb - psi_ub, 0.0)  # [..., num_cells, out_dim]
            nu_contrib = nu(neg_lb_points, neg_ub_points, neg_pred_mean, pred_std)

            stacked_factors = tf.concat(
                [tf.expand_dims(psi_lb2ub, -2), tf.expand_dims(nu_contrib, -2)], axis=-2
            )  # Take the cross product of psi_diff and nu across all outcomes
            # [..., num_cells, 2(operation_num, refer Eq. 45), num_obj]

            factor_combinations = tf.linalg.diag_part(
                tf.gather(stacked_factors, self._cross_index, axis=-2)
            )  # [..., num_cells, 2^d, 2(operation_num), num_obj]

            return tf.reduce_sum(tf.reduce_prod(factor_combinations, axis=-1), axis=-1)

        candidate_mean, candidate_var = self._model.predict(tf.squeeze(x, -2))
        candidate_std = tf.sqrt(candidate_var)

        neg_candidate_mean = -tf.expand_dims(candidate_mean, 1)  # [..., 1, out_dim]
        candidate_std = tf.expand_dims(candidate_std, 1)  # [..., 1, out_dim]

        ehvi_cells_based = ehvi_based_on_partitioned_cell(neg_candidate_mean, candidate_std)

        return tf.reduce_sum(
            ehvi_cells_based,
            axis=-1,
            keepdims=True,
        )


class BatchMonteCarloExpectedHypervolumeImprovement(
    SingleModelAcquisitionBuilder[HasReparamSampler]
):
    """
    Builder for the batch expected hypervolume improvement acquisition function.
    The implementation of the acquisition function largely
    follows :cite:`daulton2020differentiable`
    """

    def __init__(
            self,
            sample_size: int,
            reference_point_spec: Sequence[float]
                                  | TensorType
                                  | Callable[..., TensorType] = get_reference_point,
            *,
            jitter: float = DEFAULTS.JITTER,
            qMC: bool = True
    ):
        """
        :param sample_size: The number of samples from model predicted distribution for
            each batch of points.
        :param reference_point_spec: this method is used to determine how the reference point is
            calculated. If a Callable function specified, it is expected to take existing
            posterior mean-based observations (to screen out the observation noise) and return
            a reference point with shape [D] (D represents number of objectives). If the Pareto
            front location is known, this arg can be used to specify a fixed reference point
            in each bo iteration. A dynamic reference point updating strategy is used by
            default to set a reference point according to the datasets.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        self._sample_size = sample_size
        self._jitter = jitter
        if callable(reference_point_spec):
            self._ref_point_spec: tf.Tensor | Callable[..., TensorType] = reference_point_spec
        else:
            self._ref_point_spec = tf.convert_to_tensor(reference_point_spec)
        self._ref_point = None
        self._qMC = qMC

    def __repr__(self) -> str:
        """"""
        if callable(self._ref_point_spec):
            return (
                f"BatchMonteCarloExpectedHypervolumeImprovement({self._sample_size!r},"
                f" {self._ref_point_spec.__name__},"
                f" jitter={self._jitter!r})"
            )
        else:
            return (
                f"BatchMonteCarloExpectedHypervolumeImprovement({self._sample_size!r},"
                f" {self._ref_point_spec!r}"
                f" jitter={self._jitter!r})"
            )

    def prepare_acquisition_function(
            self,
            model: HasReparamSampler,
            dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model. Must have event shape [1].
        :param dataset: The data from the observer. Must be populated.
        :return: The batch expected hypervolume improvement acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)

        if callable(self._ref_point_spec):
            self._ref_point = tf.cast(self._ref_point_spec(mean), dtype=mean.dtype)
        else:
            self._ref_point = tf.cast(self._ref_point_spec, dtype=mean.dtype)

        _pf = Pareto(mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point, screened_front
        )

        if not isinstance(model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {model.__repr__()}"
            )

        sampler = model.reparam_sampler(self._sample_size)
        return batch_ehvi(sampler, self._jitter, _partition_bounds, qMC = self._qMC)


def batch_ehvi(
        sampler: BatchReparametrizationSampler[HasReparamSampler],
        sampler_jitter: float,
        partition_bounds: tuple[TensorType, TensorType],
        qMC: bool = True
) -> AcquisitionFunction:
    """
    :param sampler: The posterior sampler, which given query points `at`, is able to sample
        the possible observations at 'at'.
    :param sampler_jitter: The size of the jitter to use in sampler when stabilising the Cholesky
        decomposition of the covariance matrix.
    :param partition_bounds: with shape ([N, D], [N, D]), partitioned non-dominated hypercell
        bounds for hypervolume improvement calculation
    :return: The batch expected hypervolume improvement acquisition
        function for objective minimisation.
    """

    def acquisition(at: TensorType) -> TensorType:
        _batch_size = at.shape[-2]  # B

        def gen_q_subset_indices(q: int) -> tf.RaggedTensor:
            # generate all subsets of [1, ..., q] as indices
            indices = list(range(q))
            return tf.ragged.constant([list(combinations(indices, i)) for i in range(1, q + 1)])

        samples = sampler.sample(at, jitter=sampler_jitter, qMC=qMC)  # [..., S, B, num_obj]

        q_subset_indices = gen_q_subset_indices(_batch_size)

        hv_contrib = tf.zeros(tf.shape(samples)[:-2], dtype=samples.dtype)
        lb_points, ub_points = partition_bounds

        def hv_contrib_on_samples(
                obj_samples: TensorType,
        ) -> TensorType:  # calculate samples overlapped area's hvi for obj_samples
            # [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
            overlap_vertices = tf.reduce_max(obj_samples, axis=-2)

            overlap_vertices = tf.maximum(  # compare overlap vertices and lower bound of each cell:
                tf.expand_dims(overlap_vertices, -3),  # expand a cell dimension
                lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :],
            )  # [..., S, K, Cq_j, num_obj]

            lengths_j = tf.maximum(  # get hvi length per obj within each cell
                (ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :] - overlap_vertices), 0.0
            )  # [..., S, K, Cq_j, num_obj]

            areas_j = tf.reduce_sum(  # sum over all subsets Cq_j -> [..., S, K]
                tf.reduce_prod(lengths_j, axis=-1), axis=-1  # calc hvi within each K
            )

            return tf.reduce_sum(areas_j, axis=-1)  # sum over cells -> [..., S]

        for j in tf.range(1, _batch_size + 1):  # Inclusion-Exclusion loop
            q_choose_j = tf.gather(q_subset_indices, j - 1).to_tensor()
            # gather all combinations having j points from q batch points (Cq_j)
            j_sub_samples = tf.gather(samples, q_choose_j, axis=-2)  # [..., S, Cq_j, j, num_obj]
            hv_contrib += tf.cast((-1) ** (j + 1), dtype=samples.dtype) * hv_contrib_on_samples(
                j_sub_samples
            )

        return tf.reduce_mean(hv_contrib, axis=-1, keepdims=True)  # average through MC

    return acquisition


class ExpectedConstrainedHypervolumeImprovement(
    ExpectedConstrainedImprovement[ProbabilisticModelType]
):
    """
    Builder for the constrained expected hypervolume improvement acquisition function.
    This function essentially combines ExpectedConstrainedImprovement and
    ExpectedHypervolumeImprovement.
    """

    def __init__(
        self,
        objective_tag: Tag,
        constraint_builder: AcquisitionFunctionBuilder[ProbabilisticModelType],
        min_feasibility_probability: float | TensorType = 0.5,
        reference_point_spec: Sequence[float]
        | TensorType
        | Callable[..., TensorType] = get_reference_point,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :param reference_point_spec: this method is used to determine how the reference point is
            calculated. If a Callable function specified, it is expected to take existing posterior
            mean-based feasible observations (to screen out the observation noise) and return a
            reference point with shape [D] (D represents number of objectives). If the feasible
            Pareto front location is known, this arg can be used to specify a fixed reference
            point in each bo iteration. A dynamic reference point updating strategy is used by
            default to set a reference point according to the datasets.
        """
        super().__init__(objective_tag, constraint_builder, min_feasibility_probability)
        if callable(reference_point_spec):
            self._ref_point_spec: tf.Tensor | Callable[..., TensorType] = reference_point_spec
        else:
            self._ref_point_spec = tf.convert_to_tensor(reference_point_spec)
        self._ref_point = None

    def __repr__(self) -> str:
        """"""
        if callable(self._ref_point_spec):
            return (
                f"ExpectedConstrainedHypervolumeImprovement({self._objective_tag!r},"
                f" {self._constraint_builder!r}, {self._min_feasibility_probability!r},"
                f" {self._ref_point_spec.__name__})"
            )
        else:
            return (
                f"ExpectedConstrainedHypervolumeImprovement({self._objective_tag!r}, "
                f" {self._constraint_builder!r}, {self._min_feasibility_probability!r},"
                f" ref_point_specification={repr(self._ref_point_spec)!r}"
            )

    def _update_expected_improvement_fn(
            self, objective_model: ProbabilisticModelType, feasible_mean: TensorType
    ) -> None:
        """
        Set or update the unconstrained expected improvement function.

        :param objective_model: The objective model.
        :param feasible_mean: The mean of the feasible query points.
        """
        if callable(self._ref_point_spec):
            self._ref_point = tf.cast(
                self._ref_point_spec(feasible_mean),
                dtype=feasible_mean.dtype,
            )
        else:
            self._ref_point = tf.cast(self._ref_point_spec, dtype=feasible_mean.dtype)

        _pf = Pareto(feasible_mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point,
            screened_front,
        )

        self._expected_improvement_fn: Optional[AcquisitionFunction]
        if self._expected_improvement_fn is None:
            self._expected_improvement_fn = expected_hv_improvement(
                objective_model, _partition_bounds
            )
        else:
            tf.debugging.Assert(
                isinstance(self._expected_improvement_fn, expected_hv_improvement), []
            )
            self._expected_improvement_fn.update(_partition_bounds)  # type: ignore


class HIPPO(GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]):
    r"""
    HIPPO: HIghly Parallelizable Pareto Optimization

    Builder of the acquisition function for greedily collecting batches by HIPPO
    penalization in multi-objective optimization by penalizing batch points
    by their distance in the objective space. The resulting acquistion function
    takes in a set of pending points and returns a base multi-objective acquisition function
    penalized around those points.

    Penalization is applied to the acquisition function multiplicatively. However, to
    improve numerical stability, we perform additive penalization in a log space.
    """

    def __init__(
        self,
        objective_tag: Tag = OBJECTIVE,
        base_acquisition_function_builder: AcquisitionFunctionBuilder[ProbabilisticModelType]
        | SingleModelAcquisitionBuilder[ProbabilisticModelType]
        | None = None,
    ):
        """
        Initializes the HIPPO acquisition function builder.

        :param objective_tag: The tag for the objective data and model.
        :param base_acquisition_function_builder: Base acquisition function to be
            penalized. Defaults to Expected Hypervolume Improvement, also supports
            its constrained version.
        """
        self._objective_tag = objective_tag
        if base_acquisition_function_builder is None:
            self._base_builder: AcquisitionFunctionBuilder[
                ProbabilisticModelType
            ] = ExpectedHypervolumeImprovement().using(self._objective_tag)
        else:
            if isinstance(base_acquisition_function_builder, SingleModelAcquisitionBuilder):
                self._base_builder = base_acquisition_function_builder.using(self._objective_tag)
            else:
                self._base_builder = base_acquisition_function_builder

        self._base_acquisition_function: Optional[AcquisitionFunction] = None
        self._penalization: Optional[PenalizationFunction] = None
        self._penalized_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        Creates a new instance of the acquisition function.

        :param models: The models.
        :param datasets: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: The HIPPO acquisition function.
        :raise tf.errors.InvalidArgumentError: If the ``dataset`` is empty.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        datasets = cast(Mapping[Tag, Dataset], datasets)
        tf.debugging.Assert(datasets[self._objective_tag] is not None, [tf.constant([])])
        tf.debugging.assert_positive(
            len(datasets[self._objective_tag]),
            message=f"{self._objective_tag} dataset must be populated.",
        )

        acq = self._update_base_acquisition_function(models, datasets)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_penalization(acq, models[self._objective_tag], pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        Updates the acquisition function.

        :param function: The acquisition function to update.
        :param models: The models.
        :param datasets: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(datasets is not None, [tf.constant([])])
        datasets = cast(Mapping[Tag, Dataset], datasets)
        tf.debugging.Assert(datasets[self._objective_tag] is not None, [tf.constant([])])
        tf.debugging.assert_positive(
            len(datasets[self._objective_tag]),
            message=f"{self._objective_tag} dataset must be populated.",
        )
        tf.debugging.Assert(self._base_acquisition_function is not None, [tf.constant([])])

        if new_optimization_step:
            self._update_base_acquisition_function(models, datasets)

        if pending_points is None or len(pending_points) == 0:
            # no penalization required if no pending_points
            return cast(AcquisitionFunction, self._base_acquisition_function)

        return self._update_penalization(function, models[self._objective_tag], pending_points)

    def _update_penalization(
            self,
            function: AcquisitionFunction,
            model: ProbabilisticModel,
            pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._penalized_acquisition is not None and isinstance(
                self._penalization, hippo_penalizer
        ):
            # if possible, just update the penalization function variables
            # (the type ignore is due to mypy getting confused by tf.function)
            self._penalization.update(pending_points)  # type: ignore[unreachable]
            return self._penalized_acquisition
        else:
            # otherwise construct a new penalized acquisition function
            self._penalization = hippo_penalizer(model, pending_points)

        @tf.function
        def penalized_acquisition(x: TensorType) -> TensorType:
            log_acq = tf.math.log(
                cast(AcquisitionFunction, self._base_acquisition_function)(x)
            ) + tf.math.log(cast(PenalizationFunction, self._penalization)(x))
            return tf.math.exp(log_acq)

        self._penalized_acquisition = penalized_acquisition
        return penalized_acquisition

    def _update_base_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        if self._base_acquisition_function is None:
            self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                models, datasets
            )
        else:
            self._base_acquisition_function = self._base_builder.update_acquisition_function(
                self._base_acquisition_function, models, datasets
            )
        return self._base_acquisition_function


class hippo_penalizer:
    r"""
    Returns the penalization function used for multi-objective greedy batch Bayesian
    optimization.

    A candidate point :math:`x` is penalized based on the Mahalanobis distance to a
    given pending point :math:`p_i`. Since we assume objectives to be independent,
    the Mahalanobis distance between these points becomes a Eucledian distance
    normalized by standard deviation. Penalties for multiple pending points are multiplied,
    and the resulting quantity is warped with the arctan function to :math:`[0, 1]` interval.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :return: The penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    def __init__(self, model: ProbabilisticModel, pending_points: TensorType):
        """Initialize the MO penalizer.

        :param model: The model.
        :param pending_points: The points we penalize with respect to.
        :raise ValueError: If pending points are empty or None.
        :return: The penalization function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one."""
        tf.debugging.Assert(
            pending_points is not None and len(pending_points) != 0, [tf.constant([])]
        )

        self._model = model
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means = tf.Variable(pending_means, shape=[None, *pending_means.shape[1:]])
        self._pending_vars = tf.Variable(pending_vars, shape=[None, *pending_vars.shape[1:]])

    def update(self, pending_points: TensorType) -> None:
        """Update the penalizer with new pending points."""
        tf.debugging.Assert(
            pending_points is not None and len(pending_points) != 0, [tf.constant([])]
        )

        self._pending_points.assign(pending_points)
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means.assign(pending_means)
        self._pending_vars.assign(pending_vars)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        # x is [N, 1, D]
        x = tf.squeeze(x, axis=1)  # x is now [N, D]

        x_means, x_vars = self._model.predict(x)
        # x_means is [N, K], x_vars is [N, K]
        # where K is the number of models/objectives

        # self._pending_points is [B, D] where B is the size of the batch collected so far
        tf.debugging.assert_shapes(
            [
                (x, ["N", "D"]),
                (self._pending_points, ["B", "D"]),
                (self._pending_means, ["B", "K"]),
                (self._pending_vars, ["B", "K"]),
                (x_means, ["N", "K"]),
                (x_vars, ["N", "K"]),
            ],
            message="""Encountered unexpected shapes while calculating mean and variance
                       of given point x and pending points""",
        )

        x_means_expanded = x_means[:, None, :]
        pending_means_expanded = self._pending_means[None, :, :]
        pending_vars_expanded = self._pending_vars[None, :, :]
        pending_stddevs_expanded = tf.sqrt(pending_vars_expanded)

        # this computes Mahalanobis distance between x and pending points
        # since we assume objectives to be independent
        # it reduces to regular Eucledian distance normalized by standard deviation
        standardize_mean_diff = (
                tf.abs(x_means_expanded - pending_means_expanded) / pending_stddevs_expanded
        )  # [N, B, K]
        d = tf.norm(standardize_mean_diff, axis=-1)  # [N, B]

        # warp the distance so that resulting value is from 0 to (nearly) 1
        warped_d = (2.0 / math.pi) * tf.math.atan(d)
        penalty = tf.reduce_prod(warped_d, axis=-1)  # [N,]

        return tf.reshape(penalty, (-1, 1))


class MESMO(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the maximum entropy search for multi objective
    optimization acquisition function. The implementation of the acquisition
    function largely follows :cite:`belakaria2019max`

    MESMO is just summation of MES on each single outcome,
    """

    def __init__(
            self,
            search_space: SearchSpace,
            *,
            sample_pf_num: int = 5,
            moo_solver: str = "nsga2",
            moo_iter_for_approx_pf_searching: int = 500,
            population_size_for_approx_pf_searching: int = 50,
            discretize_input_sample_size: int = 5000,
    ):
        """
        :param search_space: search space of the acquisition function
        :param sample_pf_num Pareto frontier sample number
        :param moo_solver
        :param moo_iter_for_approx_pf_searching
        :param population_size_for_approx_pf_searching
        :param discretize_input_sample_size: how many input sample numbers are
            used to approximate the Pareto frontier, only used when "monte_carlo"
            is specified as moo_solver

        """
        if sample_pf_num <= 0:
            raise ValueError(f"num_samples must be positive, got {sample_pf_num}")
        self._num_pf_samples = sample_pf_num
        self._search_space = search_space
        self._pop_size = population_size_for_approx_pf_searching
        self._moo_iter = moo_iter_for_approx_pf_searching
        self._moo_solver = moo_solver
        self._discretize_input_sample_size = discretize_input_sample_size

    def __repr__(self) -> str:
        """"""
        return (
            f"MESMO({self._search_space!r}, sample_pf_num={self._num_pf_samples!r},"
            f"moo_solver={self._moo_solver!r}, moo_iter_for_approx_pf_searching={self._moo_iter!r}"
            f""
            f" population_size_for_approx_pf_searching={self._pop_size!r}, "
            f"discretize_input_sample_size={self._discretize_input_sample_size!r})"
        )

    def prepare_acquisition_function(
            self,
            model: [HasTrajectorySamplerModelStack, ModelStack],
            dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model over the specified ``dataset``.
        :return: The pareto frontier entropy search acquisition function.
        """
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        mean, _ = model.predict(dataset.query_points)
        obj_number = tf.shape(mean)[-1]

        obj_wise_min_samples = []
        pf_samples: list[
            TensorType
        ] = sample_pareto_fronts_from_parametric_gp_posterior(  # sample pareto frontier
            model,
            obj_number,
            self._num_pf_samples,
            self._search_space,
            popsize=self._pop_size,
            num_moo_iter=self._moo_iter,
            moo_solver=self._moo_solver,
            discretize_input_sample_size=self._discretize_input_sample_size,
        )
        for pf_sample in pf_samples:  # pf_samples: [N, obj_number]
            obj_wise_min_samples.append(tf.reduce_min(pf_sample, axis=0))
        return multi_objective_min_value_entropy_search(model, obj_wise_min_samples)


class multi_objective_min_value_entropy_search(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, obj_wise_min_samples: list[TensorType]):
        r"""
        min value entropy search of :cite:`belakaria2019max` paper.

        :param model: The model of the objective function.
        :param obj_wise_min_samples: with shape ([D], ..., [D])
        :return: The multi objective min value entropy search acquisition function modified for objective
            minimisation. This function will raise :exc:`ValueError` or
            :exc:`~tf.errors.InvalidArgumentError` if used with a batch size greater than one.
        """
        self._model = model
        self._obj_wise_min_samples = tf.constant(
            tf.expand_dims(tf.transpose(tf.stack(obj_wise_min_samples)), -1)
        )  # [D, sample_num, 1]
        self._single_obj_mes_stack = [
            min_value_entropy_search(model, obj_min_samples)
            for model, obj_min_samples in zip(self._model._models, self._obj_wise_min_samples)
        ]

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        """
        Calculation of MESMO
        note: since MESMO is the average of summation of mes through PF samples, we
        alternatively do summation of average into each mes itself so that we can use
        the origin mes implementation
        """
        single_obj_mes_val_stack = tf.stack(
            [_single_obj_mes(x) for _single_obj_mes in self._single_obj_mes_stack], axis=-1
        )  # [N, 1, D]
        return tf.reduce_sum(single_obj_mes_val_stack, axis=-1)


class PFES(AcquisitionFunctionBuilder[HasTrajectorySamplerModelStack]):
    """
    Implementation of Pareto Frontier Entropy Search (PFES) by :cite:`suzuki2020multi`
    """

    def __init__(
            self,
            search_space: SearchSpace,
            *,
            objective_tag: str = OBJECTIVE,
            sample_pf_num: int = 5,
            moo_solver: str = "nsga2",
            moo_iter_for_approx_pf_searching: int = 500,
            population_size_for_approx_pf_searching: int = 50,
            discretize_input_sample_size: int = 5000,
            reference_point_setting: [str, TensorType] = "default",
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param sample_pf_num: Pareto frontier MC sample number to approximate acq function
        :param reference_point_setting: str of a TensorType which is the reference point, by
            default it will use [1e10]^M
        :raise ValueError (or InvalidArgumentError): If ``min_feasibility_probability`` is not a
            scalar in the unit interval :math:`[0, 1]`.
        """
        self._search_space = search_space
        self._objective_tag = objective_tag
        self._num_pf_samples = sample_pf_num
        self._ref_pts_setting = reference_point_setting
        self._moo_solver = moo_solver
        self._pf_samples: Optional[list] = None  # sampled pareto frontier
        self._partitioned_bounds = None
        self._pop_size = population_size_for_approx_pf_searching
        self._moo_iter = moo_iter_for_approx_pf_searching
        self._discretize_input_sample_size = discretize_input_sample_size
        self._parametric_obj_sampler = None

    def prepare_acquisition_function(
            self,
            models: Mapping[str, TrainableHasTrajectorySamplerModelStack],
            datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models
        :param datasets
        :param pending_points If `pending_points` is not None: perform a greedy batch acquisition function optimization,
        otherwise perform a joint batch acquisition function optimization
        """

        tf.debugging.assert_positive(len(datasets), message="Dataset must be populated.")
        obj_model = models[self._objective_tag]
        obj_number = len(obj_model._models)  # assume each obj has only

        if datasets is not None:
            current_pf_x, _ = extract_pf_from_data(datasets)
        else:
            current_pf_x = None

        self._pf_samples = (
            sample_pareto_fronts_from_parametric_gp_posterior(  # sample pareto frontier
                obj_model,
                obj_number,
                self._num_pf_samples,
                self._search_space,
                popsize=self._pop_size,
                num_moo_iter=self._moo_iter,
                moo_solver=self._moo_solver,
                discretize_input_sample_size=self._discretize_input_sample_size,
                reference_pf_inputs=current_pf_x,
            )
        )

        # get partition bounds
        self._partitioned_bounds = [
            prepare_default_dominated_partition_bounds(
                observations=_pf,
                reference=tf.constant([1e20] * obj_number, dtype=_pf.dtype),
            )
            for _pf in self._pf_samples
        ]

        return pareto_frontier_entropy_search(
            models=obj_model,
            partition_bounds=self._partitioned_bounds,
        )


class pareto_frontier_entropy_search(AcquisitionFunctionClass):
    def __init__(
            self,
            models: HasTrajectorySamplerModelStack,
            partition_bounds: list[tuple[TensorType, TensorType]],
    ):
        """
         PFES based on dominated partition bounds
        :param partition_bounds: dominated space partition bounds
        :param models OBJECT ModelStack
        """

        self._models = models
        self._partition_bounds = partition_bounds

    def __call__(self, x: TensorType) -> TensorType:
        def truncated_entropy_second_term(
                mean: TensorType, std: TensorType, lower_bound: TensorType, upper_bound: TensorType
        ):
            """
            calculate Gamma defined in Theorem 3.1.

            :param mean posterior mean
            :param std posterior standard deviation
            :param lower_bound partitioned lower bound of dominated region
            :param upper_bound partitioned upper bound of dominated region
            """
            helper_normal = tfp.distributions.Normal(tf.cast(0, mean.dtype), tf.cast(1, mean.dtype))
            alpha_u = (upper_bound - mean) / std  # [num_mc, Batch_size,  N, L]
            alpha_l = (lower_bound - mean) / std  # [num_mc, Batch_size,  N, L]
            z_ml = helper_normal.cdf(alpha_u) - helper_normal.cdf(
                alpha_l
            )  # [num_mc, Batch_size,  N, L]
            z_ml = tf.math.maximum(z_ml, 1e-10)  # clip to improve numerical stability
            gamma_ml = (
                               alpha_l * helper_normal.prob(alpha_l) - alpha_u * helper_normal.prob(alpha_u)
                       ) / (
                           (2 * z_ml)
                       )  # [..., N, L]
            tf.debugging.assert_all_finite(
                alpha_l * helper_normal.prob(alpha_l), "lower_bound calc has nan"
            )
            tf.debugging.assert_all_finite(
                alpha_u * helper_normal.prob(alpha_u), "upper_bound calc has nan"
            )
            tf.debugging.assert_all_finite(gamma_ml, "gamma_ml calc has nan")
            return tf.reduce_sum(gamma_ml, -1)  # [..., N]

        def box_prob(
                mean: TensorType, std: TensorType, lower_bound: TensorType, upper_bound: TensorType
        ) -> Tuple[TensorType, TensorType, TensorType]:
            """
            Calculate the probability that the objective distribution is in hyper cubes
            :param mean posterior mean
            :param std posterior standard deviation
            :param lower_bound partitioned lower bound of dominated region
            :param upper_bound partitioned upper bound of dominated region

            return tuple(z_ml: for each dim l, the probability in each slice,
                          z_m: for each hyper cube, the probability inside,
                          z: the probability inside the dominated region)
            """
            helper_normal = tfp.distributions.Normal(tf.cast(0, mean.dtype), tf.cast(1, mean.dtype))
            alpha_u = (upper_bound - mean) / std  # [num_mc, Batch_size,  N, L]
            alpha_l = (lower_bound - mean) / std  # [num_mc, Batch_size,  N, L]
            alpha_u = tf.clip_by_value(
                alpha_u, alpha_u.dtype.min, alpha_u.dtype.max
            )  # clip to improve numerical stability
            alpha_l = tf.clip_by_value(
                alpha_l, alpha_l.dtype.min, alpha_l.dtype.max
            )  # clip to improve numerical stability
            z_ml = helper_normal.cdf(alpha_u) - helper_normal.cdf(
                alpha_l
            )  # [num_mc, Batch_size,  N, L]
            # z_ml = tf.math.maximum(z_ml, 1e-10)  # clip to improve numerical stability
            z_m = tf.reduce_prod(
                z_ml, axis=-1
            )  # [num_mc, Batch_size,  N, L] -> [num_mc, Batch_size,  N]
            z = tf.reduce_sum(
                z_m, axis=-1, keepdims=True
            )  # [num_mc, Batch_size,  N] -> [num_mc, Batch_size, 1]
            z = tf.maximum(z, 1e-10)  # clip to improve numerical stability
            z = tf.minimum(z, 1.0 - 1e-10)  # clip to improve numerical stability
            return z_ml, z_m, z

        def analytic_box_entropy(model, x, lower_bound, upper_bound) -> TensorType:
            """
            Calculate the box entropy assuming minimization of the problem: Refer Eq. 3
            :param: lower_bound: [N, L]
            :param: upper_bound: [N, L]
            return
            """
            mean, var = model.predict(x)
            mean = tf.expand_dims(mean, -2)  # [N, 1, D]
            var = tf.expand_dims(var, -2)  # [N, 1, D]
            std = tf.sqrt(var)  # [N, 1, D]
            z_ml, z_m, z = box_prob(mean, std, lower_bound, upper_bound)
            truncated_h = truncated_entropy_second_term(mean, std, lower_bound, upper_bound)
            tf.debugging.assert_all_finite(z_ml, "zml calc has nan")
            term1 = tf.math.log(z)
            term2 = tf.reduce_sum(z_m / z * truncated_h, axis=-1, keepdims=True)
            return term1 + term2

        # note: the unconstrained entropy (i.e., log(\sqrt(2\pi e)^Lσ^L)) has been canceled out by
        # part of the 1st term in box entropy, refer Eq. 3 of https://arxiv.org/pdf/1906.00127.pdf
        constraint_h = tf.zeros(shape=(tf.shape(x)[0], 1), dtype=x.dtype)  # [N, 1]

        # pareto class is not yet supported batch, we have to hence rely on a loop
        for lb_points, ub_points in self._partition_bounds:
            # The upper bound is also a placeholder: as idealy it is inf
            cons_h = -analytic_box_entropy(
                self._models, tf.squeeze(x, axis=-2), lb_points, ub_points
            )
            constraint_h = tf.concat([constraint_h, cons_h], axis=-1)
        # return tf.reduce_mean(tf.math.reduce_mean(constraint_h[..., 1:], axis=-1, keepdims=True), axis=0)
        return tf.reduce_mean(constraint_h[..., 1:], axis=-1, keepdims=True)


class PF2ES(AcquisitionFunctionBuilder[HasReparamSampler]):
    """
    Implementation of Parallel Feasible Pareto Frontier Entropy Search by :cite:`qing2022text`
    """

    def __init__(
            self,
            search_space: SearchSpace,
            *,
            objective_tag: str = OBJECTIVE,
            constraint_tag: Optional[str] = None,
            sample_pf_num: int = 5,
            moo_solver: str = "nsga2",
            moo_iter_for_approx_pf_searching: int = 500,
            population_size_for_approx_pf_searching: int = 50,
            discretize_input_sample_size: int = 5000,
            reference_point_setting: [
                str,
                TensorType,
            ] = "default",  # TODO: Support Reference Point Setting
            parallel_sampling: bool = False,
            extreme_cons_ref_value: Optional[TensorType] = None,
            batch_mc_sample_size: int = 64,
            temperature_tau=1e-3,
            remove_log: bool = False,
            pareto_epsilon: float = 0.04,
            remove_augmenting_region: bool = False,
            mean_field_pf_approx: bool = False,
            averaging_partition: bool = False,
            use_dbscan_for_conservative_epsilon: bool = False,
            qMC: bool = True
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_tag: The tag for the constraint data and model.
        :param sample_pf_num: Pareto frontier MC sample number to approximate acq function
        :param reference_point_setting: str of a TensorType which is the reference point, by
            default it will use [1e10]^M
        :param extreme_cons_ref_value: in case no feasible Pareto frontier exists (in constraint case),
            use this value as a reference value
        :param batch_mc_sample_size: Monte Carlo sample size for joint batch acquisition function calculation,
            only used when doing batch optimization
        :param remove_log: whether to remove logarithm in acquisition function calculation, note that since logarithm
            is a monotonic acquisition function, whether to have it doesn't affect the maximum of acquisition function
            itself, but will have a big impact on acquisition function optimization, since numerical issues are often
            encountered if a logarithm has been taken
        :param parallel_sampling whether to use Batch acquisition function
        :param pareto_epsilon, heuristically used to make Pareto frontier bit better, this can enforce exploration of
            the Pareto Frontier itself. By default we use 0.04
        :raise ValueError (or InvalidArgumentError): If ``min_feasibility_probability`` is not a
            scalar in the unit interval :math:`[0, 1]`.
        """
        self._search_space = search_space
        self._objective_tag = objective_tag
        self._constraint_tag = constraint_tag
        self._num_pf_samples = sample_pf_num
        self._ref_pts_setting = reference_point_setting
        self._extreme_cons_ref_value = extreme_cons_ref_value
        self._moo_solver = moo_solver
        self._pf_samples: Optional[list] = None  # sampled pareto frontier
        self._pf_samples_inputs: Optional[
            list
        ] = None  # sampled pareto frontier corresponding input
        self._partitioned_bounds = None
        self._pop_size = population_size_for_approx_pf_searching
        self._moo_iter = moo_iter_for_approx_pf_searching
        self._discretize_input_sample_size = discretize_input_sample_size
        self._parametric_obj_sampler = None
        self._parametric_con_sampler = None
        self._q_mc_sample_size = batch_mc_sample_size
        self._tau = temperature_tau
        self._remove_log = remove_log
        assert 0.0 <= pareto_epsilon <= 1.0, ValueError(
            f"Pareto Epsilon is a percentage value must between [0, 1] but received: {pareto_epsilon}"
        )
        self._percentage_pareto_epsilon = pareto_epsilon
        self._pareto_epsilon = 0.0
        self._remove_augmenting_region = remove_augmenting_region
        self._mean_field_pf_approx = mean_field_pf_approx
        self._averaging_partition = averaging_partition
        self._use_dbscan_for_conservative_epsilon = use_dbscan_for_conservative_epsilon
        self._parallel_sampling = parallel_sampling
        self._qMC = qMC

    def prepare_acquisition_function(
            self,
            models: Mapping[str, TrainableHasTrajectoryAndPredictJointReparamModelStack],
            datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models
        :param datasets
        :param pending_points If `pending_points` is not None: perform a greedy batch acquisition function optimization,
        otherwise perform a joint batch acquisition function optimization
        """
        if self._constraint_tag is None:  # prepare unconstrained acquisition function
            return self.prepare_unconstrained_acquisition_function(datasets, models)
        else:  # prepare constraint acquisition function
            if self._remove_augmenting_region is True:
                raise NotImplementedError
            return self.prepare_constrained_acquisition_function(datasets, models)

    def estimate_pareto_frontier_ranges(self, obj_num: int, dtype, sample_wise_maximum: bool = True) -> TensorType:
        """
        Estimate Pareto Frontier ranges based on sampled Pareto Frontier
        :param obj_num: number of objective functions
        :param sample_wise_maximum: whether to use pf sample wise maximum, if enabled to True, the ranges will be
            calculated per pf sample, if set to False, only the maximum range w.r.t all the pf samples will be used,
            this will make a even more conservative estimation of the feasible non-dominated region
        """

        if sample_wise_maximum is False:
            obj_wise_max = tf.zeros(obj_num, dtype=dtype)
            obj_wise_min = tf.zeros(obj_num, dtype=dtype)
            for pf in self._pf_samples:
                if pf is not None:  # handle strong constraint scenario
                    obj_wise_max = tf.maximum(tf.cast(obj_wise_max, pf.dtype), tf.reduce_max(pf, axis=-2))
                    obj_wise_min = tf.minimum(tf.cast(obj_wise_min, pf.dtype), tf.reduce_min(pf, axis=-2))
            return tf.stack([[obj_wise_max - obj_wise_min] * len(self._pf_samples)], axis=0)
        else:
            pareto_frontier_ranges = []
            for pf in self._pf_samples:
                if pf is not None:  # handle strong constraint scenario
                    pareto_frontier_ranges.append(
                        tf.reduce_max(pf, axis=-2) - tf.reduce_min(pf, axis=-2))
                else:
                    pareto_frontier_ranges.append(tf.zeros(shape=obj_num, dtype=dtype))
            return tf.stack(pareto_frontier_ranges, axis=0)

    def calculate_maximum_discrepancy_objective_vise(self, obj_num: int) -> TensorType:
        """
        Calculate Maximum Discrepancy for each sub Pareto Frontier
        """
        maximum_discrepancy_obj_wise = []
        for pf in self._pf_samples:
            max_discrepancy_obj_wise_per_pf = tf.zeros(obj_num, dtype=pf.dtype)
            # handle strong constraint scenario, if pf size is smaller than 2, there is no need to do clustering and
            # we assume the discrepancy in this case is 0

            # none clustering version
            if pf is not None and pf.shape[0] > 2:
                sorted_sub_pf = tf.sort(pf, axis=0)
                sub_maximum_discrepancy = tf.reduce_max(sorted_sub_pf[1:] - sorted_sub_pf[:-1], axis=0)
                max_discrepancy_obj_wise_per_pf = \
                    tf.maximum(tf.cast(sub_maximum_discrepancy, pf.dtype), max_discrepancy_obj_wise_per_pf)
            maximum_discrepancy_obj_wise.append(max_discrepancy_obj_wise_per_pf)
            # clustering version
            # if pf is not None and pf.shape[0] > 2:
            #     # no need to perform cluster
            #     # print(f'pf shape: {pf.shape}')
            #     # print(f'pf: {pf}')
            #     scaled_pf = (pf - tf.reduce_min(pf, axis=0)) / (tf.reduce_max(pf, axis=0) - tf.reduce_min(pf, axis=0))
            #     # print(f'scaled pf: {scaled_pf}')
            #     _, cluster_labels = dbscan(scaled_pf, eps=0.1, min_samples=1)  # eps=0.1 is heuristically chosen
            #     cluster_number = tf.reduce_max(cluster_labels)
            #     print(f'cluster_number: {cluster_number + 1}')
            #     if cluster_number == -1:  # not classified successfully in dbscan
            #         pass
            #     else:
            #         for cluster_idx in range(cluster_number + 1):
            #             # we only calculate discrepancy for clusters with at least 2 data
            #             if pf[cluster_labels == cluster_idx].shape[0] >= 2:
            #                 sorted_sub_pf = tf.sort(pf[cluster_labels == cluster_idx], axis=0)
            #                 sub_maximum_discrepancy = tf.reduce_max(sorted_sub_pf[1:] - sorted_sub_pf[:-1], axis=0)
            #                 max_discrepancy_obj_wise_per_pf = \
            #                     tf.maximum(tf.cast(sub_maximum_discrepancy, pf.dtype), max_discrepancy_obj_wise_per_pf)
            # maximum_discrepancy_obj_wise.append(max_discrepancy_obj_wise_per_pf)
        return tf.stack(maximum_discrepancy_obj_wise, axis=0)

    def prepare_unconstrained_acquisition_function(
            self,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, TrainableHasTrajectoryAndPredictJointReparamModelStack],
    ) -> AcquisitionFunction:
        """
        prepare parallel pareto frontier entropy search acquisition function
        :param datasets
        :param models
        """
        tf.debugging.assert_positive(len(datasets), message="Dataset must be populated.")
        obj_model = models[self._objective_tag]
        obj_number = len(obj_model._models)  # assume each obj has only

        current_pf_x, _ = extract_pf_from_data(datasets)

        (
            self._pf_samples,
            self._pf_samples_inputs,
        ) = sample_pareto_fronts_from_parametric_gp_posterior(  # sample pareto frontier
            obj_model,
            obj_number,
            self._num_pf_samples,
            self._search_space,
            popsize=self._pop_size,
            num_moo_iter=self._moo_iter,
            moo_solver=self._moo_solver,
            discretize_input_sample_size=self._discretize_input_sample_size,
            reference_pf_inputs=current_pf_x,
            return_pf_input=True,
            mean_field_approx=self._mean_field_pf_approx
        )
        if self._use_dbscan_for_conservative_epsilon is not True:
            self._pareto_epsilon = \
                self.estimate_pareto_frontier_ranges(obj_num=obj_number, dtype=current_pf_x.dtype) * \
                tf.convert_to_tensor(self._percentage_pareto_epsilon, dtype=current_pf_x.dtype)

        else:  # use dbscan
            self._pareto_epsilon = self.calculate_maximum_discrepancy_objective_vise(obj_num=obj_number)

        # get partition bounds
        self._partitioned_bounds = [
            prepare_default_non_dominated_partition_bounds(
                tf.constant([1e20] * obj_number, dtype=_pf.dtype),
                non_dominated(_pf - _pf_epsilon)[0],
                remove_augmentation_part=self._remove_augmenting_region,
                averaging_partition=self._averaging_partition
            )
            for _pf, _pf_epsilon in zip(self._pf_samples, self._pareto_epsilon)
        ]

        if self._averaging_partition:
            self._partitioned_bounds = list(sum(self._partitioned_bounds, ()))
            self._pf_samples = [pf_sample for pf_sample in self._pf_samples for _ in (0, 1)]

        if not isinstance(obj_model, HasReparamSampler):
            raise ValueError(
                f"The PF2ES function only supports "
                f"models that implement a reparam_sampler method; received {obj_model.__repr__()}"
            )

        sampler = obj_model.reparam_sampler(self._q_mc_sample_size)

        if self._parallel_sampling:
            return parallel_pareto_frontier_entropy_search(
                models=obj_model,
                partition_bounds=self._partitioned_bounds,
                pf_samples=self._pf_samples,
                sampler=sampler,
                remove_log=self._remove_log,
                tau=self._tau
            )
        else:
            return sequential_pareto_frontier_entropy_search(
                models=obj_model,
                partition_bounds=self._partitioned_bounds,
                pf_samples=self._pf_samples,
                sampler=sampler,
                remove_log=self._remove_log,
                tau=self._tau
            )

    def prepare_constrained_acquisition_function(
            self,
            datasets: Mapping[str, Dataset],
            models: Mapping[str, ModelStack],
    ) -> AcquisitionFunction:
        """
        prepare parallel feasible pareto frontier entropy search acquisition function
        :param datasets
        :param models
        """
        obj_model = models[self._objective_tag]
        cons_model = models[self._constraint_tag]
        obj_number = len(obj_model._models)
        cons_number = len(cons_model._models)
        _constraint_threshold = tf.zeros(
            shape=cons_number, dtype=datasets[self._objective_tag].query_points.dtype
        )
        current_pf_x, current_pf_y = extract_pf_from_data(
            datasets, objective_tag=self._objective_tag, constraint_tag=self._constraint_tag
        )
        # Note: constrain threshold is not used in sample Feasible PF
        (self._pf_samples, self._pf_samples_inputs) = \
            sample_pareto_fronts_from_parametric_gp_posterior(  # sample pareto frontier
                obj_model,
                obj_number,
                constraint_models=cons_model,
                cons_num=cons_number,
                sample_pf_num=self._num_pf_samples,
                search_space=self._search_space,
                popsize=self._pop_size,
                num_moo_iter=self._moo_iter,
                moo_solver=self._moo_solver,
                discretize_input_sample_size=self._discretize_input_sample_size,
                reference_pf_inputs=current_pf_x,
                return_pf_input=True,
                mean_field_approx=self._mean_field_pf_approx
            )

        # tf.print("cMOO optimization finished")
        if self._use_dbscan_for_conservative_epsilon is not True:
            self._pareto_epsilon = \
                self.estimate_pareto_frontier_ranges(obj_num=obj_number, dtype=_constraint_threshold.dtype) * \
                tf.convert_to_tensor(self._percentage_pareto_epsilon, dtype=_constraint_threshold.dtype)

        else:
            self._pareto_epsilon = self.calculate_maximum_discrepancy_objective_vise(obj_num=obj_number)

            for _fea_pf, _id in zip(self._pf_samples, range(len(self._pf_samples))):
                if _fea_pf is None or tf.size(_fea_pf) == 0:
                    print(f"no feasible obs in this {_id}th PF sample ")
        # get partition bounds
        self._partitioned_bounds = [
            prepare_default_non_dominated_partition_bounds(
                tf.constant([1e20] * obj_number, dtype=datasets[self._objective_tag].query_points.dtype),
                non_dominated(_fea_pf - _pf_epsilon)[0]  if _fea_pf is not None else None,
                remove_augmentation_part=self._remove_augmenting_region,
                averaging_partition=self._averaging_partition
            )
            for _fea_pf, _pf_epsilon in zip(self._pf_samples, self._pareto_epsilon)
        ]

        if not isinstance(obj_model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {obj_model.__repr__()}"
            )
        if not isinstance(cons_model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {cons_model.__repr__()}"
            )

        obj_sampler = obj_model.reparam_sampler(self._q_mc_sample_size)
        cons_sampler = cons_model.reparam_sampler(self._q_mc_sample_size)
        # print(f'constraint data: {cons_model._models[0].get_internal_data()}')
        if self._parallel_sampling:
            return parallel_feasible_pareto_frontier_entropy_search(
                objective_models=obj_model,
                constraint_models=cons_model,
                partition_bounds=self._partitioned_bounds,
                constraint_threshold=_constraint_threshold,
                obj_sampler=obj_sampler,
                con_sampler=cons_sampler,
                remove_log=self._remove_log,
                tau=self._tau,
                qMC = self._qMC
            )
        else:
            return sequential_feasible_pareto_frontier_entropy_search(
                objective_models=obj_model,
                constraint_models=cons_model,
                partition_bounds=self._partitioned_bounds,
                constraint_threshold=_constraint_threshold,
                obj_sampler=obj_sampler,
                con_sampler=cons_sampler,
                remove_log=self._remove_log,
                tau=self._tau,
            )


class sequential_pareto_frontier_entropy_search(AcquisitionFunctionClass):
    def __init__(
            self,
            models: ProbabilisticModelType,
            partition_bounds: list[tuple[TensorType, TensorType]],
            pf_samples: list[TensorType],
            sampler: Optional[ReparametrizationSampler] = None,
            remove_log: bool = False,
            tau: float = 1e-2,
            qMC: bool = True
    ):
        """
        :param partition_bounds
        :param models
        :param sampler reparameterization sampler for obj model
        :param remove_log: whether to remove logarithm in acquisition function calculation, note that since logarithm
            is a monotonic acquisition function, whether to have it doesn't affect the maximum of acquisition function
            itself, but will have a big impact on acquisition function optimization, since numerical issues are often
            encountered in a logarithm has been taken
        :param tau temperature parameter, used to soft handle 0-1 event
        """
        assert len(partition_bounds) == len(pf_samples)
        self._model = models
        self._partition_bounds = partition_bounds
        self._sampler = sampler
        self._tau = tau
        self._remove_log = remove_log
        self._pf_samples = pf_samples
        self._qMC = qMC

    @tf.function
    def __call__(self, x: TensorType):
        prob_improve = tf.zeros(shape=(tf.shape(x)[0], 1), dtype=x.dtype)  # [N, 1]
        for (lb_points, ub_points), pareto_frontier in zip(
                self._partition_bounds, self._pf_samples
        ):
            # partition the dominated region
            # The upper bound is also a placeholder: as idealy it is inf
            lb_points = tf.maximum(lb_points, -1e100)  # increase numerical stability
            prob_iprv = analytic_non_dominated_prob(
                self._model,
                x,
                lb_points,
                ub_points,
                clip_to_enable_numerical_stability=~self._remove_log,
                remove_triangle_part=False,
                pareto_frontier=pareto_frontier,
            )
            prob_improve = tf.concat([prob_improve, prob_iprv], axis=-1)  # [..., N, pf_mc_size + 1]

        # [N, 1 + pf_mc_size] -> [N, 1]
        if not self._remove_log:
            # return tf.reduce_mean(
            #     -tf.math.log(1 - prob_improve[..., 1:] + 1e-10), axis=-1, keepdims=True
            # )
            return tf.reduce_mean(-tf.math.log(1 - prob_improve[..., 1:]), axis=-1, keepdims=True)
        else:
            return tf.reduce_mean(prob_improve[..., 1:], axis=-1, keepdims=True)


class parallel_pareto_frontier_entropy_search(sequential_pareto_frontier_entropy_search):
    """q-PF2ES for MOO problem"""
    # @tf.function
    def __call__(self, x: TensorType):
        prob_improve = tf.zeros(shape=(tf.shape(x)[0], 1), dtype=x.dtype)  # [N, 1]
        for (lb_points, ub_points), pareto_frontier in zip(
                self._partition_bounds, self._pf_samples
        ):
            # partition the dominated region
            # The upper bound is also a placeholder: as idealy it is inf
            lb_points = tf.maximum(lb_points, -1e100)  # increase numerical stability
            prob_iprv = monte_carlo_non_dominated_prob(
                self._sampler, x, lb_points, ub_points, self._tau, qMC=self._qMC
            )
            prob_improve = tf.concat([prob_improve, prob_iprv], axis=-1)  # [..., N, pf_mc_size + 1]

        # [N, 1 + pf_mc_size] -> [N, 1]
        if not self._remove_log:
            # return tf.reduce_mean(
            #     -tf.math.log(1 - prob_improve[..., 1:] + 1e-10), axis=-1, keepdims=True
            # )
            return tf.reduce_mean(-tf.math.log(1 - prob_improve[..., 1:]), axis=-1, keepdims=True)
        else:
            return tf.reduce_mean(prob_improve[..., 1:], axis=-1, keepdims=True)


class sequential_feasible_pareto_frontier_entropy_search(AcquisitionFunctionClass):
    def __init__(
            self,
            objective_models: ProbabilisticModelType,
            constraint_models: ProbabilisticModelType,
            partition_bounds: list[tuple[TensorType, TensorType]],
            constraint_threshold: TensorType,
            obj_sampler: Optional[ReparametrizationSampler] = None,
            con_sampler: Optional[ReparametrizationSampler] = None,
            remove_log: bool = False,
            tau: float = 1e-2,
            qMC: bool = True
    ):
        """
        :param objective_models
        :param constraint_models
        :param partition_bounds
        :param obj_sampler
        :param con_sampler
        :param remove_log: whether to remove logarithm in acquisition function calculation, note that since logarithm
            is a monotonic acquisition function, whether to have it doesn't affect the maximum of acquisition function
            itself, but will have a big impact on acquisition function optimization, since numerical issues are often
            encountered in a logarithm has been taken
        :param tau
        """
        self._obj_model = objective_models
        self._con_model = constraint_models
        self._partition_bounds = partition_bounds
        self._obj_sampler = obj_sampler
        self._con_sampler = con_sampler
        self._tau = tau
        self._constraint_threshold = constraint_threshold
        self._remove_log = remove_log
        self._qMC = qMC

    @tf.function
    def __call__(self, x: TensorType):
        prob_improve = tf.zeros(shape=(x.shape[0], 1), dtype=x.dtype)  # [Batch_dim, 1]

        # pareto class is not yet supported batch, we have to hence rely on a loop
        for lb_points, ub_points in self._partition_bounds:
            # partition the dominated region
            lb_points = tf.maximum(lb_points, -1e100)
            _analytic_pof = analytic_pof(self._con_model, x, self._constraint_threshold)
            prob_iprv = (
                    analytic_non_dominated_prob(
                        self._obj_model,
                        x,
                        lb_points,
                        ub_points,
                        clip_to_enable_numerical_stability=~self._remove_log,
                    )
                    * _analytic_pof
            )

            # improve stability
            prob_iprv = prob_iprv * tf.cast(tf.greater_equal(_analytic_pof, 1e-5), dtype=prob_iprv.dtype)
            tf.debugging.assert_all_finite(prob_iprv, f"prob_iprv: {prob_iprv} has NaN!")
            prob_improve = tf.concat(
                [prob_improve, prob_iprv], axis=-1
            )  # [pending_mc_size, N, pf_mc_size]
        if not self._remove_log:
            return tf.reduce_mean(-tf.math.log(1 - prob_improve[..., 1:]), axis=-1, keepdims=True)
        else:
            return tf.reduce_mean(prob_improve[..., 1:], axis=-1, keepdims=True)


class parallel_feasible_pareto_frontier_entropy_search(sequential_feasible_pareto_frontier_entropy_search):
    # @tf.function
    def __call__(self, x: TensorType):
        prob_improve = tf.zeros(shape=(x.shape[0], 1), dtype=x.dtype)  # [Batch_dim, 1]

        # pareto class is not yet supported batch, we have to hence rely on a loop
        for lb_points, ub_points in self._partition_bounds:
            # partition the dominated region
            lb_points = tf.maximum(lb_points, -1e100)
            prob_iprv = monte_carlo_non_dominated_feasible_prob(
                self._obj_sampler,
                self._con_sampler,
                x,
                lb_points,
                ub_points,
                self._constraint_threshold,
                self._tau,
                qMC = self._qMC
            )
            # tf.debugging.assert_all_finite(prob_iprv, f"prob_iprv: {prob_iprv} has NaN!")
            prob_improve = tf.concat(
                [prob_improve, prob_iprv], axis=-1
            )  # [pending_mc_size, N, pf_mc_size]
        if not self._remove_log:
            return tf.reduce_mean(-tf.math.log(1 - prob_improve[..., 1:]), axis=-1, keepdims=True)
        else:
            return tf.reduce_mean(prob_improve[..., 1:], axis=-1, keepdims=True)


# TODO: We need a way to batch this! since we generally do not have such a way, we maybe try to only calculate one
def prob_being_in_triangle_region(
        model: ProbabilisticModelType, input: TensorType, pareto_frontier: TensorType
):
    def analytical_calculation_of_the_probability(mu_x, mu_y, sigma_x, sigma_y, l1, l2):
        """
        Test of being in the triangular region
        """
        __b = -l2 * sigma_x / (l1 * sigma_y)
        __a = (l1 * l2 - l1 * mu_y - l2 * mu_x) / (l1 * sigma_y)
        rho = -__b / (1 + __b ** 2) ** 0.5
        rv = multivariate_normal(
            [
                0.0,
                0.0,
            ],
            [[1.0, rho], [rho, 1.0]],
        )
        cdf_diff = rv.cdf([__a / ((1 + __b ** 2) ** 0.5), (l1 - mu_x) / sigma_x]) - rv.cdf(
            [__a / ((1 + __b ** 2) ** 0.5), (-mu_x) / sigma_x]
        )
        sub_part = norm().cdf(-mu_y / sigma_y) * (
                norm().cdf((l1 - mu_x) / sigma_x) - norm().cdf((-mu_x) / sigma_x)
        )
        return cdf_diff - sub_part

    from scipy.stats import multivariate_normal, norm

    means, vars = model.predict(input)
    means = tf.squeeze(means, -2)
    vars = tf.squeeze(vars, -2)
    stds = tf.sqrt(vars)
    # sort input front
    sorted_pareto_frontier = tf.gather_nd(
        pareto_frontier, tf.argsort(pareto_frontier[:, :1], axis=0)
    )
    element_res = []
    for mean, std in zip(means, stds):
        element_prob = 0.0
        for pf_point_a, pf_point_b in zip(sorted_pareto_frontier, sorted_pareto_frontier[1:]):
            _l1 = (pf_point_b - pf_point_a)[0]
            _l2 = (pf_point_a - pf_point_b)[1]
            # since we derive the expresion in 1st quadrant， we transform our problem there
            moved_mean = -(mean - tf.convert_to_tensor([pf_point_b[0], pf_point_a[-1]]))
            element_prob += analytical_calculation_of_the_probability(
                moved_mean[0], moved_mean[1], std[0], std[1], _l1, _l2
            )
        element_res.append(element_prob)
        # print(len(element_res))

    res = tf.convert_to_tensor(element_res)[..., tf.newaxis]
    print(f"max prob in triangle region is: {tf.reduce_max(res)}")
    return res


def analytic_non_dominated_prob(
        model: ProbabilisticModelType,
        inputs: TensorType,
        lower_bounds: TensorType,
        upper_bounds: TensorType,
        clip_to_enable_numerical_stability: bool = True,
        remove_triangle_part: bool = False,
        pareto_frontier: Optional[TensorType] = None,
) -> TensorType:
    """
    Calculate the probability of non-dominance given the mean and std, this is the
    same as hyper-volume probability of improvemet unless remove_triangle_part is set to True
    :param model
    :param inputs: [N, D]
    :param: lower_bounds: [N, M]
    :param: upper_bounds: [N, M]
    :param clip_to_enable_numerical_stability: if set to True, clip small
        amount ensure numerical stability under logarithm
    :param remove_triangle_part: remove the probability of being in the corner part of non-dominated region
    :param pareto_frontier
    return
    """
    tf.debugging.assert_shapes(
        [(inputs, ["N", 1, None])],
        message="This acquisition function only supports batch sizes of one.",
    )
    standard_normal = tfp.distributions.Normal(tf.cast(0, inputs.dtype), tf.cast(1, inputs.dtype))
    fmean, fvar = model.predict(tf.squeeze(inputs, -2))
    fvar = tf.clip_by_value(fvar, 1e-100, 1e100)  # clip below to improve numerical stability

    mean = tf.expand_dims(fmean, -2)
    std = tf.expand_dims(tf.sqrt(fvar), -2)
    alpha_u = (upper_bounds - mean) / std  # [..., N_data, N, L]
    alpha_l = (lower_bounds - mean) / std  # [..., N_data, N, L]
    alpha_u = tf.clip_by_value(
        alpha_u, alpha_u.dtype.min, alpha_u.dtype.max
    )  # clip to improve numerical stability
    alpha_l = tf.clip_by_value(
        alpha_l, alpha_l.dtype.min, alpha_l.dtype.max
    )  # clip to improve numerical stability
    z_ml = standard_normal.cdf(alpha_u) - standard_normal.cdf(alpha_l)  # [..., N_Data, N, M+C]
    z_m = tf.reduce_prod(z_ml, axis=-1)  # [..., N_Data, N, M+C] -> [..., N_Data, N]
    z = tf.reduce_sum(z_m, axis=-1, keepdims=True)  # [..., N_Data, 1]
    if clip_to_enable_numerical_stability:
        z = tf.maximum(z, 1e-10)  # clip to improve numerical stability
        z = tf.minimum(z, 1 - 1e-10)  # clip to improve numerical stability
    if remove_triangle_part:
        assert lower_bounds.shape[-1] == 2, NotImplementedError(
            "Remove Triangle Part only support Bi objective problem"
        )
        z -= prob_being_in_triangle_region(model, inputs, pareto_frontier)

    return z  # tf.reduce_mean(z, keepdims=True, axis=-3)  # [..., N_Data, 1] -> [..., N_Data, 1]


def monte_carlo_non_dominated_prob(
        sampler, inputs: TensorType, lower_bound: TensorType, upper_bound: TensorType, epsilon, qMC: bool = True
) -> TensorType:
    """
    In order to enable batch, we need to change this a bit
    Calculate the probability of non-dominance given the mean and std, this is the
    same as hyper-volume probability of improvemet
    :param sampler
    :param inputs
    :param lower_bound: [N_bd, M]
    :param upper_bound: [N_bd, M]
    :param epsilon
    return
    """
    observations = sampler.sample(inputs, qMC=qMC)  # [N, mc_num, q, M]
    expand_obs = tf.expand_dims(observations, -3)  # [N, mc_num, 1, q, M]
    # calculate the probability that it is non-dominated
    expand_lower_bound = tf.expand_dims(lower_bound, -2)  # [N_bd, 1, M]
    expand_upper_bound = tf.expand_dims(upper_bound, -2)  # [N_bd, 1, M]
    soft_above_lower_bound = tf.sigmoid(
        (expand_obs - expand_lower_bound) / epsilon
    )  # [N, mc_num, 1, q, M] - [N_bd, 1, M] -> [N, mc_num, N_bd, q, M]
    soft_below_upper_bound = tf.sigmoid(
        (expand_upper_bound - expand_obs) / epsilon
    )  # [N, mc_num, N_bd, q, M]
    soft_any_of_q_in_cell = tf.reduce_prod(
        soft_above_lower_bound * soft_below_upper_bound, axis=-1, keepdims=True
    )  # [N, mc_num, N_bd, q, 1]
    soft_any_of_q_in_cell = tf.reduce_max(soft_any_of_q_in_cell, axis=-2)  # [N, mc_num, N_bd, 1]
    prob_any_non_dominated = tf.reduce_max(soft_any_of_q_in_cell, -2)  # [N, mc_num, 1]
    prob_any_non_dominated = tf.reduce_mean(prob_any_non_dominated, -2)  # [N, 1]
    return prob_any_non_dominated


# @tf.function
def monte_carlo_non_dominated_feasible_prob(
        obj_sampler,
        cons_sampler,
        input: TensorType,
        lower_bound: TensorType,
        upper_bound: TensorType,
        constraint_threshold: TensorType,
        tau: float,
        qMC: bool =True
) -> TensorType:
    """
     Note, this represent the probability that
    "probability that at least one of q is feasible and satisfy the constraint"
    """
    observations_smp = obj_sampler.sample(input, qMC=qMC)  # [N, mc_num, q, M]
    constraints_smp = cons_sampler.sample(input, qMC=qMC)  # [N, mc_num, q, C]
    tf.debugging.assert_all_finite(observations_smp, f'observations_smp samples: has NaN')
    tf.debugging.assert_all_finite(constraints_smp, f'constraints_smp samples: has NaN')
    # aug_samp = tf.concat([observations_smp, constraints_smp], axis=-1) # [N, mc_num, q, M+C]
    expand_obs = tf.expand_dims(observations_smp, -3)  # [N, mc_num, 1, q, M]
    expand_cons = tf.expand_dims(constraints_smp, -3)  # [N, mc_num, 1, q, C]
    # calculate the probability that it is 1. non-dominated, 2. feasible
    expand_lower_bound = tf.expand_dims(lower_bound, -2)  # [N_bd, 1, M+C]
    expand_upper_bound = tf.expand_dims(upper_bound, -2)  # [N_bd, 1, M+C]

    soft_above_lower_bound = tf.sigmoid(
        (expand_obs - expand_lower_bound) / tau
    )  # [N, mc_num, N_bd, q, M]
    soft_below_upper_bound = tf.sigmoid(
        (expand_upper_bound - expand_obs) / tau
    )  # [N, mc_num, N_bd, q, M]
    # calc feasibility
    soft_satisfy_constraint = tf.reduce_prod(
        tf.sigmoid((expand_cons - constraint_threshold) / tau), -1, keepdims=True
    )  # [N, mc_num, 1, q, 1]
    soft_of_any_cand_in_cell_and_feasible = tf.reduce_prod(
        soft_above_lower_bound * soft_below_upper_bound, axis=-1, keepdims=True
    ) * soft_satisfy_constraint  # [N, mc_num, N_bd, q, 1]

    soft_of_any_cand_in_cell_and_feasible = tf.reduce_max(soft_of_any_cand_in_cell_and_feasible, axis=-2) # [N, mc_num, N_bd, 1]
    soft_of_any_cand_in_any_cell = tf.reduce_max(soft_of_any_cand_in_cell_and_feasible, axis=-2)  # [N, mc_num, 1]
    prob_any_non_dominated = tf.reduce_mean(soft_of_any_cand_in_any_cell, axis=-2) # [N, 1]
    prob_any_non_dominated = tf.minimum(
        prob_any_non_dominated, 1 - 1e-10
    )  # clip to improve numerical stability
    return prob_any_non_dominated


def analytic_pof(
        constraint_predictor: ProbabilisticModelType, input: TensorType, constraint_threshold
) -> TensorType:
    cmean, cvar = constraint_predictor.predict(tf.squeeze(input, -2))

    cvar = tf.clip_by_value(cvar, 1e-100, 1e100)  # clip below to improve numerical stability
    pof = tf.reduce_prod(
        (1 - tfp.distributions.Normal(cmean, tf.sqrt(cvar)).cdf(constraint_threshold)),
        axis=-1,
        keepdims=True,
    )  # [MC_size, batch_size, 1]
    return pof


class MESMOC(AcquisitionFunctionBuilder[HasTrajectorySamplerModelStack]):
    """
    Implementation of MESMOC
    """

    def __init__(
            self,
            search_space: SearchSpace,
            *,
            objective_tag: str = OBJECTIVE,
            constraint_tag: Optional[str] = None,
            sample_pf_num: int = 5,
            moo_solver: str = "nsga2",
            moo_iter_for_approx_pf_searching: int = 500,
            population_size_for_approx_pf_searching: int = 50,
            discretize_input_sample_size: int = 5000,
            reference_point_setting: [
                str,
                TensorType,
            ] = "default",
            extreme_cons_ref_value: Optional[TensorType] = None,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_tag: The tag for the constraint data and model.
        :param sample_pf_num: Pareto frontier MC sample number to approximate acq function
        :param reference_point_setting: str of a TensorType which is the reference point, by
            default it will use [1e10]^M
        :param extreme_cons_ref_value: in case no feasible Pareto frontier exists (in constraint case),
            use this value as a reference value
        :raise ValueError (or InvalidArgumentError): If ``min_feasibility_probability`` is not a
            scalar in the unit interval :math:`[0, 1]`.
        """
        self._search_space = search_space
        self._objective_tag = objective_tag
        self._constraint_tag = constraint_tag
        self._num_pf_samples = sample_pf_num
        self._ref_pts_setting = reference_point_setting
        self._extreme_cons_ref_value = extreme_cons_ref_value
        self._moo_solver = moo_solver
        self._pf_samples: Optional[list] = None  # sampled pareto frontier
        self._pf_cons: Optional[list] = None
        self._pf_samples_inputs: Optional[
            list
        ] = None  # sampled pareto frontier corresponding input
        self._partitioned_bounds = None
        self._pop_size = population_size_for_approx_pf_searching
        self._moo_iter = moo_iter_for_approx_pf_searching
        self._discretize_input_sample_size = discretize_input_sample_size
        self._obj_wise_min_samples = []
        self._con_wise_max_samples = []

    def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModelType],
            datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        prepare parallel feasible pareto frontier entropy search acquisition function
        :param datasets
        :param models
        """
        obj_model = models[self._objective_tag]
        cons_model = models[self._constraint_tag]
        obj_number = len(obj_model._models)
        cons_number = len(cons_model._models)
        _constraint_threshold = tf.zeros(
            shape=cons_number, dtype=datasets[self._objective_tag].query_points.dtype
        )
        current_pf_x, current_pf_y = extract_pf_from_data(
            datasets, objective_tag=self._objective_tag, constraint_tag=self._constraint_tag
        )
        # Note: constrain threshold is not used in sample Feasible PF
        self._pf_samples, self._pf_cons = (
            sample_pareto_fronts_from_parametric_gp_posterior(  # sample pareto frontier
                obj_model,
                obj_number,
                constraint_models=cons_model,
                cons_num=cons_number,
                sample_pf_num=self._num_pf_samples,
                search_space=self._search_space,
                popsize=self._pop_size,
                num_moo_iter=self._moo_iter,
                moo_solver=self._moo_solver,
                discretize_input_sample_size=self._discretize_input_sample_size,
                reference_pf_inputs=current_pf_x,
                return_pf_input=False,
                return_pf_constraints=True
            )
        )
        # tf.print("cMOO optimization finished")

        for pf_sample, con_sample in zip(self._pf_samples, self._pf_cons):  # pf_samples: [N, obj_number]
            self._obj_wise_min_samples.append(tf.reduce_min(pf_sample, axis=0))
            self._con_wise_max_samples.append(tf.reduce_max(con_sample, axis=0))
        return maximum_entropy_search_multi_objective_constraint(
            obj_model, cons_model, self._obj_wise_min_samples, self._con_wise_max_samples)


class maximum_entropy_search_multi_objective_constraint(AcquisitionFunctionClass):
    def __init__(self, objective_model: ModelStack, constraint_model: ModelStack,
                 obj_wise_min_samples: list[TensorType], constraint_wise_max_samples: list[TensorType]):
        self._obj_model = objective_model
        self._con_model = constraint_model
        self._obj_wise_min_samples = tf.constant(
            tf.expand_dims(tf.transpose(tf.stack(obj_wise_min_samples)), -1)
        )  # [D, sample_num, 1]
        self._con_wise_max_samples = tf.constant(
            tf.expand_dims(tf.transpose(tf.stack(constraint_wise_max_samples)), -1)
        )  # [D, sample_num, 1]
        self._single_obj_mes_stack = [
                                         min_value_entropy_search(obj_model, obj_min_samples)
                                         for obj_model, obj_min_samples in
                                         zip(self._obj_model._models, self._obj_wise_min_samples)
                                     ] + \
                                     [
                                         max_value_entropy_search(con_model, con_max_samples)
                                         for con_model, con_max_samples in
                                         zip(self._con_model._models, self._con_wise_max_samples)
                                     ]

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        """
        Calculation of MESMO
        note: since MESMO is the average of summation of mes through PF samples, we
        alternatively do summation of average into each mes itself so that we can use
        the origin mes implementation
        """
        single_obj_mes_val_stack = tf.stack(
            [_single_obj_mes(x) for _single_obj_mes in self._single_obj_mes_stack], axis=-1
        )  # [N, 1, D]
        return tf.reduce_sum(single_obj_mes_val_stack, axis=-1)


def vectored_probability_of_feasibility(model: ProbabilisticModel, threshold: float | TensorType
                                        ) -> AcquisitionFunction:
    r"""
    > 0 is feasible
    """

    @tf.function
    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, var = model.predict(tf.squeeze(x, -2))
        distr = tfp.distributions.Normal(mean, tf.sqrt(var))
        tf.debugging.assert_all_finite(
            tf.reduce_prod((1 - distr.cdf(tf.cast(threshold, x.dtype))), -1, keepdims=True),
            f'PoF has NaN: {tf.reduce_prod((1 - distr.cdf(tf.cast(threshold, x.dtype))), -1, keepdims=True)}')
        return tf.reduce_prod((1 - distr.cdf(tf.cast(threshold, x.dtype))), -1, keepdims=True)

    return acquisition


class CEHVI(AcquisitionFunctionBuilder):
    def __init__(self, objective_tag, constraint_tag, min_feasibility_probability=0.5,
                 reference_point_spec: Sequence[float]
                                       | TensorType
                                       | Callable[..., TensorType] = get_reference_point):
        self._objective_tag = objective_tag
        self._constraint_tag = constraint_tag
        self._min_feasibility_probability = min_feasibility_probability
        self._ref_point_specification = reference_point_spec

    def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModelType],
            datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer. Must be populated.
        :param models: The models over each dataset in ``datasets``.
        :return: The expected constrained hypervolume improvement acquisition function.
            This function will raise :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError`
            if used with a batch size greater than one.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]
        cons_num = tf.shape(datasets[self._constraint_tag].observations)[-1]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Expected hypervolume improvement is defined with respect to existing points in"
                    " the objective data, but the objective data is empty.",
        )

        # current_obs = objective_model.predict(objective_dataset.query_points)[0]
        constraint_fn = vectored_probability_of_feasibility(models[self._constraint_tag], tf.zeros(cons_num))
        pof = constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = tf.reduce_all(pof >= self._min_feasibility_probability, axis=-1)

        feasible_query_points = tf.boolean_mask(objective_dataset.query_points, is_feasible)
        feasible_mean, _ = objective_model.predict(feasible_query_points)
        _pf = Pareto(feasible_mean)

        if isinstance(self._ref_point_specification, Callable):

            if not tf.reduce_any(is_feasible):
                return constraint_fn

            # _reference_pt = self._ref_point_specification(_pf.front)
            _reference_pt = self._ref_point_specification(feasible_mean)
            # assert tf.reduce_all(tf.less_equal(_pf.front, _reference_pt)), ValueError(
            #     "There exists pareto frontier point that not dominated by reference point."
            # )
        else:
            self._ref_point_specification = tf.convert_to_tensor(
                self._ref_point_specification, dtype=objective_dataset.query_points.dtype)
            assert isinstance(
                self._ref_point_specification, tf.Tensor
            )  # specified a fixed ref point
            _reference_pt = self._ref_point_specification
        # screen out those not within reference point
        screened_front = _pf.front[tf.reduce_all(tf.less_equal(_pf.front, _reference_pt), -1)]
        # print(f'screened_front: {screened_front}')
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            _reference_pt,
            screened_front
        )
        ehvi = expected_hv_improvement(objective_model, _partition_bounds)
        return lambda at: ehvi(at) * constraint_fn(at)


class BatchMonteCarloConstrainedExpectedHypervolumeImprovement(
    AcquisitionFunctionBuilder[HasReparamSampler]
):
    """
    Builder for the batch expected constrained hypervolume improvement acquisition function.
    The implementation of the acquisition function largely
    follows :cite:`daulton2020differentiable`

    It is assumed in `daulton2020differentiable` that a reference point is known beforehand.
    Hence, when a dynamic reference point updating strategy is specified and there is no posterior
    mean-based feasible observations yet, we use the same strategy as in
    `ExpectedConstrainedHypervolumeImprovement` to search for a feasible observation first.
    """

    def __init__(
            self,
            objective_tag: str,
            constraint_tag: str,
            sample_size: int,
            *,
            qMC: bool = True,
            jitter: float = DEFAULTS.JITTER,
            min_feasibility_probability=0.5,
            reference_point_spec: Sequence[float]
                                  | TensorType
                                  | Callable[..., TensorType] = get_constraint_moo_reference_point,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_tag: The tag for the constraint data and model.
        :param sample_size: The number of samples from model predicted distribution for
            each batch of points.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :param reference_point_spec: this method is used to determine how the reference point is
            calculated. If a Callable function specified, it is expected to take existing posterior
            mean-based feasible observations (to screen out the observation noise) and return a
            reference point with shape [D] (D represents number of objectives). If the feasible
            Pareto front location is known, this arg can be used to specify a fixed reference
            point in each bo iteration. A dynamic reference point updating strategy is used by
            default to set a reference point according to the datasets.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter
        self._objective_tag = objective_tag
        self._constraint_tag = constraint_tag
        self._ref_point_spec = reference_point_spec
        self._min_feasibility_probability = min_feasibility_probability
        self._ref_point: Optional[TensorType] = None
        self.obj_sampler = None
        self.con_sampler = None
        self._constraint_fn: Optional[AcquisitionFunction] = None
        self._qMC = qMC

    def __repr__(self) -> str:
        """"""
        if callable(self._ref_point_spec):
            return (
                f"BatchMonteCarloConstrainedExpectedHypervolumeImprovement("
                f"{self._objective_tag!r},"
                f"{self._constraint_tag!r},"
                f"{self._sample_size!r},"
                f" {self._ref_point_spec.__name__},"
                f" jitter={self._jitter!r})"
            )
        else:
            return (
                f"BatchMonteCarloConstrainedExpectedHypervolumeImprovement("
                f"{self._objective_tag!r},"
                f"{self._constraint_tag!r},"
                f"{self._sample_size!r},"
                f" {self._ref_point_spec!r}"
                f" jitter={self._jitter!r})"
            )

    def prepare_acquisition_function(
            self,
            models: Mapping[str, ProbabilisticModelType],
            datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models: The models over each tag.
        :param datasets: The data from the observer.
        :return: The batch expected constraint hypervolume improvement acquisition function.
        """

        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]
        cons_num = tf.shape(datasets[self._constraint_tag].observations)[-1]
        cons_model = models[self._constraint_tag]

        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Expected hypervolume improvement is defined with respect to existing points in"
                    " the objective data, but the objective data is empty.",
        )

        self._constraint_fn = vectored_probability_of_feasibility(
            models[self._constraint_tag], tf.zeros(cons_num)
        )

        pof = self._constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = tf.reduce_all(pof >= self._min_feasibility_probability, axis=-1)

        feasible_query_points = tf.boolean_mask(objective_dataset.query_points, is_feasible)
        feasible_mean, _ = objective_model.predict(feasible_query_points)

        if callable(self._ref_point_spec):
            if tf.reduce_any(is_feasible):
                self._ref_point = self._ref_point_spec(
                    feasible_mean,  has_feasible_observations = tf.reduce_any(is_feasible))
            else:  # if no feasible point is specified
                observations = objective_model.predict(objective_dataset.query_points)[0]
                self._ref_point = self._ref_point_spec(
                    observations,  has_feasible_observations = tf.reduce_any(is_feasible))
        else:
            self._ref_point = tf.cast(
                self._ref_point_spec, dtype=objective_dataset.query_points.dtype
            )

        _pf = Pareto(feasible_mean)
        screened_front = _pf.front[tf.reduce_all(_pf.front <= self._ref_point, -1)]
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = prepare_default_non_dominated_partition_bounds(
            self._ref_point,
            screened_front,
        )

        if not isinstance(objective_model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo constrained expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {objective_model.__repr__()}"
            )

        if not isinstance(cons_model, HasReparamSampler):
            raise ValueError(
                f"The batch Monte-Carlo constrained expected hyper-volume improvement function only supports "
                f"models that implement a reparam_sampler method; received {objective_model.__repr__()}"
            )

        obj_sampler = objective_model.reparam_sampler(self._sample_size)
        cons_sampler = cons_model.reparam_sampler(self._sample_size)

        return batch_echvi(
            obj_sampler,
            cons_sampler,
            self._jitter,
            _partition_bounds,
            qMC = self._qMC
        )


def batch_echvi(
        obj_sampler: BatchReparametrizationSampler[HasReparamSampler],
        con_sampler: BatchReparametrizationSampler[HasReparamSampler],
        sampler_jitter: float,
        partition_bounds: tuple[TensorType, TensorType],
        epsilon: TensorType = 1e-3,  # this value is borrowed from botoch
        qMC: bool = True
) -> AcquisitionFunction:
    """
    :param obj_sampler: The posterior sampler, which given query points `at`, is able to sample
        the possible observations at 'at'.
    :param con_sampler
    :param sampler_jitter: The size of the jitter to use in sampler when stabilising the Cholesky
        decomposition of the covariance matrix.
    :param partition_bounds: with shape ([N, D], [N, D]), partitioned non-dominated hypercell
        bounds for hypervolume improvement calculation
    :param epsilon
    :return: The batch constrained expected hypervolume improvement acquisition
        function for objective minimisation.
    """

    def acquisition(at: TensorType) -> TensorType:
        _batch_size = at.shape[-2]  # B

        def gen_q_subset_indices(q: int) -> tf.RaggedTensor:
            # generate all subsets of [1, ..., q] as indices
            indices = list(range(q))
            return tf.ragged.constant([list(combinations(indices, i)) for i in range(1, q + 1)])

        obj_samples = obj_sampler.sample(at, jitter=sampler_jitter, qMC = qMC)  # [..., S, B, num_obj]
        con_samples = con_sampler.sample(at, jitter=sampler_jitter, qMC = qMC)  # [..., S, B, num_obj]

        q_subset_indices = gen_q_subset_indices(_batch_size)

        hv_contrib = tf.zeros(tf.shape(obj_samples)[:-2], dtype=obj_samples.dtype)
        lb_points, ub_points = partition_bounds

        def hv_contrib_on_samples(
                obj_smp: TensorType, con_smp: TensorType
        ) -> TensorType:  # calculate samples overlapped area's hvi for obj_samples
            """
            :param obj_smp [..., S, Cq_j, j, num_obj]
            :param con_smp [..., S, Cq_j, j, num_obj]
            """
            # [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j, num_obj]
            overlap_vertices = tf.reduce_max(obj_smp, axis=-2)
            # [..., S, Cq_j, j, num_obj] -> [..., S, Cq_j]
            con_indicator = tf.reduce_prod(tf.reduce_prod(tf.sigmoid(con_smp / epsilon), -1), -1)
            overlap_vertices = tf.maximum(  # compare overlap vertices and lower bound of each cell:
                tf.expand_dims(overlap_vertices, -3),  # expand a cell dimension
                lb_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :],
            )  # [..., S, K, Cq_j, num_obj]

            lengths_j = tf.maximum(  # get hvi length per obj within each cell
                (ub_points[tf.newaxis, tf.newaxis, :, tf.newaxis, :] - overlap_vertices), 0.0
            )  # [..., S, K, Cq_j, num_obj]

            cells_contrib = tf.reduce_prod(lengths_j, axis=-1)  # [..., S, K, Cq_j]
            all_contrib_per_sub_j = tf.reduce_sum(cells_contrib, -2)  # [..., S, Cq_j]
            return tf.reduce_sum(con_indicator * all_contrib_per_sub_j, -1)

        for j in tf.range(1, _batch_size + 1):  # Inclusion-Exclusion loop
            q_choose_j = tf.gather(q_subset_indices, j - 1).to_tensor()
            # gather all combinations having j points from q batch points (Cq_j)
            j_sub_obj_samples = tf.gather(
                obj_samples, q_choose_j, axis=-2
            )  # [..., S, Cq_j, j, num_obj]
            j_sub_con_samples = tf.gather(
                con_samples, q_choose_j, axis=-2
            )  # [..., S, Cq_j, j, num_obj]
            hv_contrib += tf.cast((-1) ** (j + 1), dtype=obj_samples.dtype) * hv_contrib_on_samples(
                j_sub_obj_samples, j_sub_con_samples
            )

        return tf.reduce_mean(hv_contrib, axis=-1, keepdims=True)  # average through MC

    return acquisition
