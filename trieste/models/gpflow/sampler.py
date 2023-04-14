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
This module is the home of the sampling functionality required by Trieste's
GPflow wrappers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, TypeVar, Union, cast

import tensorflow as tf
import tensorflow_probability as tfp
from gpflux.layers.basis_functions.fourier_features import RandomFourierFeaturesCosine
from gpflux.math import compute_A_inv_b
from typing_extensions import Protocol, runtime_checkable
from ...utils.sampler import QuasiMonteCarloNormalSampler

from ...types import TensorType
from ...utils import DEFAULTS, flatten_leading_dims
from ..interfaces import (
    ProbabilisticModel,
    ReparametrizationSampler,
    SupportsGetInducingVariables,
    SupportsGetInternalData,
    SupportsGetKernel,
    SupportsGetMeanFunction,
    SupportsGetObservationNoise,
    SupportsPredictJoint,
    TrajectoryFunction,
    TrajectoryFunctionClass,
    TrajectorySampler,
)
from ..utils.manipulating import safer_cholesky
from tensorflow import linalg


class IndependentReparametrizationSampler(ReparametrizationSampler[ProbabilisticModel]):
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`ProbabilisticModel`\ 's predictive distribution as

    .. math:: x \mapsto \mu(x) + \epsilon \sigma(x)

    where :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring
    samples form a continuous curve.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples to take at each point. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        super().__init__(sample_size, model)

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None]
        )  # [S, 0]

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`IndependentReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`IndependentReparametrizationSampler` instances will produce different samples.

        :param at: Where to sample the predictive distribution, with shape `[..., 1, D]`, for points
            of dimension `D`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :return: The samples, of shape `[..., S, 1, L]`, where `S` is the `sample_size` and `L` is
            the number of latent model dimensions.
        :raise ValueError (or InvalidArgumentError): If ``at`` has an invalid shape or ``jitter``
            is negative.
        """
        tf.debugging.assert_shapes([(at, [..., 1, None])])
        tf.debugging.assert_greater_equal(jitter, 0.0)

        mean, var = self._model.predict(at[..., None, :, :])  # [..., 1, 1, L], [..., 1, 1, L]
        var = var + jitter

        if not self._initialized:
            self._eps.assign(
                tf.random.normal([self._sample_size, tf.shape(mean)[-1]], dtype=tf.float64)
            )  # [S, L]
            self._initialized.assign(True)

        return mean + tf.sqrt(var) * tf.cast(self._eps[:, None, :], var.dtype)  # [..., S, 1, L]


def _cholesky(matrix):
    """Return a Cholesky factor and boolean success."""
    try:
        chol = tf.linalg.cholesky(matrix)
        ok = tf.reduce_all(tf.math.is_finite(chol))
        return chol, ok
    except tf.errors.InvalidArgumentError:
        return matrix, False


def cholesky(matrix, max_attempts: int = 10, jitter: float = 1e-6):
    def update_diag(matrix, jitter):
        diag = tf.linalg.diag_part(matrix)
        diag_add = tf.ones_like(diag) * jitter
        new_diag = diag_add + diag
        new_matrix = tf.linalg.set_diag(matrix, new_diag)
        return new_matrix

    def cond(state):
        return state[0]

    def body(state):
        _, matrix, jitter, _ = state
        res, ok = _cholesky(matrix)
        new_matrix = tf.cond(ok, lambda: matrix, lambda: update_diag(matrix, jitter))
        break_flag = tf.logical_not(ok)
        return [(break_flag, new_matrix, jitter * 10, res)]

    jitter = tf.cast(jitter, matrix.dtype)
    init_state = (True, update_diag(matrix, jitter), jitter, matrix)
    result = tf.while_loop(cond, body, [init_state], maximum_iterations=max_attempts)

    return result[-1][-1]


class BatchReparametrizationSampler(ReparametrizationSampler[SupportsPredictJoint]):
    r"""
    This sampler employs the *reparameterization trick* to approximate batches of samples from a
    :class:`ProbabilisticModel`\ 's predictive joint distribution as

    .. math:: x \mapsto \mu(x) + \epsilon L(x)

    where :math:`L` is the Cholesky factor s.t. :math:`LL^T` is the covariance, and
    :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring samples
    form a continuous curve.
    """

    def __init__(self, sample_size: int, model: SupportsPredictJoint):
        """
        :param sample_size: The number of samples for each batch of points. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        super().__init__(sample_size, model)
        if not isinstance(model, SupportsPredictJoint):
            raise NotImplementedError(
                f"BatchReparametrizationSampler only works with models that support "
                f"predict_joint; received {model.__repr__()}"
            )

        # _eps is essentially a lazy constant. It is declared and assigned an empty tensor here, and
        # populated on the first call to sample
        self._eps = tf.Variable(
            tf.ones([0, 0, sample_size], dtype=tf.float64), shape=[None, None, sample_size]
        )  # [0, 0, S]

    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER, max_trial: int = 10,
               qMC: bool = False, seed: Optional = None) -> TensorType:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`BatchReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`BatchReparametrizationSampler` instances will produce different samples.

        If a Cholesky Decomposition Error happens, we try to remove point that is close together to possibly raise the
        Cholesky issue

        :param at: Batches of query points at which to sample the predictive distribution, with
            shape `[..., B, D]`, for batches of size `B` of points of dimension `D`. Must have a
            consistent batch size across all calls to :meth:`sample` for any given
            :class:`BatchReparametrizationSampler`.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :param max_trial: if Cholesky decomposition has failed, rerun the Cholesky decomposition with a larger Jitter
            till success if the max_trial number for Cholesky decomposition is not reached
        :param qMC: whether to enable quasi-Monte Carlo Sampling for the batch sampler
        :return: The samples, of shape `[..., S, B, L]`, where `S` is the `sample_size`, `B` the
            number of points per batch, and `L` the dimension of the model's predictive
            distribution.
        :raise ValueError (or InvalidArgumentError): If any of the following are true:
            - ``at`` is a scalar.
            - The batch size `B` of ``at`` is not positive.
            - The batch size `B` of ``at`` differs from that of previous calls.
            - ``jitter`` is negative.
        """
        tf.debugging.assert_rank_at_least(at, 2)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        batch_size = at.shape[-2]

        tf.debugging.assert_positive(batch_size)

        if self._initialized:
            if qMC is False:
                tf.debugging.assert_equal(
                    batch_size,
                    tf.shape(self._eps)[-2],
                    f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                    f" but previous batch size was {tf.shape(self._eps)[-2]}.",
                )
            else:
                pass

        mean, cov = self._model.predict_joint(at)  # [..., B, L], [..., L, B, B]

        if qMC is False:
            if not self._initialized:
                self._eps.assign(
                    tf.random.normal(
                        [tf.shape(mean)[-1], batch_size, self._sample_size], dtype=tf.float64, seed=seed
                    )  # [L, B, S]
                )
                self._initialized.assign(True)

            identity = tf.eye(batch_size, dtype=cov.dtype)  # [B, B]

            try:
                cov_cholesky = safer_cholesky(cov + jitter * identity)  # [..., L, B, B]
            except:
                _mean, _cov = self._model.predict_joint(at)
                print('Cholesky decomposition failed, try to find the nearest positive definite matrix')
                print('save cholesky failure input')
                import numpy as np
                np.save(f'Cholesky_Failure_Input_Record', at)
                np.save(f'Cholesky_Failure_Cov_plus_jitter', cov + jitter * identity)
                np.save(f'Cholesky_Failure_Cov', cov)
                for model, model_idx in zip(self._model._models, range(len(self._model._models))):
                    np.save(f'Cholesky_Failure_Model_{model_idx}_cache', model._model)
                    np.save(f'Cholesky_Failure_Model_{model_idx}_Input', model._model.data[0])
                    np.save(f'Cholesky_Failure_Model_{model_idx}_Output', model._model.data[1])
                    np.save(f'Cholesky_Failure_Model_{model_idx}_Lengthscales', model._model.kernel.lengthscales)
                    np.save(f'Cholesky_Failure_Model_{model_idx}_kernel_variance', model._model.kernel.variance)
                    np.save(f'Cholesky_Failure_Model_{model_idx}_likelihood_variance', model._model.likelihood.variance)
                    np.save(f'Cholesky_Failure_Model_{model_idx}_mean_function_val', model._model.mean_function.c)

                raise ValueError
            cov_cholesky = tf.where(tf.math.is_nan(cov_cholesky),
                                    tf.zeros_like(cov_cholesky, dtype=cov_cholesky.dtype), cov_cholesky)
            # tf.debugging.assert_all_finite(cov_cholesky, message=f'Cholesky decomposition {cov_cholesky} have NaN')

            variance_contribution = cov_cholesky @ tf.cast(self._eps, cov.dtype)  # [..., L, B, S]
            tf.debugging.assert_all_finite(variance_contribution,
                                           message=f'variance_contribution {variance_contribution} have NaN')
            leading_indices = tf.range(tf.rank(variance_contribution) - 3)
            absolute_trailing_indices = [-1, -2, -3] + tf.rank(variance_contribution)
            new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)

            return mean[..., None, :, :] + tf.transpose(variance_contribution, new_order)
        else:  # enable qMC sampler
            if not self._initialized:
                # The cholesky matrix for decomposition is BL * BL
                # note the 1st dim is just used for align with the specification of eps shape, its not used in reality
                self._eps.assign(  # NOTE!!! This is not exactly the same shape as the eps used in naive Monte Carlo!
                    tf.transpose(QuasiMonteCarloNormalSampler(dimensionality=batch_size * tf.shape(mean)[-1]).sample(
                        self._sample_size, dtype=tf.float64, seed=seed))[None, ...])  # [1, BL, S])
                # self._eps.assign(tf.random.normal([1, batch_size * tf.shape(mean)[
                #     -1], self._sample_size], dtype = tf.float64, seed = seed))  # [1, BL, S])
                self._initialized.assign(True)
            # We need to sample q * M in a whole, refer appendix C of qEHVI paper
            # first we construct the full covariance matrix [..., BL, BL], which is a block matrix with block size B * B
            splitted_cov = tf.split(cov, axis=-3, num_or_size_splits=cov.shape[-3]) # split along output dimension
            linop_blocks = [linalg.LinearOperatorFullMatrix(tf.squeeze(block, axis=-3)) for block in splitted_cov]
            aug_cov = linalg.LinearOperatorBlockDiag(linop_blocks).to_dense() # [..., B * L, B * L]
            aug_identity = tf.eye(batch_size * tf.shape(mean)[-1], dtype=aug_cov.dtype)
            cov_cholesky = safer_cholesky(aug_cov + jitter * aug_identity)  # [..., BL, BL]
            cov_cholesky = tf.where(tf.math.is_nan(cov_cholesky),
                                    tf.zeros_like(cov_cholesky, dtype=cov_cholesky.dtype), cov_cholesky)
            tf.debugging.assert_all_finite(cov_cholesky, message='qMC sampler covariance decomposition has NaN')
            # [..., BL, BL] * [..., BL, S] -> [..., BL, S]
            variance_contribution = cov_cholesky @ tf.cast(self._eps[0], cov.dtype)
            # [..., B, L] -> [..., BL]
            aug_mean = tf.reshape(
                tf.transpose(mean, perm= tf.concat([tf.range(tf.rank(mean) - 2), [-1, -2] + tf.rank(mean)], axis=0)),
                                  shape=tf.concat([mean.shape[:-2], [batch_size * tf.shape(mean)[-1]]], axis=0))
            aug_sample = tf.expand_dims(aug_mean, axis=-1) + variance_contribution # [..., BL, S]
            # [..., BL, S] -> [..., S, BL]
            leading_indices = tf.range(tf.rank(variance_contribution) - 2)
            absolute_trailing_indices = [-1, -2] + tf.rank(variance_contribution)
            new_order = tf.concat([leading_indices, absolute_trailing_indices], axis=0)
            aug_sample = tf.transpose(aug_sample, perm=new_order)
            # [..., S, BL] -> [..., S, B, L]
            new_shape = tf.concat([tf.shape(aug_sample)[:-1], [tf.shape(mean)[-1]], [batch_size]], axis=0)
            # return tf.reshape(aug_sample, shape=new_shape)
            return tf.transpose(tf.reshape(aug_sample, shape=new_shape),
                                perm=tf.concat([tf.range(tf.shape(new_shape) - 2), [-1, -2] + tf.shape(new_shape)], axis=0))


@runtime_checkable
class FeatureDecompositionInternalDataModel(
    SupportsGetKernel,
    SupportsGetMeanFunction,
    SupportsGetObservationNoise,
    SupportsGetInternalData,
    Protocol,
):
    """
    A probabilistic model that supports get_kernel, get_mean_function, get_observation_noise
    and get_internal_data methods.
    """

    pass


@runtime_checkable
class FeatureDecompositionInducingPointModel(
    SupportsGetKernel, SupportsGetMeanFunction, SupportsGetInducingVariables, Protocol
):
    """
    A probabilistic model that supports get_kernel, get_mean_function
    and get_inducing_point methods.
    """

    pass


FeatureDecompositionTrajectorySamplerModel = Union[
    FeatureDecompositionInducingPointModel,
    FeatureDecompositionInternalDataModel,
]

FeatureDecompositionTrajectorySamplerModelType = TypeVar(
    "FeatureDecompositionTrajectorySamplerModelType",
    bound=FeatureDecompositionTrajectorySamplerModel,
    contravariant=True,
)


class FeatureDecompositionTrajectorySampler(
    TrajectorySampler[FeatureDecompositionTrajectorySamplerModelType],
    ABC,
):
    r"""

    This is a general class to build functions that approximate a trajectory sampled from an
    underlying Gaussian process model.

    In particular, we approximate the Gaussian processes' posterior samples as the finite feature
    approximation

    .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i

    where :math:`\phi_i` are m features and :math:`\theta_i` are feature weights sampled from a
    given distribution

    Achieving consistency (ensuring that the same sample draw for all evalutions of a particular
    trajectory function) for exact sample draws from a GP is prohibitively costly because it scales
    cubically with the number of query points. However, finite feature representations can be
    evaluated with constant cost regardless of the required number of queries.
    """

    def __init__(
            self,
            model: FeatureDecompositionTrajectorySamplerModelType,
            feature_functions: ResampleableRandomFourierFeatureFunctions,
    ):
        """
        :param model: The model to sample from.
        :raise ValueError: If ``dataset`` is empty.
        """

        super().__init__(model)
        self._feature_functions = feature_functions
        self._weight_sampler: Optional[Callable[[int], TensorType]] = None  # lazy init
        self._mean_function = model.get_mean_function()

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}(
        {self._model!r},
        {self._feature_functions!r})
        """

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by sampling weights
        and evaluating the feature functions.

        :return: A trajectory function representing an approximate trajectory from the Gaussian
            process, taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """

        weight_sampler = self._prepare_weight_sampler()  # prep feature weight distribution

        return feature_decomposition_trajectory(
            feature_functions=self._feature_functions,
            weight_sampler=weight_sampler,
            mean_function=self._mean_function,
        )

    def update_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently update a :const:`TrajectoryFunction` to reflect an update in its
        underlying :class:`ProbabilisticModel` and resample accordingly.

        For a :class:`FeatureDecompositionTrajectorySampler`, updating the sampler
        corresponds to resampling the feature functions (taking into account any
        changed kernel parameters) and recalculating the weight distribution.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        tf.debugging.Assert(
            isinstance(trajectory, feature_decomposition_trajectory), [tf.constant([])]
        )

        self._feature_functions.resample()  # resample Fourier feature decomposition
        weight_sampler = self._prepare_weight_sampler()  # recalculate weight distribution

        cast(feature_decomposition_trajectory, trajectory).update(weight_sampler=weight_sampler)

        return trajectory  # return trajectory with updated features and weight distribution

    def resample_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently resample a :const:`TrajectoryFunction` in-place to avoid function retracing
        with every new sample.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        tf.debugging.Assert(
            isinstance(trajectory, feature_decomposition_trajectory), [tf.constant([])]
        )
        cast(feature_decomposition_trajectory, trajectory).resample()
        return trajectory  # return trajectory with resampled weights

    @abstractmethod
    def _prepare_weight_sampler(self) -> Callable[[int], TensorType]:  # [B] -> [L, B]
        """
        Calculate the posterior of the feature weights for the specified feature functions,
        returning a function that takes in a batch size `B` and returns `B` samples for
        the weights of each of the `L` features.
        """
        raise NotImplementedError


class RandomFourierFeatureTrajectorySampler(
    FeatureDecompositionTrajectorySampler[FeatureDecompositionInternalDataModel]
):
    r"""
    This class builds functions that approximate a trajectory sampled from an underlying Gaussian
    process model. For tractibility, the Gaussian process is approximated with a Bayesian
    Linear model across a set of features sampled from the Fourier feature decomposition of
    the model's kernel. See :cite:`hernandez2014predictive` for details. Currently we do not
    support models with multiple latent Gaussian processes.

    In particular, we approximate the Gaussian processes' posterior samples as the finite feature
    approximation

    .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i

    where :math:`\phi_i` are m Fourier features and :math:`\theta_i` are
    feature weights sampled from a posterior distribution that depends on the feature values at the
    model's datapoints.

    Our implementation follows :cite:`hernandez2014predictive`, with our calculations
    differing slightly depending on properties of the problem. In particular,  we used different
    calculation strategies depending on the number of considered features m and the number
    of data points n.

    If :math:`m<n` then we follow Appendix A of :cite:`hernandez2014predictive` and calculate the
    posterior distribution for :math:`\theta` following their Bayesian linear regression motivation,
    i.e. the computation revolves around an O(m^3)  inversion of a design matrix.

    If :math:`n<m` then we use the kernel trick to recast computation to revolve around an O(n^3)
    inversion of a gram matrix. As well as being more efficient in early BO
    steps (where :math:`n<m`), this second computation method allows much larger choices
    of m (as required to approximate very flexible kernels).
    """

    def __init__(
            self,
            model: FeatureDecompositionInternalDataModel,
            num_features: int = 1000,
            output_standardization: Optional =  None
    ):
        """
        :param model: The model to sample from.
        :param num_features: The number of features used to approximate the kernel. We use a default
            of 1000 as it typically perfoms well for a wide range of kernels. Note that very smooth
            kernels (e.g. RBF) can be well-approximated with fewer features.
        :param output_standardization: a scaler on model data if the output has been standardized
        :raise ValueError: If ``dataset`` is empty.
        """

        if not isinstance(model, FeatureDecompositionInternalDataModel):
            raise NotImplementedError(
                f"RandomFourierFeatureTrajectorySampler only works with models with "
                f"get_kernel, get_observation_noise and get_internal_data methods; "
                f"but received {model.__repr__()}."
            )

        tf.debugging.assert_positive(num_features)
        self._num_features = num_features
        self._model = model
        self._output_standardization = output_standardization
        feature_functions = ResampleableRandomFourierFeatureFunctions(
            self._model, self._num_features
        )
        super().__init__(self._model, feature_functions)

    def _prepare_weight_sampler(self) -> Callable[[int], TensorType]:  # [B] -> [L, B]
        """
        Calculate the posterior of theta (the feature weights) for the RFFs, returning
        a function that takes in a batch size `B` and returns `B` samples for
        the weights of each of the RFF `L` features.
        """

        dataset = self._model.get_internal_data()
        num_data = tf.shape(dataset.query_points)[0]  # n
        if (
                self._num_features < num_data
        ):  # if m < n  then calculate posterior in design space (an m*m matrix inversion)
            theta_posterior = self._prepare_theta_posterior_in_design_space()
        else:  # if n <= m  then calculate posterior in gram space (an n*n matrix inversion)
            theta_posterior = self._prepare_theta_posterior_in_gram_space()

        return lambda b: theta_posterior.sample(b)

    def _prepare_theta_posterior_in_design_space(self) -> tfp.distributions.MultivariateNormalTriL:
        r"""
        Calculate the posterior of theta (the feature weights) in the design space. This
        distribution is a Gaussian

        .. math:: \theta \sim N(D^{-1}\Phi^Ty,D^{-1}\sigma^2)

        where the [m,m] design matrix :math:`D=(\Phi^T\Phi + \sigma^2I_m)` is defined for
        the [n,m] matrix of feature evaluations across the training data :math:`\Phi`
        and observation noise variance :math:`\sigma^2`.
        """
        dataset = self._model.get_internal_data()
        phi = self._feature_functions(dataset.query_points)  # [n, m]
        D = tf.matmul(phi, phi, transpose_a=True)  # [m, m]
        s = self._model.get_observation_noise() * tf.eye(self._num_features, dtype=phi.dtype)
        # L = tf.linalg.cholesky(D + s)
        L = safer_cholesky(D + s)
        D_inv = tf.linalg.cholesky_solve(L, tf.eye(self._num_features, dtype=phi.dtype))

        residuals = dataset.observations - self._model.get_mean_function()(dataset.query_points)
        theta_posterior_mean = tf.matmul(D_inv, tf.matmul(phi, residuals, transpose_a=True))[
                               :, 0
                               ]  # [m,]
        theta_posterior_chol_covariance = safer_cholesky(
            D_inv * self._model.get_observation_noise()
        )  # [m, m]

        return tfp.distributions.MultivariateNormalTriL(
            theta_posterior_mean, theta_posterior_chol_covariance
        )

    def _prepare_theta_posterior_in_gram_space(self) -> tfp.distributions.MultivariateNormalTriL:
        r"""
        Calculate the posterior of theta (the feature weights) in the gram space.

         .. math:: \theta \sim N(\Phi^TG^{-1}y,I_m - \Phi^TG^{-1}\Phi)

        where the [n,n] gram matrix :math:`G=(\Phi\Phi^T + \sigma^2I_n)` is defined for the [n,m]
        matrix of feature evaluations across the training data :math:`\Phi` and
        observation noise variance :math:`\sigma^2`.
        """
        dataset = self._model.get_internal_data()
        num_data = tf.shape(dataset.query_points)[0]  # n
        phi = self._feature_functions(dataset.query_points)  # [n, m]
        G = tf.matmul(phi, phi, transpose_b=True)  # [n, n]
        s = self._model.get_observation_noise() * tf.eye(num_data, dtype=phi.dtype)
        # L = tf.linalg.cholesky(G + s)
        L = safer_cholesky(G + s)
        L_inv_phi = tf.linalg.triangular_solve(L, phi)  # [n, m]
        residuals = dataset.observations - self._model.get_mean_function()(
            dataset.query_points
        )  # [n, 1]
        L_inv_y = tf.linalg.triangular_solve(L, residuals)  # [n, 1]

        theta_posterior_mean = tf.tensordot(tf.transpose(L_inv_phi), L_inv_y, [[-1], [-2]])[
                               :, 0
                               ]  # [m,]
        theta_posterior_covariance = tf.eye(self._num_features, dtype=phi.dtype) - tf.tensordot(
            tf.transpose(L_inv_phi), L_inv_phi, [[-1], [-2]]
        )  # [m, m]
        # theta_posterior_chol_covariance = tf.linalg.cholesky(theta_posterior_covariance)  # [m, m]
        theta_posterior_chol_covariance = safer_cholesky(theta_posterior_covariance)  # [m, m]

        return tfp.distributions.MultivariateNormalTriL(
            theta_posterior_mean, theta_posterior_chol_covariance
        )

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by sampling weights
        and evaluating the feature functions.

        :return: A trajectory function representing an approximate trajectory from the Gaussian
            process, taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """

        weight_sampler = self._prepare_weight_sampler()  # prep feature weight distribution
        if self._output_standardization is not None:
            return standardized_feature_decomposition_trajectory(
                feature_functions=self._feature_functions,
                weight_sampler=weight_sampler,
                mean_function=self._mean_function,
                output_standardization = self._output_standardization
            )
        else:
            return feature_decomposition_trajectory(
                feature_functions=self._feature_functions,
                weight_sampler=weight_sampler,
                mean_function=self._mean_function,
            )


class DecoupledTrajectorySampler(
    FeatureDecompositionTrajectorySampler[
        Union[
            FeatureDecompositionInducingPointModel,
            FeatureDecompositionInternalDataModel,
        ]
    ]
):
    r"""

    This class builds functions that approximate a trajectory sampled from an underlying Gaussian
    process model using decoupled sampling. See :cite:`wilson2020efficiently` for an introduction
    to decoupled sampling. Currently we do not support models with multiple latent Gaussian
    processes.

    Unlike our :class:`RandomFourierFeatureTrajectorySampler` which uses a RFF decomposition to
    aprroximate the Gaussian process posterior, a :class:`DecoupledTrajectorySampler` only
    uses an RFF decomposition to approximate the Gausian process prior and instead using
    a cannonical decomposition to discretize the effect of updating the prior on the given data.

    In particular, we approximate the Gaussian processes' posterior samples as the finite feature
    approximation

    .. math:: \hat{f}(.) = \sum_{i=1}^L w_i\phi_i(.) + \sum_{j=1}^m v_jk(.,z_j)

    where :math:`\phi_i(.)` and :math:`w_i` are the Fourier features and their weights that
    discretize the prior. In contrast, `k(.,z_j)` and :math:`v_i` are the cannonical features and
    their weights that discretize the data update.

    The expression for :math:`v_i` depends on if we are using an exact Gaussian process or a sparse
    approximations. See  eq. (13) in :cite:`wilson2020efficiently` for details.

    Note that if a model is both of :class:`FeatureDecompositionInducingPointModel` type and
    :class:`FeatureDecompositionInternalDataModel` type,
    :class:`FeatureDecompositionInducingPointModel` will take a priority and inducing points
    will be used for computations rather than data.
    """

    def __init__(
            self,
            model: Union[
                FeatureDecompositionInducingPointModel,
                FeatureDecompositionInternalDataModel,
            ],
            num_features: int = 1000,
    ):
        """
        :param model: The model to sample from.
        :param num_features: The number of features used to approximate the kernel. We use a default
            of 1000 as it typically perfoms well for a wide range of kernels. Note that very smooth
            kernels (e.g. RBF) can be well-approximated with fewer features.
        :raise NotImplementedError: If the model is not of valid type.
        """
        if not isinstance(
                model, (FeatureDecompositionInducingPointModel, FeatureDecompositionInternalDataModel)
        ):
            raise NotImplementedError(
                f"DecoupledTrajectorySampler only works with models that either support "
                f"get_kernel, get_observation_noise and get_internal_data or support get_kernel "
                f"and get_inducing_variables; but received {model.__repr__()}."
            )

        tf.debugging.assert_positive(num_features)
        self._num_features = num_features
        self._model = model
        feature_functions = ResampleableDecoupledFeatureFunctions(self._model, self._num_features)

        super().__init__(self._model, feature_functions)

    def _prepare_weight_sampler(self) -> Callable[[int], TensorType]:
        """
        Prepare the sampler function that provides samples of the feature weights
        for both the RFF and cannonical feature functions, i.e. we return a function
        that takes in a batch size `B` and returns `B` samples for the weights of each of
        the `L`  RFF features and `N` cannonical features.
        """

        if isinstance(self._model, FeatureDecompositionInducingPointModel):
            (  # extract variational parameters
                inducing_points,
                q_mu,
                q_sqrt,
                whiten,
            ) = self._model.get_inducing_variables()  # [M, d], [M, 1], [1, M, 1]
            q_sqrt = q_sqrt[0, :, :]  # [M, M]
            Kmm = self._model.get_kernel().K(inducing_points, inducing_points)  # [M, M]
            Kmm += tf.eye(tf.shape(inducing_points)[0], dtype=Kmm.dtype) * DEFAULTS.JITTER
        else:  # massage quantities from GP to look like variational parameters
            internal_data = self._model.get_internal_data()
            inducing_points = internal_data.query_points  # [M, d]
            q_mu = self._model.get_internal_data().observations  # [M, 1]
            q_mu = q_mu - self._model.get_mean_function()(
                inducing_points
            )  # account for mean function
            q_sqrt = tf.eye(tf.shape(inducing_points)[0], dtype=tf.float64)  # [M, M]
            q_sqrt = tf.math.sqrt(self._model.get_observation_noise()) * q_sqrt
            whiten = False
            Kmm = (
                    self._model.get_kernel().K(inducing_points, inducing_points) + q_sqrt ** 2
            )  # [M, M]

        tf.debugging.assert_shapes(
            [
                (inducing_points, ["M", "d"]),
                (q_mu, ["M", "1"]),
                (q_sqrt, ["M", "M"]),
                (Kmm, ["M", "M"]),
            ]
        )

        def weight_sampler(batch_size: int) -> Tuple[TensorType, TensorType]:

            prior_weights = tf.random.normal(  # Non-RFF features will require scaling here
                [self._num_features, batch_size], dtype=tf.float64
            )  # [L, B]

            u_noise_sample = tf.matmul(
                q_sqrt,  # [M, M]
                tf.random.normal(
                    (tf.shape(inducing_points)[0], batch_size), dtype=tf.float64
                ),  # [ M, B]
            )  # [M, B]

            u_sample = q_mu + u_noise_sample  # [M, B]

            if whiten:
                Luu = tf.linalg.cholesky(Kmm)  # [M, M]
                u_sample = tf.matmul(Luu, u_sample)  # [M, B]

            phi_Z = self._feature_functions(inducing_points)[:, : self._num_features]  # [M, B]
            weight_space_prior_Z = phi_Z @ prior_weights  # [M, B]

            diff = u_sample - weight_space_prior_Z  # [M, B]

            v = compute_A_inv_b(Kmm, diff)  # [M, B]

            tf.debugging.assert_shapes([(v, ["M", "B"]), (prior_weights, ["L", "B"])])

            return tf.transpose(tf.concat([prior_weights, v], axis=0))  # [B, L + M]

        return weight_sampler


class ResampleableRandomFourierFeatureFunctions(RandomFourierFeaturesCosine):
    """
    A wrapper around GPFlux's random Fourier feature function that allows for
    efficient in-place updating when generating new decompositions.

    In particular, we store the bias and weights as variables, which can then be
    updated without triggering expensive graph retracing.

    Note that if a model is both of :class:`FeatureDecompositionInducingPointModel` type and
    :class:`FeatureDecompositionInternalDataModel` type,
    :class:`FeatureDecompositionInducingPointModel` will take a priority and inducing points
    will be used for computations rather than data.
    """

    def __init__(
            self,
            model: Union[
                FeatureDecompositionInducingPointModel,
                FeatureDecompositionInternalDataModel,
            ],
            n_components: int,
    ):
        """
        :param model: The model that will be approximed by these feature functions.
        :param n_components: The desired number of features.
        :raise NotImplementedError: If the model is not of valid type.
        """
        if not isinstance(
                model,
                (
                        FeatureDecompositionInducingPointModel,
                        FeatureDecompositionInternalDataModel,
                ),
        ):
            raise NotImplementedError(
                f"ResampleableRandomFourierFeatureFunctions only work with models that either"
                f"support get_kernel, get_observation_noise and get_internal_data or support "
                f"get_kernel and get_inducing_variables;"
                f"but received {model.__repr__()}."
            )

        self._kernel = model.get_kernel()
        self._n_components = n_components
        super().__init__(self._kernel, self._n_components, dtype=tf.float64)

        if isinstance(model, SupportsGetInducingVariables):
            dummy_X = model.get_inducing_variables()[0][0:1, :]
        else:
            dummy_X = model.get_internal_data().query_points[0:1, :]

        self.__call__(dummy_X)  # dummy call to force init of weights
        self.b: TensorType = tf.Variable(self.b)
        self.W: TensorType = tf.Variable(self.W)  # allow updateable weights

    def resample(self) -> None:
        """
        Resample weights and biases
        """
        self.b.assign(self._bias_init(tf.shape(self.b), dtype=self._dtype))
        self.W.assign(self._weights_init(tf.shape(self.W), dtype=self._dtype))


class ResampleableDecoupledFeatureFunctions(ResampleableRandomFourierFeatureFunctions):
    """
    A wrapper around our :class:`ResampleableRandomFourierFeatureFunctions` which rather
    than evaluates just `L` RFF functions instead evaluates the concatenation of
    `L` RFF functions with evaluations of the cannonical basis functions.

    Note that if a model is both of :class:`FeatureDecompositionInducingPointModel` type and
    :class:`FeatureDecompositionInternalDataModel` type,
    :class:`FeatureDecompositionInducingPointModel` will take a priority and inducing points
    will be used for computations rather than data.
    """

    def __init__(
            self,
            model: Union[
                FeatureDecompositionInducingPointModel,
                FeatureDecompositionInternalDataModel,
            ],
            n_components: int,
    ):
        """
        :param model: The model that will be approximed by these feature functions.
        :param n_components: The desired number of features.
        """

        if isinstance(model, SupportsGetInducingVariables):
            inducing_points = model.get_inducing_variables()[0]  # [M, D]
        else:
            inducing_points = model.get_internal_data().query_points  # [M, D]

        self._cannonical_feature_functions = lambda x: tf.linalg.matrix_transpose(
            model.get_kernel().K(inducing_points, x)
        )

        super().__init__(model, n_components)

    def __call__(self, x: TensorType) -> TensorType:  # [N,D] -> [N, L + M]
        """
        combine prior basis functions with cannonical basis functions
        """
        fourier_feature_eval = super().__call__(x)  # [N, L]
        cannonical_feature_eval = self._cannonical_feature_functions(x)  # [N, M]
        return tf.concat([fourier_feature_eval, cannonical_feature_eval], axis=-1)  # [N, L + M]


class feature_decomposition_trajectory(TrajectoryFunctionClass):
    r"""
    An approximate sample from a Gaussian processes' posterior samples represented as a
    finite weighted sum of features.

    A trajectory is given by

    .. math:: \hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i

    where :math:`\phi_i` are m feature functions and :math:`\theta_i` are
    feature weights sampled from a posterior distribution.

    The number of trajectories (i.e. batch size) is determined from the first call of the
    trajectory. In order to change the batch size, a new :class:`TrajectoryFunction` must be built.
    """

    def __init__(
            self,
            feature_functions: Callable[[TensorType], TensorType],
            weight_sampler: Callable[[int], TensorType],
            mean_function: Callable[[TensorType], TensorType],
    ):
        """
        :param feature_functions: Set of feature function.
        :param weight_sampler: New sampler that generates feature weight samples.
        :param mean_function: The underlying model's mean function.
        """
        self._feature_functions = feature_functions
        self._mean_function = mean_function
        self._weight_sampler = weight_sampler
        self._initialized = tf.Variable(False)

        self._weights_sample = tf.Variable(  # dummy init to be updated before trajectory evaluation
            tf.ones([0, 0], dtype=tf.float64), shape=[None, None]
        )

        self._batch_size = tf.Variable(
            0, dtype=tf.int32
        )  # dummy init to be updated before trajectory evaluation

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:  # [N, B, d] -> [N, B, 1]
        """Call trajectory function."""

        if not self._initialized:  # work out desired batch size from input
            self._batch_size.assign(tf.shape(x)[-2])  # B
            self.resample()  # sample B feature weights
            self._initialized.assign(True)

        tf.debugging.assert_equal(
            tf.shape(x)[-2],
            self._batch_size.value(),
            message=f"""
            This trajectory only supports batch sizes of {self._batch_size}.
            If you wish to change the batch size you must get a new trajectory
            by calling the get_trajectory method of the trajectory sampler.
            """,
        )

        flat_x, unflatten = flatten_leading_dims(x)  # [N*B, d]
        flattened_feature_evaluations = self._feature_functions(flat_x)  # [N*B, m]
        feature_evaluations = unflatten(flattened_feature_evaluations)  # [N, B, m]
        mean = self._mean_function(x)  # account for the model's mean function
        return (
                tf.reduce_sum(feature_evaluations * self._weights_sample, -1, keepdims=True) + mean
        )  # [N, B, 1]

    def resample(self) -> None:
        """
        Efficiently resample in-place without retracing.
        """
        self._weights_sample.assign(  # [B, m]
            self._weight_sampler(self._batch_size)
        )  # resample weights

    def update(self, weight_sampler: Callable[[int], TensorType]) -> None:
        """
        Efficiently update the trajectory with a new weight distribution and resample its weights.

        :param weight_sampler: New sampler that generates feature weight samples.
        """
        self._weight_sampler = weight_sampler  # update weight sampler
        self.resample()  # resample weights


class standardized_feature_decomposition_trajectory(feature_decomposition_trajectory):
    def __init__(self, *args, output_standardization: Optional = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_standardization = output_standardization

    # @tf.function
    def __call__(self, x: TensorType) -> TensorType:  # [N, B, d] -> [N, B, 1]
        """Call trajectory function."""

        if not self._initialized:  # work out desired batch size from input
            self._batch_size.assign(tf.shape(x)[-2])  # B
            self.resample()  # sample B feature weights
            self._initialized.assign(True)

        tf.debugging.assert_equal(
            tf.shape(x)[-2],
            self._batch_size.value(),
            message=f"""
            This trajectory only supports batch sizes of {self._batch_size}.
            If you wish to change the batch size you must get a new trajectory
            by calling the get_trajectory method of the trajectory sampler.
            """,
        )

        flat_x, unflatten = flatten_leading_dims(x)  # [N*B, d]
        flattened_feature_evaluations = self._feature_functions(flat_x)  # [N*B, m]
        feature_evaluations = unflatten(flattened_feature_evaluations)  # [N, B, m]
        mean = self._mean_function(x)  # account for the model's mean function
        return (
                self._output_standardization.backward_mean(tf.reduce_sum(feature_evaluations * self._weights_sample, -1, keepdims=True) + mean)
        )  # [N, B, 1]