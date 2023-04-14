from __future__ import annotations

from typing import Optional, Tuple, Union, cast

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.conditionals.util import sample_mvn
from gpflow.models import GPR
from gpflow.utilities import is_variable, multiple_assign, read_values
from gpflow.utilities.ops import leading_transpose

from ....data import Dataset
from ....types import TensorType
from ...interfaces import (
    FastUpdateModel,
    HasTrajectorySampler,
    SupportsGetInternalData,
    TrainableProbabilisticModel,
    TrajectorySampler,
)
from ...optimizer import Optimizer
from ..interface import GPflowPredictor, SupportsCovarianceBetweenPoints
from ..sampler import DecoupledTrajectorySampler, RandomFourierFeatureTrajectorySampler
from ..utils import (
    assert_data_is_compatible,
    check_optimizer,
    randomize_hyperparameters,
    squeeze_hyperparameters,
)
from .transformer import Standardization


class StandardizedGaussianProcessRegression(
    GPflowPredictor,
    TrainableProbabilisticModel,
    FastUpdateModel,
    SupportsCovarianceBetweenPoints,
    SupportsGetInternalData,
    HasTrajectorySampler,
):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.GPR`.

    As Bayesian optimization requires a large number of sequential predictions (i.e. when maximizing
    acquisition functions), rather than calling the model directly at prediction time we instead
    call the posterior objects built by these models. These posterior objects store the
    pre-computed Gram matrices, which can be reused to allow faster subsequent predictions. However,
    note that these posterior objects need to be updated whenever the underlying model is changed
    by calling :meth:`update_posterior_cache` (this
    happens automatically after calls to :meth:`update` or :math:`optimize`).
    """

    def __init__(
            self,
            model: GPR,
            optimizer: Optimizer | None = None,
            num_kernel_samples: int = 10,
            optimize_restarts: int = 20,
            num_rff_features: int = 1000,
            use_decoupled_sampler: bool = True,
            out_put_transformation = Standardization
    ):
        """
        :param model: The GPflow model to wrap.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :param num_kernel_samples: Number of randomly sampled kernels (for each kernel parameter) to
            evaluate before beginning model optimization. Therefore, for a kernel with `p`
            (vector-valued) parameters, we evaluate `p * num_kernel_samples` kernels.
        :param num_rff_features: The number of random Fourier features used to approximate the
            kernel when calling :meth:`trajectory_sampler`. We use a default of 1000 as it
            typically perfoms well for a wide range of kernels. Note that very smooth
            kernels (e.g. RBF) can be well-approximated with fewer features.
        :param use_decoupled_sampler: If True use a decoupled random Fourier feature sampler, else
            just use a random Fourier feature sampler. The decoupled sampler suffers less from
            overestimating variance and can typically get away with a lower num_rff_features.
        """
        super().__init__(optimizer)
        self._model = model
        self._output_transformation = out_put_transformation()

        check_optimizer(self.optimizer)

        if num_kernel_samples <= 0:
            raise ValueError(
                f"num_kernel_samples must be greater or equal to zero but got {num_kernel_samples}."
            )
        self._num_kernel_samples = num_kernel_samples

        if num_rff_features <= 0:
            raise ValueError(
                f"num_rff_features must be greater or equal to zero but got {num_rff_features}."
            )
        self._num_rff_features = num_rff_features
        self._use_decoupled_sampler = use_decoupled_sampler
        self._ensure_variable_model_data()
        self.create_posterior_cache()
        self.optimize_restarts = optimize_restarts

    def __repr__(self) -> str:
        """"""
        return (
            f"GaussianProcessRegression({self.model!r}, {self.optimizer!r},"
            f"{self._num_kernel_samples!r}, {self._num_rff_features!r},"
            f"{self._use_decoupled_sampler!r})"
        )

    @property
    def model(self) -> GPR:
        return self._model

    def _ensure_variable_model_data(self) -> None:
        # GPflow stores the data in Tensors. However, since we want to be able to update the data
        # without having to retrace the acquisition functions, put it in Variables instead.
        # Data has to be stored in variables with dynamic shape to allow for changes
        # Sometimes, for instance after serialization-deserialization, the shape can be overridden
        # Thus here we ensure data is stored in dynamic shape Variables

        if all(is_variable(x) and x.shape[0] is None for x in self._model.data):
            # both query points and observations are in right shape
            # nothing to do
            return

        self._model.data = (
            tf.Variable(
                self._model.data[0], trainable=False, shape=[None, *self._model.data[0].shape[1:]]
            ),
            tf.Variable(
                self._output_transformation.update_forward(self._model.data[1]),
                trainable=False, shape=[None, *self._model.data[1].shape[1:]]
            ),
        )
        # since we transformed the output data, we reassign mean function and kernel variance
        self._model.mean_function.c.assign(tf.zeros(shape=1, dtype=self._model.data[0].dtype)[0])
        self._model.kernel.variance.assign(tf.ones(shape=1, dtype=self._model.data[0].dtype)[0])

    # TODO: TEST
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        standardized_mean, standardized_cov = (self._posterior or self.model).predict_f(query_points)
        mean = self._output_transformation.backward_mean(standardized_mean)
        cov = self._output_transformation.backward_variance(standardized_cov)
        # posterior predict can return negative variance values [cf GPFlow issue #1813]
        if self._posterior is not None:
            cov = tf.clip_by_value(cov, 1e-12, cov.dtype.max)
        return mean, cov

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        standardized_mean, standardized_cov = (self._posterior or self.model).predict_f(query_points, full_cov=True)
        mean = self._output_transformation.backward_mean(standardized_mean)
        cov = self._output_transformation.backward_variance(standardized_cov)
        # posterior predict can return negative variance values [cf GPFlow issue #1813]
        if self._posterior is not None:
            cov = tf.linalg.set_diag(
                cov, tf.clip_by_value(tf.linalg.diag_part(cov), 1e-12, cov.dtype.max)
            )
        return mean, cov

    # Believed finished, visual checked
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        standardized_mean, standardized_cov = self.model.predict_f(query_points, full_cov=True, full_output_cov=False)
        mean = self._output_transformation.backward_mean(standardized_mean)
        cov = self._output_transformation.backward_variance(standardized_cov)
        # mean: [..., N, P]
        # cov: [..., P, N, N]
        mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
        samples = sample_mvn(
            mean_for_sample, cov, True, num_samples=num_samples
        )  # [..., (S), P, N]
        samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    # Decided not enabled
    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        raise NotImplementedError
        f_mean, f_var = self.predict(query_points)
        return self.model.likelihood.predict_mean_and_var(f_mean, f_var)

    # Believed finished
    def update(self, dataset: Dataset) -> None:
        self._ensure_variable_model_data()

        x, y = self.model.data[0].value(), self.model.data[1].value()

        assert_data_is_compatible(dataset, Dataset(x, y))

        if dataset.query_points.shape[-1] != x.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != y.shape[-1]:
            raise ValueError

        self.model.data[0].assign(dataset.query_points)
        self.model.data[1].assign(self._output_transformation.update_forward(dataset.observations))
        self.update_posterior_cache()

    # TODO:
    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        r"""
        Compute the posterior covariance between sets of query points.

        .. math:: \Sigma_{12} = K_{12} - K_{x1}(K_{xx} + \sigma^2 I)^{-1}K_{x2}

        Note that query_points_2 must be a rank 2 tensor, but query_points_1 can
        have leading dimensions.

        :param query_points_1: Set of query points with shape [..., N, D]
        :param query_points_2: Sets of query points with shape [M, D]
        :return: Covariance matrix between the sets of query points with shape [..., L, N, M]
            (L being the number of latent GPs = number of output dimensions)
        """
        raise NotImplementedError
        tf.debugging.assert_shapes(
            [(query_points_1, [..., "N", "D"]), (query_points_2, ["M", "D"])]
        )

        x = self.model.data[0].value()
        num_data = tf.shape(x)[0]
        s = tf.linalg.diag(tf.fill([num_data], self.model.likelihood.variance))

        K = self.model.kernel(x)  # [num_data, num_data] or [L, num_data, num_data]
        Kx1 = self.model.kernel(query_points_1, x)  # [..., N, num_data] or [..., L, N, num_data]
        Kx2 = self.model.kernel(x, query_points_2)  # [num_data, M] or [L, num_data, M]
        K12 = self.model.kernel(query_points_1, query_points_2)  # [..., N, M] or [..., L, N, M]

        if len(tf.shape(K)) == 2:
            # if single output GPR, the kernel does not return the latent dimension so
            # we add it back here
            K = tf.expand_dims(K, -3)
            Kx1 = tf.expand_dims(Kx1, -3)
            Kx2 = tf.expand_dims(Kx2, -3)
            K12 = tf.expand_dims(K12, -3)
        elif len(tf.shape(K)) > 3:
            raise NotImplementedError(
                "Covariance between points is not supported "
                "for kernels of type "
                f"{type(self.model.kernel)}."
            )

        L = tf.linalg.cholesky(K + s)  # [L, num_data, num_data]

        Kx1 = leading_transpose(Kx1, [..., -1, -2])  # [..., L, num_data, N]
        Linv_Kx1 = tf.linalg.triangular_solve(L, Kx1)  # [..., L, num_data, N]
        Linv_Kx2 = tf.linalg.triangular_solve(L, Kx2)  # [L, num_data, M]

        # The line below is just A^T*B over the last 2 dimensions.
        cov = K12 - tf.einsum("...lji,ljk->...lik", Linv_Kx1, Linv_Kx2)  # [..., L, N, M]

        num_latent = self.model.num_latent_gps
        if cov.shape[-3] == 1 and num_latent > 1:
            # For multioutput GPR with shared kernel, we need to duplicate cov
            # for each output
            cov = tf.repeat(cov, num_latent, axis=-3)

        tf.debugging.assert_shapes(
            [
                (query_points_1, [..., "N", "D"]),
                (query_points_2, ["M", "D"]),
                (cov, [..., "L", "N", "M"]),
            ]
        )

        return cov

    # Done update
    def optimize(self, dataset: Dataset, randomize_model_hyperparam: bool = True) -> None:
        """
        Optimize the model with the specified `dataset`.

        For :class:`GaussianProcessRegression`, we (optionally) try multiple randomly sampled
        kernel parameter configurations as well as the configuration specified when initializing
        the kernel. The best configuration is used as the starting point for model optimization.

        For trainable parameters constrained to lie in a finite interval (through a sigmoid
        bijector), we begin model optimization from the best of a random sample from these
        parameters' acceptable domains.

        For trainable parameters without constraints but with priors, we begin model optimization
        from the best of a random sample from these parameters' priors.

        For trainable parameters with neither priors nor constraints, we begin optimization from
        their initial values.

        :param dataset: The data with which to optimize the `model`.
        """

        num_trainable_params_with_priors_or_constraints = tf.reduce_sum(
            [
                tf.size(param)
                for param in self.model.trainable_parameters
                if param.prior is not None or isinstance(param.bijector, tfp.bijectors.Sigmoid)
            ]
        )
        # TODO: Note this hyperparam need to be set on the scaled data!
        if (
            min(num_trainable_params_with_priors_or_constraints, self._num_kernel_samples) >= 1
        ) and randomize_model_hyperparam and True == False:  # Find a promising kernel initialization
            self.find_best_model_initialization(
                self._num_kernel_samples * num_trainable_params_with_priors_or_constraints
            )

        hyper_opt_success = False
        for i in range(self.optimize_restarts):
            if i > 0:
                randomize_hyperparameters(self.model)
            try:
                self.optimizer.optimize(self.model,
                                        Dataset(dataset.query_points,
                                                self._output_transformation.forward_mean(dataset.observations)))
                self.update_posterior_cache()
                hyper_opt_success = True
                break
            except tf.errors.InvalidArgumentError:  # pragma: no cover
                print(f"Warning: optimization restart {i + 1}/{self.optimize_restarts} failed")
        if not hyper_opt_success:
            raise RuntimeError("All model hyperparameter optimization restarts failed, exiting.")

    # No need to update
    def find_best_model_initialization(self, num_kernel_samples: int) -> None:
        """
        Test `num_kernel_samples` models with sampled kernel parameters. The model's kernel
        parameters are then set to the sample achieving maximal likelihood.

        :param num_kernel_samples: Number of randomly sampled kernels to evaluate.
        """

        @tf.function
        def evaluate_loss_of_model_parameters() -> tf.Tensor:
            randomize_hyperparameters(self.model)
            return self.model.training_loss()

        squeeze_hyperparameters(self.model)
        current_best_parameters = read_values(self.model)
        min_loss = self.model.training_loss()

        for _ in tf.range(num_kernel_samples):
            try:
                train_loss = evaluate_loss_of_model_parameters()
            except tf.errors.InvalidArgumentError:  # allow badly specified kernel params
                train_loss = 1e100

            if train_loss < min_loss:  # only keep best kernel params
                min_loss = train_loss
                current_best_parameters = read_values(self.model)

        multiple_assign(self.model, current_best_parameters)

    # Done for RFF
    def trajectory_sampler(self) -> TrajectorySampler[StandardizedGaussianProcessRegression]:
        """
        Return a trajectory sampler. For :class:`GaussianProcessRegression`, we build
        trajectories using a random Fourier feature approximation.

        At the moment only models with single latent GP are supported.

        :return: The trajectory sampler.
        :raise NotImplementedError: If we try to use the
            sampler with a model that has more than one latent GP.
        """
        if self.model.num_latent_gps > 1:
            raise NotImplementedError(
                f"""
                Trajectory sampler does not currently support models with multiple latent
                GPs, however received a model with {self.model.num_latent_gps} latent GPs.
                """
            )

        if self._use_decoupled_sampler:
            raise NotImplementedError
            return DecoupledTrajectorySampler(self, self._num_rff_features)
        else:
            return RandomFourierFeatureTrajectorySampler(
                self, self._num_rff_features, output_standardization=self._output_transformation)

    # Believed Finished
    def get_internal_data(self, get_standardized_output: bool = False) -> Dataset:
        """
        Return the model's training data.
        Note!!! This is the standardized data, and this is a must need for trajectory sampler calculation

        :return: The model's training data.
        """
        if get_standardized_output:
            return Dataset(self.model.data[0], self._output_transformation.backward_mean(self.model.data[1]))
        else:
            return Dataset(self.model.data[0], self.model.data[1])

    # TODO
    def conditional_predict_f(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        Returns the marginal GP distribution at query_points conditioned on both the model
        and some additional data, using exact formula. See :cite:`chevalier2014corrected`
        (eqs. 8-10) for details.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: mean_qp_new: predictive mean at query_points, with shape [..., M, L],
                 and var_qp_new: predictive variance at query_points, with shape [..., M, L]
        """
        raise NotImplementedError
        tf.debugging.assert_shapes(
            [
                (additional_data.query_points, [..., "N", "D"]),
                (additional_data.observations, [..., "N", "L"]),
                (query_points, ["M", "D"]),
            ],
            message="additional_data must have query_points with shape [..., N, D]"
            " and observations with shape [..., N, L], and query_points "
            "should have shape [M, D]",
        )

        mean_add, cov_add = self.predict_joint(
            additional_data.query_points
        )  # [..., N, L], [..., L, N, N]
        mean_qp, var_qp = self.predict(query_points)  # [M, L], [M, L]

        cov_cross = self.covariance_between_points(
            additional_data.query_points, query_points
        )  # [..., L, N, M]

        cov_shape = tf.shape(cov_add)
        noise = self.get_observation_noise() * tf.eye(
            cov_shape[-2], batch_shape=cov_shape[:-2], dtype=cov_add.dtype
        )
        L_add = tf.linalg.cholesky(cov_add + noise)  # [..., L, N, N]
        A = tf.linalg.triangular_solve(L_add, cov_cross, lower=True)  # [..., L, N, M]
        var_qp_new = var_qp - leading_transpose(
            tf.reduce_sum(A ** 2, axis=-2), [..., -1, -2]
        )  # [..., M, L]

        mean_add_diff = additional_data.observations - mean_add  # [..., N, L]
        mean_add_diff = leading_transpose(mean_add_diff, [..., -1, -2])[..., None]  # [..., L, N, 1]
        AM = tf.linalg.triangular_solve(L_add, mean_add_diff)  # [..., L, N, 1]

        mean_qp_new = mean_qp + leading_transpose(
            (tf.matmul(A, AM, transpose_a=True)[..., 0]), [..., -1, -2]
        )  # [..., M, L]

        tf.debugging.assert_shapes(
            [
                (additional_data.observations, [..., "N", "L"]),
                (query_points, ["M", "D"]),
                (mean_qp_new, [..., "M", "L"]),
                (var_qp_new, [..., "M", "L"]),
            ],
            message="received unexpected shapes computing conditional_predict_f,"
            "check model kernel structure?",
        )

        return mean_qp_new, var_qp_new

    # TODO
    def conditional_predict_joint(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        Predicts the joint GP distribution at query_points conditioned on both the model
        and some additional data, using exact formula. See :cite:`chevalier2014corrected`
        (eqs. 8-10) for details.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: mean_qp_new: predictive mean at query_points, with shape [..., M, L],
                 and cov_qp_new: predictive covariance between query_points, with shape
                 [..., L, M, M]
        """

        raise NotImplementedError
        tf.debugging.assert_shapes(
            [
                (additional_data.query_points, [..., "N", "D"]),
                (additional_data.observations, [..., "N", "L"]),
                (query_points, ["M", "D"]),
            ],
            message="additional_data must have query_points with shape [..., N, D]"
            " and observations with shape [..., N, L], and query_points "
            "should have shape [M, D]",
        )

        leading_dims = tf.shape(additional_data.query_points)[:-2]  # [...]
        new_shape = tf.concat([leading_dims, tf.shape(query_points)], axis=0)  # [..., M, D]
        query_points_r = tf.broadcast_to(query_points, new_shape)  # [..., M, D]
        points = tf.concat([additional_data.query_points, query_points_r], axis=-2)  # [..., N+M, D]

        mean, cov = self.predict_joint(points)  # [..., N+M, L], [..., L, N+M, N+M]

        N = tf.shape(additional_data.query_points)[-2]

        mean_add = mean[..., :N, :]  # [..., N, L]
        mean_qp = mean[..., N:, :]  # [..., M, L]

        cov_add = cov[..., :N, :N]  # [..., L, N, N]
        cov_qp = cov[..., N:, N:]  # [..., L, M, M]
        cov_cross = cov[..., :N, N:]  # [..., L, N, M]

        cov_shape = tf.shape(cov_add)
        noise = self.get_observation_noise() * tf.eye(
            cov_shape[-2], batch_shape=cov_shape[:-2], dtype=cov_add.dtype
        )
        L_add = tf.linalg.cholesky(cov_add + noise)  # [..., L, N, N]
        A = tf.linalg.triangular_solve(L_add, cov_cross, lower=True)  # [..., L, N, M]
        cov_qp_new = cov_qp - tf.matmul(A, A, transpose_a=True)  # [..., L, M, M]

        mean_add_diff = additional_data.observations - mean_add  # [..., N, L]
        mean_add_diff = leading_transpose(mean_add_diff, [..., -1, -2])[..., None]  # [..., L, N, 1]
        AM = tf.linalg.triangular_solve(L_add, mean_add_diff)  # [..., L, N, 1]
        mean_qp_new = mean_qp + leading_transpose(
            (tf.matmul(A, AM, transpose_a=True)[..., 0]), [..., -1, -2]
        )  # [..., M, L]

        tf.debugging.assert_shapes(
            [
                (additional_data.observations, [..., "N", "L"]),
                (query_points, ["M", "D"]),
                (mean_qp_new, [..., "M", "L"]),
                (cov_qp_new, [..., "L", "M", "M"]),
            ],
            message="received unexpected shapes computing conditional_predict_joint,"
            "check model kernel structure?",
        )

        return mean_qp_new, cov_qp_new

    # TODO
    def conditional_predict_f_sample(
        self, query_points: TensorType, additional_data: Dataset, num_samples: int
    ) -> TensorType:
        """
        Generates samples of the GP at query_points conditioned on both the model
        and some additional data.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :param num_samples: number of samples
        :return: samples of f at query points, with shape [..., num_samples, M, L]
        """

        raise NotImplementedError
        mean_new, cov_new = self.conditional_predict_joint(query_points, additional_data)
        mean_for_sample = tf.linalg.adjoint(mean_new)  # [..., L, N]
        samples = sample_mvn(
            mean_for_sample, cov_new, full_cov=True, num_samples=num_samples
        )  # [..., (S), P, N]
        return tf.linalg.adjoint(samples)  # [..., (S), N, L]

    # TODO
    def conditional_predict_y(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        Generates samples of y from the GP at query_points conditioned on both the model
        and some additional data.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: predictive variance at query_points, with shape [..., M, L],
                 and predictive variance at query_points, with shape [..., M, L]
        """

        raise NotImplementedError
        f_mean, f_var = self.conditional_predict_f(query_points, additional_data)
        return self.model.likelihood.predict_mean_and_var(f_mean, f_var)

    # Note this is the standardized output model's likelihood
    def get_observation_noise(self) -> TensorType:
        """
        Return the variance of observation noise for homoscedastic likelihoods.

        :return: The observation noise.
        :raise NotImplementedError: If the model does not have a homoscedastic likelihood.
        """
        try:
            noise_variance = self.model.likelihood.variance
        except AttributeError:
            raise NotImplementedError(f"Model {self!r} does not have scalar observation noise")

        return noise_variance

    # Note this is the standardized output model's mean function
    def get_mean_function(self) -> gpflow.mean_functions.MeanFunction:
        """
        Return the mean function of the model.

        :return: The mean function.
        """
        return self.model.mean_function


