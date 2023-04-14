from trieste.data import Dataset
from trieste.space import SearchSpace, Box
from trieste.models.interfaces import TrainablePredictJointReparamModelStack
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
import tensorflow as tf
import numpy as np
from trieste.utils import DEFAULTS


def build_stacked_reparam_sampler_independent_objectives_model(
        data: Dataset, num_output: int, search_space: SearchSpace
) -> TrainablePredictJointReparamModelStack:
    gprs = []
    for idx in range(num_output):
        single_obj_data = Dataset(
            data.query_points, tf.gather(data.observations, [idx], axis=1)
        )
        gpr = build_gpr(single_obj_data, search_space, likelihood_variance=1e-2)
        gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=False), 1))

    return TrainablePredictJointReparamModelStack(*gprs)


xs = tf.convert_to_tensor(np.load('Cholesky_Failure_Model_0_Input.npy'))
ys = tf.convert_to_tensor(np.load('Cholesky_Failure_Model_0_Output.npy'))

pb_search_space = Box([0.0] * 2, [1.0] * 2)
model = build_stacked_reparam_sampler_independent_objectives_model(Dataset(xs, ys), 1, pb_search_space)

# set lengthscale
cholesky_failure_lengthscale = np.load('Cholesky_Failure_Model_0_Lengthscales.npy')
cholesky_failure_likelihood = np.load('Cholesky_Failure_Model_0_likelihood_variance.npy')
cholesky_failure_kernel_variance = np.load('Cholesky_Failure_Model_0_kernel_variance.npy')
cholesky_failure_mean_function = np.load('Cholesky_Failure_Model_0_mean_function_val.npy')
model._models[0].model.kernel.lengthscales.assign(cholesky_failure_lengthscale)
model._models[0].model.likelihood.variance.assign(cholesky_failure_likelihood)
model._models[0].model.mean_function.c.assign(cholesky_failure_mean_function)
model._models[0].model.kernel.variance.assign(cholesky_failure_kernel_variance)
# this is a must need to keep the predict joint the same
model._models[0].update_posterior_cache()

model._models[0]._model
test_xs = tf.convert_to_tensor(np.load('Cholesky_Failure_Input_Record.npy'))
_, covs = model.predict_joint(test_xs)

for cov, i in zip(covs, range(covs.shape[0])):
    try:
        tf.linalg.cholesky(cov[0] + DEFAULTS.JITTER)
    except:
        print('--------------')
        print(f'Cholesky error at index: {i}')
        print(f'Cholesky error for batch input:\n {test_xs[i]}')
        print(f'Cholesky error cov + jitter:\n {covs[i][0]}')
        print(f'Cholesky error for cov:\n {covs[i][0]}')