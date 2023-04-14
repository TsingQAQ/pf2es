import numpy as np
import tensorflow as tf

covs = np.load('Cholesky_Failure_Cov.npy')
covs_with_jitter = np.load('Cholesky_Failure_Cov_plus_jitter.npy')
xs = np.load('Cholesky_Failure_Input_Record.npy')
model_input = np.load('Cholesky_Failure_Model_0_Input.npy')
model_output = np.load('Cholesky_Failure_Model_0_Output.npy')
success_covs = np.load('Cholesky_Failure_Cov.npy')
success_covs_with_jitter = np.load('Cholesky_Succ_Cov_plus_jitter.npy')

for cov, i in zip(covs_with_jitter, range(covs.shape[0])):
    try:
        tf.linalg.cholesky(cov[0])
    except:
        print('--------------')
        print(f'Cholesky error at index: {i}')
        print(f'Cholesky error at batch input:\n {xs[i]}')
        print(f'Cholesky error cov + jitter:\n {covs_with_jitter[i][0]}')
        print(f'Cholesky error for cov:\n {covs[i][0]}')


print(f'A Successful Cov + jitter:\n {success_covs_with_jitter}')
print(f'A Successful Cov:\n {success_covs}')
# print(f'Model Input: {model_input}')
# print(f'Model Output: {model_output}')