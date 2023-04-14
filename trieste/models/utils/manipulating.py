"""
Related Answer
https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194
https://scicomp.stackexchange.com/questions/30631/how-to-find-the-nearest-a-near-positive-definite-from-a-given-matrix#:~:text=Quick%20sketch%20of%20an%20answer,%3DQD%2BQ%E2%8A%A4.
"""


import numpy as np
import tensorflow as tf
from numpy import linalg as la

from trieste.types import TensorType


# TODO:
def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, Vh = la.svd(B)

    H = np.dot(Vh.T, np.dot(np.diag(s), Vh))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


# -------------------------------------------------------------------
# Fully TF version
def is_pd(matrix: TensorType):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = tf.linalg.cholesky(matrix)
        return True
    except:
        return False


# TODO:
def nearest_pd(matrix: TensorType):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

    :param matrix [..., N, N]
    """

    B = (matrix + tf.linalg.matrix_transpose(matrix)) / 2  # [..., N, N]
    s, _, V = tf.linalg.svd(B)  # [..., N]; [..., N, N]

    # VΣV*
    H = tf.matmul(tf.matmul(V, tf.linalg.diag(s)),
                  tf.linalg.matrix_transpose(V, conjugate=True))  #

    A2 = (B + H) / 2

    A3 = (A2 + tf.linalg.matrix_transpose(A2)) / 2  # A3 is garanteed to be PSD

    if is_pd(A3):
        return A3

    spacing = np.spacing(tf.linalg.norm(matrix))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = tf.eye(matrix.shape[-1], batch_shape=matrix.shape[:-2], dtype=matrix.dtype)
    k = 1
    while not is_pd(A3):
        mineig = tf.reduce_min(tf.math.real(tf.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)  # A3 +
        k += 1

    return A3


def _cholesky(matrix):
    """Return a Cholesky factor and boolean success."""
    try:
        chol = tf.linalg.cholesky(matrix)
        ok = tf.reduce_all(tf.math.is_finite(chol))
        return chol, ok
    except tf.errors.InvalidArgumentError:
        return matrix, False


def safer_cholesky(matrix, max_attempts: int = 10, jitter: float = 1e-6):
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


if __name__ == '__main__':
    # Batch Test
    xx = tf.random.normal(shape=(2, 5, 5))
    xx_approx = nearest_pd(xx)
    import numpy as np
    for i in range(10):
        for j in range(2, 100):
            A = tf.random.normal(shape=(j, j))
            B = nearestPD(A.numpy())
            C = nearest_pd(A)
            print(np.sum(np.abs(B - C.numpy())))
            assert (is_pd(C))
    print('unit test passed!')