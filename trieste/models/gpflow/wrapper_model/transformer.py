from ....types import TensorType
import tensorflow as tf


class Standardization:
    def __init__(self):
        self._mean = None
        self._std = None # Note std can possibly be 0

    def __initialize_scaling_factor(self, mean: TensorType, std: TensorType):
        self._mean = tf.Variable(mean)
        self._std = tf.Variable(std)

    def update_forward(self, x:TensorType) -> TensorType:
        """
        :param x [..., N, D]
        """
        _mean = tf.reduce_mean(x, axis=-2) # [..., D]
        _std = tf.math.reduce_std(x, axis=-2) # [..., D]
        tf.debugging.assert_positive(_std)
        if self._mean is None:
            self.__initialize_scaling_factor(_mean, _std)
        self._mean.assign(_mean)
        self._std.assign(_std)
        return (x - self._mean)/self._std

    def forward_mean(self, x: TensorType):
        return (x - self._mean)/self._std

    def backward_mean(self, x: TensorType):
        return x * self._std + self._mean

    def backward_variance(self, cov: TensorType):
        return cov * self._std ** 2