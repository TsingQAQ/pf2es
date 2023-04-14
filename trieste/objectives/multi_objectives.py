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
This module contains synthetic multi-objective functions, useful for experimentation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import tensorflow as tf
from typing_extensions import Protocol

from ..space import Box
from ..types import TensorType, Callable
from .single_objectives import ObjectiveTestProblem, branin, forrester_function, sinlinear
from abc import abstractmethod


class GenParetoOptimalPoints(Protocol):
    """A Protocol representing a function that generates Pareto optimal points."""

    def __call__(self, n: int, seed: int | None = None) -> TensorType:
        """
        Generate `n` Pareto optimal points.

        :param n: The number of pareto optimal points to be generated.
        :param seed: An integer used to create a random seed for distributions that
         used to generate pareto optimal points.
        :return: The Pareto optimal points
        """


@dataclass(frozen=True)
class MultiObjectiveTestProblem(ObjectiveTestProblem):
    """
    Convenience container class for synthetic multi-objective test functions, containing
    a generator for the pareto optimal points, which can be used as a reference of performance
    measure of certain multi-objective optimization algorithms.
    """

    gen_pareto_optimal_points: GenParetoOptimalPoints
    """Function to generate Pareto optimal points, given the number of points and an optional
    random number seed."""


def vlmop2(x: TensorType, d: int) -> TensorType:
    """
    The VLMOP2 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param d: The dimensionality of the synthetic function.
    :return: The function values at ``x``, with shape [..., 2].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes(
        [(x, (..., d))],
        message=f"input x dim: {x.shape[-1]} does not align with pre-specified dim: {d}",
    )
    transl = 1 / tf.sqrt(tf.cast(d, x.dtype))
    y1 = 1 - tf.exp(-1 * tf.reduce_sum((x - transl) ** 2, axis=-1))
    y2 = 1 - tf.exp(-1 * tf.reduce_sum((x + transl) ** 2, axis=-1))
    return tf.stack([y1, y2], axis=-1)


def VLMOP2(input_dim: int) -> MultiObjectiveTestProblem:
    """
    The VLMOP2 problem, typically evaluated over :math:`[-2, 2]^d`.
    The idea pareto fronts lies on -1/sqrt(d) - 1/sqrt(d) and x1=...=xdim.

    See :cite:`van1999multiobjective` and :cite:`fonseca1995multiobjective`
    (the latter for discussion of pareto front property) for details.

    :param input_dim: The input dimensionality of the synthetic function.
    :return: The problem specification.
    """

    def gen_pareto_optimal_points(n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater(n, 0)
        transl = 1 / tf.sqrt(tf.cast(input_dim, tf.float64))
        _x = tf.tile(tf.linspace([-transl], [transl], n), [1, input_dim])
        return vlmop2(_x, input_dim)

    return MultiObjectiveTestProblem(
        name=f"VLMOP2({input_dim})",
        objective=partial(vlmop2, d=input_dim),
        search_space=Box([-2.0], [2.0]) ** input_dim,
        gen_pareto_optimal_points=gen_pareto_optimal_points,
    )


def dtlz_mkd(input_dim: int, num_objective: int) -> tuple[int, int, int]:
    """Return m/k/d values for dtlz synthetic functions."""
    tf.debugging.assert_greater(input_dim, 0)
    tf.debugging.assert_greater(num_objective, 0)
    tf.debugging.assert_greater(
        input_dim,
        num_objective,
        f"input dimension {input_dim}"
        f"  must be greater than function objective numbers {num_objective}",
    )
    M = num_objective
    k = input_dim - M + 1
    d = input_dim
    return (M, k, d)


def dtlz1(x: TensorType, m: int, k: int, d: int) -> TensorType:
    """
    The DTLZ1 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param m: The objective numbers.
    :param k: The input dimensionality for g.
    :param d: The dimensionality of the synthetic function.
    :return: The function values at ``x``, with shape [..., m].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes(
        [(x, (..., d))],
        message=f"input x dim: {x.shape[-1]} does not align with pre-specified dim: {d}",
    )
    tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")

    def g(xM: TensorType) -> TensorType:
        return 100 * (
                k
                + tf.reduce_sum(
            (xM - 0.5) ** 2 - tf.cos(20 * math.pi * (xM - 0.5)), axis=-1, keepdims=True
        )
        )

    ta = tf.TensorArray(x.dtype, size=m)
    for i in range(m):
        xM = x[..., m - 1:]
        y = 1 + g(xM)
        y *= 1 / 2 * tf.reduce_prod(x[..., : m - 1 - i], axis=-1, keepdims=True)
        if i > 0:
            y *= 1 - x[..., m - i - 1, tf.newaxis]
        ta = ta.write(i, y)

    return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)


def DTLZ1(input_dim: int, num_objective: int) -> MultiObjectiveTestProblem:
    """
    The DTLZ1 problem, the idea pareto fronts lie on a linear hyper-plane.
    See :cite:`deb2002scalable` for details.

    :param input_dim: The input dimensionality of the synthetic function.
    :param num_objective: The number of objectives.
    :return: The problem specification.
    """
    M, k, d = dtlz_mkd(input_dim, num_objective)

    def gen_pareto_optimal_points(n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater_equal(M, 2)
        rnd = tf.random.uniform([n, M - 1], minval=0, maxval=1, seed=seed, dtype=tf.float64)
        strnd = tf.sort(rnd, axis=-1)
        strnd = tf.concat(
            [tf.zeros([n, 1], dtype=tf.float64), strnd, tf.ones([n, 1], dtype=tf.float64)], axis=-1
        )
        return 0.5 * (strnd[..., 1:] - strnd[..., :-1])

    return MultiObjectiveTestProblem(
        name=f"DTLZ1({input_dim}, {num_objective})",
        objective=partial(dtlz1, m=M, k=k, d=d),
        search_space=Box([0.0], [1.0]) ** d,
        gen_pareto_optimal_points=gen_pareto_optimal_points,
    )


def dtlz2(x: TensorType, m: int, d: int) -> TensorType:
    """
    The DTLZ2 synthetic function.

    :param x: The points at which to evaluate the function, with shape [..., d].
    :param m: The objective numbers.
    :param d: The dimensionality of the synthetic function.
    :return: The function values at ``x``, with shape [..., m].
    :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
    """
    tf.debugging.assert_shapes(
        [(x, (..., d))],
        message=f"input x dim: {x.shape[-1]} does not align with pre-specified dim: {d}",
    )
    tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")

    def g(xM: TensorType) -> TensorType:
        z = (xM - 0.5) ** 2
        return tf.reduce_sum(z, axis=-1, keepdims=True)

    ta = tf.TensorArray(x.dtype, size=m)
    for i in tf.range(m):
        y = 1 + g(x[..., m - 1:])
        for j in tf.range(m - 1 - i):
            y *= tf.cos(math.pi / 2 * x[..., j, tf.newaxis])
        if i > 0:
            y *= tf.sin(math.pi / 2 * x[..., m - 1 - i, tf.newaxis])
        ta = ta.write(i, y)

    return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)


# class DTLZ3(DTLZ):
#     """
#     The DTLZ3 problem, the idea pareto fronts correspond to xi=0.5.
#     See :cite:deb2002scalable for details.
#     """
#
#     def objective(self):
#         return partial(dtlz3, m=self.M, k=self.k, d=self.dim)
#
#     def gen_pareto_optimal_points(self, n: int, seed=None):
#         pass
#
#
# def dtlz3(x: TensorType, m: int, k: int, d: int) -> TensorType:
#     """
#     The DTLZ3 synthetic function.
#     :param x: The points at which to evaluate the function, with shape [..., d].
#     :param m: The objective numbers.
#     :param d: The dimensionality of the synthetic function.
#     :return: The function values at ``x``, with shape [..., m].
#     :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
#     """
#     tf.debugging.assert_shapes(
#         [(x, (..., d))],
#         message=f"input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}",
#     )
#     tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")
#
#     def g(xM):
#         return 100 * (
#                 k
#                 + tf.reduce_sum(
#             (xM - 0.5) ** 2 - tf.cos(20 * math.pi * (xM - 0.5)), axis=-1, keepdims=True
#         )
#         )
#
#     ta = tf.TensorArray(x.dtype, size=m)
#     for i in tf.range(m):
#         y = 1 + g(x[..., m - 1:])
#         for j in tf.range(m - 1 - i):
#             y *= tf.cos(math.pi / 2 * x[..., j, tf.newaxis])
#         if i > 0:
#             y *= tf.sin(math.pi / 2 * x[..., m - 1 - i, tf.newaxis])
#         ta = ta.write(i, y)
#
#     return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)
#
#
# class DTLZ4(DTLZ):
#     """
#     The DTLZ4 problem
#     See :cite:deb2002scalable for details.
#     """
#
#     def objective(self):
#         return partial(dtlz4, m=self.M, d=self.dim)
#
#     def gen_pareto_optimal_points(self, n: int, seed=None):
#         pass
#
#
# def dtlz4(x: TensorType, m: int, d: int) -> TensorType:
#     """
#     The DTLZ4 synthetic function.
#     :param x: The points at which to evaluate the function, with shape [..., d].
#     :param m: The objective numbers.
#     :param d: The dimensionality of the synthetic function.
#     :return: The function values at ``x``, with shape [..., m].
#     :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
#     """
#     alpha = 100
#     tf.debugging.assert_shapes(
#         [(x, (..., d))],
#         message=f"input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}",
#     )
#     tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")
#
#     def g(xM):
#         """
#         The g function for DTLZ4 is the same as DTLZ2
#         """
#         z = (xM - 0.5) ** 2
#         return tf.reduce_sum(z, axis=-1, keepdims=True)
#
#     ta = tf.TensorArray(x.dtype, size=m)
#     for i in tf.range(m):
#         y = 1 + g(x[..., m - 1:])
#         for j in tf.range(m - 1 - i):
#             y *= tf.cos(math.pi / 2 * (x[..., j, tf.newaxis]) ** alpha)
#         if i > 0:
#             y *= tf.sin(math.pi / 2 * (x[..., m - 1 - i, tf.newaxis]) ** alpha)
#         ta = ta.write(i, y)
#
#     return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)
#
#
# class DTLZ5(DTLZ):
#     """
#     The DTLZ5 problem
#     See :cite:deb2002scalable for details.
#     """
#
#     def objective(self):
#         return partial(dtlz5, m=self.M, d=self.dim)
#
#     def gen_pareto_optimal_points(self, n: int, seed=None):
#         pass
#
#
# def dtlz5(x: TensorType, m: int, d: int) -> TensorType:
#     """
#     The DTLZ5 synthetic function.
#     :param x: The points at which to evaluate the function, with shape [..., d].
#     :param m: The objective numbers.
#     :param d: The dimensionality of the synthetic function.
#     :return: The function values at ``x``, with shape [..., m].
#     :raise ValueError (or InvalidArgumentError): If ``x`` has an invalid shape.
#     """
#     tf.debugging.assert_shapes(
#         [(x, (..., d))],
#         message=f"input x dim: {x.shape[-1]} is not align with pre-specified dim: {d}",
#     )
#     tf.debugging.assert_greater(m, 0, message=f"positive objective numbers expected but found {m}")
#
#     def g(xM):
#         """
#         The g function for DTLZ5 is the same as DTLZ4
#         """
#         return tf.reduce_sum((xM - 0.5) ** 2, axis=-1, keepdims=True)
#
#     ta = tf.TensorArray(x.dtype, size=m)
#     X_, X_M = x[:, : m - 1], x[:, m - 1:]
#     g_val = g(X_M)
#
#     theta = 1 / (2 * (1 + g_val)) * (1 + 2 * g_val * X_)
#     theta = tf.concat([x[:, 0, tf.newaxis], theta[:, 1:]], axis=-1)
#
#     for i in range(0, m):
#         _f = 1 + g_val
#         _f *= tf.reduce_prod(
#             tf.math.cos(theta[:, : theta.shape[1] - i] * math.pi / 2.0), axis=-1, keepdims=True
#         )
#         if i > 0:
#             _f *= tf.math.sin(theta[:, theta.shape[1] - i, tf.newaxis] * math.pi / 2.0)
#         ta = ta.write(i, _f)
#
#     return tf.squeeze(tf.concat(tf.split(ta.stack(), m, axis=0), axis=-1), axis=0)


class VehicleCrashSafety(MultiObjectiveTestProblem):
    bounds = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
    dim = 5

    def objective(self) -> Callable[[TensorType], TensorType]:
        return vehicle_safety

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


_vehicle_bounds = [[1, 1, 1, 1, 1], [3, 3, 3, 3, 3]]


def vehicle_safety(x: TensorType) -> TensorType:
    x = x * (tf.constant(_vehicle_bounds[-1], dtype=x.dtype) - tf.constant(_vehicle_bounds[-2], dtype=x.dtype)) + \
        tf.constant(_vehicle_bounds[-2], dtype=x.dtype)
    X1, X2, X3, X4, X5 = tf.split(x, 5, axis=-1)
    f1 = (
            1640.2823
            + 2.3573285 * X1
            + 2.3220035 * X2
            + 4.5688768 * X3
            + 7.7213633 * X4
            + 4.4559504 * X5
    )
    f2 = (
            6.5856
            + 1.15 * X1
            - 1.0427 * X2
            + 0.9738 * X3
            + 0.8364 * X4
            - 0.3695 * X1 * X4
            + 0.0861 * X1 * X5
            + 0.3628 * X2 * X4
            - 0.1106 * X1 ** 2
            - 0.3437 * X3 ** 2
            + 0.1764 * X4 ** 2
    )
    f3 = (
            -0.0551
            + 0.0181 * X1
            + 0.1024 * X2
            + 0.0421 * X3
            - 0.0073 * X1 * X2
            + 0.024 * X2 * X3
            - 0.0118 * X2 * X4
            - 0.0204 * X3 * X4
            - 0.008 * X3 * X5
            - 0.0241 * X2 ** 2
            + 0.0109 * X4 ** 2
    )
    f_X = tf.concat([f1, f2, f3], axis=-1)
    return f_X


class BraninCurrin(MultiObjectiveTestProblem):
    """
    The BraninCurrin problem, typically evaluated over :math:`[-2, 2]^2`.
    """

    bounds = [[0.0] * 2, [1.0] * 2]
    dim = 2

    def objective(self):
        return branincurrin

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass


class CBraninCurrin(BraninCurrin):

    def constraint(self):
        return evaluate_slack_true


def branincurrin(x):
    return tf.concat([branin(x), Currin(x)], axis=-1)


def Currin(x):
    """
    copied from  https://github.com/belakaria/MESMO/blob/8cccda6ccabfbce214ca49a5c990c2f8456f3297/benchmark_functions.py#L7-L8
    """
    return (
            (1 - tf.math.exp(-0.5 * (1 / (x[..., 1] + 1e-100))))
            * (
                    (2300 * x[..., 0] ** 3 + 1900 * x[..., 0] ** 2 + 2092 * x[..., 0] + 60)
                    / (100 * x[..., 0] ** 3 + 500 * x[..., 0] ** 2 + 4 * x[..., 0] + 20)
            )
    )[..., tf.newaxis]


def evaluate_slack_true(X: TensorType) -> TensorType:
    # unnormalize X
    # [(-5.0, 10.0), (0.0, 15.0)]
    X = X * (tf.constant([10.0, 15.0], dtype=X.dtype) - tf.constant([-5.0, 0.0], dtype=X.dtype)) + \
        tf.constant([-5.0, 0.0], dtype=X.dtype)
    return 50 - (X[..., 0:1] - 2.5) ** 2 - (X[..., 1:2] - 7.5) ** 2


# class ZDT(MultiObjectiveTestProblem):
#     def __init__(self, input_dim: int = 30):
#         self._dim = input_dim
#         self._bounds = [[0.0] * input_dim, [1.0] * input_dim]
#
#     @property
#     def dim(self) -> int:
#         return self._dim
#
#     @property
#     def bounds(self) -> list[list[float]]:
#         return self._bounds


# class ZDT1(ZDT):
#     """
#     ZDT series multi-objective test problem.
#     """
#
#     def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
#         pass
#
#     def objective(self) -> Callable[[TensorType], TensorType]:
#         return partial(zdt1, n_variable=self._dim)
#
#
# def zdt1(x: TensorType, n_variable: int = 30):
#     f1 = x[:, 0, tf.newaxis]
#     g = 1.0 + 9.0 / (n_variable - 1) * tf.reduce_sum(x[:, 1:], axis=1, keepdims=True)
#     # TODO: F2 is weird, not the same as https://pymoo.org/problems/multi/zdt.html
#     f2 = g * (1 - (f1 / g) ** 0.5)
#     return tf.concat([f1, f2], axis=-1)
#
#
# class ZDT2(ZDT):
#     """
#     ZDT series multi-objective test problem.
#     """
#
#     def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
#         pass
#
#     def objective(self) -> Callable[[TensorType], TensorType]:
#         return partial(zdt2, n_variable=self._dim)
#
#
# def zdt2(x: TensorType, n_variable: int = 30):
#     f1 = x[:, 0, tf.newaxis]
#     c = tf.reduce_sum(x[:, 1:], axis=-1, keepdims=True)
#     g = 1.0 + 9.0 * c / (n_variable - 1)
#     f2 = g * (1 - (f1 * 1.0 / g) ** 2)
#     return tf.concat([f1, f2], axis=-1)
#
#
# class ZDT3(ZDT):
#     """
#     ZDT series multi-objective test problem.
#     """
#
#     def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
#         pass
#
#     def objective(self) -> Callable[[TensorType], TensorType]:
#         return partial(zdt3, n_variable=self._dim)
#
#
# def zdt3(x: TensorType, n_variable: int = 30) -> TensorType:
#     f1 = x[:, 0, tf.newaxis]
#     c = tf.reduce_sum(x[:, 1:], axis=1, keepdims=True)
#     g = 1.0 + 9.0 * c / (n_variable - 1)
#     f2 = g * (1 - (f1 * 1.0 / g) ** 0.5 - (f1 * 1.0 / g) * tf.math.sin(10 * math.pi * f1))
#
#     return tf.concat([f1, f2], axis=-1)
#
#
# class ZDT4(ZDT):
#     """
#     ZDT series multi-objective test problem.
#     This is a scaled ZDT4, except for the first input dim, the rest has been scaled from [-10, 10] to [0, 1]
#     """
#
#     def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
#         pass
#
#     def objective(self) -> Callable[[TensorType], TensorType]:
#         return partial(zdt4, n_variable=self._dim)
#
#
# def zdt4(x: TensorType, n_variable: int = 30):
#     f1 = x[:, 0, tf.newaxis]
#     g = (
#             1
#             + 10 * (n_variable - 1)
#             + tf.reduce_sum(
#         (x[:, 1:] * 20 - 10) ** 2 - 10.0 * tf.math.cos(4.0 * math.pi * (x[:, 1:] * 20 - 10)),
#         axis=-1,
#         keepdims=True,
#     )
#     )
#     h = 1.0 - (f1 / g) ** 0.5
#     f2 = g * h
#     return tf.concat([f1, f2], axis=-1)


import numpy as np


# class FourBarTruss(MultiObjectiveTestProblem):
#     bounds = [[0.0] * 4, [1.0] * 4]
#     dim = 4
#     """
#     from https://github.com/ryojitanabe/reproblems/blob/master/reproblem_python_ver/reproblem.py
#     """
#
#     def __init__(self):
#         self.problem_name = 'RE21'
#         self.n_objectives = 2
#         self.n_variables = 4
#         self.n_constraints = 0
#         self.n_original_constraints = 0
#
#         F = 10.0
#         sigma = 10.0
#         tmp_val = F / sigma
#
#         self.ubound = np.full(self.n_variables, 3 * tmp_val)
#         self.lbound = np.zeros(self.n_variables)
#         self.lbound[0] = tmp_val
#         self.lbound[1] = np.sqrt(2.0) * tmp_val
#         self.lbound[2] = np.sqrt(2.0) * tmp_val
#         self.lbound[3] = tmp_val
#
#     def objective(self):
#         return partial(four_bar_truss_obj, lbounds=self.lbound, ubounds=self.ubound)
#
#     def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
#         pass
#
#
# def four_bar_truss_obj(x, lbounds, ubounds):
#     '''
#     In trieste we assume [0, 1]^d
#     '''
#
#     def evaluate_sequential(x):
#         f = np.zeros(2)
#         x1 = x[0]
#         x2 = x[1]
#         x3 = x[2]
#         x4 = x[3]
#
#         F = 10.0
#         sigma = 10.0
#         E = 2.0 * 1e5
#         L = 200.0
#
#         f[0] = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
#         f[1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))
#
#         return f
#
#     res = []
#     for seq_x in x:
#         unscaled_seq_x = seq_x * (ubounds - lbounds) + lbounds
#         res.append(evaluate_sequential(unscaled_seq_x))
#     return tf.stack(res, 0)


class SinLinearForrester(MultiObjectiveTestProblem):
    """
    The SinLinear Forrester problem, typically evaluated over :math:`[0, 1]`.
    """

    bounds = [[0.0] * 1, [1.0] * 1]
    dim = 1

    def objective(self):
        return sinlinarforrester

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass


def sinlinarforrester(x):
    return tf.concat([sinlinear(x), forrester_function(x)], axis=-1)


class GMMForrester(MultiObjectiveTestProblem):
    bounds = [[0.0] * 1, [1.0] * 1]
    dim = 1

    def objective(self) -> Callable[[TensorType], TensorType]:
        return gmmforrester

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


def gmmforrester(x: TensorType) -> TensorType:
    return tf.concat([h(x), forrester_function(x)], axis=-1)


def h(x: TensorType) -> TensorType:
    return 20 * (
            0.5 - tf.exp(-(((x - 0.1) / 0.02) ** 2)) - 0.3 * tf.exp(-(((x - 0.8) / 0.35) ** 2))
    )


class Quadratic(MultiObjectiveTestProblem):
    bounds = [[0.0] * 1, [1.0] * 1]
    dim = 1

    def objective(self):
        return quadratic_function

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass


def quadratic_function(x):
    return tf.concat([_quadratic(x), -_quadratic(x)], axis=-1)


def _quadratic(x):
    return x ** 2


class ConstraintMultiObjectiveTestProblem(MultiObjectiveTestProblem):
    @abstractmethod
    def constraint(self) -> Callable[[TensorType], TensorType]:
        """
        Get the synthetic constraint function.
        :return: A callable synthetic function
        """


def sim_constraint(input_data):
    if tf.rank(input_data) == 1:
        input_data = tf.expand_dims(input_data, axis=-2)
    x, y = input_data[:, -2], input_data[:, -1]
    z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
    return 0.75 - z[:, None]


class CVLMOP2(ConstraintMultiObjectiveTestProblem):
    bounds = [[-2.0] * 2, [2.0] * 2]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return vlmop2

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return sim_constraint

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


_osy_lb = [0, 0, 1, 0, 1, 0]
_osy_ub = [10, 10, 5, 6, 5, 10]


class Osy(ConstraintMultiObjectiveTestProblem):
    bounds = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]  # [[0, 0, 1, 0, 1, 0], [10, 10, 5, 6, 5, 10]]
    dim = 6

    def objective(self) -> Callable[[TensorType], TensorType]:
        return osy_obj

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return osy_cons

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass


def osy_obj(x: TensorType) -> TensorType:
    x = x * (tf.constant(_osy_ub, dtype=x.dtype) - tf.constant(_osy_lb, dtype=x.dtype)) + \
        tf.constant(_osy_lb, dtype=x.dtype)
    obj1 = (
            -25 * (x[:, 0] - 2) ** 2
            - (x[:, 1] - 2) ** 2
            - (x[:, 2] - 1) ** 2
            - (x[:, 3] - 2) ** 2
            - (x[:, 4] - 2) ** 2
    )
    obj2 = tf.reduce_sum(x ** 2, axis=-1)
    return tf.stack([obj1, obj2], axis=-1)


def osy_cons(x: TensorType) -> TensorType:
    x = x * (tf.constant(_osy_ub, dtype=x.dtype) - tf.constant(_osy_lb, dtype=x.dtype)) + \
        tf.constant(_osy_lb, dtype=x.dtype)
    c1 = x[:, 0] + x[:, 1] - 2
    c2 = 6 - x[:, 0] - x[:, 1]
    c3 = 2 - x[:, 1] + x[:, 0]
    c4 = 2 - x[:, 0] + 3 * x[:, 1]
    c5 = 4 - (x[:, 2] - 3) ** 2 - x[:, 3]
    c6 = (x[:, 4] - 3) ** 2 + x[:, 5] - 4
    return tf.stack([c1, c2, c3, c4, c5, c6], axis=-1)


_tnk_lb = [0, 1e-30]  # for numerical stability
_tnk_ub = [math.pi, math.pi]


class TNK(ConstraintMultiObjectiveTestProblem):
    bounds = [[0, 0], [1, 1]]  # [[0, 0], [math.pi, math.pi]]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return TNK_obj

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return TNK_cons

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass


def TNK_obj(x: TensorType) -> TensorType:
    x = x * (tf.constant(_tnk_ub, dtype=x.dtype) - tf.constant(_tnk_lb, dtype=x.dtype))
    return tf.stack([x[:, 0], x[:, 1]], axis=-1)


def TNK_cons(x: TensorType) -> TensorType:
    x = x * (tf.constant(_tnk_ub, dtype=x.dtype) - tf.constant(_tnk_lb, dtype=x.dtype)) + \
        tf.constant(_tnk_lb, dtype=x.dtype)
    c1 = x[:, 0] ** 2 + x[:, 1] ** 2 - 1 - 0.1 * tf.cos(16 * tf.math.atan(x[:, 0] / x[:, 1]))
    c2 = 0.5 - ((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2)
    return tf.stack([c1, c2], axis=-1)


_constr_ex_lb = [0.1, 0]
_constr_ex_ub = [1, 5]


class Constr_Ex(ConstraintMultiObjectiveTestProblem):
    bounds = [[0, 0], [1, 1]]  # [[0.1, 0], [1, 5]]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return constr_ex

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return constr_ex_cons_func


def constr_ex(x: TensorType):
    x = x * (tf.constant(_constr_ex_ub, dtype=x.dtype) - tf.constant(_constr_ex_lb, dtype=x.dtype)) + \
        tf.constant(_constr_ex_lb, dtype=x.dtype)
    y1 = x[:, 0]
    y2 = (1 + x[:, 1]) / x[:, 0]
    return tf.stack([y1, y2], axis=-1)


def constr_ex_cons_func(x: TensorType):
    x = x * (tf.constant(_constr_ex_ub, dtype=x.dtype) - tf.constant(_constr_ex_lb, dtype=x.dtype)) + \
        tf.constant(_constr_ex_lb, dtype=x.dtype)
    c1 = x[:, 1] + 9 * x[:, 0] - 6
    c2 = -x[:, 1] + 9 * x[:, 0] - 1
    return tf.stack([c1, c2], axis=-1)


_srn_lb = [-20, -20]
_srn_ub = [20, 20]


class SRN(ConstraintMultiObjectiveTestProblem):
    """
    SRN function, refer :cite:`chafekar2003constrained`
    """

    bounds = [[0, 0], [1, 1]]  # [[-20, -20], [20, 20]]
    dim = 2

    def objective(self) -> Callable[[TensorType], TensorType]:
        return srn_obj

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return srn_cons

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass


def srn_obj(x: TensorType) -> TensorType:
    x = x * (tf.constant(_srn_ub, dtype=x.dtype) - tf.constant(_srn_lb, dtype=x.dtype)) + \
        tf.constant(_srn_lb, dtype=x.dtype)
    obj1 = 2 + (x[..., 0] - 2) ** 2 + (x[..., 1] - 2) ** 2
    obj2 = 9 * x[..., 0] - (x[..., 1] - 1) ** 2
    return tf.stack([obj1, obj2], axis=-1)


def srn_cons(x: TensorType) -> TensorType:
    x = x * (tf.constant(_srn_ub, dtype=x.dtype) - tf.constant(_srn_lb, dtype=x.dtype)) + \
        tf.constant(_srn_lb, dtype=x.dtype)
    c1 = 225 - (x[..., 0]) ** 2 - (x[..., 1]) ** 2
    c2 = -x[..., 0] + 3 * x[..., 1] - 10.0
    return tf.stack([c1, c2], axis=-1)


class Penicillin(MultiObjectiveTestProblem):
    r"""A penicillin production simulator from [Liang2021]_.
    This implementation is adapted from
    https://github.com/HarryQL/TuRBO-Penicillin.
    The goal is to maximize the penicillin yield while minimizing
    time to ferment and the CO2 byproduct.
    The function is defined for minimization of all objectives.
    The reference point was set using the `infer_reference_point` heuristic
    on the Pareto frontier over a large discrete set of random designs.
    """
    bounds = [[0.0] * 7, [1.0] * 7]

    dim = 7

    def objective(self) -> Callable[[TensorType], TensorType]:
        return Penicillin_objective


    def gen_pareto_optimal_points(self, n: int, seed = None) -> TensorType:
        pass


_penicillin_lb = [60.0, 0.05, 293.0, 0.05, 0.01, 500.0, 5.0]
_penicillin_ub = [120.0, 18.0, 303.0, 18.0, 0.5, 700.0, 6.5]


def Penicillin_objective(X_input: TensorType) -> TensorType:
     r"""Penicillin simulator, simplified and vectorized.
     The 7 input parameters are (in order): culture volume, biomass
     concentration, temperature, glucose concentration, substrate feed
     rate, substrate feed concentration, and H+ concentration.
     Args:
         X_input: A `n x 7`-dim tensor of inputs.
     Returns:
         An `n x 3`-dim tensor of (negative) penicillin yield, CO2 and time.
     """

     _ref_point = [1.85, 86.93, 514.70]

     Y_xs = 0.45
     Y_ps = 0.90
     K_1 = 10 ** (-10)
     K_2 = 7 * 10 ** (-5)
     m_X = 0.014
     alpha_1 = 0.143
     alpha_2 = 4 * 10 ** (-7)
     alpha_3 = 10 ** (-4)
     mu_X = 0.092
     K_X = 0.15
     mu_p = 0.005
     K_p = 0.0002
     K_I = 0.10
     K = 0.04
     k_g = 7.0 * 10 ** 3
     E_g = 5100.0
     k_d = 10.0 ** 33
     E_d = 50000.0
     lambd = 2.5 * 10 ** (-4)
     T_v = 273.0  # Kelvin
     T_o = 373.0
     R = 1.9872  # CAL/(MOL K)
     V_max = 180.0

     X_input = (X_input * (tf.constant(_penicillin_ub, dtype=X_input.dtype) -
                          tf.constant(_penicillin_lb, dtype=X_input.dtype)) + \
                          tf.constant(_penicillin_lb, dtype=X_input.dtype)).numpy()
     V, X, T, S, F, s_f, H_ = np.split(X_input, 7, axis=-1)
     P, CO2 = tf.zeros_like(V).numpy(), tf.zeros_like(V).numpy()

     # H = torch.full_like(H_, 10.0).pow(-H_)
     H = ((tf.ones_like(H_) * 10.0) ** (-H_)).numpy()

     active = tf.ones_like(V, dtype=tf.bool).numpy()
     t_tensor = (2500 * tf.ones_like(V)).numpy()

     for t in range(1, 2501):
         if np.sum(active) == 0:
             break
         F_loss = (
             V[active]
             * lambd
             * (tf.exp(5 * ((T[active] - T_o) / (T_v - T_o))) - 1)
         )
         dV_dt = F[active] - F_loss
         mu = (
             (mu_X / (1 + K_1 / H[active] + H[active] / K_2))
             * (S[active] / (K_X * X[active] + S[active]))
             * (
                 (k_g * tf.exp(-E_g / (R * T[active])))
                 - (k_d * tf.exp(-E_d / (R * T[active])))
             )
         )
         dX_dt = mu * X[active] - (X[active] / V[active]) * dV_dt
         mu_pp = mu_p * (
             S[active] / (K_p + S[active] + S[active] ** 2 / K_I)
         )
         dS_dt = (
             -(mu / Y_xs) * X[active]
             - (mu_pp / Y_ps) * X[active]
             - m_X * X[active]
             + F[active] * s_f[active] / V[active]
             - (S[active] / V[active]) * dV_dt
         )
         dP_dt = (
             (mu_pp * X[active])
             - K * P[active]
             - (P[active] / V[active]) * dV_dt
         )
         dCO2_dt = alpha_1 * dX_dt + alpha_2 * X[active] + alpha_3

         # UPDATE
         # P = tf.where(active, tf.expand_dims(P[active] + dP_dt, -1), P)
         P[active] = P[active] + dP_dt  # Penicillin concentration
         # P[active] = P[active] + dP_dt  # Penicillin concentration
         # V = tf.where(active, tf.expand_dims(V[active] + dV_dt, -1), V)  # Culture medium volume
         V[active] = V[active] + dV_dt  # Culture medium volume
         # X = tf.where(active, tf.expand_dims(X[active] + dX_dt, -1), X)  # Culture medium volume
         X[active] = X[active] + dX_dt  # Biomass concentration
         # S = tf.where(active, tf.expand_dims(S[active] + dS_dt, -1), S)  # Culture medium volume
         S[active] = S[active] + dS_dt  # Glucose concentration
         # CO2 = tf.where(active, tf.expand_dims(CO2[active] + dCO2_dt, -1), CO2)  # Culture medium volume
         CO2[active] = CO2[active] + dCO2_dt  # CO2 concentration

         # Update active indices
         full_dpdt = np.ones_like(P)
         full_dpdt[active] = dP_dt
         inactive = (V > V_max) + (S < 0) + (full_dpdt < 10e-12)
         t_tensor[inactive] = np.minimum(
             t_tensor[inactive], np.ones_like(t_tensor[inactive]) * t
         )
         active[inactive] = 0

     return tf.convert_to_tensor(np.concatenate([-P, CO2, t_tensor], axis=-1), dtype=X_input.dtype)


class TwoBarTruss(ConstraintMultiObjectiveTestProblem):
    """
    Two Bar Truss problem, refer :cite:`chafekar2003constrained`
    Note there are some zero division in the design space: when either one of x1, x2 is 0
    """

    bounds = [[0, 0, 0], [1, 1, 1]]  # [[0, 0, 1], [0.01, 0.01, 3]]
    dim = 3

    def objective(self) -> Callable[[TensorType], TensorType]:
        return two_bar_truss_obj

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return two_bar_truss_con

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass


_two_bar_truss_lb = [1e-5, 1e-5, 1]  # the 1e-5 is used to prevent having zero division in problem
_two_bar_truss_ub = [0.01, 0.01, 3]


def two_bar_truss_obj(x: TensorType) -> TensorType:
    x = x * (tf.constant(_two_bar_truss_ub, dtype=x.dtype) - tf.constant(_two_bar_truss_lb, dtype=x.dtype)) + \
        tf.constant(_two_bar_truss_lb, dtype=x.dtype)
    obj1 = x[..., 0] * tf.sqrt(16 + x[..., 2] ** 2) + x[..., 1] * tf.sqrt(1 + x[..., 2] ** 2)
    sigma_1 = 20 * tf.sqrt(16 + x[..., 2] ** 2) / (x[..., 0] * x[..., 2])
    sigma_2 = 80 * tf.sqrt(1 + x[..., 2] ** 2) / (x[..., 1] * x[..., 2])
    obj2 = tf.maximum(sigma_1, sigma_2)
    return tf.stack([obj1, obj2], axis=-1)


def two_bar_truss_con(x: TensorType) -> TensorType:
    x = x * (tf.constant(_two_bar_truss_ub, dtype=x.dtype) - tf.constant(_two_bar_truss_lb, dtype=x.dtype)) + \
        tf.constant(_two_bar_truss_lb, dtype=x.dtype)
    sigma_1 = 20 * tf.sqrt(16 + x[..., 2] ** 2) / (x[..., 0] * x[..., 2])
    sigma_2 = 80 * tf.sqrt(1 + x[..., 2] ** 2) / (x[..., 1] * x[..., 2])
    con = 10 ** 5 - tf.maximum(sigma_1, sigma_2)
    return con[..., tf.newaxis]


_welded_beam_design_lb = [0.125, 0.125, 0.1, 0.1]
_welded_beam_design_ub = [5, 5, 10, 10]


class WeldedBeamDesign(ConstraintMultiObjectiveTestProblem):
    """
    Welded Beam Design problem, refer :cite:`chafekar2003constrained`
    This is a difficult problem even for non-BO (genetic algorithm)
    """

    bounds = [[0, 0, 0, 0], [1, 1, 1, 1]]  # [[0.125, 0.125, 0.1, 0.1], [5, 5, 10, 10]]
    dim = 4

    def objective(self) -> Callable[[TensorType], TensorType]:
        return welded_beam_design_obj

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return welded_beam_design_con

    def gen_pareto_optimal_points(self, n: int, seed=None) -> tf.Tensor:
        pass


def welded_beam_design_obj(x: TensorType) -> TensorType:
    x = x * (tf.constant(_welded_beam_design_ub, dtype=x.dtype) - tf.constant(_welded_beam_design_lb, dtype=x.dtype)) + \
        tf.constant(_welded_beam_design_lb, dtype=x.dtype)
    h = x[..., -4]
    b = x[..., -3]
    l = x[..., -2]
    t = x[..., -1]
    obj1 = 1.10471 * h ** 2 * l + 0.04811 * t * b * (14 + l)
    obj2 = 2.1952 / (t ** 3 * b)

    return tf.stack([obj1, obj2], axis=-1)


def welded_beam_design_con(x: TensorType) -> TensorType:
    x = x * (tf.constant(_welded_beam_design_ub, dtype=x.dtype) - tf.constant(_welded_beam_design_lb, dtype=x.dtype)) + \
        tf.constant(_welded_beam_design_lb, dtype=x.dtype)
    h = x[..., -4]
    b = x[..., -3]
    l = x[..., -2]
    t = x[..., -1]
    tau_prime = 6000.0 / (tf.cast(tf.sqrt(2.0), dtype=x.dtype) * h * l)
    tau_double_prime = (6000.0 * (14 + 0.5 * l) * tf.sqrt(0.25 * (l ** 2 + (h + t) ** 2))) / (
            2 * tf.cast(tf.sqrt(2.0), dtype=x.dtype) * h * l * (l ** 2 / 12 + 0.25 * (h + t) ** 2)
    )
    tau = tf.sqrt(
        tau_prime ** 2
        + tau_double_prime ** 2
        + l * tau_prime * tau_double_prime / tf.sqrt(0.25 * (l ** 2 + (h + t) ** 2))
    )
    sigma = 504000.0 / (t ** 2 * b)
    p_c = 64746.022 * (1.0 - 0.0282346 * t) * (t * b ** 3)
    c1 = 13600.0 - tau
    c2 = 30000.0 - sigma
    c3 = b - h
    c4 = p_c - 6000.0
    return tf.stack([c1, c2, c3, c4], axis=-1)


_disc_brake_design_lb = [55, 75, 1000, 11]
_disc_brake_design_ub = [80, 110, 3000, 20]

class DiscBrakeDesign(ConstraintMultiObjectiveTestProblem):
    """
    Refer: https://github.com/ryojitanabe/reproblems/blob/master/doc/re-supplementary_file.pdf
    """
    bounds = [[0.0] * 4, [1.0] * 4]  # [[0.125, 0.125, 0.1, 0.1], [5, 5, 10, 10]]
    dim = 4

    def objective(self) -> Callable[[TensorType], TensorType]:
        return disc_brake_design_object

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return disc_brake_design_constraint

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


def disc_brake_design_object(x: TensorType) -> TensorType:
    x = x * (tf.constant(_disc_brake_design_ub, dtype=x.dtype) - tf.constant(_disc_brake_design_lb, dtype=x.dtype)) + \
        tf.constant(_disc_brake_design_lb, dtype=x.dtype)
    x1 = x[:, 0, None]
    x2 = x[:, 1, None]
    x3 = x[:, 2, None]
    x4 = x[:, 3, None]

    # First original objective function
    f_0 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
    # Second original objective function
    f_1 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

    return tf.concat([f_0, f_1], axis=1)


def disc_brake_design_constraint(x: TensorType) -> TensorType:
    x = x * (tf.constant(_disc_brake_design_ub, dtype=x.dtype) - tf.constant(_disc_brake_design_lb, dtype=x.dtype)) + \
        tf.constant(_disc_brake_design_lb, dtype=x.dtype)
    x1 = x[:, 0, None]
    x2 = x[:, 1, None]
    x3 = x[:, 2, None]
    x4 = x[:, 3, None]
    # Reformulated objective functions
    g_0 = (x2 - x1) - 20.0
    g_1 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
    g_2 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
    g_3 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0

    return tf.concat([g_0, g_1, g_2, g_3], axis=1)


class ConecptualMarineDesign(ConstraintMultiObjectiveTestProblem):
    pass


class WaterProblem(ConstraintMultiObjectiveTestProblem):
    """
    Note there are https://github.com/ryojitanabe/reproblems/blob/28845742cc72910e301d5c9d1d806ef54185c074/reproblem_python_ver/reproblem.py#L1206

    Refer:
        Jain, H., & Deb, K. (2013). An evolutionary many-objective optimization algorithm using
    reference-point based nondominated sorting approach, part II: Handling constraints and
    extending to an adaptive approach. IEEE Transactions on evolutionary computation, 18(4), 602-622.
    """
    bounds = [[0.0] * 3, [1.0] * 3]  # [[0.125, 0.125, 0.1, 0.1], [5, 5, 10, 10]]
    dim = 3

    def objective(self) -> Callable[[TensorType], TensorType]:
        return waterproblem_objective

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return waterproblem_constraint

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


_waterproblem_lb = [0.01, 0.01, 0.01]
_waterproblem_ub = [0.45, 0.1, 0.1]


def waterproblem_objective(x: TensorType) -> TensorType:
    x = x * (tf.constant(_waterproblem_ub, dtype=x.dtype) - tf.constant(_waterproblem_lb, dtype=x.dtype)) + \
        tf.constant(_waterproblem_lb, dtype=x.dtype)
    x1 = x[:, 0, None]
    x2 = x[:, 1, None]
    x3 = x[:, 2, None]

    # According to "MULTIOBJECTIVE DESIGN OPTIMIZATION BY AN EVOLUTIONARY ALGORITHM" 2001, the result are scaled
    f1 = 106780.37 * (x2 + x3) + 61704.67
    f2 = 3000.0 * x1
    # f3 = 305700 * 2289 * x2 / (0.06 * 2289) ** 0.65 # 14 年是0， 之前的是00
    f3 = 305700 * 2289 * x2 / (0.06 * 2289) ** 0.65 # 14 年是0， 之前的是00
    f4 = 250 * 2289 * tf.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
    f5 = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80.0 )
    return  tf.concat([f1 / 80000, f2 / 1500, f3 / 3000000, f4 / 6000000, f5 / 32000], axis=1) # we modified the scaling coefficient for f5


def waterproblem_constraint(x: TensorType) -> TensorType:
    x = x * (tf.constant(_waterproblem_ub, dtype=x.dtype) - tf.constant(_waterproblem_lb, dtype=x.dtype)) + \
        tf.constant(_waterproblem_lb, dtype=x.dtype)
    x1 = x[:, 0, None]
    x2 = x[:, 1, None]
    x3 = x[:, 2, None]

    g1 = 1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)
    g2 = 1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)
    g3 = 50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)
    g4 = 16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)
    g5 = 10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)
    g6 = 2000 - (0.417 / (x1 * x2) + 1721.26 * x3 - 136.54)
    g7 = 550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48)

    return tf.concat([g1, g2, g3/50000, g4/17000, g5/11000, g6/2100, g7/600], axis=1)


class VLMOP2BraninCurrin(ConstraintMultiObjectiveTestProblem):
    bounds = [[0.0] * 2, [1.0] * 2]  # [[0.125, 0.125, 0.1, 0.1], [5, 5, 10, 10]]
    dim = 2
    def objective(self) -> Callable[[TensorType], TensorType]:
        return vlmop2branin_objective

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return vlmop2branin_constraint

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


vlmop2_lbs = [-2.0] * 2
vlmop2_ubs = [2.0] * 2

def vlmop2branin_objective(x: TensorType) -> TensorType:
    vlmop2_x = x * (tf.constant(vlmop2_ubs, dtype=x.dtype) - tf.constant(vlmop2_lbs, dtype=x.dtype)) + \
        tf.constant(vlmop2_lbs, dtype=x.dtype)
    return tf.concat([vlmop2(vlmop2_x) / tf.constant([1.2, 1.2], dtype=x.dtype), branincurrin(x) / tf.constant([180.0, 6.0], dtype=x.dtype)], axis=-1)


def vlmop2branin_constraint(x: TensorType) -> TensorType:
    return evaluate_slack_true(x)


class VLMOP2ConstrEx(ConstraintMultiObjectiveTestProblem):
    bounds = [[0.0] * 2, [1.0] * 2]  # [[0.125, 0.125, 0.1, 0.1], [5, 5, 10, 10]]
    dim = 2
    def objective(self) -> Callable[[TensorType], TensorType]:
        return vlmop2constr_ex_objective

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return vlmop2constr_ex_constraint

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


def vlmop2constr_ex_objective(x:TensorType) -> TensorType:
    vlmop2_x = x * (tf.constant(vlmop2_ubs, dtype=x.dtype) - tf.constant(vlmop2_lbs, dtype=x.dtype)) + \
               tf.constant(vlmop2_lbs, dtype=x.dtype)
    return tf.concat([vlmop2(vlmop2_x),
                      constr_ex(x) / tf.constant([9.0, 8.0], dtype=x.dtype)], axis=-1)


def vlmop2constr_ex_constraint(x: TensorType) -> TensorType:
    return constr_ex_cons_func(x)



# class C1DTLZ3(DTLZ3, ConstraintMultiObjectiveTestProblem):
#     """
#     Eq. 5 of
#     Jain, H., & Deb, K. (2013). An evolutionary many-objective optimization algorithm using
#     reference-point based nondominated sorting approach, part II: Handling constraints and
#     extending to an adaptive approach. IEEE Transactions on evolutionary computation, 18(4), 602-622.
#     """
#
#     def constraint(self):
#         return partial(c1dtlz3_constraint, objective_func=self.objective())
#
#
# def c1dtlz3_constraint(x: TensorType, objective_func: Callable, r=11) -> TensorType:
#     fi = objective_func(x)
#     first_part = tf.reduce_sum(fi ** 2, axis=-1, keepdims=True)  - 16
#     second_part = tf.reduce_sum(fi ** 2, axis=-1, keepdims=True)  - r ** 2
#     return (first_part * second_part) / 1E7


# class C2DTLZ2(DTLZ2, ConstraintMultiObjectiveTestProblem):
#     """
#     from https://link.springer.com/content/pdf/10.1007/978-3-319-91341-4.pdf
#     page 110
#     """
#
#     def constraint(self):
#         return partial(c2dtlz2_constraint, objective_func=self.objective())
#
#
# def c2dtlz2_constraint(x: TensorType, objective_func: Callable, r=0.2) -> TensorType:
#     """
#     The param `r` is copied from https://github.com/pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py#:~:text=num_constraints%20%3D%201-,_r%20%3D%200.2,-%23%20approximate%20from%20nsga
#     Verified the same expected output as botorch's code
#     """
#     f_X = objective_func(x)
#     M = f_X.shape[-1]
#     term1 = (f_X - 1) ** 2
#     # mask = ~(torch.eye(f_X.shape[-1], device=f_X.device).bool())
#     # indices = torch.arange(f_X.shape[1], device=f_X.device).repeat(f_X.shape[1], 1)
#     # indexer = indices[mask].view(f_X.shape[1], f_X.shape[-1] - 1)
#     index = tf.ones(shape=(M, M), dtype=tf.int32) - tf.eye(M, dtype=tf.int32)
#     # term2_inner = (
#     #     f_X.unsqueeze(1)
#     #     .expand(f_X.shape[0], f_X.shape[-1], f_X.shape[-1])
#     #     .gather(dim=-1, index=indexer.repeat(f_X.shape[0], 1, 1))
#     # )
#     # [M, M-1]
#     aug_f_X = tf.tile(tf.expand_dims(f_X, -2), [1, M, 1])
#     term2_inner = tf.ragged.boolean_mask(  # [..., M, M-1]
#         aug_f_X, tf.cast(tf.tile(tf.expand_dims(index, -3),
#                                  [f_X.shape[0], 1, 1]), dtype=tf.bool)).to_tensor()
#
#     # term2_inner = tf.gather(f_X, index)
#     # (f - 1/\sqrt{M})^2 - r^2
#     term2 = tf.reduce_sum(term2_inner ** 2 - r ** 2, axis=-1)
#     # min1: min_{i=1}^M [fi(x)-1]^2 + Sum (fj - r)
#     min1 = tf.reduce_min(term1 + term2, axis=-1, keepdims=True)
#     # min2:
#     min2 = tf.reduce_sum((f_X - 1 / math.sqrt(M)) ** 2 - r ** 2, axis=-1, keepdims=True)
#     return -tf.minimum(min1, min2)
#
#
# class C3DTLZ1(DTLZ4, ConstraintMultiObjectiveTestProblem):
#     def constraint(self):
#         return partial(c3dtlz1_constraint, objective_func=self.objective())
#
#
# def c3dtlz1_constraint(x: TensorType, objective_func: Callable, r=0.2) -> TensorType:
#     raise NotImplementedError
#
#
# class C3DTLZ4(DTLZ4, ConstraintMultiObjectiveTestProblem):
#     """
#     Eq. 8 of
#     Jain, H., & Deb, K. (2013). An evolutionary many-objective optimization algorithm using
#     reference-point based nondominated sorting approach, part II: Handling constraints and
#     extending to an adaptive approach. IEEE Transactions on evolutionary computation, 18(4), 602-622.
#     """
#     def constraint(self):
#         return partial(c3dtlz4_constraint, objective_func=self.objective(), objective_number = self.M)
#
#
# def c3dtlz4_constraint(x: TensorType, objective_func: Callable, objective_number:int) -> TensorType:
#     fi = objective_func(x)
#     fj = (fi ** 2)/4  # [N, D]
#     cj = tf.concat([fj[:, idx, None] + tf.reduce_sum(
#         tf.concat([fi[:, :idx] ** 2, fi[:, idx+1:] ** 2], axis=-1), axis=-1, keepdims=True) - 1
#                     for idx in range(objective_number)], axis=-1)
#     return cj
#
#
# class C3DTLZ5(DTLZ5, ConstraintMultiObjectiveTestProblem):
#     """
#     Eq. 8 of
#     Jain, H., & Deb, K. (2013). An evolutionary many-objective optimization algorithm using
#     reference-point based nondominated sorting approach, part II: Handling constraints and
#     extending to an adaptive approach. IEEE Transactions on evolutionary computation, 18(4), 602-622.
#     """
#     def constraint(self):
#         return partial(c3dtlz5_constraint, objective_func=self.objective(), objective_number = self.M)
#
#
# def c3dtlz5_constraint(x: TensorType, objective_func: Callable, objective_number:int) -> TensorType:
#     fi = objective_func(x)
#     fj = (fi ** 2)/4  # [N, D]
#     cj = tf.concat([fj[:, idx, None] + tf.reduce_sum(
#         tf.concat([fi[:, :idx] ** 2, fi[:, idx+1:] ** 2], axis=-1), axis=-1, keepdims=True) - 1
#                     for idx in range(objective_number)], axis=-1)
#     return cj


# from microwaveopt.lib_example.ex6.ex6_blackbox import blackbox_constr

ee_6_bounds = [[6.5, 2.5, 0.2, 9.0], [8.0, 3.5, 0.3, 11]]
ee_6_extended_vebounds = [[6.5, 2.0, 0.040, 0.8, 0.2, 9.0], [8.0, 3.5, 0.08, 1.2, 0.3, 11]]


# class EE6(ConstraintMultiObjectiveTestProblem):
#     dim = None
#     restricted_bounds = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
#     bounds = restricted_bounds
#     restricted_dim = 4
#     extended_bounds = [[0.0] * 6, [1.0] * 6]
#     extended_dim = 6
#
#     def __init__(self, input_dim = 4, exp_id = None):
#         if input_dim == 4:
#             self._use_restricted_pb = True
#         elif input_dim == 6:
#             self._use_restricted_pb = False
#         else:
#             raise NotImplementedError
#         self.exp_id = exp_id
#
#     def objective(self) -> Callable[[TensorType], TensorType]:
#         pass
#
#     def constraint(self) -> Callable[[TensorType], TensorType]:
#         pass
#
#     def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
#         pass
#
#     # /usr/local/ADS2015_01
#     def joint_objective_con(self) -> Callable:
#         def obj_con_wrapper(xs: TensorType) -> tuple[TensorType, TensorType]:
#             if self._use_restricted_pb:
#                 pass
#                 # xs = xs * (tf.constant(ee_6_bounds[-1], dtype=xs.dtype) - tf.constant(ee_6_bounds[0], dtype=xs.dtype)) + \
#                 #      tf.constant(ee_6_bounds[0], dtype=xs.dtype)
#             else:
#                 xs = xs * (tf.constant(ee_6_extended_vebounds[-1], dtype=xs.dtype) - tf.constant(ee_6_extended_vebounds[0], dtype=xs.dtype)) + \
#                      tf.constant(ee_6_extended_vebounds[0], dtype=xs.dtype)
#             obj1s = []
#             obj2s = []
#             constrs = []
#             for x in xs:
#                 if self._use_restricted_pb:
#                     x_aug = [x[0], x[1], 0.06, 1, x[2], x[3]]
#                 else:
#                     x_aug = [x[0], x[1], x[2], x[3], x[4], x[5]]
#                 print(f'input: {tf.convert_to_tensor(x_aug)}')
#                 print(f'exp id: {self.exp_id}')
#                 while True:
#                     obj1, obj2, constr = blackbox_constr(x_aug, debug=False, afs=False, exp_id=self.exp_id)
#                     if obj2 != 4.0 and obj1 != 2.0:
#                         obj1s.append(obj1)
#                         obj2s.append(obj2)
#                         constrs.append(constr)
#                         break
#                     else:
#                         print('Random sampling for a new input')
#                         if self._use_restricted_pb:
#                             x = Box(*self.restricted_bounds).sample(1)[0]
#                             x_aug = [x[0], x[1], 0.06, 1, x[2], x[3]]
#                         else:
#                             x = Box(*self.extended_bounds).sample(1)[0]
#                             x_aug = [x[0], x[1], x[2], x[3], x[4], x[5]]
#             print(f'obj1: {obj1s}')
#             print(f'obj2s: {obj2s}')
#             print(f'constrs: {constrs}')
#             return tf.stack([tf.constant(obj1s, dtype=xs.dtype), tf.constant(obj2s, dtype=xs.dtype)], axis=-1), \
#                    0.12 - tf.constant(constrs, dtype=xs.dtype)[..., tf.newaxis]
#                    # - tf.constant(constrs, dtype=xs.dtype)[..., tf.newaxis] - 0.12
#
#         return obj_con_wrapper


class RocketInjectorDesign(MultiObjectiveTestProblem):
    bounds = [[0.0] * 4, [1.0] * 4]  # [[0.125, 0.125, 0.1, 0.1], [5, 5, 10, 10]]
    dim = 4
    def objective(self) -> Callable[[TensorType], TensorType]:
        return rocket_objective

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


def rocket_objective(xs: TensorType) -> TensorType:
    xAlpha = xs[:, 0, None]
    xHA = xs[:, 1, None]
    xOA = xs[:, 2, None]
    xOPTT = xs[:, 3, None]

    # f1 (TF_max)
    f0 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (
                0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (
                       0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (
                       0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
    # f2 (X_cc)
    f1 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (
                0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (
                       0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (
                       0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
    # f3 (TT_max)
    f2 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (
                0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (
                       0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (
                       0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                       0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                       0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

    return tf.concat([f0, f1, f2], axis=-1)


conceptual_marine_design_lb = [150.0, 20.0, 13.0, 10.0, 14.0, 0.63]
conceptual_marine_design_ub = [274.32, 32.31, 25.0, 11.71, 18.0, 0.75]


class ConceptualMarineDesign(ConstraintMultiObjectiveTestProblem):
    bounds = [[0.0] * 6, [1.0] * 6]
    dim = 6

    def objective(self) -> Callable[[TensorType], TensorType]:
        return conceptual_marine_design_objective

    def constraint(self) -> Callable[[TensorType], TensorType]:
        return conceptual_marine_design_constraint

    def gen_pareto_optimal_points(self, n: int, seed: int | None = None) -> TensorType:
        pass


def conceptual_marine_design_objective(xs: TensorType):
    # NOT g
    constraintFuncs = np.zeros(9)
    xs = xs * (tf.constant(conceptual_marine_design_ub, dtype=xs.dtype) -
               tf.constant(conceptual_marine_design_lb, dtype=xs.dtype)) + \
         tf.constant(conceptual_marine_design_lb, dtype=xs.dtype)
    x_L = xs[:, 0, None]
    x_B = xs[:, 1, None]
    x_D = xs[:, 2, None]
    x_T = xs[:, 3, None]
    x_Vk = xs[:, 4, None]
    x_CB = xs[:, 5, None]

    displacement = 1.025 * x_L * x_B * x_T * x_CB
    V = 0.5144 * x_Vk
    g = 9.8065
    Fn = V / np.power(g * x_L, 0.5)
    a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
    b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

    power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
    outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
    steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
    machinery_weight = 0.17 * np.power(power, 0.9)
    light_ship_weight = steel_weight + outfit_weight + machinery_weight

    ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (
                2400.0 * np.power(power, 0.8)))
    capital_costs = 0.2 * ship_cost

    DWT = displacement - light_ship_weight

    running_costs = 40000.0 * np.power(DWT, 0.3)

    round_trip_miles = 5000.0
    sea_days = (round_trip_miles / 24.0) * x_Vk
    handling_rate = 8000.0

    daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
    fuel_price = 100.0
    fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
    port_cost = 6.3 * np.power(DWT, 0.8)

    fuel_carried = daily_consumption * (sea_days + 5.0)
    miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)

    cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
    port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
    RTPA = 350.0 / (sea_days + port_days)

    voyage_costs = (fuel_cost + port_cost) * RTPA
    annual_costs = capital_costs + running_costs + voyage_costs
    annual_cargo = cargo_DWT * RTPA

    f0 = annual_costs / annual_cargo
    f1 = light_ship_weight
    # f_2 is dealt as a minimization problem
    f2 = -annual_cargo

    return tf.concat([f0, f1, f2], axis=-1)


def conceptual_marine_design_constraint(xs: TensorType):
    # NOT g
    xs = xs * (tf.constant(conceptual_marine_design_ub, dtype=xs.dtype) -
               tf.constant(conceptual_marine_design_lb, dtype=xs.dtype)) + \
         tf.constant(conceptual_marine_design_lb, dtype=xs.dtype)
    x_L = xs[:, 0, None]
    x_B = xs[:, 1, None]
    x_D = xs[:, 2, None]
    x_T = xs[:, 3, None]
    x_Vk = xs[:, 4, None]
    x_CB = xs[:, 5, None]

    displacement = 1.025 * x_L * x_B * x_T * x_CB
    V = 0.5144 * x_Vk
    g = 9.8065
    Fn = V / np.power(g * x_L, 0.5)
    a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
    b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

    power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
    outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
    steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
    machinery_weight = 0.17 * np.power(power, 0.9)
    light_ship_weight = steel_weight + outfit_weight + machinery_weight

    DWT = displacement - light_ship_weight


    # Reformulated objective functions
    cons0 = (x_L / x_B) - 6.0
    cons1 = -(x_L / x_D) + 15.0
    cons2 = -(x_L / x_T) + 19.0
    cons3 = 0.45 * np.power(DWT, 0.31) - x_T
    cons4 = 0.7 * x_D + 0.7 - x_T
    cons5 = 500000.0 - DWT
    cons6 = DWT - 3000.0
    cons7 = 0.32 - Fn

    KB = 0.53 * x_T
    BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
    KG = 1.0 + 0.52 * x_D
    cons8 = (KB + BMT - KG) - (0.07 * x_B)
    # constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)

    return tf.concat([cons0, cons1, cons2, cons3, cons4, cons5, cons6, cons7, cons8], axis=-1)

def DTLZ2(input_dim: int, num_objective: int) -> MultiObjectiveTestProblem:
    """
    The DTLZ2 problem, the idea pareto fronts lie on (part of) a unit hyper sphere.
    See :cite:`deb2002scalable` for details.

    :param input_dim: The input dimensionality of the synthetic function.
    :param num_objective: The number of objectives.
    :return: The problem specification.
    """
    M, k, d = dtlz_mkd(input_dim, num_objective)

    def gen_pareto_optimal_points(n: int, seed: int | None = None) -> TensorType:
        tf.debugging.assert_greater_equal(M, 2)
        rnd = tf.random.normal([n, M], seed=seed, dtype=tf.float64)
        samples = tf.abs(rnd / tf.norm(rnd, axis=-1, keepdims=True))
        return samples

    return MultiObjectiveTestProblem(
        name=f"DTLZ2({input_dim}, {num_objective})",
        objective=partial(dtlz2, m=M, d=d),
        search_space=Box([0.0], [1.0]) ** d,
        gen_pareto_optimal_points=gen_pareto_optimal_points,
    )
