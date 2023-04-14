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
from __future__ import annotations

from typing import Callable, Optional

import pytest
import tensorflow as tf

from trieste.acquisition.multi_objective.pareto import Pareto, get_reference_point
from trieste.acquisition.multi_objective.utils import moo_nsga2_pymoo
from trieste.objectives.multi_objectives import VLMOP2
from trieste.types import TensorType


@pytest.mark.parametrize(
    "pb_objective_function, reference_pareto_front, "
    "input_dim, obj_num, bounds, popsize, pb_constraint_function, "
    "cons_num, tolerable_threshold",
    [
        (
            VLMOP2().objective(),
            VLMOP2().gen_pareto_optimal_points(100),
            2,
            2,
            [tf.constant([-2, -2]), tf.constant([2, 2])],
            100,
            None,
            0,
            2e-3,
        ),
    ],
)
def test_moo_nsga2_pymoo_can_find_true_Pareto_front(
    pb_objective_function: Callable[[TensorType], TensorType],
    reference_pareto_front: TensorType,
    input_dim: int,
    obj_num: int,
    bounds: tuple,
    popsize: int,
    pb_constraint_function: Optional[Callable[[TensorType], TensorType]],
    cons_num: int,
    tolerable_threshold: float,
) -> None:

    pf = moo_nsga2_pymoo(
        f=pb_objective_function,
        input_dim=input_dim,
        obj_num=obj_num,
        bounds=bounds,
        popsize=popsize,
        cons_num=cons_num,
    )

    ref_point = get_reference_point(reference_pareto_front)
    assert (
        Pareto(reference_pareto_front).hypervolume_indicator(ref_point)
        - Pareto(tf.cast(pf, reference_pareto_front.dtype)).hypervolume_indicator(ref_point)
    ) < tolerable_threshold


def test_sample_pareto_fronts_from_parametric_gp_posterior_works_with_discrete_strategy():
    pass


def test_sample_pareto_fronts_from_parametric_gp_posterior_works_with_continuous_strategy():
    pass
