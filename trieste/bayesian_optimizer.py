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
This module contains the :class:`BayesianOptimizer` class, used to perform Bayesian optimization.
"""

from __future__ import annotations

import copy
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Dict,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    TypeVar,
    cast,
    overload,
)

import absl
import dill
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist

from .acquisition.multi_objective import non_dominated

try:
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError:
    pd = None
    sns = None

import os

from matplotlib import pyplot as plt

from trieste.experimental.plotting import (
    plot_mobo_points_in_obj_space,
    plot_pf_dominated_region_boarder_2d,
    view_2D_function_in_contourf,
    view_2D_function_in_contour
)
from trieste.utils.optimal_recommenders.multi_objective import inference_pareto_fronts_from_gp_mean

from . import logging
from .acquisition.function.multi_objective import PF2ES
from .acquisition.multi_objective.utils import sample_pareto_fronts_from_parametric_gp_posterior
from .acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from .data import Dataset
from .observer import CONSTRAINT
from .models import TrainableProbabilisticModel
from .observer import OBJECTIVE, Observer
from .space import SearchSpace
from .types import State, Tag, TensorType
from .utils import Err, Ok, Result, Timer

StateType = TypeVar("StateType")
""" Unbound type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace)
""" Type variable bound to :class:`SearchSpace`. """

TrainableProbabilisticModelType = TypeVar(
    "TrainableProbabilisticModelType", bound=TrainableProbabilisticModel, contravariant=True
)
""" Contravariant type variable bound to :class:`TrainableProbabilisticModel`. """

EarlyStopCallback = Callable[
    [Mapping[Tag, Dataset], Mapping[Tag, TrainableProbabilisticModelType], Optional[StateType]],
    bool,
]
""" Early stop callback type, generic in the model and state types. """


@dataclass(frozen=True)
class Record(Generic[StateType]):
    """Container to record the state of each step of the optimization process."""

    datasets: Mapping[Tag, Dataset]
    """ The known data from the observer. """

    models: Mapping[Tag, TrainableProbabilisticModel]
    """ The models over the :attr:`datasets`. """

    acquisition_state: StateType | None
    """ The acquisition state. """

    @property
    def dataset(self) -> Dataset:
        """The dataset when there is just one dataset."""
        if len(self.datasets) == 1:
            return next(iter(self.datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(self.datasets)}")

    @property
    def model(self) -> TrainableProbabilisticModel:
        """The model when there is just one dataset."""
        if len(self.models) == 1:
            return next(iter(self.models.values()))
        else:
            raise ValueError(f"Expected a single model, found {len(self.models)}")

    def save(self, path: Path | str) -> FrozenRecord[StateType]:
        """Save the record to disk. Will overwrite any existing file at the same path."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)
        return FrozenRecord(Path(path))


@dataclass(frozen=True)
class FrozenRecord(Generic[StateType]):
    """
    A Record container saved on disk.

    Note that records are saved via pickling and are therefore neither portable nor secure.
    Only open frozen records generated on the same system.
    """

    path: Path
    """ The path to the pickled Record. """

    def load(self) -> Record[StateType]:
        """Load the record into memory."""
        with open(self.path, "rb") as f:
            return dill.load(f)

    @property
    def datasets(self) -> Mapping[Tag, Dataset]:
        """The known data from the observer."""
        return self.load().datasets

    @property
    def models(self) -> Mapping[Tag, TrainableProbabilisticModel]:
        """The models over the :attr:`datasets`."""
        return self.load().models

    @property
    def acquisition_state(self) -> StateType | None:
        """The acquisition state."""
        return self.load().acquisition_state

    @property
    def dataset(self) -> Dataset:
        """The dataset when there is just one dataset."""
        return self.load().dataset

    @property
    def model(self) -> TrainableProbabilisticModel:
        """The model when there is just one dataset."""
        return self.load().model


# this should be a generic NamedTuple, but mypy doesn't support them
#  https://github.com/python/mypy/issues/685
@dataclass(frozen=True)
class OptimizationResult(Generic[StateType]):
    """The final result, and the historical data of the optimization process."""

    final_result: Result[Record[StateType]]
    """
    The final result of the optimization process. This contains either a :class:`Record` or an
    exception.
    """

    history: list[Record[StateType] | FrozenRecord[StateType]]
    r"""
    The history of the :class:`Record`\ s from each step of the optimization process. These
    :class:`Record`\ s are created at the *start* of each loop, and as such will never
    include the :attr:`final_result`. The records may be either in memory or on disk.
    """

    @staticmethod
    def step_filename(step: int, num_steps: int) -> str:
        """Default filename for saved optimization steps."""
        return f"step.{step:0{len(str(num_steps - 1))}d}.pickle"

    STEP_GLOB: ClassVar[str] = "step.*.pickle"
    RESULTS_FILENAME: ClassVar[str] = "results.pickle"

    def astuple(
        self,
    ) -> tuple[Result[Record[StateType]], list[Record[StateType] | FrozenRecord[StateType]]]:
        """
        **Note:** In contrast to the standard library function :func:`dataclasses.astuple`, this
        method does *not* deepcopy instance attributes.

        :return: The :attr:`final_result` and :attr:`history` as a 2-tuple.
        """
        return self.final_result, self.history

    def try_get_final_datasets(self) -> Mapping[Tag, Dataset]:
        """
        Convenience method to attempt to get the final data.

        :return: The final data, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().datasets

    def try_get_final_dataset(self) -> Dataset:
        """
        Convenience method to attempt to get the final data for a single dataset run.

        :return: The final data, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single dataset run.
        """
        datasets = self.try_get_final_datasets()
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(datasets)}")

    def try_get_optimal_point(self) -> tuple[TensorType, TensorType, TensorType]:
        """
        Convenience method to attempt to get the optimal point for a single dataset,
        single objective run.

        :return: Tuple of the optimal query point, observation and its index.
        """
        dataset = self.try_get_final_dataset()
        if tf.rank(dataset.observations) != 2 or dataset.observations.shape[1] != 1:
            raise ValueError("Expected a single objective")
        arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
        return dataset.query_points[arg_min_idx], dataset.observations[arg_min_idx], arg_min_idx

    def try_get_final_models(self) -> Mapping[Tag, TrainableProbabilisticModel]:
        """
        Convenience method to attempt to get the final models.

        :return: The final models, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().models

    def try_get_final_model(self) -> TrainableProbabilisticModel:
        """
        Convenience method to attempt to get the final model for a single model run.

        :return: The final model, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single model run.
        """
        models = self.try_get_final_models()
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected single model, found {len(models)}")

    @property
    def loaded_history(self) -> list[Record[StateType]]:
        """The history of the optimization process loaded into memory."""
        return [record if isinstance(record, Record) else record.load() for record in self.history]

    def save_result(self, path: Path | str) -> None:
        """Save the final result to disk. Will overwrite any existing file at the same path."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            dill.dump(self.final_result, f, dill.HIGHEST_PROTOCOL)

    def save(self, base_path: Path | str) -> None:
        """Save the optimization result to disk. Will overwrite existing files at the same path."""
        path = Path(base_path)
        num_steps = len(self.history)
        self.save_result(path / self.RESULTS_FILENAME)
        for i, record in enumerate(self.loaded_history):
            record_path = path / self.step_filename(i, num_steps)
            record.save(record_path)

    @classmethod
    def from_path(cls, base_path: Path | str) -> OptimizationResult[StateType]:
        """Load a previously saved OptimizationResult."""
        try:
            with open(Path(base_path) / cls.RESULTS_FILENAME, "rb") as f:
                result = dill.load(f)
        except FileNotFoundError as e:
            result = Err(e)

        history: list[Record[StateType] | FrozenRecord[StateType]] = [
            FrozenRecord(file) for file in sorted(Path(base_path).glob(cls.STEP_GLOB))
        ]
        return cls(result, history)


class BayesianOptimizer(Generic[SearchSpaceType]):
    """
    This class performs Bayesian optimization, the data-efficient optimization of an expensive
    black-box *objective function* over some *search space*. Since we may not have access to the
    objective function itself, we speak instead of an *observer* that observes it.
    """

    def __init__(
        self,
        observer: Observer,
        search_space: SearchSpaceType,
        pb_name_prefix: Optional[str] = "",
        acq_name: Optional[str] = "",
    ):
        """
        :param observer: The observer of the objective function.
        :param search_space: The space over which to search. Must be a
            :class:`~trieste.space.SearchSpace`.
        :param pb_name_prefix the problem name, used as prefix of auxiliary files
        :param acq_name acquisition function name, used as identifier in auxiliary files
        """
        self._observer = observer
        self._search_space = search_space
        self._pb_name_prefix = pb_name_prefix
        self._acq_name = acq_name

    def __repr__(self) -> str:
        """"""
        return f"BayesianOptimizer({self._observer!r}, {self._search_space!r})"

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModel],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModel, object]
        ] = None,
    ) -> OptimizationResult[None]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, object]
        ] = None,
        # this should really be OptimizationResult[None], but tf.Tensor is untyped so the type
        # checker can't differentiate between TensorType and State[S | None, TensorType], and
        # the return types clash. object is close enough to None that object will do.
    ) -> OptimizationResult[object]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, object]
        ] = None,
    ) -> OptimizationResult[object]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
    ) -> OptimizationResult[StateType]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
    ) -> OptimizationResult[StateType]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModel,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModel, object]
        ] = None,
    ) -> OptimizationResult[None]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, object]
        ] = None,
    ) -> OptimizationResult[object]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, object]
        ] = None,
    ) -> OptimizationResult[object]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
    ) -> OptimizationResult[StateType]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
    ) -> OptimizationResult[StateType]:
        ...

    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset] | Dataset,
        models: Mapping[Tag, TrainableProbabilisticModelType] | TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            TensorType | State[StateType | None, TensorType],
            SearchSpaceType,
            TrainableProbabilisticModelType,
        ]
        | None = None,
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_initial_model: bool = True,
        inspect_in_obj_space: bool = False,
        inspect_inferred_pareto: bool = False,
        inspect_acq_contour: bool = False,
        kwargs_for_input_space_plot: dict = {},
        kwargs_for_obj_space_plot: dict = {},
        kwargs_for_inferred_pareto: dict = {},
        auxiliary_file_output_path: Optional[str] = "",
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
        generate_recommendation: bool = False
    ) -> OptimizationResult[StateType] | OptimizationResult[None]:
        """
        Attempt to find the minimizer of the ``observer`` in the ``search_space`` (both specified at
        :meth:`__init__`). This is the central implementation of the Bayesian optimization loop.

        For each step in ``num_steps``, this method:
            - Finds the next points with which to query the ``observer`` using the
              ``acquisition_rule``'s :meth:`acquire` method, passing it the ``search_space``,
              ``datasets``, ``models``, and current acquisition state.
            - Queries the ``observer`` *once* at those points.
            - Updates the datasets and models with the data from the ``observer``.

        If any errors are raised during the optimization loop, this method will catch and return
        them instead and print a message (using `absl` at level `absl.logging.ERROR`).
        If ``track_state`` is enabled, then in addition to the final result, the history of the
        optimization process will also be returned. If ``track_path`` is also set, then
        the history and final result will be saved to disk rather than all being kept in memory.

        **Type hints:**
            - The ``acquisition_rule`` must use the same type of
              :class:`~trieste.space.SearchSpace` as specified in :meth:`__init__`.
            - The ``acquisition_state`` must be of the type expected by the ``acquisition_rule``.
              Any acquisition state in the optimization result will also be of this type.

        :param num_steps: The number of optimization steps to run.
        :param datasets: The known observer query points and observations for each tag.
        :param models: The model to use for each :class:`~trieste.data.Dataset` in
            ``datasets``.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments. Note that if the default is used, this implies the tags must be
            `OBJECTIVE`, the search space can be any :class:`~trieste.space.SearchSpace`, and the
            acquisition state returned in the :class:`OptimizationResult` will be `None`.
        :param acquisition_state: The acquisition state to use on the first optimization step.
            This argument allows the caller to restore the optimization process from an existing
            :class:`Record`.
        :param track_state: If `True`, this method saves the optimization state at the start of each
            step. Models and acquisition state are copied using `copy.deepcopy`.
        :param track_path: If set, the optimization state is saved to disk at this path,
            rather than being copied in memory.
        :param fit_initial_model: If `False`, this method assumes that the initial models have
            already been optimized on the datasets and so do not require optimization before the
            first optimization step.
        :param inspect_in_obj_space: bool = True,
        :param inspect_inferred_pareto: bool = True,
        :param inspect_acq_contour: bool = True, if True, plot the acquisition function contour,
            if num_query_points = 1, plot the contour (only support input_dim = 2)
            if num_query_points = 2, plot the contour with batch input (only support q=2, input_dim=1)
        :param kwargs_for_obj_space_plot: dict = {},
        :param kwargs_for_inferred_pareto: dict = {},
        :param early_stop_callback: An optional callback that is evaluated with the current
            datasets, models and optimization state before every optimization step. If this
            returns `True` then the optimization loop is terminated early.
        :return: An :class:`OptimizationResult`. The :attr:`final_result` element contains either
            the final optimization data, models and acquisition state, or, if an exception was
            raised while executing the optimization loop, it contains the exception raised. In
            either case, the :attr:`history` element is the history of the data, models and
            acquisition state at the *start* of each optimization step (up to and including any step
            that fails to complete). The history will never include the final optimization result.
        :raise ValueError: If any of the following are true:

            - ``num_steps`` is negative.
            - the keys in ``datasets`` and ``models`` do not match
            - ``datasets`` or ``models`` are empty
            - the default `acquisition_rule` is used and the tags are not `OBJECTIVE`.
        """
        if isinstance(datasets, Dataset):
            datasets = {OBJECTIVE: datasets}
            models = {OBJECTIVE: models}  # type: ignore[dict-item]

        # reassure the type checker that everything is tagged
        datasets = cast(Dict[Tag, Dataset], datasets)
        models = cast(Dict[Tag, TrainableProbabilisticModelType], models)

        if num_steps < 0:
            raise ValueError(f"num_steps must be at least 0, got {num_steps}")

        if datasets.keys() != models.keys():
            raise ValueError(
                f"datasets and models should contain the same keys. Got {datasets.keys()} and"
                f" {models.keys()} respectively."
            )

        if not datasets:
            raise ValueError("dicts of datasets and models must be populated.")

        if acquisition_rule is None:
            if datasets.keys() != {OBJECTIVE}:
                raise ValueError(
                    f"Default acquisition rule EfficientGlobalOptimization requires tag"
                    f" {OBJECTIVE!r}, got keys {datasets.keys()}"
                )

            acquisition_rule = EfficientGlobalOptimization[
                SearchSpaceType, TrainableProbabilisticModelType
            ]()

        history: list[FrozenRecord[StateType] | Record[StateType]] = []
        query_plot_dfs: dict[int, pd.DataFrame] = {}
        observation_plot_dfs = observation_plot_init(datasets)

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=0):
                write_summary_init(
                    self._observer,
                    self._search_space,
                    acquisition_rule,
                    datasets,
                    models,
                    num_steps,
                )

        for step in range(1, num_steps + 1):
            start = time.time()
            print(f"BO iter: {step}")
            logging.set_step_number(step)

            if early_stop_callback and early_stop_callback(datasets, models, acquisition_state):
                tf.print("Optimization terminated early", output_stream=absl.logging.INFO)
                break

            try:

                if track_state:
                    try:
                        if track_path is None:
                            datasets_copy = copy.deepcopy(datasets)
                            models_copy = copy.deepcopy(models)
                            acquisition_state_copy = copy.deepcopy(acquisition_state)
                            record = Record(datasets_copy, models_copy, acquisition_state_copy)
                            history.append(record)
                        else:
                            track_path = Path(track_path)
                            record = Record(datasets, models, acquisition_state)
                            file_name = OptimizationResult.step_filename(step, num_steps)
                            history.append(record.save(track_path / file_name))
                    except Exception as e:
                        raise NotImplementedError(
                            "Failed to save the optimization state. Some models do not support "
                            "deecopying or serialization and cannot be saved. "
                            "(This is particularly common for deep neural network models, though "
                            "some of the model wrappers accept a model closure as a workaround.) "
                            "For these models, the `track_state`` argument of the "
                            ":meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method "
                            "should be set to `False`. This means that only the final model "
                            "will be available."
                        ) from e

                if step == 1 and fit_initial_model:
                    with Timer() as initial_model_fitting_timer:
                        for tag, model in models.items():
                            dataset = datasets[tag]
                            model.update(dataset)
                            model.optimize(dataset)
                    if summary_writer:
                        logging.set_step_number(0)
                        with summary_writer.as_default(step=0):
                            write_summary_initial_model_fit(
                                datasets, models, initial_model_fitting_timer
                            )
                        logging.set_step_number(step)

                with Timer() as total_step_wallclock_timer:
                    with Timer() as query_point_generation_timer:
                        points_or_stateful = acquisition_rule.acquire(
                            self._search_space, models, datasets=datasets
                        )
                        if callable(points_or_stateful):
                            acquisition_state, query_points = points_or_stateful(acquisition_state)
                        else:
                            query_points = points_or_stateful

                    pfs = None
                    input_dim = query_points.shape[-1]
                    if acquisition_rule._num_query_points == 1:
                        plot_sequential = True  # If num_query = 1: just plot acq input space;
                        batch_q = 1
                    else:
                        plot_sequential = (
                            False  # If num_query = 2: plot batch input space (only support q=2):
                        )
                        batch_q = acquisition_rule._num_query_points
                    if inspect_acq_contour and input_dim <= 2:
                        # assert acquisition_rule._num_query_points == 2, NotImplementedError
                        if not plot_sequential:
                            assert input_dim == 1 and batch_q == 2, NotImplementedError(
                                "If plot joint batch acq, only q=2 and "
                                "input_dim=1 is valid for a 2 dim input "
                                f"contour plot, but fond input_dim={input_dim} and q={batch_q}"
                            )

                        def acq_f(at) -> TensorType:
                            """
                            Acquisition Function for plotting (expand the batch dim)
                            """
                            from .acquisition.utils import split_acquisition_function
                            return split_acquisition_function(acquisition_rule._acquisition_function, split_size=1000)(
                                tf.expand_dims(at, axis=-2)
                            )

                        def batch_acq_f(at) -> TensorType:
                            """
                            Acquisition Function for plotting (expand the batch dim)
                            [batch_dim, 2] -> [batch_dim, 2, 1]
                            """
                            return acquisition_rule._acquisition_function(
                                tf.expand_dims(at, axis=-1)
                            )

                        if plot_sequential:
                            _acq_f = acq_f
                        else:
                            _acq_f = batch_acq_f

                        # ============================
                        if input_dim == 1 and plot_sequential:
                            raise NotImplementedError
                        else:  # 2D contour (input_dim=1 & plot_batch; input_dim=2 & plot_sequential)
                            plt.figure(figsize=(10, 8))
                            plt_inst = view_2D_function_in_contourf(
                                _acq_f,
                                list(
                                    zip(
                                        self._search_space.lower.numpy(),
                                        self._search_space.upper.numpy(),
                                    )
                                )
                                * acquisition_rule._num_query_points,
                                show=False,
                                colorbar=True,
                                plot_fidelity=100,
                                title=None,
                            )
                            plt_inst.title(f'Batch Iteration: {step}', fontsize=40)
                            if plot_sequential:  # 2D contour
                                assert input_dim == 2
                                if "plot_acq_pf_sample" in kwargs_for_input_space_plot:
                                    if isinstance(acquisition_rule._builder, PF2ES):
                                        pf_inputs = acquisition_rule._builder._pf_samples_inputs
                                    else:
                                        (
                                            pfs,
                                            pf_inputs,
                                        ) = sample_pareto_fronts_from_parametric_gp_posterior(
                                            models[OBJECTIVE],
                                            obj_num=kwargs_for_input_space_plot[
                                                "kwargs_for_pf_sample"
                                            ]["obj_num"],
                                            sample_pf_num=kwargs_for_input_space_plot[
                                                "kwargs_for_pf_sample"
                                            ]["sample_pf_num"],
                                            search_space=kwargs_for_input_space_plot[
                                                "kwargs_for_pf_sample"
                                            ]["search_space"],
                                            cons_num=kwargs_for_input_space_plot[
                                                "kwargs_for_pf_sample"
                                            ]["cons_num"],
                                            constraint_models=models[CONSTRAINT]
                                            if CONSTRAINT in models.keys()
                                            else None,
                                            return_pf_input=True,
                                        )
                                    for pf_input in pf_inputs:
                                        if pf_input is not None:
                                            plt_inst.scatter(
                                                pf_input[:, 0], pf_input[:, 1], s=5, marker="x"
                                            )

                                plt_inst.scatter(
                                    datasets[OBJECTIVE].query_points[:, 0],
                                    datasets[OBJECTIVE].query_points[:, 1],
                                    s=40,
                                    marker="x",
                                    label="Existing data",
                                    color="r",
                                )
                                plt_inst.scatter(
                                    query_points[:, 0],
                                    query_points[:, 1],
                                    s=40,
                                    label="Sampled Batch Data",
                                    color="w",
                                )
                            else:
                                assert input_dim == 1 and plot_sequential is False
                                plt_inst.scatter(
                                    datasets[OBJECTIVE].query_points[:, 0],
                                    datasets[OBJECTIVE].query_points[:, 0],
                                    s=50,
                                    marker="x",
                                    label="Training data",
                                    color="r",
                                )
                                plt_inst.scatter(
                                    query_points[0, 0],
                                    query_points[1, 0],
                                    marker="^",
                                    s=50,
                                    label="Batch Sample",
                                    color="k",
                                )
                                plt_inst.scatter(
                                    query_points[:, 0],
                                    query_points[:, 0],
                                    s=50,
                                    label="Each Data in Batch",
                                    color="k",
                                )
                        plt_inst.legend(fontsize=30, framealpha=0.5)
                        plt.xticks([], [])
                        plt_inst.gca().set_xticks([])
                        plt_inst.gca().set_xticks([], minor=True)
                        plt.yticks([], [])
                        plt_inst.gca().set_yticks([])
                        plt_inst.gca().set_yticks([], minor=True)
                        plt_inst.gca().patch.set_edgecolor('black')
                        plt_inst.gca().patch.set_linewidth('1')
                        plt_inst.savefig(
                            os.path.join(
                                auxiliary_file_output_path,
                                f"{self._pb_name_prefix}_iter{step}_input_{self._acq_name}.png",
                            )
                        )
                        plt_inst.close()

                    observer_output = self._observer(query_points)

                    if inspect_in_obj_space:  # plot added point in objective space
                        if CONSTRAINT in datasets.keys():
                            mask_fail = tf.reduce_any(datasets[CONSTRAINT].observations < 0, -1)
                        else:
                            mask_fail = None
                        # _, ax = plot_mobo_points_in_obj_space(
                        #     datasets[OBJECTIVE].observations,
                        #     num_init=tf.shape(datasets[OBJECTIVE].observations)[0] - step,
                        #     mask_fail=mask_fail,
                        #     m_init="^",
                        #     m_add="^",
                        #     xlabel = None,
                        #     ylabel = None,
                        #     title = None
                        # )
                        fig, ax = plt.subplots()
                        ax.scatter(- datasets[OBJECTIVE].observations[:, 0],
                                   - datasets[OBJECTIVE].observations[:, 1],
                                   marker='x',
                                   s=40,
                                   color='r',
                                   label='Training Data',
                                   )
                        ax.set_xlabel('Objective 1', fontsize=20)
                        ax.set_ylabel('Objective 2', fontsize=20)
                        ax.tick_params(axis='x', labelsize=20)
                        ax.tick_params(axis='y', labelsize=20)
                        ax.set_title('Batch Sample In Objective Space' , fontsize=20)

                        # ---------------------------------------------------
                        ax.scatter(
                            - models[OBJECTIVE].predict(query_points)[0][:, 0],
                            - models[OBJECTIVE].predict(query_points)[0][:, 1],
                            s=40,
                            label="Predict Batch Data ",
                            color="k",
                        )
                        ax.scatter(
                            - observer_output[OBJECTIVE].observations[:, 0],
                            - observer_output[OBJECTIVE].observations[:, 1],
                            s=40,
                            marker="^",
                            label="Actual Batch Data",
                            color="k",
                        )
                        if "plot_acq_pf_samples" in kwargs_for_obj_space_plot.keys():
                            prop_cycle = plt.rcParams["axes.prop_cycle"]
                            colors = (
                                prop_cycle.by_key()["color"] * 100
                            )  # hard code here, default prop_cycle only
                            if isinstance(acquisition_rule._builder, PF2ES):
                                pfs = acquisition_rule._builder._pf_samples
                            elif pfs is None:
                                pfs = sample_pareto_fronts_from_parametric_gp_posterior(
                                    models[OBJECTIVE],
                                    obj_num=kwargs_for_obj_space_plot["kwargs_for_pf_sample"][
                                        "obj_num"
                                    ],
                                    sample_pf_num=kwargs_for_obj_space_plot["kwargs_for_pf_sample"][
                                        "sample_pf_num"
                                    ],
                                    search_space=kwargs_for_obj_space_plot["kwargs_for_pf_sample"][
                                        "search_space"
                                    ],
                                    cons_num=kwargs_for_obj_space_plot["kwargs_for_pf_sample"][
                                        "cons_num"
                                    ],
                                    constraint_models=models[CONSTRAINT]
                                    if CONSTRAINT in models.keys()
                                    else None,
                                )
                            for pf, pf_idx, default_color in zip(pfs, np.arange(len(pfs)), colors):
                                if pf is not None:
                                    ax.scatter(
                                        - pf[:, 0],
                                        - pf[:, 1],
                                        s=5,
                                        # label=f"Acq Sampled PF: {pf_idx}",
                                    )

                                    # plot_pf_dominated_region_boarder_2d(
                                    #     pf,
                                    #     tf.constant([1.2] * 2, dtype=pf.dtype),
                                    #     ax=ax,
                                    #     color=default_color,
                                    #     linewidth=1.0,
                                    # )
                        # if plot_sequential:
                        for query_point in query_points:
                            query_point = (
                                tf.expand_dims(query_point, axis=-2)
                                if tf.rank(query_point) == 1
                                else query_point
                            )
                            from matplotlib.patches import Ellipse

                            var1 = tf.squeeze(models[OBJECTIVE].predict(query_point)[1][:, 0])
                            var2 = tf.squeeze(models[OBJECTIVE].predict(query_point)[1][:, 1])
                            pred_obj1, pred_obj2 = tf.split(
                                tf.squeeze(models[OBJECTIVE].predict(query_point)[0]),
                                num_or_size_splits=2,
                            )
                            ellipse = Ellipse(
                                (- pred_obj1, - pred_obj2),
                                2 * tf.sqrt(var1),
                                2 * tf.sqrt(var2),
                                angle=0,
                                alpha=0.2,
                                edgecolor="k",
                            )
                            ax.add_artist(ellipse)
                        plt.legend(framealpha=0.5, fontsize=20)
                        if "obj1_lim" in kwargs_for_obj_space_plot.keys():
                            plt.xlim(kwargs_for_obj_space_plot["obj1_lim"])
                        if "obj2_lim" in kwargs_for_obj_space_plot.keys():
                            plt.ylim(kwargs_for_obj_space_plot["obj2_lim"])
                        # plt.show(block=True)

                        box = ax.get_position()
                        ax.set_position([box.x0 + box.width * 0.1, box.y0 + box.height * 0.1,
                                         box.width * 0.9, box.height * 0.9])
                        # plt.tight_layout()
                        plt.savefig(
                            os.path.join(
                                auxiliary_file_output_path,
                                f"{self._pb_name_prefix}_iter{step}_objspace_{self._acq_name}.png",
                            ),
                            dpi=400,
                        )
                        plt.close()

                    if inspect_inferred_pareto:  # plot inferred Pareto front in objective space
                        # perform NSGA2 on the problem
                        res, resx = inference_pareto_fronts_from_gp_mean(
                            models[OBJECTIVE],
                            self._search_space,
                            popsize=kwargs_for_inferred_pareto.get("popsize", 50),
                            cons_models=models[CONSTRAINT] if CONSTRAINT in models.keys() else None,
                            num_moo_iter=kwargs_for_inferred_pareto.get("num_moo_iter", 500),
                            min_feasibility_probability=0.95,
                            constraint_enforce_percentage=1e-3,
                            monte_carlo_input = False
                        )
                        if resx is not None:
                            np.savetxt(f'Recommend PF Input_BO_iter_{step}.txt', resx)
                        if kwargs_for_inferred_pareto.get('plot_rec_pf_actual_obx') == True:
                            # plot input space
                            if input_dim == 2:
                                _, _axis = plt.subplots()
                                if CONSTRAINT in models.keys() and kwargs_for_input_space_plot.get('plot_constraint_contour'):
                                    def constraint_func(x):  # only support 1 constraint!!!
                                        return self._observer(x)[CONSTRAINT].observations

                                    plt_inst = view_2D_function_in_contour(constraint_func,
                                                                list(
                                            zip(
                                                self._search_space.lower.numpy(),
                                                self._search_space.upper.numpy(),
                                            )
                                        )
                                        * acquisition_rule._num_query_points,
                                        show=False,
                                        colorbar=True,
                                        plot_fidelity=100,
                                        title=f"iter:{step}", line_number=2, color='k', levels=[0.0, 5.0])
                                else:
                                    plt_inst = plt
                                plt_inst.scatter(datasets[OBJECTIVE].query_points[:-batch_q, 0], datasets[OBJECTIVE].query_points[:-batch_q, 1],
                                           label='Training Data')
                                plt_inst.scatter(datasets[OBJECTIVE].query_points[-batch_q:, 0], datasets[OBJECTIVE].query_points[-batch_q:, 1],
                                           label='Added Data In this Iter')
                                if resx is not None:
                                    plt_inst.scatter(resx[:, 0], resx[:, 1], label='NSGA2 recommendation', s=5, color='g')
                                plt.legend()
                                plt.savefig(
                                    os.path.join(
                                        auxiliary_file_output_path,
                                        f"{self._pb_name_prefix}_iter{step}_pareto_set_infer_{self._acq_name}.png",
                                    )
                                )
                                plt.close()

                            # plot output space
                            if res is None:  # no feasible obs at all
                                assert datasets[OBJECTIVE].observations.shape[-1] == 2
                                _ = plt.subplots()

                            else:
                                if CONSTRAINT in models.keys():
                                    mask_fail = tf.reduce_any(
                                        self._observer(resx)[CONSTRAINT].observations < 0.0, axis=-1
                                    )
                                else:
                                    mask_fail = None
                                actual_obs = self._observer(resx)[OBJECTIVE].observations
                                _, ax = plot_mobo_points_in_obj_space(
                                    actual_obs, num_init=None, mask_fail=mask_fail, xlabel=None, ylabel=None, inverse_plot=True
                                )
                                box = ax.get_position()
                                ax.set_position([box.x0 + box.width * 0.1, box.y0 + box.height * 0.1,
                                                      box.width * 0.9, box.height * 0.9])
                                from trieste.acquisition.multi_objective.pareto import non_dominated
                                _pf, dominance = non_dominated(actual_obs)
                                # from trieste.experimental.plotting.plotting import format_point_markers
                                # col_pts, mark_pts = format_point_markers(actual_obs.shape[0], 100, dominance==0, mask_fail )


                                ax.scatter(- _pf[0, 0], - _pf[0, 1], color='tab:purple', marker='x', label='Pareto Frontier')
                                if tf.size(actual_obs[dominance!=0])!= 0:
                                    ax.scatter(- actual_obs[dominance!=0][0, 0], - actual_obs[dominance!=0][0, 1],marker='x',
                                               color='tab:green', label='False Positive Pareto Frontier')
                                ax.set_xlabel('Objective 1', fontsize=20)
                                ax.set_ylabel('Objective 2', fontsize=20)
                                ax.tick_params(axis='x', labelsize=20)
                                ax.tick_params(axis='y', labelsize=20)
                                ax.set_title('Out-of-Sample Recommendation', fontsize=20)
                            plt.legend(fontsize=20)
                            if "obj1_lim" in kwargs_for_inferred_pareto.keys():
                                plt.xlim(kwargs_for_inferred_pareto["obj1_lim"])
                            if "obj2_lim" in kwargs_for_inferred_pareto.keys():
                                plt.ylim(kwargs_for_inferred_pareto["obj2_lim"])
                            plt.savefig(
                                os.path.join(
                                    auxiliary_file_output_path,
                                    f"{self._pb_name_prefix}_iter{step}_pareto_infer_{self._acq_name}.png",
                                )
                            )
                            plt.close()
                        try:
                            np.savetxt(f'BO_Iter_{step}_constraint.txt',
                                       self._observer(resx)[CONSTRAINT].observations.numpy())
                        except: #  NO FEASIBLE REC
                            pass

                    if generate_recommendation:
                        res, resx = inference_pareto_fronts_from_gp_mean(
                            models[OBJECTIVE],
                            self._search_space,
                            popsize=kwargs_for_inferred_pareto.get("popsize", 50),
                            cons_models=models[CONSTRAINT] if CONSTRAINT in models.keys() else None,
                            num_moo_iter=kwargs_for_inferred_pareto.get("num_moo_iter", 100),
                            min_feasibility_probability=0.95,
                            constraint_enforce_percentage=5e-3,
                            monte_carlo_input=False
                        )
                        if resx is not None:
                            np.savetxt(f'Recommend PF Input_BO_iter_{step}.txt', resx)
                        np.savetxt(f'BO_query_points_{step}.txt', models[OBJECTIVE].get_internal_data().query_points)

                    tagged_output = (
                        observer_output
                        if isinstance(observer_output, Mapping)
                        else {OBJECTIVE: observer_output}
                    )

                    datasets = {tag: datasets[tag] + tagged_output[tag] for tag in tagged_output}

                    with Timer() as model_fitting_timer:
                        for tag, model in models.items():
                            dataset = datasets[tag]
                            tf.debugging.assert_all_finite(
                                dataset.query_points,
                                f"There are inf in obs: {dataset.query_points}",
                            )
                            tf.debugging.assert_all_finite(
                                dataset.observations,
                                f"There are inf in obs: {dataset.observations}",
                            )
                            model.update(dataset)
                            model.optimize(dataset, randomize_model_hyperparam = True)

                if summary_writer:
                    with summary_writer.as_default(step=step):
                        write_summary_observations(
                            datasets,
                            models,
                            tagged_output,
                            model_fitting_timer,
                            observation_plot_dfs,
                        )
                        write_summary_query_points(
                            datasets,
                            models,
                            self._search_space,
                            query_points,
                            query_point_generation_timer,
                            query_plot_dfs,
                        )
                        logging.scalar("wallclock/step", total_step_wallclock_timer.time)

            except Exception as error:  # pylint: disable=broad-except
                tf.print(
                    f"\nOptimization failed at step {step}, encountered error with traceback:"
                    f"\n{traceback.format_exc()}"
                    f"\nTerminating optimization and returning the optimization history. You may "
                    f"be able to use the history to restart the process from a previous successful "
                    f"optimization step.\n",
                    output_stream=absl.logging.ERROR,
                )
                if isinstance(error, MemoryError):
                    tf.print(
                        "\nOne possible cause of memory errors is trying to evaluate acquisition "
                        "\nfunctions over large datasets, e.g. when initializing optimizers. "
                        "\nYou may be able to word around this by splitting up the evaluation "
                        "\nusing split_acquisition_function or split_acquisition_function_calls.",
                        output_stream=absl.logging.ERROR,
                    )
                result = OptimizationResult(Err(error), history)
                if track_state and track_path is not None:
                    result.save_result(Path(track_path) / OptimizationResult.RESULTS_FILENAME)
                return result
            end = time.time()
            print(f"BO iter: {step} takes: {end - start}")
        tf.print("Optimization completed without errors", output_stream=absl.logging.INFO)

        record = Record(datasets, models, acquisition_state)
        result = OptimizationResult(Ok(record), history)
        if track_state and track_path is not None:
            result.save_result(Path(track_path) / OptimizationResult.RESULTS_FILENAME)
        return result


def write_summary_init(
    observer: Observer,
    search_space: SearchSpace,
    acquisition_rule: AcquisitionRule[
        TensorType | State[StateType | None, TensorType],
        SearchSpaceType,
        TrainableProbabilisticModelType,
    ],
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, TrainableProbabilisticModel],
    num_steps: int,
) -> None:
    """Write initial BO loop TensorBoard summary."""
    devices = tf.config.list_logical_devices()
    logging.text(
        "metadata",
        f"Observer: `{observer}`\n\n"
        f"Number of steps: `{num_steps}`\n\n"
        f"Number of initial points: "
        f"`{dict((k, len(v)) for k, v in datasets.items())}`\n\n"
        f"Search Space: `{search_space}`\n\n"
        f"Acquisition rule:\n\n    {acquisition_rule}\n\n"
        f"Models:\n\n    {models}\n\n"
        f"Available devices: `{dict(Counter(d.device_type for d in devices))}`",
    )


def write_summary_initial_model_fit(
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, TrainableProbabilisticModel],
    model_fitting_timer: Timer,
) -> None:
    """Write TensorBoard summary for the model fitting to the initial data."""
    for tag, model in models.items():
        with tf.name_scope(f"{tag}.model"):
            model.log(datasets[tag])
    logging.scalar(
        "wallclock/model_fitting",
        model_fitting_timer.time,
    )


def observation_plot_init(
    datasets: Mapping[Tag, Dataset],
) -> dict[Tag, pd.DataFrame]:
    """Initialise query point pairplot dataframes with initial observations.
    Also logs warnings if pairplot dependencies are not installed."""
    observation_plot_dfs: dict[Tag, pd.DataFrame] = {}
    if logging.get_tensorboard_writer():

        seaborn_warning = False
        if logging.include_summary("query_points/_pairplot") and not (pd and sns):
            seaborn_warning = True
        for tag in datasets:
            if logging.include_summary(f"{tag}.observations/_pairplot"):
                output_dim = tf.shape(datasets[tag].observations)[-1]
                if output_dim >= 2:
                    if not (pd and sns):
                        seaborn_warning = True
                    else:
                        columns = [f"x{i}" for i in range(output_dim)]
                        observation_plot_dfs[tag] = pd.DataFrame(
                            datasets[tag].observations, columns=columns
                        ).applymap(float)
                        observation_plot_dfs[tag]["observations"] = "initial"

        if seaborn_warning:
            tf.print(
                "\nPairplot TensorBoard summaries require seaborn to be installed."
                "\nOne way to do this is to install 'trieste[plotting]'.",
                output_stream=absl.logging.INFO,
            )
    return observation_plot_dfs


def write_summary_observations(
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, TrainableProbabilisticModel],
    tagged_output: Mapping[Tag, TensorType],
    model_fitting_timer: Timer,
    observation_plot_dfs: MutableMapping[Tag, pd.DataFrame],
) -> None:
    """Write TensorBoard summary for the current step observations."""
    for tag in datasets:

        with tf.name_scope(f"{tag}.model"):
            models[tag].log(datasets[tag])

        output_dim = tf.shape(tagged_output[tag].observations)[-1]
        for i in tf.range(output_dim):
            suffix = f"[{i}]" if output_dim > 1 else ""
            if tf.size(tagged_output[tag].observations) > 0:
                logging.histogram(
                    f"{tag}.observation{suffix}/new_observations",
                    tagged_output[tag].observations[..., i],
                )
                logging.scalar(
                    f"{tag}.observation{suffix}/best_new_observation",
                    np.min(tagged_output[tag].observations[..., i]),
                )
            if tf.size(datasets[tag].observations) > 0:
                logging.scalar(
                    f"{tag}.observation{suffix}/best_overall",
                    np.min(datasets[tag].observations[..., i]),
                )

        if logging.include_summary(f"{tag}.observations/_pairplot") and (
            pd and sns and output_dim >= 2
        ):
            columns = [f"x{i}" for i in range(output_dim)]
            observation_new_df = pd.DataFrame(
                tagged_output[tag].observations, columns=columns
            ).applymap(float)
            observation_new_df["observations"] = "new"
            observation_plot_df = pd.concat(
                (observation_plot_dfs.get(tag), observation_new_df),
                copy=False,
                ignore_index=True,
            )

            hue_order = ["initial", "old", "new"]
            palette = {"initial": "tab:green", "old": "tab:green", "new": "tab:orange"}
            markers = {"initial": "X", "old": "o", "new": "o"}

            # assume that any OBJECTIVE- or single-tagged multi-output dataset => multi-objective
            # more complex scenarios (e.g. constrained data) need to be plotted by the acq function
            if len(datasets) > 1 and tag != OBJECTIVE:
                observation_plot_df["observation type"] = observation_plot_df.apply(
                    lambda x: x["observations"],
                    axis=1,
                )
            else:
                observation_plot_df["pareto"] = non_dominated(datasets[tag].observations)[1]
                observation_plot_df["observation type"] = observation_plot_df.apply(
                    lambda x: x["observations"] + x["pareto"] * " (non-dominated)",
                    axis=1,
                )
                hue_order += [hue + " (non-dominated)" for hue in hue_order]
                palette.update(
                    {
                        "initial (non-dominated)": "tab:purple",
                        "old (non-dominated)": "tab:purple",
                        "new (non-dominated)": "tab:red",
                    }
                )
                markers.update(
                    {
                        "initial (non-dominated)": "X",
                        "old (non-dominated)": "o",
                        "new (non-dominated)": "o",
                    }
                )

            pairplot = sns.pairplot(
                observation_plot_df,
                vars=columns,
                hue="observation type",
                hue_order=hue_order,
                palette=palette,
                markers=markers,
            )
            logging.pyplot(f"{tag}.observations/_pairplot", pairplot.fig)
            observation_plot_df.loc[
                observation_plot_df["observations"] == "new", "observations"
            ] = "old"
            observation_plot_dfs[tag] = observation_plot_df

    logging.scalar(
        "wallclock/model_fitting",
        model_fitting_timer.time,
    )


def write_summary_query_points(
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, TrainableProbabilisticModel],
    search_space: SearchSpace,
    query_points: TensorType,
    query_point_generation_timer: Timer,
    query_plot_dfs: MutableMapping[int, pd.DataFrame],
) -> None:
    """Write TensorBoard summary for the current step query points."""

    if tf.rank(query_points) == 2:
        for i in tf.range(tf.shape(query_points)[1]):
            if len(query_points) == 1:
                logging.scalar(f"query_points/[{i}]", float(query_points[0, i]))
            else:
                logging.histogram(f"query_points/[{i}]", query_points[:, i])
        logging.histogram("query_points/euclidean_distances", lambda: pdist(query_points))

    if pd and sns and logging.include_summary("query_points/_pairplot"):
        columns = [f"x{i}" for i in range(tf.shape(query_points)[1])]
        qp_preds = query_points
        for tag in datasets:
            pred = models[tag].predict(query_points)[0]
            qp_preds = tf.concat([qp_preds, tf.cast(pred, query_points.dtype)], 1)
            output_dim = tf.shape(pred)[-1]
            for i in range(output_dim):
                columns.append(f"{tag}{i if (output_dim > 1) else ''} predicted")
        query_new_df = pd.DataFrame(qp_preds, columns=columns).applymap(float)
        query_new_df["query points"] = "new"
        query_plot_df = pd.concat(
            (query_plot_dfs.get(0), query_new_df), copy=False, ignore_index=True
        )
        pairplot = sns.pairplot(
            query_plot_df, hue="query points", hue_order=["old", "new"], height=2.25
        )
        padding = 0.025 * (search_space.upper - search_space.lower)
        upper_limits = search_space.upper + padding
        lower_limits = search_space.lower - padding
        for i in range(search_space.dimension):
            pairplot.axes[0, i].set_xlim((lower_limits[i], upper_limits[i]))
            pairplot.axes[i, 0].set_ylim((lower_limits[i], upper_limits[i]))
        logging.pyplot("query_points/_pairplot", pairplot.fig)
        query_plot_df["query points"] = "old"
        query_plot_dfs[0] = query_plot_df

    logging.scalar(
        "wallclock/query_point_generation",
        query_point_generation_timer.time,
    )


def stop_at_minimum(
    minimum: Optional[tf.Tensor] = None,
    minimizers: Optional[tf.Tensor] = None,
    minimum_atol: float = 0,
    minimum_rtol: float = 0.05,
    minimizers_atol: float = 0,
    minimizers_rtol: float = 0.05,
    objective_tag: Tag = OBJECTIVE,
) -> EarlyStopCallback[TrainableProbabilisticModel, object]:
    """
    Generate an early stop function that terminates a BO loop when it gets close enough to the
    given objective minimum and/or minimizer points.

    :param minimum: Optional minimum to stop at, with shape [1].
    :param minimizers: Optional minimizer points to stop at, with shape [N, D].
    :param minimum_atol: Absolute tolerance for minimum.
    :param minimum_rtol: Relative tolerance for minimum.
    :param minimizers_atol: Absolute tolerance for minimizer point.
    :param minimizers_rtol: Relative tolerance for minimizer point.
    :param objective_tag: The tag for the objective data.
    :return: An early stop function that terminates if we get close enough to both the minimum
        and any of the minimizer points.
    """

    def early_stop_callback(
        datasets: Mapping[Tag, Dataset],
        _models: Mapping[Tag, TrainableProbabilisticModel],
        _acquisition_state: object,
    ) -> bool:
        dataset = datasets[objective_tag]
        arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
        if minimum is not None:
            best_y = dataset.observations[arg_min_idx]
            close_y = np.isclose(best_y, minimum, atol=minimum_atol, rtol=minimum_rtol)
            if not tf.reduce_all(close_y):
                return False
        if minimizers is not None:
            best_x = dataset.query_points[arg_min_idx]
            close_x = np.isclose(best_x, minimizers, atol=minimizers_atol, rtol=minimizers_rtol)
            if not tf.reduce_any(tf.reduce_all(close_x, axis=-1), axis=0):
                return False
        return True

    return early_stop_callback
