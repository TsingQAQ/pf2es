from collections.abc import Callable

from typing import Optional


def serial_benchmarker(
        process_identifier: int,
        *,
        acq,
        num_obj: int = 1,
        num_con: int = 0,
        kwargs_for_optimize: dict = {},
        kwargs_for_acq: dict = {},
        kwargs_for_model: dict = {},
        post_profiler: Optional[Callable] = None,
        q: int = 1,
        is_return: bool = True,
        save_result_to_file: bool = True,
        file_info_prefix: str = "",
        path_prefix: str = "",
        seed: Optional[int] = None,
        **kwargs,
):
    """
    Sequential benchmarker of Bayesian Optimization framework
    :param process_identifier
    :param acq acquisition function
    :param num_obj: objective number
    :param num_con: constraint number
    :kwargs methods:
    :param kwargs_for_optimize
    :param kwargs_for_acq
    :param post_profiler: callable function that has input: (mo_result, post_recommenders, performance_metrics),
        for each bo iter from mo_result, for each recommender in the post_recommenders, it will extract the
        recommendation and calculate the performance metric value from all performance metric, by default it uses
        get_performance_metric_from_optimize_result.
    :param is_return
    :param q Batch size
    :param save_result_to_file
    :param file_info_prefix
    :param path_prefix
    :param seed for serial benchmarker
    """

    import os

    import numpy as np
    import tensorflow as tf

    import trieste
    from trieste.acquisition.function import (
        MESMO,
        PF2ES,
        PFES,
        MESMOC,
    )
    from trieste.models.gpflow import GaussianProcessRegression
    from trieste.models import TrainableHasTrajectoryAndPredictJointReparamModelStack
    from trieste.acquisition.rule import Random
    from trieste.bayesian_optimizer import BayesianOptimizer, EfficientGlobalOptimization
    from trieste.data import Dataset
    from trieste.objectives import multi_objectives, single_objectives
    from trieste.observer import CONSTRAINT, OBJECTIVE
    from trieste.models.gpflow.builders import build_gpr, build_standardized_gpr
    from trieste.models.gpflow.wrapper_model.transformed_models import StandardizedGaussianProcessRegression
    from trieste.space import SearchSpace
    from trieste.utils.optimal_recommenders.multi_objective import MODEL_BELIEVE
    from trieste.acquisition.function.greedy_batch import Fantasizer
    from trieste.logging import set_tensorboard_writer, set_summary_filter
    from trieste.acquisition.optimizer import automatic_optimizer_selector
    from trieste.acquisition.utils import split_acquisition_function_calls
    # ---------------------------------- Settings -----------------------------------
    process_identifier_str = str(process_identifier)
    print(process_identifier_str)
    # logging
    _log_path = os.path.join(path_prefix)
    os.makedirs(_log_path, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(
        os.path.join(_log_path, 'logs', 'tensorboard', kwargs.get("acq_name"), process_identifier_str),
        name=f'log_{process_identifier_str}')
    set_tensorboard_writer(summary_writer)
    set_summary_filter(lambda name: True)

    #
    if seed is not None:  # reproducible experiment
        tf.random.set_seed(seed + process_identifier)
    tf.debugging.assert_greater_equal(num_obj, 1)
    num_objective = num_obj
    # Unpack settings
    if "benchmark" not in kwargs:
        if num_objective == 1:
            func_inst = getattr(single_objectives, kwargs["benchmark_name"])(
                **kwargs.get("kwargs_for_benchmark", {})
            )
        else:
            func_inst = getattr(multi_objectives, kwargs["benchmark_name"])(
                **kwargs.get("kwargs_for_benchmark", {})
            )
    else:  # this is the hard code only for matlab benchmark
        func_inst = kwargs["benchmark"]

    obj = func_inst.objective()

    if hasattr(func_inst, "constraint"):
        has_constraints = True
        con = func_inst.constraint()
        if hasattr(func_inst, "joint_objective_con"):
            has_joint_obj_con = True
            joint_obj_con = func_inst.joint_objective_con()
        else:
            has_joint_obj_con = False
    else:
        has_constraints = False
        has_joint_obj_con = False
    bounds = func_inst.bounds

    total_iter = kwargs.get("total_iter", 20)  # This is used to control the maximum func eval
    tf.debugging.assert_greater_equal(total_iter, 0)

    def observer(query_points):
        """
        Black box function observer
        """
        if has_constraints:
            if not has_joint_obj_con:
                return {
                    OBJECTIVE: Dataset(query_points, obj(query_points)),
                    CONSTRAINT: Dataset(query_points, con(query_points)),
                }
            else:  # obs and constraint obtained simultaneously, usually happens in real-life problem
                _obj, _con = joint_obj_con(query_points)
                return {
                    OBJECTIVE: Dataset(query_points, _obj),
                    CONSTRAINT: Dataset(query_points, _con),
                }
        else:
            return {OBJECTIVE: Dataset(query_points, obj(query_points))}

    search_space = trieste.space.Box(bounds[0], bounds[1])

    if kwargs.get("initial_x") is not None:
        print("init x provided, evaluate on it as doe")
        if kwargs["initial_x"] == "Stored":
            # note: hard code
            print(f"use stored initial x: xs_{process_identifier_str}.txt")
            path = os.path.join(
                ".",
                "cfg",
                "initial_xs",
                kwargs["benchmark_name"],
                f"xs_{process_identifier_str}.txt",
            )
            xs = np.loadtxt(path)
            xs = tf.convert_to_tensor(xs, dtype=tf.float64)

        else:
            xs = kwargs["initial_x"]
        initial_data = observer(xs)
    else:
        assert "doe_num" in kwargs, ValueError("doe_num must be specified if no init x provided")
        doe_num = kwargs["doe_num"]
        num_initial_points = doe_num
        x_init = search_space.sample(num_initial_points)
        initial_data = observer(x_init)

    def build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            data: Dataset, num_output: int, search_space: SearchSpace
    ) -> TrainableHasTrajectoryAndPredictJointReparamModelStack:
        gprs = []

        for idx in range(num_output):
            single_obj_data = Dataset(
                data.query_points, tf.gather(data.observations, [idx], axis=1)
            )
            if 'likelihood_variances' in kwargs_for_model:
                likelihood_variance = kwargs_for_model['likelihood_variances'].pop(0)
            else:
                likelihood_variance = 1e-7
            gpr = build_gpr(single_obj_data, search_space, likelihood_variance=likelihood_variance \
                            , trainable_likelihood=False)
            gprs.append((GaussianProcessRegression(gpr, use_decoupled_sampler=False), 1))
            # gpr = build_standardized_gpr(single_obj_data, search_space, likelihood_variance=likelihood_variance \
                            # , trainable_likelihood=False)
            # gprs.append((StandardizedGaussianProcessRegression(gpr, use_decoupled_sampler=False), 1))

        return TrainableHasTrajectoryAndPredictJointReparamModelStack(*gprs)

    # choose different models according to different acq function
    models = (
        {
            OBJECTIVE: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
                initial_data[OBJECTIVE], num_output=num_objective, search_space=search_space),
            CONSTRAINT: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
                initial_data[CONSTRAINT], num_output=num_con, search_space=search_space),
        }
        if has_constraints
        else {OBJECTIVE: build_stacked_has_traj_and_reparam_sampler_independent_objectives_model(
            initial_data[OBJECTIVE], num_output=num_objective, search_space=search_space)}
    )

    # HARD CODE: we need to specify search space for these acquisition functions
    if (
            acq == PF2ES
            or acq == PFES
            or acq == MESMO
            or acq == MESMOC
    ):
        if kwargs.get('fantasizer') is True:  # We only use this for PF2ES, so no wrap for other acq
            _acq = Fantasizer(acq(search_space=search_space, **kwargs_for_acq))
        else:
            _acq = acq(search_space=search_space, **kwargs_for_acq)
    elif acq is None:
        print("No acq specified, run random search")
        _acq = None
    else:
        print(acq)
        _acq = acq(**kwargs_for_acq)

    if _acq is not None:
        _rule = EfficientGlobalOptimization(
            builder=_acq, num_query_points=q,
            optimizer= split_acquisition_function_calls(
                automatic_optimizer_selector,  split_size=kwargs.get('split_size', 1000000)))
    else:
        _rule = Random(num_query_points=q)

    bo = BayesianOptimizer(
        observer, search_space, pb_name_prefix=kwargs['benchmark_name'], acq_name=kwargs['acq_name'])

    try:
        mo_result = bo.optimize(
            total_iter, initial_data, models, acquisition_rule=_rule, **kwargs_for_optimize
        )

        if post_profiler is not None:
            recommenders_history = post_profiler(mo_result)
            if save_result_to_file:
                _path = os.path.join(path_prefix,
                                     kwargs.get("acq_name"),
                                     file_info_prefix)

                # save performance metric according to recommenders
                for recommender_name in recommenders_history.keys():
                    if recommender_name != MODEL_BELIEVE:
                        for metric_name, metric_val in recommenders_history[recommender_name].items():
                            _file_name = "_".join(
                                [
                                    kwargs["benchmark_name"],
                                    process_identifier_str,
                                    recommender_name,
                                    metric_name,
                                    f'q{q}',
                                    ".txt",
                                ]
                            )
                            os.makedirs(_path, exist_ok=True)
                            np.savetxt(os.path.join(_path, _file_name),
                                       np.atleast_1d(np.asarray(metric_val)))
                    else:  # save as npy file, refer https://stackoverflow.com/questions/59971982/save-3d-numpy-array-with-high-speed-into-the-disk
                        _file_name = "_".join(
                            [
                                kwargs["benchmark_name"],
                                process_identifier_str,
                                recommender_name,
                                "_model_inferred_pfs",
                                f'q{q}',
                                ".npy",
                            ]
                        )
                        os.makedirs(_path, exist_ok=True)
                        np.save(os.path.join(_path, _file_name), np.asarray(
                            recommenders_history[recommender_name]['_model_inferred_pfs']))

                final_data = mo_result.try_get_final_datasets()["OBJECTIVE"]
                query, obs = final_data.query_points, final_data.observations
                # save queried data points
                np.savetxt(
                    os.path.join(
                        _path,
                        "_".join([kwargs["benchmark_name"], process_identifier_str, f'q{q}', "queried_X.txt"]),
                    ),
                    query.numpy(),
                )
                np.savetxt(
                    os.path.join(
                        _path,
                        "_".join([kwargs["benchmark_name"], process_identifier_str, f'q{q}', "queried_Y.txt"]),
                    ),
                    obs.numpy(),
                )

                # if recommender is None:
                #     # save final pareto optimal x and
                #     final_model = mo_result.try_get_final_models()["OBJECTIVE"]
                #     _optimum_y, dominance = non_dominated(final_model.predict(query)[0])
                #     _optimum_x = tf.gather_nd(query, tf.where(tf.equal(dominance, 0)))
                # else:
                #     _optimum_x = recommender(
                #         mo_result.try_get_final_models(),
                #         mo_result.try_get_final_datasets()[OBJECTIVE].query_points,
                #     )
                # np.savetxt(
                #     os.path.join(
                #         path_prefix,
                #         kwargs.get("acq_name"),
                #         file_info_prefix,
                #         "_".join([kwargs["benchmark_name"], file_identifier_str, f'q{q}', "pfx.txt"]),
                #     ),
                #     _optimum_x,
                # )

        if is_return:
            return mo_result
    # except NotImplementedError as e:
    #     pass
    except Exception as e:
        print(e)
        print(f"Exp {process_identifier_str} failed, skip")
        return


def parallel_benchmarker(workers: int, exp_repeat: int, seed: Optional[int] = 1817, **kwargs):
    """
    Parallel Bayesian Optimization benchmarker
    """
    from multiprocessing import Pool, set_start_method

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    import numpy as np
    import tensorflow as tf

    tf.debugging.assert_greater_equal(workers, 1)
    tf.debugging.assert_greater_equal(exp_repeat, 1)
    if workers == 1:  # serial
        for i in range(exp_repeat):
            serial_benchmarker(i, **kwargs)
    else:  # parallel
        workers = exp_repeat if workers > exp_repeat else workers
        from functools import partial

        pb = partial(serial_benchmarker, **kwargs, is_return=False, seed=seed)
        for parallel_work in np.arange(kwargs.get('start_exp_id', 0), kwargs.get('start_exp_id', 0) + exp_repeat, workers):
            with Pool(workers) as p:
                _ = p.map_async(pb, np.arange(parallel_work, parallel_work + workers)).get()
