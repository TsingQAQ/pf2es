from collections.abc import Callable
from typing import Mapping, Optional

import tensorflow as tf
from trieste.bayesian_optimizer import OptimizationResult
from trieste.utils.optimal_recommenders.multi_objective import IN_SAMPLE, OUT_OF_SAMPLE, MODEL_BELIEVE
from trieste.utils.performance_metrics.multi_objective import HypervolumeIndicator, NegLogMarginalParetoFrontier


def get_performance_metric_from_optimize_result(
        optimization_result: OptimizationResult,
        recommenders: Mapping[str, Callable],
        metrics: Mapping[str, Callable],
        true_func_inst: Optional[Callable] = None,
) -> Mapping[str, Mapping[str, list]]:
    """
    Extract metric history from optimization history
    :param recommenders: {"in_sample": callable_recommender, "Out-of-sample": callable_recommender}
    :param true_func_inst: the real black box function
    :param optimization_result
    :param metrics: {"HV": hv_indicator, "IGD": IGD_indicator}
    """

    # history is the beginning state of a bo iter
    # this history is 0-num bo iter
    opt_history = optimization_result.history + [optimization_result.final_result.unwrap()]

    # initialize result dictionary
    recommender_history = {}
    for rec_key in recommenders.keys():
        recommender_history[rec_key] = {}
        for metric_key in metrics.keys():
            recommender_history[rec_key][metric_key] = []
    # hard code
    if MODEL_BELIEVE in recommender_history.keys():
        recommender_history[MODEL_BELIEVE]['_model_inferred_pfs'] = []
    # TODO: This is messy, can we clean it bit
    # extracting:
    for hist, idx in zip(opt_history, tf.range(len(opt_history))):
        models = hist.models  # hard code
        datas = hist.datasets
        for recommender_key, recommender in recommenders.items():
            if recommender_key == IN_SAMPLE:  # IN_SAMPLE RECOMMENDATION
                try:
                    _optimum_x, _optimum_f = recommender(models, datas)
                except:
                    _optimum_f = None
                    _optimum_x = recommender(models, datas)
                if tf.rank(_optimum_x) == 1:
                    _optimum_x = tf.expand_dims(_optimum_x, axis=-2)
                for metric_key, metric in metrics.items():
                    if true_func_inst is not None:  # TODO
                        if metric_key == HypervolumeIndicator:
                            recommender_history[recommender_key][metric_key].append(
                                metric(recommendation_input=_optimum_x,
                                       true_func_inst=true_func_inst.joint_objective_con(),
                                       recommendation_ouput=_optimum_f)
                            )
                        elif metric_key == NegLogMarginalParetoFrontier:
                            recommender_history[recommender_key][metric_key].append(
                                metric(models)
                            )
                        else:
                            recommender_history[recommender_key][metric_key].append(
                                metric(recommendation_input=_optimum_x, true_func_inst=true_func_inst)
                            )
                    else:
                        raise NotImplementedError  #
            elif recommender_key == OUT_OF_SAMPLE:  # OUT_OF_SAMPLE
                if HypervolumeIndicator in metrics.keys() and len(metrics.keys()) == 1:
                    if idx != 0 and idx % 25 == 0:
                        _optimum_x = recommender(models, datas)
                        recommender_history[recommender_key][metric_key].append(
                            metric(recommendation_input=_optimum_x, true_func_inst=true_func_inst.joint_objective_con())
                        )
                    else:
                        pass
                else:
                    _optimum_x = recommender(models, datas)
                    for metric_key, metric in metrics.items():
                        if true_func_inst is not None:
                            if metric_key == NegLogMarginalParetoFrontier:
                                recommender_history[recommender_key][metric_key].append(
                                    metric(models)
                                )
                            else:
                                recommender_history[recommender_key][metric_key].append(
                                    metric(recommendation_input=_optimum_x, true_func_inst=true_func_inst)
                                )
                        else:
                            recommender_history[recommender_key][metric_key].append(
                                metric(recommendation_input=_optimum_x))
            elif recommender_key == MODEL_BELIEVE:
                _model_inferred_pfs = recommender(models)
                # here, for experimental reason, we only store it instead of extracting it to a performance metric
                recommender_history[recommender_key]['_model_inferred_pfs'].append(_model_inferred_pfs)
            else:
                raise ValueError(f'Recommender must be either {IN_SAMPLE}, '
                                 f'{OUT_OF_SAMPLE} or {MODEL_BELIEVE} but found: {recommender_key}')
    return recommender_history
