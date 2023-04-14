import os
from functools import partial
import numpy as np
from os.path import dirname, abspath

import tensorflow as tf
from trieste.utils.performance_metrics.multi_objective import \
    log_hypervolume_difference, \
    average_hausdoff_distance, \
    additive_epsilon_indicator, \
    negative_log_marginal_likelihood_of_target_pareto_frontier, \
    LogHypervolumeDifference, \
    AverageHausdauffDistance, \
    AdditiveEpsilonIndicator, \
    NegLogMarginalParetoFrontier


from trieste.objectives.multi_objectives import VLMOP2, BraninCurrin, DTLZ2, DTLZ4, DTLZ5, ZDT1, ZDT2, ZDT3, ZDT4,\
    VehicleCrashSafety, FourBarTruss, RocketInjectorDesign, Penicillin
from trieste.utils.optimal_recommenders.multi_objective import IN_SAMPLE, OUT_OF_SAMPLE, MODEL_BELIEVE, \
    recommend_pareto_front_from_existing_data, recommend_pareto_front_from_model_prediction, \
    inspecting_pareto_front_distributions_from_model


VLMOP2_cfg = {
    "pb": VLMOP2,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        # OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model # Not used for time profiling
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VLMOP2_PF_F.txt')),
                ref_point=tf.constant([1.2, 1.2], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VLMOP2_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VLMOP2_PF_F.txt')),
            ),},
            # NegLogMarginalParetoFrontier: partial(
            #     negative_log_marginal_likelihood_of_target_pareto_frontier,
            #     reference_pf_input=np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VLMOP2_PF_X.txt')),
            #     reference_pf=np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VLMOP2_PF_F.txt')),
            #     numerical_stability_term = 1e-7)
            # },
    "path_prefix": os.path.join(".", "exp_res", "VLMOP2"),
}


BraninCurrin_cfg = {
    "pb": BraninCurrin,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        # OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_hv=tf.convert_to_tensor(60.0, dtype=tf.float64),
                ref_point=tf.constant([18.0, 6.0], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'BraninCurrin_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'BraninCurrin_PF_F.txt')),
            ),
            # NegLogMarginalParetoFrontier: partial(
            #     negative_log_marginal_likelihood_of_target_pareto_frontier,
            #     reference_pf_input=np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'BraninCurrin_PF_NLL_X.txt')),
            #     reference_pf=np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'BraninCurrin_PF_NLL_F.txt')),
            #     numerical_stability_term = 1e-7)
            },
    "path_prefix": os.path.join(".", "exp_res", "BraninCurrin"),
}


DTLZ2_7_Input_2_Output_cfg = {
    "pb": DTLZ2,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 7, "num_objective": 2},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_7I_2O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_7I_2O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_7I_2O_PF_F.txt')),
            ),
            NegLogMarginalParetoFrontier: partial(
                negative_log_marginal_likelihood_of_target_pareto_frontier,
                reference_pf_input=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_7I_2O_PF_X.txt')),
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_7I_2O_PF_F.txt')),
                numerical_stability_term = 1e-7)
            },
    "path_prefix": os.path.join(".", "exp_res", "DTLZ2_7I_2O"),
}


DTLZ2_4_Input_3_Output_cfg = {
    "pb": DTLZ2,
    "num_obj": 3,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 4, "num_objective": 3},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_4I_3O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_4I_3O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_4I_3O_PF_F.txt')),
            ),
            NegLogMarginalParetoFrontier: partial(
                negative_log_marginal_likelihood_of_target_pareto_frontier,
                reference_pf_input=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_4I_3O_PF_X.txt')),
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_4I_3O_PF_F.txt')),
                numerical_stability_term = 1e-7)
            },
    "path_prefix": os.path.join(".", "exp_res", "DTLZ2_4I_3O"),
}


DTLZ2_3_Input_2_Output_cfg = {
    "pb": DTLZ2,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 3, "num_objective": 2},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_3I_2O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_3I_2O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_3I_2O_PF_F.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "DTLZ2_3I_2O"),
}


DTLZ2_8_Input_2_Output_cfg = {
    "pb": DTLZ2,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 8, "num_objective": 2},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_8I_2O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_8I_2O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ2_8I_2O_PF_F.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "DTLZ2_8I_2O"),
}


DTLZ4_8_Input_2_Output_cfg = {
    "pb": DTLZ4,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 8, "num_objective": 2},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_8I_2O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_8I_2O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_8I_2O_PF_F.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "DTLZ4_8I_2O"),
}

# fixme: 2022/10/11 the input dim is wrongly specified as 8!
DTLZ4_6_Input_2_Output_cfg = {
    "pb": DTLZ4,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 8, "num_objective": 2},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_6I_2O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_6I_2O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_6I_2O_PF_F.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "DTLZ4_6I_2O"),
}


DTLZ4_6_Input_4_Output_cfg = {
    "pb": DTLZ4,
    "num_obj": 4,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 6, "num_objective": 4},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_6I_4O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5, 2.5, 2.5], dtype=tf.float64),
            )},
    "path_prefix": os.path.join(".", "exp_res", "DTLZ4_6I_4O"),
}


DTLZ4_5_Input_3_Output_cfg = {
    "pb": DTLZ4,
    "num_obj": 3,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 5, "num_objective": 3},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_5I_3O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5, 2.5], dtype=tf.float64),
            )},
    "path_prefix": os.path.join(".", "exp_res", "DTLZ4_5I_3O"),
}


DTLZ4_5_Input_4_Output_cfg = {
    "pb": DTLZ4,
    "num_obj": 4,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 5, "num_objective": 4},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ4_5I_4O_PF_F.txt')),
                ref_point=tf.constant([2.5] * 4, dtype=tf.float64),
            )},
    "path_prefix": os.path.join(".", "exp_res", "DTLZ4_5I_4O"),
}


DTLZ5_5_Input_4_Output_cfg = {
    "pb": DTLZ5,
    "num_obj": 4,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 5, "num_objective": 4},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ5_5I_4O_PF_F.txt')),
                ref_point=tf.constant([2.5] * 4, dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ5_5I_4O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ5_5I_4O_PF_F.txt')),
            )
            },
    "path_prefix": os.path.join(".", "exp_res", "DTLZ5_5I_4O"),
}


DTLZ5_4_Input_3_Output_cfg = {
    "pb": DTLZ5,
    "num_obj": 3,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 4, "num_objective": 3},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model

    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ5_4I_3O_PF_F.txt')),
                ref_point=tf.constant([2.5] * 3, dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ5_4I_3O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DTLZ5_4I_3O_PF_F.txt')),
            )
            },
    "path_prefix": os.path.join(".", "exp_res", "DTLZ5_4I_3O"),
}


ZDT1_5_Input_2_Output_cfg = {
    "pb": ZDT1,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 5},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT1_5I_2O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT1_5I_2O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT1_5I_2O_PF_F.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "ZDT1_5I_2O"),
}


ZDT2_5_Input_2_Output_cfg = {
    "pb": ZDT2,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 5},
    "recommenders": {
        # OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT2_5I_2O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT2_5I_2O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT2_5I_2O_PF_F.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "ZDT2_5I_2O"),
}


ZDT4_5_Input_2_Output_cfg = {
    "pb": ZDT4,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim": 5},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT4_5I_2O_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT4_5I_2O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT4_5I_2O_PF_F.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "ZDT4_5I_2O"),
}

VehicleCrashSafty_cfg = {
    "pb": VehicleCrashSafety,
    "num_obj": 3,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VehicleCrashSafety_PF_F.txt')),
                ref_point=tf.constant([1695, 11, 0.30], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VehicleCrashSafety_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VehicleCrashSafety_PF_F.txt')),
            ), },
    "path_prefix": os.path.join(".", "exp_res", "VehicleCrashSafety"),
}


FourBarTruss_cfg = {
    "pb": FourBarTruss,
    "num_obj": 2,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'FourBarTruss_PF_F.txt')),
                ref_point=tf.constant([3400, 0.05], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'FourBarTruss_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'FourBarTruss_PF_F.txt')),
            ),
            },
    "path_prefix": os.path.join(".", "exp_res", "FourBarTruss"),
}



RocketInjectorDesign_cfg = {
    "pb": RocketInjectorDesign,
    "num_obj": 3,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'RocketInjectorDesign_PF_F.txt')),
                ref_point=tf.constant([2] * 3, dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'RocketInjectorDesign_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'RocketInjectorDesign_PF_F.txt')),
            ),
            },
    "path_prefix": os.path.join(".", "exp_res", "RocketInjectorDesign"),
}


Penicillin_7_Input_3_Output_cfg = {
    "pb": Penicillin,
    "num_obj": 3,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'Penicillin_7I_3O_PF_F.txt')),
                ref_point=tf.constant([1.85, 86.93, 514.70], dtype=tf.float64),
            )},
    "path_prefix": os.path.join(".", "exp_res", "Penicillin_7I_3O"),
}


# ZDT3_5_Input_2_Output_cfg = {
#     "pb": ZDT3,
#     "num_obj": 2,
#     "initial_x": "Stored",
#     "kwargs_for_benchmark": {"input_dim": 5},
#     "recommenders": {
#         OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
#         IN_SAMPLE: recommend_pareto_front_from_existing_data,
#     },
#     "post_metrics":
#         {
#             LogHypervolumeDifference: partial(
#                 log_hypervolume_difference,
#                 reference_pf=np.loadtxt(
#                     os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT3_5I_2O_PF_F.txt')),
#                 ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
#             ),
#             AverageHausdauffDistance: partial(
#                 average_hausdoff_distance,
#                 reference_pf=np.loadtxt(
#                     os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT3_5I_2O_PF_F.txt')),
#             ),
#             AdditiveEpsilonIndicator: partial(
#                 additive_epsilon_indicator,
#                 reference_pf = np.loadtxt(
#                     os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT3_5I_2O_PF_F.txt')),
#             ),
#             NegLogMarginalParetoFrontier: partial(
#                 negative_log_marginal_likelihood_of_target_pareto_frontier,
#                 reference_pf_input=np.loadtxt(
#                     os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT3_5I_2O_NLL_PF_X.txt')),
#                 reference_pf=np.loadtxt(
#                     os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ZDT3_5I_2O_NLL_PF_F.txt')),
#                 numerical_stability_term = 1e-7)
#             },
#     "path_prefix": os.path.join(".", "exp_res", "ZDT3_5I_2O"),
# }

