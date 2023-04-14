import os
from os.path import dirname, abspath
import numpy as np
from functools import partial

import tensorflow as tf
from trieste.utils.performance_metrics.multi_objective import \
    log_hypervolume_difference, \
    average_hausdoff_distance, \
    additive_epsilon_indicator, \
    LogHypervolumeDifference, \
    AverageHausdauffDistance, \
    AdditiveEpsilonIndicator, \
    hypervolume_indicator, \
    HypervolumeIndicator


from trieste.objectives.multi_objectives import CVLMOP2, CBraninCurrin, TNK, Osy, Constr_Ex, C2DTLZ2, \
    WaterProblem, C3DTLZ4, VLMOP2BraninCurrin, VLMOP2ConstrEx, \
    C1DTLZ3, SRN, C3DTLZ5,\
    DiscBrakeDesign, EE6, WeldedBeamDesign, ConceptualMarineDesign
from trieste.utils.optimal_recommenders.multi_objective import IN_SAMPLE, OUT_OF_SAMPLE, MODEL_BELIEVE, \
    recommend_pareto_front_from_existing_data, recommend_pareto_front_from_model_prediction, \
    inspecting_pareto_front_distributions_from_model


CVLMOP2_cfg = {
    "pb": CVLMOP2,
    "num_obj": 2,
    "num_con": 1,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'CVLMOP2_PF.txt')),
                ref_point=tf.constant([1.2, 1.2], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'CVLMOP2_PF.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'CVLMOP2_PF.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "CVLMOP2"),
}


CBraninCurrin_cfg = {
    "pb": CBraninCurrin,
    "num_obj": 2,
    "num_con": 1,
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
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C_BraninCurrin_PF_F.txt')),
                ref_point=tf.constant([80.0, 12.0], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C_BraninCurrin_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C_BraninCurrin_PF_F.txt')),
            ),},
    "path_prefix": os.path.join(".", "exp_res", "C_BraninCurrin"),
}


TNK_cfg = {
    "pb": TNK,
    "num_obj": 2,
    "num_con": 2,
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'TNK_PF_F.txt')),
                ref_point=tf.constant([1.2, 1.2], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'TNK_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'TNK_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "TNK"),
               }


Osy_cfg = {
    "pb": Osy,
    "num_obj": 2,
    "num_con": 6,
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'Osy_PF_F.txt')),
                ref_point=tf.constant([50.0, 100.0], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'Osy_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'Osy_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "Osy"),
               }


SRN_cfg = {
    "pb": SRN,
    "num_obj": 2,
    "num_con": 2,
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
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'SRN_PF_F.txt')),
                ref_point=tf.constant([250, 50], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'SRN_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'SRN_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "SRN"),
               }


Constr_Ex_cfg = {
    "pb": Constr_Ex,
    "num_obj": 2,
    "num_con": 2,
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
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'Constr_Ex_PF_F.txt')),
                ref_point=tf.constant([1.1, 10.0], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'Constr_Ex_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'Constr_Ex_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "Constr_Ex"),
               }


C2DTLZ2_cfg = {
    "pb": C2DTLZ2,
    "num_obj": 2,
    "num_con": 1,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim":12, "num_objective":2},
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_PF_F.txt')),
                ref_point=tf.constant([2.5, 2.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "C2DTLZ2"),
               }


WATER_3I_5O_cfg = {
    "pb": WaterProblem,
    "num_obj": 5,
    "num_con": 7,
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'WATER_PF_F.txt')),
                ref_point=tf.constant([1.5] * 5, dtype=tf.float64),
            ),
            # AverageHausdauffDistance: partial(
            #     average_hausdoff_distance,
            #     reference_pf=np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'WATER_PF_F.txt')),
            # ),
            # AdditiveEpsilonIndicator: partial(
            #     additive_epsilon_indicator,
            #     reference_pf = np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'WATER_PF_F.txt')),
            # ),
        },
    "path_prefix": os.path.join(".", "exp_res", "WATER"),
               }


C2DTLZ2_5I_4O_cfg = {
    "pb": C2DTLZ2,
    "num_obj": 4,
    "num_con": 1,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim":5, "num_objective":4},
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_5I_4O_PF_F.txt')),
                ref_point=tf.constant([2.5] * 4, dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_5I_4O_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_5I_4O_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "C2DTLZ2_5I_4O"),
               }


CVLMOP2BraninCurrin_5I_4O_cfg = {
    "pb": VLMOP2BraninCurrin,
    "num_obj": 4,
    "num_con": 2,
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'CVLMOP2BraninCurrin_PF_F.txt')),
                ref_point=tf.constant([2.5] * 4, dtype=tf.float64),
                reference_hv = tf.convert_to_tensor(22.5, dtype=tf.float64)
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "CVLMOP2BraninCurrin"),
               }


VLMOP2ConstrEx_2I_4O_cfg = {
    "pb": VLMOP2ConstrEx,
    "num_obj": 4,
    "num_con": 2,
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'VLMOP2ConstrEx_PF_F.txt')),
                ref_point=tf.constant([2.0] * 4, dtype=tf.float64),
                reference_hv = tf.convert_to_tensor(10.0, dtype=tf.float64)
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "VLMOP2ConstrEx"),
               }



C1DTLZ3_5I_4O_cfg = {
    "pb": C1DTLZ3,
    "num_obj": 4,
    "num_con": 1,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim":5, "num_objective":4},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C1DTLZ3_PF_F.txt')),
                ref_point=tf.constant([2.5] * 4, dtype=tf.float64),
                reference_hv =tf.convert_to_tensor(39.0, dtype=tf.float64)
            ),
            # AverageHausdauffDistance: partial(
            #     average_hausdoff_distance,
            #     reference_pf=np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C1DTLZ3_PF_F.txt')),
            # ),
            # AdditiveEpsilonIndicator: partial(
            #     additive_epsilon_indicator,
            #     reference_pf = np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C1DTLZ3_PF_F.txt')),
            # ),
        },
    "path_prefix": os.path.join(".", "exp_res", "C1DTLZ3"),
               }


C3DTLZ4_5I_4O_cfg = {
    "pb": C3DTLZ4,
    "num_obj": 4,
    "num_con": 4,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim":5, "num_objective":4},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C3DTLZ4_PF_F.txt')),
                ref_point=tf.constant([2.5] * 4, dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C3DTLZ4_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C3DTLZ4_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "C3DTLZ4"),
               }


C3DTLZ5_5I_4O_cfg = {
    "pb": C3DTLZ5,
    "num_obj": 4,
    "num_con": 4,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim":5, "num_objective":4},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C3DTLZ5_PF_F.txt')),
                ref_point=tf.constant([2.0] * 4, dtype=tf.float64),
                reference_hv = tf.convert_to_tensor(8.5, dtype=tf.float64)
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "C3DTLZ5"),
               }


DiscBrakeDesign_cfg = {
    "pb": DiscBrakeDesign,
    "num_obj": 2,
    "num_con": 4,
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DiscBrakeDesign_PF_F.txt')),
                ref_point=tf.constant([8.0, 4.0], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DiscBrakeDesign_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'DiscBrakeDesign_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "DiscBrakeDesign"),
               }


WeldedBeamDesign_cfg = {
    "pb": WeldedBeamDesign,
    "num_obj": 2,
    "num_con": 4,
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'WeldedBeamDesign_PF_F.txt')),
                ref_point=tf.constant([40, 0.02], dtype=tf.float64),
            ),
            # AverageHausdauffDistance: partial(
            #     average_hausdoff_distance,
            #     reference_pf=np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'WeldedBeamDesign_PF_F.txt')),
            # ),
            # AdditiveEpsilonIndicator: partial(
            #     additive_epsilon_indicator,
            #     reference_pf = np.loadtxt(
            #         os.path.join(dirname(abspath(__file__)), 'ref_opts', 'WeldedBeamDesign_PF_F.txt')),
            # ),
        },
    "path_prefix": os.path.join(".", "exp_res", "WeldedBeamDesign"),
               }


C2DTLZ2_4D_cfg = {
    "pb": C2DTLZ2,
    "num_obj": 2,
    "num_con": 1,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {"input_dim":4, "num_objective":2},
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
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_4D_PF_F.txt')),
                ref_point=tf.constant([1.5, 1.5], dtype=tf.float64),
            ),
            AverageHausdauffDistance: partial(
                average_hausdoff_distance,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_4D_PF_F.txt')),
            ),
            AdditiveEpsilonIndicator: partial(
                additive_epsilon_indicator,
                reference_pf = np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'C2DTLZ2_4D_PF_F.txt')),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "C2DTLZ2_4D"),
               }

EE6_cfg = {
    "pb": EE6,
    "num_obj": 2,
    "num_con": 1,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: partial(recommend_pareto_front_from_existing_data, return_direct_obs=True),
        # MODEL_BELIEVE: inspecting_pareto_front_distributions_from_model
    },
    "post_metrics":
        {
            HypervolumeIndicator: partial(
                hypervolume_indicator,
                ref_point=tf.constant([1, 1], dtype=tf.float64),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "EE6"),
}


MarineDesign_cfg = {
    "pb": ConceptualMarineDesign,
    "num_obj": 3,
    "num_con": 9,
    "initial_x": "Stored",
    "kwargs_for_benchmark": {},
    "recommenders": {
        OUT_OF_SAMPLE: recommend_pareto_front_from_model_prediction,
        IN_SAMPLE: recommend_pareto_front_from_existing_data,
    },
    "post_metrics":
        {
            LogHypervolumeDifference: partial(
                log_hypervolume_difference,
                reference_pf=np.loadtxt(
                    os.path.join(dirname(abspath(__file__)), 'ref_opts', 'ConceptualMarineDesign_6I_3O_PF_F.txt')),
                ref_point=tf.constant([-700, 13000, 3500], dtype=tf.float64),
            ),
        },
    "path_prefix": os.path.join(".", "exp_res", "ConceptualMarineDesign"),
               }