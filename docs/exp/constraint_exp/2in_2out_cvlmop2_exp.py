import os
import tensorflow as tf

from docs.exp.interface import (
    main
)


def run_PF2ES_original():
    print("Run PF2ES Original")
    main(
        cfg="CVLMOP2_cfg",
        pb='CVLMOP2',
        r=1,
        c=1,
        acq="PF2ES",
        acq_name='PF2ES_original',
        n=5,
        q=1,
        fp="",
        kw_acq= {"pareto_epsilon": 0.00},
        kw_bench={"doe_num": 10, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_PF2ES_without_augmentation():
    print("Run PF2ES Without Augmentation")
    main(
        cfg="CVLMOP2_cfg",
        pb='CVLMOP2',
        r=10,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_without_augmentation',
        n=25,
        q=1,
        fp="",
        kw_acq= {"pareto_epsilon": 0.00, "remove_augmenting_region": True},
        kw_bench={"doe_num": 10, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_PF2ES_epsilon1():
    print("Run PF2ES + 0.01")
    main(
        cfg="CVLMOP2_cfg",
        pb='CVLMOP2',
        r=10,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_epsilon0.01',
        n=25,
        q=1,
        fp="",
        kw_acq= {"pareto_epsilon": 0.01},
        kw_bench={"doe_num": 10, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_PF2ES_epsilon5():
    print("Run PF2ES + 0.05")
    main(
        cfg="CVLMOP2_cfg",
        pb='CVLMOP2',
        r=10,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_epsilon0.05',
        n=25,
        q=1,
        fp="",
        kw_acq= {"pareto_epsilon": 0.05},
        kw_bench={"doe_num": 10, "initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_EHVI():
    print("Run EHVI")
    raise NotImplementedError
    # TODO
    main(
        cfg="branin",
        r=30,
        c=15,
        acq="Random",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 10, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        kw_metrics={"worst_obj_val": 337.7120990791997},
        ref_optf="Branin_MV_Normal_0.01_Opt_F.txt",
        vc=160.0,
    )


def run_Random():
    print("Run Random")
    raise NotImplementedError
    # TODO
    main(
        cfg="branin",
        r=30,
        c=15,
        acq="US",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
            "infer_mc_num": 10000,
        },
        kw_bench={"doe_num": 10, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimize={"acquire_return_builder": True},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.01, 0.01], tf.float64)),
            "kw_al": {"which_al_obj": "std"},
        },
        kw_metrics={"worst_obj_val": 337.7120990791997},
        ref_optf="Branin_MV_Normal_0.01_Opt_F.txt",
        vc=160.0,
    )


def run_PFES():
    print("Run PFES")
    raise NotImplementedError
    # TODO
    main(
        cfg="branin",
        r=30,
        c=15,
        acq="CO-MVA-BO",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
            "approx_mc_num": 100,
            "tau": 0.05,
            "variance_threshold": 160.0,
        },
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_optimizer={
            "x_delta": 1.96 * tf.sqrt(tf.constant([0.01, 0.01], tf.float64)),
            "kw_al": {"which_al_obj": "P_std"},
        },
        kw_bench={"doe_num": 10, "initial_x": "Stored"},
        kw_metrics={"worst_obj_val": 337.7120990791997},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Branin_MV_Normal_0.01_Opt_F.txt",
        vc=160.0,
    )


def run_MESMO():
    print("Run MESMO")
    raise NotImplementedError
    # TODO
    main(
        cfg="branin",
        r=30,
        c=15,
        acq="FF-MV-qECI",
        n=50,
        n_tp="normal",
        n_p=tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
        q=1,
        fp=os.path.join("normal", "q1_rff"),
        kw_acq={
            "noise_type": "normal",
            "noise_param": tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
            "ff_method": "RFF",
            "opt_ff_num": 900,
            "infer_mc_num": 10000,
            "variance_threshold": 160.0,
            "mc_num": 128,
            "max_batch_element": 80,
            "rec_var_prob_threshold": 0.8,
            "implicit_sample": True,
            "pseudo_min": 337.7120990791997,
        },
        kw_bench={"doe_num": 10, "initial_x": "Stored"},
        kw_rule={"num_initial_samples": 20, "num_optimization_runs": 1},
        kw_metrics={"worst_obj_val": 337.7120990791997},
        kw_optimize={"acquire_return_builder": True},
        ref_optf="Branin_MV_Normal_0.01_Opt_F.txt",
        vc=160.0,
        which_rec="acquisition.recommend",
    )


exp_cfg = {
    1: run_PF2ES_original,
    2: run_PF2ES_epsilon1,
    3: run_PF2ES_epsilon5,
    4: run_PF2ES_without_augmentation,
    5: run_EHVI,
    6: run_Random,
    7: run_PFES,
    8: run_MESMO
}


if __name__ == "__main__":
    which_to_run = [1]
    for which in which_to_run:
        exp_cfg[which]()
