from docs.exp.interface import (
    main
)


def run_PF2ES():
    print("Run PF2ES + 0.04")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES',
        n=50,
        q=1,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_PF2ES_exp_dbscan():
    print("Run PF2ES + dbscan")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_epsilon_dbscan',
        n=50,
        q=1,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.00, 'use_dbscan_for_conservative_epsilon': True},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_EHVI():
    print("Run EHVI")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="EHVI",
        acq_name="EHVI",
        n=50,
        q=1,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_Random():
    print("Run EHVI")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="Random",
        acq_name="Random",
        n=50,
        q=1,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_MESMO():
    print("Run MESMO")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="MESMO",
        acq_name="MESMO",
        n=50,
        q=1,
        fp="",
        kw_acq={"sample_pf_num": 5},
        kw_bench={"initial_x": "Stored"},
    )


def run_PFES():
    print("Run PFES")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="PFES",
        acq_name="PFES",
        n=50,
        q=1,
        fp="",
        kw_acq={"sample_pf_num": 5},
        kw_bench={"initial_x": "Stored"},
    )



def run_PF2ES_KB_q2():
    print("Run PF2ES + 0.04 Kriging Believer q2")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="PF2ES-KB",
        acq_name='PF2ES_KB_q2',
        n=25,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
    )


def run_Random_q2():
    print("Run Random q2")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="Random",
        acq_name="Random_q2",
        n=25,
        q=2,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
    )


def run_qPF2ES_q2():
    print("Run q-PF2ES + 0.04 q2")
    main(
        cfg="FourBarTruss_cfg",
        pb='FourBarTruss',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q2',
        n=25,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "parallel_sampling": True, "batch_mc_sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
    )



def run_qEHVI_q2():
    print("Run q-EHVI")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="qEHVI",
        acq_name='qEHVI_q2',
        n=25,
        q=2,
        fp="",
        kw_acq= {"sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
    )



def run_PF2ES_KB_q4():
    print("Run PF2ES + 0.04 Kriging Believer q2")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="PF2ES-KB",
        acq_name='PF2ES_KB_q4',
        n=13,
        q=4,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
    )


def run_Random_q4():
    print("Run Random q2")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="Random",
        acq_name="Random_q4",
        n=13,
        q=4,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
    )


def run_qPF2ES_q4():
    print("Run q-PF2ES + 0.04 q2")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_epsilon0.04_q4',
        n=13,
        q=4,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
    )


def run_qEHVI_q4():
    print("Run q-EHVI")
    main(
        cfg="VehicleCrashSafty_cfg",
        pb='VehicleCrashSafety',
        r=30,
        c=15,
        acq="qEHVI",
        acq_name='qEHVI_q4',
        n=13,
        q=4,
        fp="",
        kw_acq= {"sample_size": 64},
        kw_bench={"initial_x": "Stored"},
    )

exp_cfg = {
    1: run_PF2ES,
    2: run_EHVI,
    3: run_Random,
    4: run_MESMO,
    5: run_PFES,
    11: run_PF2ES_KB_q2,
    12: run_Random_q2,
    13: run_qPF2ES_q2,
    14: run_qEHVI_q2,
    21: run_PF2ES_KB_q4,
    22: run_Random_q4,
    23: run_qPF2ES_q4,
    24: run_qEHVI_q4,
    0: run_PF2ES_exp_dbscan
}


if __name__ == "__main__":
    which_to_run = [1, 2, 3, 4, 5]
    for which in which_to_run:
        exp_cfg[which]()
