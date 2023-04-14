from docs.exp.interface import (
    main
)
from trieste.observer import CONSTRAINT, OBJECTIVE


def run_PF2ES():
    print("Run PF2ES + 0.04")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES',
        n=120,
        q=1,
        fp="",
        kw_acq= {"constraint_tag": CONSTRAINT},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_EHVI_PoF():
    print("Run EHVI_PoF")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="EHVI_PoF",
        acq_name="EHVI_PoF",
        n=120,
        q=1,
        fp="",
        kw_acq= {"objective_tag": OBJECTIVE, "constraint_tag": CONSTRAINT},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_Random():
    print("Run EHVI")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="Random",
        acq_name="Random",
        n=120,
        q=1,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_MESMOC():
    print("Run MESMOC")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="MESMOC",
        acq_name="MESMOC",
        n=120,
        q=1,
        fp="",
        kw_acq={"sample_pf_num": 5, "constraint_tag": CONSTRAINT},
        kw_bench={"initial_x": "Stored"},
    )


def run_qPF2ES_q2():
    print("qRun PF2ES")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q2',
        n=60,
        q=2,
        fp="",
        kw_acq={"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "parallel_sampling": True,
                "batch_mc_sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_qPF2ES_q3():
    print("qRun PF2ES")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q3',
        n=40,
        q=3,
        fp="",
        kw_acq={"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "parallel_sampling": True,
                "batch_mc_sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_qPF2ES_q4():
    print("qRun PF2ES")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q4',
        n=30,
        q=4,
        fp="",
        kw_acq={"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "parallel_sampling": True,
                "batch_mc_sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_PF2ES_KB_q2():
    print("Run PF2ES + 0.04 Kriging Believer q2")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="PF2ES-KB",
        acq_name='PF2ES_KB_q2',
        n=60,
        q=2,
        fp="",
        kw_acq={"sample_pf_num": 5, "constraint_tag": CONSTRAINT, "pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
    )


def run_Random_q2():
    print("Run Random q2")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="Random",
        acq_name="Random_q2",
        n=60,
        q=2,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
    )


def run_qCEHVI_q2():
    print('Run qCEHVI')
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="qCEHVI",
        acq_name='qCEHVI_q2',
        n=60,
        q=2,
        fp="",
        kw_acq={"objective_tag": OBJECTIVE, "constraint_tag": CONSTRAINT, "sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
    )


def run_qCEHVI_q3():
    print('Run qCEHVI')
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="qCEHVI",
        acq_name='qCEHVI_q3',
        n=40,
        q=3,
        fp="",
        kw_acq={"objective_tag": OBJECTIVE, "constraint_tag": CONSTRAINT, "sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
    )


def run_qCEHVI_q4():
    print('Run qCEHVI')
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="qCEHVI",
        acq_name='qCEHVI_q4',
        n=30,
        q=4,
        fp="",
        kw_acq={"objective_tag": OBJECTIVE, "constraint_tag": CONSTRAINT, "sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
    )


def run_Random_q4():
    print("Run Random q4")
    main(
        cfg="C2DTLZ2_4D_cfg",
        pb='C2DTLZ2',
        r=30,
        c=15,
        acq="Random",
        acq_name="Random_q4",
        n=40,
        q=4,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
    )


exp_cfg = {
    1: run_PF2ES,
    2: run_EHVI_PoF,
    3: run_Random,
    4: run_MESMOC,
    11: run_qPF2ES_q2,
    12: run_PF2ES_KB_q2,
    13: run_Random_q2,
    14: run_qCEHVI_q2,
    21: run_qPF2ES_q3,
    22: run_qCEHVI_q3,
    31: run_qPF2ES_q4,
    32: run_qCEHVI_q4,
    33: run_Random_q4
}

if __name__ == "__main__":
    which_to_run = [31, 22, 32]
    for which in which_to_run:
        exp_cfg[which]()
