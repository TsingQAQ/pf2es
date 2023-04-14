from docs.exp.interface import (
    main
)
from trieste.observer import CONSTRAINT, OBJECTIVE


def run_PF2ES():
    print("Run PF2ES + 0.04")
    main(
        cfg="Osy_cfg",
        pb='Osy',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES',
        n=60,
        q=1,
        fp="",
        kw_acq= {"constraint_tag": CONSTRAINT},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_EHVI_PoF():
    print("Run EHVI_PoF")
    main(
        cfg="Osy_cfg",
        pb='Osy',
        r=30,
        c=15,
        acq="EHVI_PoF",
        acq_name="EHVI_PoF",
        n=60,
        q=1,
        fp="",
        kw_acq= {"objective_tag": OBJECTIVE, "constraint_tag": CONSTRAINT},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_Random():
    print("Run EHVI")
    main(
        cfg="Osy_cfg",
        pb='Osy',
        r=30,
        c=15,
        acq="Random",
        acq_name="Random",
        n=60,
        q=1,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_MESMOC():
    print("Run MESMOC")
    main(
        cfg="Osy_cfg",
        pb='Osy',
        r=30,
        c=15,
        acq="MESMOC",
        acq_name="MESMOC",
        n=60,
        q=1,
        fp="",
        kw_acq={"sample_pf_num": 5, "constraint_tag": CONSTRAINT},
        kw_bench={"initial_x": "Stored"},
    )


def run_qPF2ES_q4():
    print("Run qPF2ES + 0.04")
    main(
        cfg="Osy_cfg",
        pb='Osy',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q4',
        n=30,
        q=4,
        fp="",
        kw_acq={"constraint_tag": CONSTRAINT, "parallel_sampling": True, "batch_mc_sample_size": 128},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_qPF2ES_q3():
    print("Run qPF2ES + 0.04")
    main(
        cfg="Osy_cfg",
        pb='Osy',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q3',
        n=40,
        q=3,
        fp="",
        kw_acq={"constraint_tag": CONSTRAINT, "parallel_sampling": True, "batch_mc_sample_size": 128},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_qPF2ES_q2():
    print("Run qPF2ES + 0.04")
    main(
        cfg="Osy_cfg",
        pb='Osy',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q2',
        n=60,
        q=2,
        fp="",
        kw_acq={"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "parallel_sampling": True, "batch_mc_sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_Random_q2():
    print("Run Random q2")
    main(
        cfg="Osy_cfg",
        pb='Osy',
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
        cfg="Osy_cfg",
        pb='Osy',
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


exp_cfg = {
    1: run_PF2ES,
    2: run_EHVI_PoF,
    3: run_Random,
    4: run_MESMOC,
    5: run_qPF2ES_q4,
    11: run_qPF2ES_q2,
    13: run_Random_q2,
    14: run_qCEHVI_q2,
    15: run_qPF2ES_q3
}

if __name__ == "__main__":
    which_to_run = [15, 5]
    for which in which_to_run:
        exp_cfg[which]()
