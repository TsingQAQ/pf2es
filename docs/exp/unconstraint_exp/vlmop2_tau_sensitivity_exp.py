from docs.exp.interface import (
    main
)

"""
This is experiment conducted to test the sensitivity of tau
"""


def run_PF2ES_1EM5_q2():
    print("Run PF2ES tau 1e-5")
    main(
        cfg="VLMOP2_cfg",
        pb='VLMOP2',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_tau1E-5_q2',
        n=25,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "temperature_tau": 1e-5},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_1EM4_q2():
    print("Run PF2ES tau 1e-4")
    main(
        cfg="VLMOP2_cfg",
        pb='VLMOP2',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_tau1E-4_q2',
        n=25,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "temperature_tau": 1e-4},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_1EM3_q2():
    print("Run PF2ES tau 1e-3")
    main(
        cfg="VLMOP2_cfg",
        pb='VLMOP2',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_tau1E-3_q2',
        n=25,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "temperature_tau": 1e-3},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_1EM2_q2():
    print("Run PF2ES tau 1e-2")
    main(
        cfg="VLMOP2_cfg",
        pb='VLMOP2',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_tau1E-2_q2',
        n=25,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "temperature_tau": 1e-2},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_1EM1_q2():
    print("Run PF2ES tau 1e-1")
    main(
        cfg="VLMOP2_cfg",
        pb='VLMOP2',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_tau1E-1_q2',
        n=25,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "temperature_tau": 1e-1},
        kw_bench={"initial_x": "Stored"},
    )



def run_PF2ES_1EM3_q4():
    print("Run PF2ES + 0.01")
    main(
        cfg="VLMOP2_cfg",
        pb='VLMOP2',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_tau1E-3_q4',
        n=13,
        q=4,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "temperature_tau": 1e-3},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_1EM2_q4():
    print("Run PF2ES + 0.05")
    main(
        cfg="VLMOP2_cfg",
        pb='VLMOP2',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_tau1E-2_q4',
        n=13,
        q=4,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "temperature_tau": 1e-2},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_1EM1_q4():
    print("Run PF2ES + 0.10")
    main(
        cfg="VLMOP2_cfg",
        pb='VLMOP2',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_tau1E-1_q4',
        n=13,
        q=4,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "temperature_tau": 1e-1},
        kw_bench={"initial_x": "Stored"},
    )

exp_cfg = {
    -1: run_PF2ES_1EM5_q2,
    0: run_PF2ES_1EM4_q2,
    1: run_PF2ES_1EM3_q2,
    2: run_PF2ES_1EM2_q2,  # we have runned 0.05, no need to run again
    3: run_PF2ES_1EM1_q2,
    4: run_PF2ES_1EM3_q4,
    5: run_PF2ES_1EM2_q4,  # we have runned 0.05, no need to run again
    6: run_PF2ES_1EM1_q4,

}


if __name__ == "__main__":
    which_to_run = [-1]
    for which in which_to_run:
        exp_cfg[which]()
