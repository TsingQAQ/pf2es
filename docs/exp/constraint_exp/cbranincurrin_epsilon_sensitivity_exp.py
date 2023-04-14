from docs.exp.interface import (
    main
)
from trieste.observer import CONSTRAINT

"""
This is experiment conducted to test the sensitivity of epsilon
"""


def run_PF2ES():
    print("Run PF2ES Origin")
    main(
        cfg="CBraninCurrin_cfg",
        pb='CBraninCurrin',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_origin',
        n=50,
        q=1,
        fp="",
        kw_acq= {"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "pareto_epsilon": 0.00},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_epsilon001():
    print("Run PF2ES + 0.01")
    main(
        cfg="CBraninCurrin_cfg",
        pb='CBraninCurrin',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_001',
        n=50,
        q=1,
        fp="",
        kw_acq= {"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "pareto_epsilon": 0.01},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_epsilon003():
    print("Run PF2ES + 0.03")
    main(
        cfg="CBraninCurrin_cfg",
        pb='CBraninCurrin',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_003',
        n=50,
        q=1,
        fp="",
        kw_acq= {"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "pareto_epsilon": 0.03},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_epsilon004():
    print("Run PF2ES + 0.04")
    main(
        cfg="CBraninCurrin_cfg",
        pb='CBraninCurrin',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_004',
        n=50,
        q=1,
        fp="",
        kw_acq= {"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_epsilon005():
    print("Run PF2ES + 0.05")
    main(
        cfg="CBraninCurrin_cfg",
        pb='CBraninCurrin',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_005',
        n=50,
        q=1,
        fp="",
        kw_acq= {"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "pareto_epsilon": 0.05},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_epsilon010():
    print("Run PF2ES + 0.10")
    main(
        cfg="CBraninCurrin_cfg",
        pb='CBraninCurrin',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_010',
        n=50,
        q=1,
        fp="",
        kw_acq= {"constraint_tag": CONSTRAINT, "sample_pf_num": 5, "pareto_epsilon": 0.10},
        kw_bench={"initial_x": "Stored"},
    )


exp_cfg = {
    0: run_PF2ES,
    1: run_PF2ES_epsilon001,
    3: run_PF2ES_epsilon003,
    4: run_PF2ES_epsilon004,
    5: run_PF2ES_epsilon005,  # we have runned 0.05, no need to run again
    10: run_PF2ES_epsilon010,
}


if __name__ == "__main__":
    # which_to_run = [0, 1, 3, 4, 5, 10]
    which_to_run = [3, 4, 5, 10]  # run on sumo-ai 2022/07/13
    for which in which_to_run:
        exp_cfg[which]()
