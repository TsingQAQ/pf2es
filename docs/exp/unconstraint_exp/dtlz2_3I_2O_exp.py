from docs.exp.interface import (
    main
)


def run_PF2ES():
    print("Run PF2ES + 0.05")
    main(
        cfg="DTLZ2_3_Input_2_Output_cfg",
        pb='DTLZ2',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES_epsilon0.05',
        n=50,
        q=1,
        fp="",
        kw_acq= {"pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_EHVI():
    print("Run EHVI")
    main(
        cfg="DTLZ2_3_Input_2_Output_cfg",
        pb='DTLZ2',
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
    print("Run Random")
    main(
        cfg="DTLZ2_3_Input_2_Output_cfg",
        pb='DTLZ2',
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
        cfg="DTLZ2_3_Input_2_Output_cfg",
        pb='DTLZ2',
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
        cfg="DTLZ2_3_Input_2_Output_cfg",
        pb='DTLZ2',
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


exp_cfg = {
    1: run_PF2ES,
    2: run_EHVI,
    3: run_Random,
    4: run_MESMO,
    5: run_PFES
}


if __name__ == "__main__":
    which_to_run = [1, 2, 3, 4, 5]
    for which in which_to_run:
        exp_cfg[which]()
