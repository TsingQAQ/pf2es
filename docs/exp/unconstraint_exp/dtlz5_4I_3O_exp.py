from docs.exp.interface import (
    main
)


def run_PF2ES_epsilon5():
    print("Run PF2ES + 0.05")
    main(
        cfg="DTLZ5_4_Input_3_Output_cfg",
        pb='DTLZ5',
        r=30,
        c=10,
        acq="PF2ES",
        acq_name='PF2ES_epsilon0.05',
        n=40,
        q=1,
        fp="",
        kw_acq= {"pareto_epsilon": 0.05},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_EHVI():
    print("Run EHVI")
    main(
        cfg="DTLZ5_4_Input_3_Output_cfg",
        pb='DTLZ5',
        r=30,
        c=10,
        acq="EHVI",
        acq_name="EHVI",
        n=40,
        q=1,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_Random():
    print("Run Random")
    main(
        cfg="DTLZ5_4_Input_3_Output_cfg",
        pb='DTLZ5',
        r=30,
        c=10,
        acq="Random",
        acq_name="Random",
        n=50,
        q=1,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
    )


def run_MESMO():
    print("Run MESMO")
    main(
        cfg="DTLZ5_4_Input_3_Output_cfg",
        pb='DTLZ5',
        r=30,
        c=10,
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
        cfg="DTLZ5_4_Input_3_Output_cfg",
        pb='DTLZ5',
        r=30,
        c=10,
        acq="PFES",
        acq_name="PFES",
        n=50,
        q=1,
        fp="",
        kw_acq={"sample_pf_num": 5},
        kw_bench={"initial_x": "Stored"},
    )


exp_cfg = {
    1: run_PF2ES_epsilon5,
    2: run_EHVI,
    3: run_Random,
    4: run_MESMO,
    5: run_PFES
}


if __name__ == "__main__":
    which_to_run = [1, 2, 3, 4, 5]
    for which in which_to_run:
        exp_cfg[which]()
