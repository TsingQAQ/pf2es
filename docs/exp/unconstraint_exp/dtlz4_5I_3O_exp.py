from docs.exp.interface import (
    main
)


def run_MESMO():
    print("Run MESMO")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="MESMO",
        acq_name="MESMO",
        n=60,
        q=1,
        fp="",
        kw_acq={"sample_pf_num": 5},
        kw_bench={"initial_x": "Stored"},
    )


def run_PFES():
    print("Run PFES")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="PFES",
        acq_name="PFES",
        n=60,
        q=1,
        fp="",
        kw_acq={"sample_pf_num": 5},
        kw_bench={"initial_x": "Stored"},
    )
    
    
def run_PF2ES():
    print("Run PF2ES + 0.04")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='PF2ES',
        n=60,
        q=1,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_EHVI():
    print("Run EHVI")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="EHVI",
        acq_name="EHVI",
        n=60,
        q=1,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
        kw_optimize={"acquire_return_builder": True},
    )


def run_Random():
    print("Run EHVI")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
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


def run_qPF2ES_q2():
    print("Run q-PF2ES + 0.04 q2")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q2',
        n=30,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "batch_mc_sample_size": 128,
                 'parallel_sampling': True, "qMC": True},
        kw_bench={"initial_x": "Stored"},
    )


def run_qEHVI_q2():
    print("Run q-EHVI")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="qEHVI",
        acq_name='qEHVI_q2',
        n=30,
        q=2,
        fp="",
        kw_acq= {"sample_size": 128, "qMC": True},
        kw_bench={"initial_x": "Stored"},
    )


def run_PF2ES_KB_q2():
    print("Run PF2ES + 0.04 Kriging Believer q2")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="PF2ES-KB",
        acq_name='PF2ES_KB_q2',
        n=30,
        q=2,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04},
        kw_bench={"initial_x": "Stored"},
    )


def run_Random_q2():
    print("Run Random q2")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="Random",
        acq_name="Random_q2",
        n=30,
        q=2,
        fp="",
        kw_acq={},
        kw_bench={"initial_x": "Stored"},
    )



def run_qPF2ES_q3():
    print("Run q-PF2ES + 0.04 q2")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q3',
        n=20,
        q=3,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "batch_mc_sample_size": 128,
                 'parallel_sampling': True, "qMC": True},
        kw_bench={"initial_x": "Stored"},
    )
    
    
def run_qPF2ES_q4():
    print("Run q-PF2ES + 0.04 q2")
    main(
        cfg="DTLZ4_5_Input_3_Output_cfg",
        pb='DTLZ4',
        r=30,
        c=15,
        acq="PF2ES",
        acq_name='qPF2ES_q4',
        n=15,
        q=4,
        fp="",
        kw_acq= {"sample_pf_num": 5, "pareto_epsilon": 0.04, "batch_mc_sample_size": 128,
                 'parallel_sampling': True, "qMC": True},
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
}


if __name__ == "__main__":
    which_to_run = [13, 12, 14, 11]
    for which in which_to_run:
        exp_cfg[which]()
