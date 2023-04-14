import argparse
import json
from functools import partial

from docs.exp.multi_objective_benchmarker import parallel_benchmarker
from docs.exp.unconstraint_exp.cfg import pb_cfgs as unconstrained_pb_cfgs
from docs.exp.constraint_exp.cfg import pb_cfgs as constrained_pb_cfgs
from docs.exp.utils.bo_result_processor import get_performance_metric_from_optimize_result
from trieste.acquisition.function import (
    MESMO,
    MESMOC,
    PFES,
    PF2ES,
    ExpectedHypervolumeImprovement,
    BatchMonteCarloExpectedHypervolumeImprovement,
    BatchMonteCarloConstrainedExpectedHypervolumeImprovement,
    CEHVI
)
from trieste.space import Box
from trieste.utils.optimal_recommenders.multi_objective import OUT_OF_SAMPLE, MODEL_BELIEVE
from trieste.utils.performance_metrics.multi_objective import HypervolumeIndicator


def main(**kwargs):
    if kwargs is not None:
        cfg_name = kwargs['cfg']
        pb_name = kwargs['pb']
        exp_repeat = kwargs['r']
        workers = kwargs['c']
        acq = kwargs['acq']
        acq_name = kwargs['acq_name']
        total_iter = kwargs['n']
        file_info_prefix = kwargs['file_info_prefix'] if 'file_info_prefix' in kwargs else ""
        q = kwargs['q'] if 'q' in kwargs else 1
        kwargs_for_acq = kwargs['kw_acq'] if 'kw_acq' in kwargs else {}
        kwargs_for_model = kwargs['kw_model'] if 'kw_model' in kwargs else {}
    else:

        parser = argparse.ArgumentParser()

        parser.add_argument("-cfg")
        parser.add_argument("-pb")
        parser.add_argument("-n", "--total_iter", type=int)
        parser.add_argument("-c", "--core", type=int)
        parser.add_argument("-r", "--repeat", type=int)
        parser.add_argument("-acq")
        parser.add_argument("-acq_n")
        parser.add_argument("-kw_acq", type=json.loads)
        parser.add_argument("-kw_model", type=json.loads)

        # optional args
        # FIXME: Must specify one of it, otherwise there is a weird behavior
        parser.add_argument("-fp", "--file_info_prefix", default="", type=str)
        parser.add_argument("-q", "--batch_query", default=1, type=int)

        _args = parser.parse_args()
        cfg_name = _args.cfg
        pb_name = _args.pb
        exp_repeat = _args.repeat
        workers = _args.core
        acq = _args.acq
        acq_name = _args.acq_n
        total_iter = _args.total_iter

        file_info_prefix = _args.file_info_prefix if _args.file_info_prefix else ""
        q = _args.batch_query if _args.batch_query else 1
        kwargs_for_acq = _args.kw_acq if _args.kw_acq else {}
        kwargs_for_model = _args.kw_model if _args.kw_model else {}
    try:
        pb_cfg = getattr(unconstrained_pb_cfgs, cfg_name)
    except AttributeError:
        try:
            pb_cfg = getattr(constrained_pb_cfgs, cfg_name)
        except AttributeError:
            raise NotImplementedError(
                rf"NotImplemented Problem: {pb_name} specified, however, it doesn\'t mean this "
                r"benchmark cannot be used for a new problem, in order to do so,"
                r"you may need to first write your own problem cfg in cfg/pb_cfgs.py"
            )

    # Maybe can stored on local before hand
    fantasizer = False
    if acq == "PF2ES":
        acq = PF2ES
    elif acq == "PF2ES-KB":
        acq = PF2ES
        fantasizer = True
    elif acq == 'PFES':
        acq = PFES
    elif acq == "MESMO":
        acq = MESMO
    elif acq == "EHVI":
        acq = ExpectedHypervolumeImprovement
    elif acq == "qEHVI":
        acq = BatchMonteCarloExpectedHypervolumeImprovement
    elif acq == "qCEHVI":
        acq = BatchMonteCarloConstrainedExpectedHypervolumeImprovement
    elif acq == "Random" or acq == "Random_q2" or acq == "Random_q4":
        acq = None
    elif acq_name == 'EHVI_PoF':
        acq = CEHVI
    elif acq == 'MESMOC':
        acq = MESMOC
    else:
        raise NotImplementedError(rf"NotImplemented Acquisition: {acq} specified")
    # We put it here for parallel issue
    pb = pb_cfg["pb"](**pb_cfg["kwargs_for_benchmark"])
    if OUT_OF_SAMPLE in pb_cfg["recommenders"].keys():
        pb_cfg["recommenders"][OUT_OF_SAMPLE] = \
            partial(pb_cfg["recommenders"][OUT_OF_SAMPLE],
                    search_space=Box(pb.bounds[0], pb.bounds[1]), min_feasibility_probability=0.95,
                    hard_constraint_threshold_perc=2e-2)

    if MODEL_BELIEVE in pb_cfg["recommenders"].keys():
        pb_cfg["recommenders"][MODEL_BELIEVE] = \
            partial(pb_cfg["recommenders"][MODEL_BELIEVE],
                    search_space=Box(pb.bounds[0], pb.bounds[1]),
                    obj_num = pb_cfg['num_obj'],
                    sample_pf_num =10,
                    cons_num = pb_cfg['num_con'] if 'num_con' in pb_cfg.keys() else 0,
                    )

    # if HypervolumeIndicator in pb_cfg['post_metrics'].keys():
    #     pb_cfg['post_metrics'][HypervolumeIndicator] = \
    #         partial(pb_cfg['post_metrics'][HypervolumeIndicator],
    #                 true_func_inst = pb_cfg['pb']().joint_objective_con())

    # have a default seed 1817
    parallel_benchmarker(
        benchmark_name=pb_name,
        acq_name=acq_name,
        file_info_prefix=file_info_prefix,
        exp_repeat=exp_repeat,
        workers=workers,
        acq=acq,
        q=q,
        total_iter=total_iter,
        fantasizer = fantasizer,
        **pb_cfg,
        post_profiler=partial(
            get_performance_metric_from_optimize_result,
            recommenders=pb_cfg["recommenders"],
            metrics=pb_cfg["post_metrics"],
            true_func_inst=pb,
        ),
        kwargs_for_acq=kwargs_for_acq,
        kwargs_for_model=kwargs_for_model,
        # start_exp_id = 14
    )


if __name__ == "__main__":
    main()
    # main(cfg="VLMOP2_cfg",
    #      pb="VLMOP2",
    #      c=1,
    #      r=1,
    #      n=3,
    #      fp="MC5",
    #      acq="BCPFES_IBO",
    #      kw_acq={"objective_tag": "OBJECTIVE", "pf_mc_sample_num": 2,
    #              "kwargs_for_pf_sampler": {"num_moo_iter": 500}})
