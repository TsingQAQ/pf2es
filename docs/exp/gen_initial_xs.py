"""
Generate initial DOE for benchmarking
This is to assure that the comparison of different acq starts from exactly the same data
"""
import json
import os

import numpy as np

from trieste.objectives import multi_objectives
from trieste.space import Box
import tensorflow as tf


def gen_doe():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--number_of_initial_xs", type=int)
    parser.add_argument("-b", "--batch_size_of_doe", type=int)
    parser.add_argument("-pb", "--problem", type=str)
    parser.add_argument("-pt", "--problem_type", type=str)
    parser.add_argument("-kw_pb", type=json.loads, default={})
    parser.add_argument("-sd", "--seed", type=int, default=1817)

    _args = parser.parse_args()
    doe_num = _args.number_of_initial_xs
    doe_repeat = _args.batch_size_of_doe
    pb_name = _args.problem
    pb_kw = _args.kw_pb
    problem_type = _args.problem_type
    seed = _args.seed
    tf.random.set_seed(seed)
    assert problem_type in ['unconstraint_exp', 'constraint_exp']
    pb = getattr(multi_objectives, pb_name)(**pb_kw)
    assert isinstance(pb, multi_objectives.MultiObjectiveTestProblem)
    for i in range(doe_repeat):
        xs = Box(*pb.bounds).sample(doe_num)
        _path = os.path.join(problem_type, "cfg", "initial_xs", pb_name)
        _file_name = f"xs_{i}.txt"
        os.makedirs(_path, exist_ok =True)
        np.savetxt(os.path.join(_path, _file_name), xs)


if __name__ == "__main__":
    gen_doe()