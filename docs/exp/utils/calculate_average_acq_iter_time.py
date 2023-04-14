import tensorflow as tf
from tensorflow.core.util import event_pb2
from os.path import dirname, abspath
import os
import glob
import numpy as np


def calculate_acq_iter_time(exp_type, exp_name, acq_name, exp_repeat):
    assert exp_type in ['unconstraint_exp', 'constraint_exp']
    d = dirname(dirname(abspath(__file__)))
    _path_prefix = os.path.join(d, exp_type, 'exp_res', exp_name, 'logs', 'tensorboard', acq_name)
    acq_iter_times = []
    for i in range(exp_repeat):
        _path = os.path.join(_path_prefix, f'{i}')
        tf_log_file_abs_path = glob.glob(os.path.join(_path, '*'))
        serialized_examples = tf.data.TFRecordDataset(tf_log_file_abs_path)
        for serialized_example in serialized_examples:
            event = event_pb2.Event.FromString(serialized_example.numpy())
            for value in event.summary.value:
                if value.tag == 'wallclock/query_point_generation':
                    t = tf.make_ndarray(value.tensor)
                    print(value.tag, event.step, t, type(t))
                    acq_iter_times.append(t)
    print(len(acq_iter_times))
    print(np.mean(acq_iter_times))
    print(np.std(acq_iter_times))
    return acq_iter_times


if __name__ == '__main__':
    calculate_acq_iter_time('constraint_exp','SRN', 'qPF2ES_q2', 30)