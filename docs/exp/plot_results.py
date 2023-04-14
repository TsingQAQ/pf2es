from docs.exp.utils.bo_history_plot import *
from typing import Optional


def plot_convergence_curve_for_exp(pb_name: str, formulation: str, noise_type: str, fix_min: Optional[float] = None, x_space: Optional[int] = 1):
    assert formulation in ['smv', 'vc', 'mo']
    cfg = eval('_'.join([pb_name, formulation, 'config']))
    title_dict = eval('_'.join([pb_name, formulation, 'titles']))
    cfg["res_path_prefix"] = path_res[noise_type]
    cfg['plot_cfg']["title"] = title_dict[noise_type]
    cfg['which_noise'] = noise_type
    cfg['x_space'] = x_space
    plot_convergence_curve_for_cfg(cfg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-cfg", type=str)
    parser.add_argument("-x_sp", "--x_space", type=int, default=1)

    _args = parser.parse_args()
    cfg = _args.cfg
    formulation = _args.formulation
    noise_type = _args.noise_type
    fix_min = _args.fix_min
    x_space = _args.x_space
    plot_convergence_curve_for_exp(cfg, formulation, noise_type, fix_min=fix_min, x_space=x_space)