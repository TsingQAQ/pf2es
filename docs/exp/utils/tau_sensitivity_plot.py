import os
from typing import List, Optional
import numpy as np
from matplotlib import pyplot as plt
from os.path import dirname, abspath
import tensorflow as tf
from trieste.acquisition.multi_objective.pareto import Pareto

# matplotlib.rc("text", usetex=True)
# matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}" r"\usepackage{amsfonts}"]


def get_performance_result_from_file(path_prefix: str, exp_repeat: int, file_prefix: str, file_suffix: str):
    regret_list = []
    for i in range(exp_repeat):
        res_path = os.path.join(path_prefix, "".join([file_prefix, f"_{i}_", file_suffix, ".txt"]))
        try:
            regret_list.append(np.loadtxt(res_path)[None, ...])
        except:
            print(f"Cannot load {res_path}, skip")
    regrets = np.concatenate(regret_list, axis=0)
    return regrets


def plot_convergence_curve_for_cfg(cfg: dict, ax: Optional = None, plot_ylabel: bool = True,
                                   plot_xlabel: bool = True, plot_title: bool = True,
                                   multi_panel_plot = None, move_up=0.15,
                                   box_scaling=0.8, move_right=0.0, uncertainty_quantification: bool = False):
    path_prefix = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        cfg['exp_type'],
        "exp_res",
        cfg["pb_name"],
    )
    legend_handles = []
    legend_label = []

    color_idx = 0
    color_list = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"] # Not use for color blindness

    if ax is None:
        multi_panel_plot = False
        _, ax = plt.subplots(figsize=cfg["plot_cfg"]["fig_size"])
    else:
        multi_panel_plot = True

    for acq_exp_label, _ in cfg["acq"].items():
        # print(acq_exp_label)
        acq_exp_label_for_plot = acq_exp_label if acq_exp_label != "PF2ES_tau1E-1" else r'{PF}$^2$ES-$\tau(1e-1)$'
        if acq_exp_label == "PF2ES_tau1E-5_q2":
            acq_exp_label_for_plot = r'{PF}$^2$ES-$\tau(1e-5)$'
        if acq_exp_label == "PF2ES_tau1E-4_q2":
            acq_exp_label_for_plot = r'{PF}$^2$ES-$\tau(1e-4)$'
        if acq_exp_label == "PF2ES_tau1E-3_q2":
            acq_exp_label_for_plot = r'{PF}$^2$ES-$\tau(1e-3)$'
        if acq_exp_label == "PF2ES_tau1E-2_q2":
            acq_exp_label_for_plot = r'{PF}$^2$ES-$\tau(1e-2)$'

        if uncertainty_quantification is False:
            x_space = 1 if 'x_space' not in cfg else cfg['x_space']
            _path_prefix = os.path.join(path_prefix, acq_exp_label, cfg["res_path_prefix"])
            exp_regrets = get_performance_result_from_file(
                _path_prefix,
                exp_repeat=cfg["exp_repeat"],
                file_prefix=cfg["pb_name"],
                file_suffix=cfg["file_suffix"],
            )
            if "max_iter" in cfg.keys():
                exp_regrets = exp_regrets[:, : cfg["max_iter"] + 1]
            if "fix_threshold" in cfg.keys():
                exp_regrets += cfg["fix_threshold"]
            a1 = ax.plot(np.arange(exp_regrets.shape[-1]), np.percentile(exp_regrets, 50, axis=0), color=color_list[color_idx], zorder=5)
            ax.fill_between(
                np.arange(exp_regrets.shape[-1]),
                np.percentile(exp_regrets, 25, axis=0),
                np.percentile(exp_regrets, 75, axis=0),
                color=color_list[color_idx],
                alpha=0.2,
                label=acq_exp_label_for_plot,
                zorder=3
            )
            a2 = ax.fill(np.NaN, np.NaN, alpha=0.2, color=color_list[color_idx])
            legend_handles.append((a1[0], a2[0]))
            legend_label.append(acq_exp_label_for_plot)
            color_idx += 1
        else:
            ground_truth_pf = \
                np.loadtxt(
                    os.path.join(dirname(dirname(abspath(__file__))), cfg['exp_type'], 'cfg', 'ref_opts',
                                 cfg['PF_file_name']))
            ground_truth_hv = Pareto(tf.convert_to_tensor(ground_truth_pf, dtype=tf.float64)).hypervolume_indicator(
                cfg['ref_point'])

            x_space = 1 if 'x_space' not in cfg else cfg['x_space']
            _path_prefix = os.path.join(path_prefix, acq_exp_label, cfg["res_path_prefix"])
            from docs.exp.utils.pf_uncertainty_calibration_post_inspectation import _get_performance_result_from_file
            exp_regrets = _get_performance_result_from_file(
                _path_prefix,
                exp_repeat=cfg["exp_repeat"],
                file_prefix=cfg["pb_name"],
                file_suffix=cfg["file_suffix"],
                ref_point=cfg['ref_point']
            )  # [exp_repeat, bo_iter, pf_samples]

            exp_regrets = np.log(exp_regrets)  # we use loge here
            a1 = ax.plot(np.arange(exp_regrets.shape[-2]), np.mean(np.percentile(exp_regrets, 50, axis=-1), axis=0),
                         color=color_list[color_idx], zorder=5)
            ax.fill_between(
                np.arange(exp_regrets.shape[-2]),
                np.mean(np.percentile(exp_regrets, 10, axis=-1), axis=0),
                np.mean(np.percentile(exp_regrets, 90, axis=-1), axis=0),
                color=color_list[color_idx],
                alpha=0.2,
                label=acq_exp_label_for_plot,
                zorder=3
            )
            a2 = ax.fill(np.NaN, np.NaN, alpha=0.2, color=color_list[color_idx])
            legend_handles.append((a1[0], a2[0]))
            legend_label.append(acq_exp_label_for_plot)
            color_idx += 1
    if uncertainty_quantification:
        ax.hlines(np.log(ground_truth_hv), 0, exp_regrets.shape[-2], linestyles='--', color='r',
                  label='Reference Hypervolume')
    if plot_xlabel:
        ax.set_xlabel(
            "BO Iterations", fontsize=cfg["plot_cfg"].get("label_fontsize", 10)
        )
    if plot_ylabel:
        ax.set_ylabel(
            cfg["plot_cfg"]["plot_ylabel"],
            fontsize=cfg["plot_cfg"].get("label_fontsize", 10),
        )
    if cfg["plot_cfg"]["log_y"] == True:
        plt.yscale("log")
    # plt.yticks([1e-1, 1e-2, 1e-3])
    ax.tick_params(labelsize=cfg["plot_cfg"].get("tick_fontsize", 5))
    box = ax.get_position()
    if multi_panel_plot is True:
        ax.set_position([box.x0 + box.width * move_right, box.y0 + box.height * move_up,
                                box.width, box.height * box_scaling])

    # Put a legend below current axis
    if multi_panel_plot is False:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   fancybox=True, shadow=False, ncol=2, fontsize=cfg["plot_cfg"].get("lgd_fontsize", 10))
    # plt.legend(fontsize=cfg["plot_cfg"].get("lgd_fontsize", 10), framealpha=0.3)
    if plot_title:
        ax.set_title(cfg["plot_cfg"]["title"], fontsize=cfg["plot_cfg"].get("title_fontsize", 10))
    ax.grid(zorder=1)
    # plt.tight_layout()
    # plt.show(block=True)
    ax.grid(True, color="w", linestyle="-", linewidth=2)
    ax.patch.set_facecolor("0.85")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # plt.tight_layout()
    if multi_panel_plot is False:
        plt.show(block=True)
    # plt.savefig(os.path.join(path_prefix, "".join([cfg["pb_name"], cfg["plot_file_suffix"], ".png"])), dpi=300)


def multi_panel_bo_history_curve(cfgs: List[dict], figsize: tuple, legend_fontsize: int = 20,
                                              ncol: int = None, save_figure_name: str='Exp_Res'):
    """
    Create Multi-Panel Plot for Bayesian Optimization
    """
    assert len(cfgs) >= 2
    fig, axis = plt.subplots(figsize=figsize, nrows=1, ncols=len(cfgs))
    for ax, cfg, id in zip(axis, cfgs, range(len(cfgs))):
        if id == 0:
            plot_convergence_curve_for_cfg(cfg, ax=ax, plot_ylabel=True)
        else:
            plot_convergence_curve_for_cfg(cfg, ax=ax, plot_ylabel=False)
    handles, labels = axis[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), fancybox=False, ncol=ncol, fontsize=legend_fontsize)
    plt.show(block=True)
    # plt.savefig(f'{save_figure_name}.png', dpi=500)


vlmop2_tau_sensitivity_check_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_tau1E-2_q2": {},
            "PF2ES_tau1E-3_q2": {},
            "PF2ES_tau1E-4_q2": {},
            "PF2ES_tau1E-5_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "VLMOP2",
        "title": 'Out-of-sample',
        "title_fontsize": 14,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
    "max_iter": 25,
}


vlmop2_tau_sensitivity_check_in_sample_log_hv_diff_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_tau1E-2_q2": {},
            "PF2ES_tau1E-3_q2": {},
            "PF2ES_tau1E-4_q2": {},
            "PF2ES_tau1E-5_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "VLMOP2",
        "title": 'In-sample',
        "title_fontsize": 14,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
    "max_iter": 25,
}



vlmop2_pf_uc_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "PF_file_name": 'VLMOP2_PF_F.txt',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_tau1E-2_q2": {},
            "PF2ES_tau1E-3_q2": {},
            "PF2ES_tau1E-4_q2": {},
            "PF2ES_tau1E-5_q2": {},
    },
    "ref_point": tf.constant([3.0, 3.0], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 3,
    "file_suffix": "Model_Believe__model_inferred_pfs_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "VLMOP2",
        "title": 'Uncertainty Calibration',
        "title_fontsize": 14,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
    "max_iter": 25,
}


cbranincurrin_tau_sensitivity_check_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "CBraninCurrin",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES_tau1E-2_q2": {},
            "PF2ES_tau1E-3_q2": {},
            "PF2ES_tau1E-4_q2": {},
            "PF2ES_tau1E-5_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "C-BraninCurrin",
        "title": 'Out-of-sample',
        "title_fontsize": 14,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
    "max_iter": 25,
}

cbranincurrin_tau_sensitivity_check_in_sample_log_hv_diff_exp_config = {
    "pb_name": "CBraninCurrin",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES_tau1E-2_q2": {},
            "PF2ES_tau1E-3_q2": {},
            "PF2ES_tau1E-4_q2": {},
            "PF2ES_tau1E-5_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "C-BraninCurrin",
        "title": "In-sample",
        "title_fontsize": 14,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
    "max_iter": 25,
}



cbranincurrin_pf_uc_exp_config = {
    "pb_name": "CBraninCurrin",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "exp_type": "constraint_exp",
    "PF_file_name": 'C_BraninCurrin_PF_F.txt',
    "acq": {
            "PF2ES_tau1E-2_q2": {},
            "PF2ES_tau1E-3_q2": {},
            "PF2ES_tau1E-4_q2": {},
            "PF2ES_tau1E-5_q2": {},
    },
    "ref_point": tf.constant([300, 20], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 3,
    "file_suffix": "Model_Believe__model_inferred_pfs_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "C-BraninCurrin",
        "title": 'Uncertainty Calibration',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
    "max_iter": 25,
}


if __name__ == '__main__':
    # epsilon sensitivity check
    # multi_panel_bo_history_curve(
    #     cfgs=[vlmop2_epsilon_sensitivity_check_out_of_sample_log_hv_diff_exp_config,
    #           cbranincurrin_epsilon_sensitivity_check_out_of_sample_log_hv_diff_exp_config, ],
    #     figsize=(15, 4.5), ncol=6, legend_fontsize=14,
    #     save_figure_name='Unconstraint_Sequential_Exp_Res_out_of_sample_log_hv')
    # multi_panel_bo_history_curve(
    #     cfgs=[vlmop2_epsilon_sensitivity_check_in_sample_log_hv_diff_exp_config,
    #           cbranincurrin_epsilon_sensitivity_check_in_sample_log_hv_diff_exp_config, ],
    #     figsize=(15, 4.5), ncol=6, legend_fontsize=14,
    #     save_figure_name='Unconstraint_Sequential_Exp_Res_in_sample_log_hv')
    # we manually plot 4 figure here
    cfgs = [vlmop2_tau_sensitivity_check_out_of_sample_log_hv_diff_exp_config,
            vlmop2_tau_sensitivity_check_in_sample_log_hv_diff_exp_config,
            vlmop2_pf_uc_exp_config,
            cbranincurrin_tau_sensitivity_check_out_of_sample_log_hv_diff_exp_config,
            cbranincurrin_tau_sensitivity_check_in_sample_log_hv_diff_exp_config,
            cbranincurrin_pf_uc_exp_config]
    fig, axis = plt.subplots(figsize=(10, 5), nrows=2, ncols=3)
    plot_convergence_curve_for_cfg(cfgs[0], ax=axis[0, 0], plot_ylabel=True, plot_xlabel=False, move_up=0.35, box_scaling=0.8)
    plot_convergence_curve_for_cfg(cfgs[1], ax=axis[0, 1], plot_ylabel=False, plot_xlabel=False,move_up=0.35,  box_scaling=0.8)
    plot_convergence_curve_for_cfg(cfgs[2], ax=axis[0, 2], plot_ylabel=False, plot_xlabel=False,move_up=0.35,  move_right=0.2, box_scaling=0.8, uncertainty_quantification=True)
    plot_convergence_curve_for_cfg(cfgs[3], ax=axis[1, 0], plot_ylabel=True, plot_title=False, move_up=0.5, box_scaling=0.8)
    plot_convergence_curve_for_cfg(cfgs[4], ax=axis[1, 1], plot_ylabel=False, plot_title=False, move_up=0.5, box_scaling=0.8)
    plot_convergence_curve_for_cfg(cfgs[5], ax=axis[1, 2], plot_ylabel=False, plot_title=False, move_up=0.5, move_right=0.2, box_scaling=0.8, uncertainty_quantification=True)
    axis[1, 0].text(-13, 1, 'log Hypervolume Difference', rotation=90, fontsize=15)
    axis[1, 0].text(61.5, 3, 'Hypervolume Indicator', rotation=90, fontsize=15)
    handles, labels = axis[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), fancybox=False, ncol=3, fontsize=12)
    plt.show(block=True)
    # plt.savefig('tau_sensitivity_analysis.png', dpi=300)