import os
from typing import List, Optional
import numpy as np
from matplotlib import pyplot as plt

# matplotlib.rc("text", usetex=True)
# matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}" r"\usepackage{amsfonts}"]


def get_performance_result_from_file(path_prefix: str, exp_repeat: int, file_prefix: str, file_suffix: str):
    regret_list = []
    for i in range(exp_repeat):
        if path_prefix.split('\\')[-2] == 'PESMO' or path_prefix.split('\\')[-2] == 'MESMOC_PLUS' or \
                path_prefix.split('\\')[-2] == 'PPESMOC':
            if 'Out_Of_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_os_'
            elif 'In_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_is_'
            res_path = os.path.join(path_prefix, "".join([file_suffix, f'{i+1}', ".txt"]))
        elif path_prefix.split('\\')[-2] == 'PPESMOC_with_concate':
            if 'Out_Of_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_os_with_concate_'
            res_path = os.path.join(path_prefix, "".join([file_suffix, f'{i + 1}', ".txt"]))
        elif path_prefix.split('\\')[-2] == 'PPESMOC_without_concate':
            if 'Out_Of_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_os_without_concate_'
            res_path = os.path.join(path_prefix, "".join([file_suffix, f'{i + 1}', ".txt"]))
        else:
            res_path = os.path.join(path_prefix, "".join([file_prefix, f"_{i}_", file_suffix, ".txt"]))
        try:
            print(i)
            # nan fixing
            res = np.loadtxt(res_path)[None, ...]
            res[np.isnan(res)] = np.min(res[~np.isnan(res)])
            regret_list.append(res)
        except:
            print(f"Cannot load {res_path}, skip")
    # try:
    if path_prefix.split('\\')[-2] == 'PPESMOC' and file_prefix == 'C2DTLZ2':
        regrets = np.concatenate([regret[:, :60] for regret in regret_list], axis=0)
    elif path_prefix.split('\\')[-2] == 'MESMOC_PLUS' and file_prefix == 'C2DTLZ2':
        regrets = np.concatenate([regret[:, :121] for regret in regret_list], axis=0)
    elif path_prefix.split('\\')[-2] == 'MESMOC_PLUS' and file_prefix == 'SRN':
        regrets = np.concatenate([regret[:, :31] for regret in regret_list], axis=0)
    else:
        regrets = np.concatenate(regret_list, axis=0)
    # except:
    return regrets


def plot_convergence_curve_for_cfg(cfg: dict, ax: Optional = None, plot_ylabel: bool = True):
    path_prefix = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        cfg['exp_type'],
        "exp_res",
        cfg["pb_name"],
    )
    legend_handles = []
    legend_label = []

    color_idx = 0
    color_list = cfg['plot_cfg'].get('color_list', ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]) # Not use for color blindness

    if ax is None:
        multi_panel_plot = False
        _, ax = plt.subplots(figsize=cfg["plot_cfg"]["fig_size"])
    else:
        multi_panel_plot = True

    for acq_exp_label, _ in cfg["acq"].items():
        acq_exp_label_for_plot = acq_exp_label if acq_exp_label != "PF2ES_epsilon0.05" else '{PF}$^2$ES'
        if acq_exp_label == 'PF2ES':
            acq_exp_label_for_plot = '{PF}$^2$ES'
        if acq_exp_label == 'PF2ES_q2' or acq_exp_label == 'PF2ES_epsilon0.04_q2' or acq_exp_label == 'qPF2ES_q2':
            acq_exp_label_for_plot = 'q-{PF}$^2$ES'
        if acq_exp_label == 'PF2ES_KB_q2':
            acq_exp_label_for_plot = '{PF}$^2$ES-KB'
        if acq_exp_label == 'qEHVI_q2' or acq_exp_label == 'qCEHVI_q2':
            acq_exp_label_for_plot = 'qEHVI'
        if acq_exp_label == 'EHVI_PoF':
            acq_exp_label_for_plot = 'EHVI-PoF'
        if acq_exp_label == "Random_q2":
            acq_exp_label_for_plot = 'Random'
        if acq_exp_label == "MESMOC_PLUS":
            acq_exp_label_for_plot = 'MESMOC+'
        if acq_exp_label == "PPESMOC_without_concate":
            acq_exp_label_for_plot = 'PPESMOC'
        # print(acq_exp_label)

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
        if acq_exp_label =='PF2ES' or acq_exp_label =='PF2ES_q2' or acq_exp_label =='qPF2ES_q2':
            a1 = ax.plot(np.arange(exp_regrets.shape[-1]),
                         np.percentile(exp_regrets, 50, axis=0),
                         # np.mean(exp_regrets, axis=0),
                         color=color_list[color_idx], zorder=555)
        else:
            a1 = ax.plot(np.arange(exp_regrets.shape[-1]),
                         np.percentile(exp_regrets, 50, axis=0),
                         # np.mean(exp_regrets, axis=0),
                         color=color_list[color_idx], zorder=5)
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
    ax.set_xlabel(
        cfg["plot_cfg"].get('xlabel', "BO Iterations"), fontsize=cfg["plot_cfg"].get("label_fontsize", 10)
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
        # ax.set_position([box.x0, box.y0 + box.height * 0.15,
        #                         box.width, box.height * 0.85])
        ax.set_position([box.x0 - box.width * 0.28, box.y0 + box.height * 0.12,
                                box.width, box.height * 0.8])

    # Put a legend below current axis
    if multi_panel_plot is False:
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        #            fancybox=True, shadow=False, ncol=2, fontsize=cfg["plot_cfg"].get("lgd_fontsize", 3))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   fancybox=True, shadow=False, ncol=2, fontsize=cfg["plot_cfg"].get("lgd_fontsize", 10))
    # plt.legend(fontsize=cfg["plot_cfg"].get("lgd_fontsize", 10), framealpha=0.3)
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
    return legend_handles
    # plt.savefig(os.path.join(path_prefix, "".join([cfg["pb_name"], cfg["plot_file_suffix"], ".png"])), dpi=300)


def multi_panel_bo_history_curve(cfgs: List[dict], figsize: tuple, legend_fontsize: int = 20,
                                              ncol: int = None, save_figure_name: str='Exp_Res',
                                 bbox_to_anchor=(0.98, 0.5), savefig: bool=False):
    """
    Create Multi-Panel Plot for Bayesian Optimization
    """
    assert len(cfgs) >= 2
    fig, axis = plt.subplots(figsize=figsize, nrows=1, ncols=len(cfgs))
    for ax, cfg, id in zip(axis, cfgs, range(len(cfgs))):
        if id == 0:
            handles = plot_convergence_curve_for_cfg(cfg, ax=ax, plot_ylabel=True)
        else:
            plot_convergence_curve_for_cfg(cfg, ax=ax, plot_ylabel=False)
    _, labels = axis[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=bbox_to_anchor,
    #            fancybox=False, ncol=ncol, fontsize=legend_fontsize)
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=bbox_to_anchor, fancybox=False, ncol=ncol,
               fontsize=legend_fontsize)
    if savefig:
        plt.savefig(f'{save_figure_name}.png', dpi=500)
    else:
        plt.show(block=True)


vlmop2_NLLPF_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'NLLPF',
    "exp_type": "unconstraint_exp",
    "acq": {
    "PF2ES_epsilon0.05": {},
    "EHVI": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_NegLogMarginalParetoFrontier_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "Negative Log Marginal Likelihood",
        "title": 'VLMOP2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


vlmop2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "MESMO": {},
            "PFES": {},
            'PESMO': {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'VLMOP2\n $d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50

}


vlmop2_out_of_sample_avd_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'VLMOP2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


vlmop2_out_of_sample_eps_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'VLMOP2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


vlmop2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "MESMO": {},
            "PFES": {},
            'PESMO': {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        'title': 'VLMOP2\n $d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}


vlmop2_in_sample_avd_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        'VLMOP2': '$d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


vlmop2_in_sample_eps_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        'VLMOP2': '$d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


vlmop2_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "xlabel": "Batch BO Iterations",
        'title': 'VLMOP2 \n $d=2, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    },
    "max_iter": 15
}


vlmop2_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        'title': 'VLMOP2\n $d=2, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    },
    "max_iter": 15
}

cvlmop2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "CVLMOP2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": { "PF2ES_epsilon0.05": {}, "PF2ES_original": {}, "PF2ES_without_augmentation": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'CVLMOP2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


cvlmop2_out_of_sample_avd_exp_config = {
    "pb_name": "CVLMOP2",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": { "PF2ES_epsilon0.05": {}, "PF2ES_original": {}, "PF2ES_without_augmentation": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'CVLMOP2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


cvlmop2_out_of_sample_eps_exp_config = {
    "pb_name": "CVLMOP2",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": { "PF2ES_epsilon0.05": {}, "PF2ES_original": {}, "PF2ES_without_augmentation": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'CVLMOP2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


cvlmop2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "CVLMOP2",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "constraint_exp",
    "acq": { "PF2ES_epsilon0.05": {}, "PF2ES_original": {}, "PF2ES_without_augmentation": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'CVLMOP2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


cvlmop2_in_sample_avd_exp_config = {
    "pb_name": "CVLMOP2",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "constraint_exp",
    "acq": { "PF2ES_epsilon0.05": {}, "PF2ES_original": {}, "PF2ES_without_augmentation": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'CVLMOP2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


cvlmop2_in_sample_eps_exp_config = {
    "pb_name": "CVLMOP2",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "constraint_exp",
    "acq": { "PF2ES_epsilon0.05": {}, "PF2ES_original": {}, "PF2ES_without_augmentation": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'CVLMOP2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


branincurrin_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "MESMO": {},
            "PFES": {},
            'PESMO': {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'BraninCurrin\n $d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}


branincurrin_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "xlabel": "Batch BO Iterations",
        "title": 'BraninCurrin\n $d=2, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    },
    'max_iter': 20
}


branincurrin_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'BraninCurrin\n $d=2, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
'color_list': ['C1', 'C2', 'C0', 'C6']
    },
    'max_iter': 20
}



branincurrin_out_of_sample_avd_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'BraninCurrin Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


branincurrin_out_of_sample_eps_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'BraninCurrin Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


branincurrin_in_sample_log_hv_diff_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "MESMO": {},
            "PFES": {},
            'PESMO': {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'BraninCurrin\n $d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}


branincurrin_in_sample_avd_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'BraninCurrin $d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


branincurrin_in_sample_eps_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'BraninCurrin $d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}

dtlz2_7i_2o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_7i_2o_out_of_sample_avd_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'DTLZ2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_7i_2o_out_of_sample_eps_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'DTLZ2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_7i_2o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_7i_2o_in_sample_avd_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'DTLZ2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_7i_2o_in_sample_eps_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'DTLZ2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_7i_2o_NLLPF_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'NLLPF',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_NegLogMarginalParetoFrontier_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "Negative Log Marginal Likelihood",
        "title": 'DTLZ2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_4i_3o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_4i_3o_out_of_sample_avd_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'DTLZ2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_4i_3o_out_of_sample_eps_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'DTLZ2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_4i_3o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_4i_3o_in_sample_avd_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'DTLZ2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_4i_3o_in_sample_eps_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'DTLZ2',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz2_4i_3o_NLLPF_exp_config = {
    "pb_name": "DTLZ2",
    "plot_file_suffix": 'NLLPF',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_NegLogMarginalParetoFrontier_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "Negative Log Marginal Likelihood",
        "title": 'DTLZ2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz5_5i_4o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ5 In Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
}


dtlz5_5i_4o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ5 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
}


dtlz5_5i_4o_out_of_sample_avd_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'DTLZ5 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz5_5i_4o_out_of_sample_eps_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'DTLZ5 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}

dtlz5_5i_4o_in_sample_avd_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'DTLZ5',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


dtlz5_5i_4o_in_sample_eps_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'DTLZ5',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}

dtlz5_4i_3o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ5 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}

dtlz5_4i_3o_out_of_sample_avd_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'DTLZ5 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}

dtlz5_4i_3o_out_of_sample_eps_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'DTLZ5 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}

dtlz5_4i_3o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ5',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}

dtlz5_4i_3o_in_sample_avd_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'DTLZ5',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}

dtlz5_4i_3o_in_sample_eps_exp_config = {
    "pb_name": "DTLZ5",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'DTLZ5',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt1_5i_2o_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.04_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "xlabel": "Batch BO Iterations",
        "title": 'ZDT1 \n$d=5, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    }}


zdt1_5i_2o_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.04_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "xlabel": "Batch BO Iterations",
        "title": 'ZDT1 \n$d=5, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    }}



zdt1_5i_2o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "PFES": {},
            "MESMO": {},
            'PESMO': {},
        "Random": {},},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT1\n $d=5, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}


zdt1_5i_2o_out_of_sample_avd_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "PFES": {},
            "MESMO": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'ZDT1 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt1_5i_2o_out_of_sample_eps_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "PFES": {},
            "MESMO": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'ZDT1 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt1_5i_2o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "PFES": {},
            "MESMO": {},
            'PESMO': {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT1\n $d=5, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}


zdt1_5i_2o_in_sample_avd_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "PFES": {},
            "MESMO": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'ZDT1 $d=5 M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt1_5i_2o_in_sample_eps_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "PFES": {},
            "MESMO": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'ZDT1 $d=5 M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt2_5i_2o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {"PF2ES": {},
            "EHVI": {},
            "PFES": {},
            "MESMO": {},
            'PESMO': {},

            "Random": {},
            },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT2\n $d=5, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50,
}


zdt2_5i_2o_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.04_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "xlabel": "Batch BO Iterations",
        "title": 'ZDT2\n $d=5, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    },
    "max_iter": 10
}

zdt2_5i_2o_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.04_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT2\n $d=5, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
'color_list': ['C1', 'C2', 'C0', 'C6']
    },
    "max_iter": 15
}

zdt2_5i_2o_out_of_sample_avd_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'ZDT2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt2_5i_2o_out_of_sample_eps_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'ZDT2 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt2_5i_2o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "MESMO": {},
            "PFES": {},
            "PESMO": {},
        "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT2\n $d=5, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },

    'max_iter': 50
}


zdt2_5i_2o_in_sample_avd_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'ZDT2 $d=5, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt2_5i_2o_in_sample_eps_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'ZDT2 $d=5, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt3_5i_2o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT3",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},



            "PF2ES_pf_10_popsize_50": {},

            "EHVI": {}},
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT3 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt3_5i_2o_out_of_sample_avd_exp_config = {
    "pb_name": "ZDT3",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'ZDT3 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt3_5i_2o_out_of_sample_eps_exp_config = {
    "pb_name": "ZDT3",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'ZDT3 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt3_5i_2o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT3",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT3',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt3_5i_2o_in_sample_avd_exp_config = {
    "pb_name": "ZDT3",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'ZDT3',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt3_5i_2o_in_sample_eps_exp_config = {
    "pb_name": "ZDT3",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'ZDT3',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt4_5i_2o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT4",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT4 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt4_5i_2o_out_of_sample_avd_exp_config = {
    "pb_name": "ZDT4",
    "plot_file_suffix": 'AVD_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'ZDT4 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt4_5i_2o_out_of_sample_eps_exp_config = {
    "pb_name": "ZDT4",
    "plot_file_suffix": 'EPS_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'ZDT4 Out OF Sample Inspectation',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt4_5i_2o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "ZDT4",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ZDT4',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt4_5i_2o_in_sample_avd_exp_config = {
    "pb_name": "ZDT4",
    "plot_file_suffix": 'AVD_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AverageHausdauffDistance_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Average Hausdauff Distance",
        "title": 'ZDT4',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


zdt4_5i_2o_in_sample_eps_exp_config = {
    "pb_name": "ZDT4",
    "plot_file_suffix": 'EPS_In_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.05": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_AdditiveEpsilonIndicator_q1_",
    "plot_cfg": {
        "log_y": True,
        "label_fontsize": 13,
        "plot_ylabel": "Additive Epsilon Indicator",
        "title": 'ZDT4',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


vehicle_crash_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "VehicleCrashSafety",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.04": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
            # "PF2ES_epsilon_dbscan": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Vehicle Crash Safety',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


forbartruss_in_sample_log_hv_diff_exp_config = {
    "pb_name": "FourBarTruss",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            # "PF2ES_epsilon_dbscan": {},
            "PF2ES": {},
            "EHVI": {},
            "MESMO": {},
            "PFES": {},
            "Random": {},
            'PESMO': {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Four Bar Truss',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
    'max_iter': 40
}


fourbartruss_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "FourBarTruss",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "MESMO": {},
            "PFES": {},
            'PESMO': {},
            "Random": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Four Bar Truss \n $d=4, M=2 $',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },

    'max_iter': 40
}


RocketInjectorDesign_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "RocketInjectorDesign",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI": {},
            "MESMO": {},
            # "PFES": {},
            # 'PESMO': {},
            "Random": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'RocketInjectorDesign \n $d=4, M=3 $',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },

    'max_iter': 40
}

fourbartruss_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "FourBarTruss",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qEHVI_q2": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Four Bar Truss \n $d=4, M=2 $',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    },

    'max_iter': 15
}


vehicle_crash_in_sample_log_hv_diff_exp_config = {
    "pb_name": "VehicleCrashSafety",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            "PF2ES_epsilon0.04": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
            "PF2ES_epsilon_dbscan": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Vehicle Crash Safety',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


cbranincurrin_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "CBraninCurrin",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            "MESMOC_PLUS": {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C-BraninCurrin \n $d=2, M=2, C=1$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    "max_iter": 50
}


cbranincurrin_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "CBraninCurrin",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qCEHVI_q2": {},
            "PPESMOC_without_concate": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "xlabel": "Batch BO Iterations",
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C-BraninCurrin \n $d=2, M=2, C=1, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
    },
    "max_iter": 25
}


cbranincurrin_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "CBraninCurrin",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qCEHVI_q2": {},
            "PPESMOC": {},

        "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
"xlabel": "Batch BO Iterations",
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C-BraninCurrin \n $d=2, M=2, C=1, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
        # 'color_list': ['C1', 'C2', 'C7', 'C9', 'C6']
    },
    "max_iter": 25
}


cbranincurrin_in_sample_log_hv_diff_exp_config = {
    "pb_name": "CBraninCurrin",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            "MESMOC_PLUS": {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C-BraninCurrin\n $d=2, M=2, C=1$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    'max_iter': 50
}

Constr_Ex_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "Constr_Ex",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            "MESMOC_PLUS": {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Constr-Ex \n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    "max_iter": 60
}


Constr_Ex_in_sample_log_hv_diff_exp_config = {
    "pb_name": "Constr_Ex",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            "MESMOC_PLUS": {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Constr-Ex\n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    "max_iter": 60
}


Constr_Ex_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "Constr_Ex",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qCEHVI_q2": {},
            "PPESMOC_without_concate": {},

        "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "xlabel": "Batch BO Iterations",
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Constr-Ex \n $d=2, M=2, C=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
    },
    "max_iter": 25
}


Constr_Ex_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "Constr_Ex",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qCEHVI_q2": {},
        "PPESMOC": {},

        "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "xlabel": "Batch BO Iterations",
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Constr-Ex\n $d=2, M=2, C=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        # 'color_list': ['C1', 'C2', 'C7', 'C9', 'C6']
'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
    },
    "max_iter": 25
}

TNK_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "TNK",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "Random": {},
            "MESMOC": {},
            "MESMOC_PLUS": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'TNK \n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    "max_iter": 60
}


TNK_in_sample_log_hv_diff_exp_config = {
    "pb_name": "TNK",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "Random": {},
            "MESMOC": {},
            "MESMOC_PLUS": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'TNK\n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    "max_iter": 60
}


SRN_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "SRN",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            'MESMOC_PLUS': {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'SRN \n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    "max_iter": 30
}


SRN_in_sample_log_hv_diff_exp_config = {
    "pb_name": "SRN",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            'MESMOC_PLUS': {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'SRN\n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    "max_iter": 30
}


SRN_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "SRN",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qCEHVI_q2": {},
            "PPESMOC_without_concate": {},

        "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "xlabel": "Batch BO Iterations",
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'SRN \n $d=2, M=2, C=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
    },
    "max_iter": 60
}


SRN_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "SRN",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qCEHVI_q2": {},
            "PPESMOC": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "xlabel": "Batch BO Iterations",
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'SRN\n $d=2, M=2, C=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        # 'color_list': ['C1', 'C2', 'C7', 'C9', 'C6']
'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
    },
    "max_iter": 60
}


TNK_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "TNK",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "Random_q2": {},
            "qCEHVI_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'TNK \n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    "max_iter": 60
}


TNK_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "TNK",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "Random_q2": {},
            "qCEHVI_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'TNK\n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    "max_iter": 60
}


DiscBrakeDesign_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DiscBrakeDesign",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "Random": {},
            "MESMOC": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Disc Brake Design\n $d=4, M=2, C=6$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    "max_iter": 60
}


DiscBrakeDesign_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DiscBrakeDesign",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "Random": {},
            "MESMOC": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Disc Brake Design\n $d=4, M=2, C=6$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    "max_iter": 60
}


DiscBrakeDesign_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DiscBrakeDesign",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "Random_q2": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Disc Brake Design\n $d=4, M=2, C=6$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    "max_iter": 60
}


DiscBrakeDesign_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DiscBrakeDesign",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
        "qPF2ES_q2": {},
        "PF2ES_KB_q2": {},
        "qCEHVI_q2": {},
        "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Disc Brake Design\n $d=4, M=2, C=6$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    "max_iter": 25
}

C2DTLZ2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "Random": {},
            "MESMOC": {},
            'MESMOC_PLUS': {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C2DTLZ2 \n $d=12, M=2, C=1$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    'max_iter': 40
}


C2DTLZ2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "Random": {},
            "MESMOC": {},
            'MESMOC_PLUS': {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C2DTLZ2 \n $d=12, M=2, C=1$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    'max_iter': 40
}


C2DTLZ2_4D_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            "MESMOC_PLUS": {},
            "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C2DTLZ2 \n $d=4, M=2, C=1$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    'max_iter': 120
}


C2DTLZ2_4D_in_sample_log_hv_diff_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'LogHvDiff_In_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            "MESMOC_PLUS": {},

        "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C2DTLZ2 \n $d=4, M=2, C=1$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    'max_iter': 120
}


C2DTLZ2_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "Random_q2": {},
            "qCEHVI_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C2DTLZ2 \n $d=12, M=2, C=1, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    'max_iter': 40
}

C2DTLZ2_4D_q2_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qCEHVI_q2": {},
            "PPESMOC_without_concate": {},
            "Random_q2": {},

    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "xlabel": "Batch BO Iterations",
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C2DTLZ2 \n $d=4, M=2, C=1, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
    },
}


C2DTLZ2_4D_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "qCEHVI_q2": {},
        "PPESMOC": {},
            "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "xlabel": "Batch BO Iterations",
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C2DTLZ2 \n $d=4, M=2, C=1, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        # 'color_list': ['C1', 'C2', 'C7', 'C9', 'C6']
'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
    },
}


C2DTLZ2_q2_in_sample_log_hv_diff_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "qPF2ES_q2": {},
            "PF2ES_KB_q2": {},
            "Random_q2": {},
            "qCEHVI_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 13,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'C2DTLZ2 \n $d=12, M=2, C=1, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C7', 'C8']
    },
    'max_iter': 40
}


if __name__ == "__main__":
    # Sequential MOO plot
    # in-sample logHV
    # multi_panel_bo_history_curve(
    #    cfgs=[vlmop2_in_sample_log_hv_diff_exp_config,
    #          branincurrin_in_sample_log_hv_diff_exp_config,
    #          zdt1_5i_2o_in_sample_log_hv_diff_exp_config,
    #          zdt2_5i_2o_in_sample_avd_exp_config],
    #    figsize=(15, 2.6), ncol=1, legend_fontsize=14,
    #     save_figure_name='Seq_MOO_IS_res', savefig=False)
    # out-of-sample logHV
    # multi_panel_bo_history_curve(
    #     cfgs=[vlmop2_out_of_sample_log_hv_diff_exp_config,
    #           branincurrin_out_of_sample_log_hv_diff_exp_config,
    #           zdt1_5i_2o_out_of_sample_log_hv_diff_exp_config,
    #           vehicle_crash_out_of_sample_log_hv_diff_exp_config],
    #     figsize=(15, 2.6), ncol=1, legend_fontsize=14,
    #     save_figure_name='Seq_MOO_OS_res', savefig=False)

    # Batch MOO Plot
    ## Out-of-sample log HV
    # multi_panel_bo_history_curve(
    #     cfgs=[vlmop2_q2_out_of_sample_log_hv_diff_exp_config,
    #           branincurrin_q2_out_of_sample_log_hv_diff_exp_config,
    #           zdt1_5i_2o_q2_out_of_sample_log_hv_diff_exp_config,
    #           zdt2_5i_2o_q2_out_of_sample_log_hv_diff_exp_config],
    #     figsize=(15, 2.6), ncol=1, legend_fontsize=14,
    #     save_figure_name='Batch_MOO_q2_res', bbox_to_anchor=(1, 0.5),
    #     savefig=True)
    ## In-Sample log HV
    multi_panel_bo_history_curve(
        cfgs=[vlmop2_q2_in_sample_log_hv_diff_exp_config,
              branincurrin_q2_in_sample_log_hv_diff_exp_config,
              zdt1_5i_2o_q2_in_sample_log_hv_diff_exp_config,
              zdt2_5i_2o_q2_in_sample_log_hv_diff_exp_config],
        figsize=(15, 2.6), save_figure_name='Batch_MOO_IS_res', ncol=1, bbox_to_anchor=(1, 0.5),
        legend_fontsize=14, savefig=True)

    # Sequential Constrained Plot
    # Out-of-Sample
    # multi_panel_bo_history_curve(
    #     cfgs=[cbranincurrin_out_of_sample_log_hv_diff_exp_config,
    #           Constr_Ex_out_of_sample_log_hv_diff_exp_config,
    #           SRN_out_of_sample_log_hv_diff_exp_config,
    #           C2DTLZ2_4D_out_of_sample_log_hv_diff_exp_config],
    #     save_figure_name='Seq_CMOO_OS_res',
    #     figsize=(15, 2.6), ncol=1, legend_fontsize=14, savefig=True)
    # In-Sample
    # multi_panel_bo_history_curve(
    #     cfgs=[cbranincurrin_in_sample_log_hv_diff_exp_config,
    #           Constr_Ex_in_sample_log_hv_diff_exp_config,
    #           SRN_in_sample_log_hv_diff_exp_config,
    #           C2DTLZ2_4D_in_sample_log_hv_diff_exp_config],
    #     save_figure_name='Seq_CMOO_IS_res',
    #     figsize=(15, 2.6), ncol=1, legend_fontsize=14, savefig=True)

    # Batch CMOO Plot
    ## Out-of-sample log HV
    # multi_panel_bo_history_curve(
    #          cfgs=[cbranincurrin_q2_out_of_sample_log_hv_diff_exp_config,
    #                Constr_Ex_q2_out_of_sample_log_hv_diff_exp_config,
    #                SRN_q2_out_of_sample_log_hv_diff_exp_config,
    #                C2DTLZ2_4D_q2_out_of_sample_log_hv_diff_exp_config,
    #                ],
    #     figsize=(15, 2.6), ncol=1, legend_fontsize=14,
    #     save_figure_name='Batch_CMOO_q2_os_res', savefig=True)

    ## In-Sample log HV
    # multi_panel_bo_history_curve(
    #     cfgs=[cbranincurrin_q2_in_sample_log_hv_diff_exp_config,
    #           Constr_Ex_q2_in_sample_log_hv_diff_exp_config,
    #           SRN_q2_in_sample_log_hv_diff_exp_config,
    #           C2DTLZ2_4D_q2_in_sample_log_hv_diff_exp_config,
    #           ],
    #     figsize=(15, 2.6), ncol=1, legend_fontsize=14,
    #     save_figure_name='Batch_CMOO_q2_is_res', savefig=True)