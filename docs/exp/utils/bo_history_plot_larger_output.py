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
            elif 'In_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_is_'
            res_path = os.path.join(path_prefix, "".join([file_suffix, f'{i + 1}', ".txt"]))
        elif path_prefix.split('\\')[-2] == 'PPESMOC_without_concate':
            if 'Out_Of_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_os_without_concate_'
            elif 'In_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_is_'
            res_path = os.path.join(path_prefix, "".join([file_suffix, f'{i + 1}', ".txt"]))
        elif path_prefix.split('\\')[-2] == 'PPESMOC_with_concate50':
            if 'Out_Of_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_os_with_concate_50_'
            elif 'In_Sample' in file_suffix:
                file_suffix = 'log_Hv_difference_is_'
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
    if path_prefix.split('\\')[-2] == 'PPESMOC_without_concate' or path_prefix.split('\\')[-2] == 'PPESMOC_with_concate' or path_prefix.split('\\')[-2] == 'PPESMOC_with_concate50':
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
        print(acq_exp_label)
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
        if acq_exp_label == "PPESMOC_without_concate" or acq_exp_label == "PPESMOC_with_concate":
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
        if acq_exp_label =='PF2ES' or acq_exp_label =='PF2ES_q2':
            a1 = ax.plot(np.arange(exp_regrets.shape[-1]),
                         np.percentile(exp_regrets, 50, axis=0),
                         color=color_list[color_idx], zorder=555)
        else:
            a1 = ax.plot(np.arange(exp_regrets.shape[-1]),
                         np.percentile(exp_regrets, 50, axis=0),
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
        cfg["plot_cfg"].get('xlabel', "BO Iterations"), fontsize=cfg["plot_cfg"].get("label_fontsize", 25)
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
    if multi_panel_plot is True and plot_ylabel is True:
        # ax.set_position([box.x0, box.y0 + box.height * 0.15,
        #                         box.width, box.height * 0.85])
        ax.set_position([box.x0 - box.width * 0.1, box.y0 + box.height * 0.1,
                                box.width * 0.9, box.height * 0.8])
    elif multi_panel_plot is True:
        ax.set_position([box.x0 - box.width * 0.3, box.y0 + box.height * 0.1,
                                box.width * 0.9, box.height * 0.8])

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




RocketInjectorDesign_in_sample_log_hv_diff_exp_config = {
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
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 25,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Rocket Injector Design \n $d=4, M=3 $',
        "title_fontsize": 12,
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
        "label_fontsize": 25,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Rocket Injector Design \n $d=4, M=3 $',
        "title_fontsize": 12,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },

    'max_iter': 40
}

dtlz4_6i_4o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ4",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
        "PF2ES": {},
        "EHVI": {},
        "MESMO": {},
        "Random": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 12,
        "plot_ylabel": "DTLZ4 \n $d=8, M=2$",
        "title": 'Out-of-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C6']
    },
    "max_iter": 80,
}

dtlz4_6i_4o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ4",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
        "PF2ES": {},
        "EHVI": {},
        "Random": {},
        "MESMO": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 12,
        "plot_ylabel": "DTLZ4 \n $d=6, M=4$",
        "title": 'Out-of-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C6', 'C1', 'C2', 'C3']
    },
    "max_iter": 80,
}



dtlz4_5i_3o_seq_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ4",
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
        "label_fontsize": 12,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'DTLZ4 $(d=5, M=3)$ \nOut-of-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5','C6']
    },
    "max_iter": 60,
}

dtlz4_5i_3o_seq_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ4",
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
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 12,
        "plot_ylabel": "DTLZ4 \n $d=5, M=3$",
        "title": 'DTLZ4 $(d=5, M=3)$ \n In-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5','C6']
    },
    "max_iter": 60,
}


dtlz4_5i_3o_batch_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ4",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
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
        "xlabel": "Batch BO Iterations",
        "label_fontsize": 12,
        "plot_ylabel": "log Hypervolume Difference", #
        "title": 'DTLZ4 ($d=5, M=3$) \n Out-of-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    },
    "max_iter": 80,
}

dtlz4_5i_3o_batch_in_sample_log_hv_diff_exp_config = {
    "pb_name": "DTLZ4",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
        "qPF2ES_q2": {},
        "PF2ES_KB_q2": {},
        "qEHVI_q2": {},
        "Random_q2": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "xlabel": "Batch BO Iterations",
        "label_fontsize": 12,
        "plot_ylabel": "DTLZ4 \n $d=5, M=3$",
        "title": 'DTLZ4 ($d=5, M=3$)\n In-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C6']
    },
    "max_iter": 80,
}


penicillin_7i_3o_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "Penicillin",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
        "PF2ES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 12,
        "plot_ylabel": "Penicillin \n $d=7, M=3$",
        "title": 'Out-of-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C6', 'C1', 'C2', 'C3']
    },
    "max_iter": 80,
}



penicillin_7i_3o_in_sample_log_hv_diff_exp_config = {
    "pb_name": "Penicillin",
    "plot_file_suffix": 'LogHvDiff_Out_Of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
        "PF2ES": {},
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 12,
        "plot_ylabel": "Penicillin \n $d=7, M=3$",
        "title": 'Out-of-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 5,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C6', 'C1', 'C2', 'C3']
    },
    "max_iter": 80,
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
        "label_fontsize": 25,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Vehicle Crash Safety \n $d=5, M=3$',
        "title_fontsize": 12,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


vehicle_crash_in_sample_log_hv_diff_exp_config = {
    "pb_name": "VehicleCrashSafety",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "unconstraint_exp",
    "acq": {
            # "PF2ES_epsilon0.04": {},
            "EHVI": {},
            "Random": {},
            "MESMO": {},
            "PFES": {},
            "PF2ES": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 25,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'Vehicle Crash Safety\n $d=5, M=3$',
        "title_fontsize": 12,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
    },
}


marine_design_seq_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "ConceptualMarineDesign",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
            "MESMOC_PLUS": {},
            "Random": {},
            # "PFES": {},
            # "PF2ES_epsilon_dbscan": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Out_Of_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 12,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ConceptualMarineDesign $(d=5, $ \n $M=3, C=9)$ Out-of-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    'max_iter': 60
}


marine_design_seq_in_sample_log_hv_diff_exp_config = {
    "pb_name": "ConceptualMarineDesign",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
    "exp_type": "constraint_exp",
    "acq": {
            "PF2ES": {},
            "EHVI_PoF": {},
            "MESMOC": {},
"MESMOC_PLUS": {},
            "Random": {},
            # "MESMO": {},
            # "PFES": {},
            # "PF2ES_epsilon_dbscan": {}
    },
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "In_Sample_LogHypervolumeDifference_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 12,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ConceptualMarineDesign \n $(d=5, M=3, C=9)$ In-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C7', 'C8', 'C6']
    },
    'max_iter': 60
}



marine_design_batch_out_of_sample_log_hv_diff_exp_config = {
    "pb_name": "ConceptualMarineDesign",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
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
        "label_fontsize": 12,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ConceptualMarineDesign $(d=5, $\n $M=3, C=9)$ Out-of-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C9', 'C6'],
        'xlabel': 'Batch BO Iterations'
    },
    'max_iter': 60
}


marine_design_batch_in_sample_log_hv_diff_exp_config = {
    "pb_name": "ConceptualMarineDesign",
    "plot_file_suffix": 'LogHvDiff_Out_of_Sample',
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
    "file_suffix": "In_Sample_LogHypervolumeDifference_q2_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 12,
        "plot_ylabel": "log Hypervolume Difference",
        "title": 'ConceptualMarineDesign \n $(d=5, M=3, C=9)$  In-sample',
        "title_fontsize": 12,
        "lgd_fontsize": 2,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C2', 'C0', 'C9', 'C6'],
        'xlabel': 'Batch BO Iterations'
        # 'color_list': ['C1', 'C2', 'C0', 'C9', 'C6']
    },
    'max_iter': 60
}
if __name__ == "__main__":

    # out-of-sample logHV
    # multi_panel_bo_history_curve(
    #     cfgs=[
    #             dtlz4_5i_3o_seq_out_of_sample_log_hv_diff_exp_config,
    #             dtlz4_5i_3o_seq_in_sample_log_hv_diff_exp_config,
    #         ],
    #     figsize=(7.5, 2.6),ncol=1, legend_fontsize=14,
    #     save_figure_name='seq_MOO_larger_output', savefig=True)

    # Batch
    # multi_panel_bo_history_curve(
    #     cfgs=[
    #             dtlz4_5i_3o_batch_out_of_sample_log_hv_diff_exp_config,
    #             dtlz4_5i_3o_batch_in_sample_log_hv_diff_exp_config,
    #         ],
    #     figsize=(7.5, 2.6),ncol=1, legend_fontsize=14,
    #     save_figure_name='batch_MOO_larger_output', savefig=True, bbox_to_anchor=(1.01, 0.5))

    # CMOO
    multi_panel_bo_history_curve(
        cfgs=[
                marine_design_seq_out_of_sample_log_hv_diff_exp_config,
                marine_design_seq_in_sample_log_hv_diff_exp_config,
            ],
        figsize=(7.5, 2.6),ncol=1, legend_fontsize=14,
        save_figure_name='seq_CMOO_larger_output', savefig=False, bbox_to_anchor=(1.005, 0.5))

    # Batch CMOO
    # multi_panel_bo_history_curve(
    #     cfgs=[
    #             marine_design_batch_out_of_sample_log_hv_diff_exp_config,
    #             marine_design_batch_in_sample_log_hv_diff_exp_config,
    #         ],
    #     figsize=(7.5, 2.6),ncol=1, legend_fontsize=14,
    #     save_figure_name='batch_CMOO_larger_output', savefig=True, bbox_to_anchor=(1.005, 0.5))
