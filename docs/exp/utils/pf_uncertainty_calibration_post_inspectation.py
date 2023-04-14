import os
import numpy as np
from matplotlib import pyplot as plt
from trieste.acquisition.multi_objective.pareto import Pareto
import tensorflow as tf
from os.path import dirname, abspath
from typing import List, Optional
from numpy import ndarray
import pickle

'''
Let's plot the median of 50%, 10-90 percentile 
'''

# TODO: Fix when no feasible PF
def _get_performance_result_from_file(path_prefix: str, exp_repeat: int, file_prefix: str, file_suffix: str, ref_point):
    pf_list = []
    for i in range(exp_repeat):
        res_path = os.path.join(path_prefix, "".join([file_prefix, f"_{i}_", file_suffix, ".npy"]))
        try:
            if path_prefix.split('\\')[-2] == 'PESMO' or path_prefix.split('\\')[-2] == 'MESMOC_PLUS':
                res_path = os.path.join(path_prefix, "".join(['UC_pfs', f"_{i+1}"]))
                with open(res_path, 'rb') as handle:
                    pf_list.append(pickle.load(handle, encoding='latin1'))
            else:
                pf_list.append(np.load(res_path, allow_pickle=True))
        except:
            print(f"Cannot load {res_path}, skip")
    pf_samples = pf_list  # [exp_repeat, bo_iter, sample_pf_size, pf_pop_size, obj_num]
    if len(pf_samples) == 0:
        return tf.zeros(shape=(0, 0, 0))
    if path_prefix.split('\\')[-2] == 'PESMO':
        hv_array = np.zeros(shape=(len(pf_samples), len(pf_samples[0]), pf_samples[0][0].shape[0]))
        for pf_sample_in_exp, exp_id in zip(pf_samples,
                                            range(len(pf_samples))):  # [bo_iter, sample_pf_size, pf_pop_size, obj_num]
            print('Here')
            for pf_sample_in_exp_per_iter, bo_iter in zip(pf_sample_in_exp, range(len(pf_sample_in_exp))):  # [sample_pf_size, pf_pop_size, obj_num]
                for pf_sample_in_exp_per_iter_per_sample, pf_sample_id in \
                        zip(pf_sample_in_exp_per_iter, range(len(pf_sample_in_exp[0]))):
                    # hv_array_per_pf_sample.append(Pareto(pf_sample_in_exp_per_iter_per_sample).hypervolume_indicator(ref_point))
                    if pf_sample_in_exp_per_iter_per_sample is None:  # no feasible samples:
                        hv_array[exp_id, bo_iter, pf_sample_id] = 0
                    else:
                        hv_array[exp_id, bo_iter, pf_sample_id] = Pareto(
                            pf_sample_in_exp_per_iter_per_sample).hypervolume_indicator(ref_point)
    elif path_prefix.split('\\')[-2] == 'MESMOC_PLUS':
        hv_array = np.zeros(shape=(len(pf_samples), len(pf_samples[0]), pf_samples[0][-1].shape[0]))
        for pf_sample_in_exp, exp_id in zip(pf_samples,
                                            range(len(pf_samples))):  # [bo_iter, sample_pf_size, pf_pop_size, obj_num]
            print('Here')
            for pf_sample_in_exp_per_iter, bo_iter in zip(pf_sample_in_exp, range(len(pf_samples[0]))):  # [sample_pf_size, pf_pop_size, obj_num]
                if pf_sample_in_exp_per_iter.shape == ():
                    pass
                else:
                    for pf_sample_in_exp_per_iter_per_sample, pf_sample_id in \
                            zip(pf_sample_in_exp_per_iter, range(len(pf_sample_in_exp[-1]))):
                        # hv_array_per_pf_sample.append(Pareto(pf_sample_in_exp_per_iter_per_sample).hypervolume_indicator(ref_point))
                        if pf_sample_in_exp_per_iter_per_sample is None:  # no feasible samples:
                            hv_array[exp_id, bo_iter, pf_sample_id] = 0
                        else:
                            hv_array[exp_id, bo_iter, pf_sample_id] = Pareto(
                                pf_sample_in_exp_per_iter_per_sample).hypervolume_indicator(ref_point)
    else:
        hv_array = np.zeros(shape=(len(pf_samples), pf_samples[0].shape[0], pf_samples[0].shape[1]))
        for pf_sample_in_exp, exp_id in zip(pf_samples,
                                            range(len(pf_samples))):  # [bo_iter, sample_pf_size, pf_pop_size, obj_num]
            print('Here')
            for pf_sample_in_exp_per_iter, bo_iter in zip(pf_sample_in_exp, range(
                    pf_sample_in_exp.shape[0])):  # [sample_pf_size, pf_pop_size, obj_num]
                for pf_sample_in_exp_per_iter_per_sample, pf_sample_id in zip(pf_sample_in_exp_per_iter,
                                                                              range(pf_sample_in_exp.shape[1])):
                    # hv_array_per_pf_sample.append(Pareto(pf_sample_in_exp_per_iter_per_sample).hypervolume_indicator(ref_point))
                    if pf_sample_in_exp_per_iter_per_sample is None:  # no feasible samples:
                        hv_array[exp_id, bo_iter, pf_sample_id] = 0
                    else:
                        hv_array[exp_id, bo_iter, pf_sample_id] = Pareto(
                            pf_sample_in_exp_per_iter_per_sample).hypervolume_indicator(ref_point)

    return hv_array


def plot_uncertainty_calibration_curve_for_cfg(cfg: dict, plot_data: Optional[ndarray] = None,
                                               ax: Optional = None, plot_ylabel: bool = True, log10 = True):
    """
    Plot

    :param plot_data: If specified plot_data it will use plot data to plot instead of reading data from scratch
    :param ax plot axis, if specified, it will just use ax to plot instead of initializing the whole plot
    """
    path_prefix = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        cfg['exp_type'],
        "exp_res",
        cfg["pb_name"],
    )
    legend_handles = []
    legend_label = []

    color_idx = 0
    color_list = cfg['plot_cfg'].get('color_list', ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])

    if ax is None:
        multi_panel_plot = False
        _, ax = plt.subplots(figsize=cfg["plot_cfg"]["fig_size"])
    else:
        multi_panel_plot = True

    ground_truth_pf = \
        np.loadtxt(
            os.path.join(dirname(dirname(abspath(__file__))), cfg['exp_type'], 'cfg', 'ref_opts', cfg['PF_file_name']))
    ground_truth_hv = Pareto(tf.convert_to_tensor(ground_truth_pf, dtype=tf.float64)).hypervolume_indicator(
        cfg['ref_point'])

    for acq_exp_label, _ in cfg["acq"].items():
        if acq_exp_label == "PF2ES":
            acq_exp_label_for_plot = '{PF}$^2$ES'
        elif acq_exp_label == "MESMOC_PLUS":
            acq_exp_label_for_plot = 'MESMOC+'
        elif acq_exp_label == 'EHVI_PoF':
            acq_exp_label_for_plot ='EHVI-PoF'
        else:
            acq_exp_label_for_plot = acq_exp_label
        # print(acq_exp_label)
        x_space = 1 if 'x_space' not in cfg else cfg['x_space']
        _path_prefix = os.path.join(path_prefix, acq_exp_label, cfg["res_path_prefix"])
        if plot_data is not None:
            exp_regrets = plot_data[acq_exp_label]
        else:
            exp_regrets = _get_performance_result_from_file(
                _path_prefix,
                exp_repeat=cfg["exp_repeat"],
                file_prefix=cfg["pb_name"],
                file_suffix=cfg["file_suffix"],
                ref_point=cfg['ref_point']
            )  # [exp_repeat, bo_iter, pf_samples]
        if "max_iter" in cfg.keys():
            exp_regrets = exp_regrets[:, : cfg["max_iter"] + 1]
        # FIXME: divided by 0
        if log10:
            exp_regrets = np.log10(exp_regrets)
        if tf.size(exp_regrets)!= 0:
            if acq_exp_label == 'PF2ES':
                print('PF2ES!')
                # a1 = ax.plot(np.arange(exp_regrets.shape[-2]), np.mean(np.percentile(exp_regrets, 50, axis=-1), axis=0),
                #              color=color_list[color_idx], zorder=500)
                a1 = ax.plot(np.arange(exp_regrets.shape[-2]), np.median(np.mean(exp_regrets, axis=-1), axis=0),
                             color=color_list[color_idx], zorder=500)
            else:
                # a1 = ax.plot(np.arange(exp_regrets.shape[-2]), np.mean(np.percentile(exp_regrets, 50, axis=-1), axis=0),
                #               color=color_list[color_idx], zorder=5)
                a1 = ax.plot(np.arange(exp_regrets.shape[-2]), np.median(np.mean(exp_regrets, axis=-1), axis=0),
                              color=color_list[color_idx], zorder=5)
            ax.fill_between(
                np.arange(exp_regrets.shape[-2]),
                np.median(np.percentile(exp_regrets, 10, axis=-1), axis=0),
                np.median(np.percentile(exp_regrets, 90, axis=-1), axis=0),
                # np.mean(np.percentile(exp_regrets, 10, axis=-1), axis=0),
                # np.mean(np.percentile(exp_regrets, 90, axis=-1), axis=0),
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
        "BO Iterations", fontsize=cfg["plot_cfg"].get("label_fontsize", 10)
    )
    if cfg["pb_name"] == 'Constr_Ex':
        ax.set_yticks([1.88, 1.91, 1.94, 1.97])
    if cfg["pb_name"] == 'C2DTLZ2':
        ax.set_yticks([3.35, 3.40, 3.45])
    if plot_ylabel:
        ax.set_ylabel(
            cfg["plot_cfg"]["plot_ylabel"],
            fontsize=cfg["plot_cfg"].get("label_fontsize", 10),
        )
    if not log10:
        a = ax.hlines(ground_truth_hv, 0, exp_regrets.shape[-2], linestyles='--', color='r', label='Reference\nHypervolume')
    else:
        a = ax.hlines(np.log10(ground_truth_hv), 0, exp_regrets.shape[-2], linestyles='--', color='r', label='Reference\nHypervolume')
    legend_handles.append(a)
    if cfg["plot_cfg"]["log_y"] == True:
        ax.set_yscale("log")
    # plt.yticks([1e-1, 1e-2, 1e-3])
    ax.tick_params(labelsize=cfg["plot_cfg"].get("tick_fontsize", 5))
    # ax.yticks(fontsize=cfg["plot_cfg"].get("tick_fontsize", 5))
    # we use this approach to move fig upward to allow lagend downward
    #  https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    box = ax.get_position()
    if multi_panel_plot is True:
        ax.set_position([box.x0 - box.width * 0.28, box.y0 + box.height * 0.12,
                                box.width, box.height * 0.8])
        # ax.set_position([box.x0, box.y0 + box.height * 0.3,
        #                         box.width, box.height * 0.65])

    # Put a legend below current axis
    if multi_panel_plot is False:
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
    else:
        return legend_handles
    # plt.savefig(os.path.join(path_prefix, "".join([cfg["pb_name"], cfg["plot_file_suffix"], ".png"])), dpi=300)


def multi_panel_uncertainty_calibration_curve(cfgs: List[dict], figsize: tuple, legend_fontsize: int = 20,
                                              ncol: int = None, savefig: bool=False, savefig_name='exp_res.png'):
    """
    Create Multi-Panel Plot for Uncertainty Calibration
    """
    assert len(cfgs) >= 2
    fig, axis = plt.subplots(figsize=figsize, nrows=1, ncols=len(cfgs))
    for ax, cfg, id in zip(axis, cfgs, range(len(cfgs))):
        if id == 0:
            handles = plot_uncertainty_calibration_curve_for_cfg(cfg, ax=ax, plot_ylabel=True)
        else:
            plot_uncertainty_calibration_curve_for_cfg(cfg, ax=ax, plot_ylabel=False)
    _, labels = axis[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5), fancybox=False, ncol=ncol, fontsize=legend_fontsize)
    # axis[0].legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, -0.5), fancybox=False, ncol=ncol, fontsize=legend_fontsize)
    # plt.tight_layout()
    # plt.show(block=True)
    if savefig is True:
        plt.savefig(savefig_name, dpi=500)
    else:
        plt.show()
    # plt.savefig('Uncertainty_Calibration_MOO.png', dpi=500)
    # plt.savefig('Uncertainty_Calibration_CMOO.png', dpi=500)


vlmop2_exp_config = {
    "pb_name": "VLMOP2",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "PF_file_name": 'VLMOP2_PF_F.txt',
    "exp_type": "unconstraint_exp",
    "acq": {"PF2ES": {},
            "EHVI": {},
            'MESMO': {},
            'PFES': {}, 
            'PESMO': {},

            "Random": {},},
    "ref_point": tf.constant([3.0, 3.0], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'VLMOP2\n $d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}

branincurrin_exp_config = {
    "pb_name": "BraninCurrin",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "exp_type": "unconstraint_exp",
    "PF_file_name": 'BraninCurrin_PF_F.txt',
    "acq": {"PF2ES": {},
            "EHVI": {},
            'MESMO': {},
            'PFES': {}, 
            'PESMO': {},

            'Random': {},},
    "ref_point": tf.constant([2000, 50], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'BraninCurrin\n $d=2, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}


zdt1_5i_2o_exp_config = {
    "pb_name": "ZDT1",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "exp_type": "unconstraint_exp",
    "PF_file_name": 'ZDT1_5I_2O_PF_F.txt',
    "acq": {"PF2ES": {},
            "EHVI": {},
            'MESMO': {},
            'PFES': {}, 
            'PESMO': {},

            'Random': {},},
    "ref_point": tf.constant([15, 15], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'ZDT1 \n$d=5, M=2, q=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}


zdt2_5i_2o_exp_config = {
    "pb_name": "ZDT2",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "exp_type": "unconstraint_exp",
    "PF_file_name": 'ZDT2_5I_2O_PF_F.txt',
    "acq": {"PF2ES": {},
            "EHVI": {},
            'MESMO': {},
            'PFES': {}, 
            'PESMO': {},

            'Random': {},},
    "ref_point": tf.constant([15, 15], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'ZDT2\n $d=5, M=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0', 'C3', 'C4', 'C5', 'C6']
    },
    'max_iter': 50
}


CBraninCurrin_exp_config = {
    "pb_name": "CBraninCurrin",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "PF_file_name": 'C_BraninCurrin_PF_F.txt',
    "exp_type": "constraint_exp",
    "acq": {"PF2ES": {},
            "EHVI_PoF": {},
            'MESMOC': {},
            'MESMOC_PLUS': {},

            "Random": {},},
    "ref_point": tf.constant([2000, 100.0], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'C-BraninCurrin\n $d=2, M=2, C=1$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0',  'C7', 'C8', 'C6']
    },
    'max_iter': 50
}


Constr_Ex_exp_config = {
    "pb_name": "Constr_Ex",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "PF_file_name": 'Constr_Ex_PF_F.txt',
    "exp_type": "constraint_exp",
    "acq": {"PF2ES": {},
            "EHVI_PoF": {},
            'MESMOC': {},
            'MESMOC_PLUS': {},

            "Random": {},
            },
    "ref_point": tf.constant([2, 50.0], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'Constr-Ex\n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0',  'C7', 'C8', 'C6']
    },
    'max_iter': 60
}


TNK_exp_config = {
    "pb_name": "TNK",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "PF_file_name": 'TNK_PF_F.txt',
    "exp_type": "constraint_exp",
    "acq": {"PF2ES": {},
            "EHVI_PoF": {},
            'MESMOC': {},
            'MESMOC_PLUS': {},

            "Random": {},},
    "ref_point": tf.constant([5, 5], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'TNK\n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0',  'C7', 'C8', 'C6']
    },
    'max_iter': 60
}


SRN_exp_config = {
    "pb_name": "SRN",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "PF_file_name": 'SRN_PF_F.txt',
    "exp_type": "constraint_exp",
    "acq": {"PF2ES": {},
            "EHVI_PoF": {},
            'MESMOC': {},
            'MESMOC_PLUS': {},
            "Random": {}},
    "ref_point": tf.constant([1500, 500], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'SRN\n $d=2, M=2, C=2$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0',  'C7', 'C8', 'C6']
    },
    'max_iter': 60
}


C2DTLZ2_exp_config = {
    "pb_name": "C2DTLZ2",
    "plot_file_suffix": 'Model_Believe__model_inferred_pfs_q1_',
    "PF_file_name": 'C2DTLZ2_PF_F.txt',
    "exp_type": "constraint_exp",
    "acq": {"PF2ES": {},
            "EHVI_PoF": {},
            'MESMOC': {},
            'MESMOC_PLUS': {},
            "Random": {},},
    "ref_point": tf.constant([10, 250], dtype=tf.float64),
    "res_path_prefix": "",
    "exp_repeat": 30,
    "file_suffix": "Model_Believe__model_inferred_pfs_q1_",
    "plot_cfg": {
        "log_y": False,
        "label_fontsize": 15,
        "plot_ylabel": "lg Hypervolume Indicator",
        "title": 'C2-DTLZ2\n $d=4, M=2, C=1$',
        "title_fontsize": 14,
        "lgd_fontsize": 15,
        "tick_fontsize": 13,
        "fig_size": (6, 4),
        'color_list': ['C1', 'C0',  'C7', 'C8', 'C6']
    },
    'max_iter': 40
}
if __name__ == '__main__':
    # plot_uncertainty_calibration_curve_for_cfg(cfg=zdt1_5i_2o_exp_config)
    # MOO Problem
    multi_panel_uncertainty_calibration_curve(
        cfgs=[vlmop2_exp_config, branincurrin_exp_config, zdt1_5i_2o_exp_config ,zdt2_5i_2o_exp_config],
        figsize=(15, 3), ncol=1, legend_fontsize=14, savefig=False, savefig_name='Uncertainty_Calibration_MOO.png')
    # CMOO Problem
    multi_panel_uncertainty_calibration_curve(
        # cfgs=[CBraninCurrin_exp_config, Constr_Ex_exp_config, SRN_exp_config, C2DTLZ2_exp_config],
        cfgs=[CBraninCurrin_exp_config, Constr_Ex_exp_config, SRN_exp_config, C2DTLZ2_exp_config],
        figsize=(15, 3), ncol=1, legend_fontsize=14, savefig=True, savefig_name='Uncertainty_Calibration_CMOO')
