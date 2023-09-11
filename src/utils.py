import json
from pathlib import Path
import sys
from time import time

import torch
import torch.nn as nn
import wandb
import numpy as np
import pandas as pd


def timeit(fun):
    def fun_(*args, **kwargs):
        start_time = time()
        f_ret = fun(*args, **kwargs)
        end_time = time()

        return end_time - start_time, f_ret

    return fun_

def load_from_wandb(net: nn.Module, run_id: str, project,
                    model_fname='model_last'):
    best_model_file = wandb.restore(
        f'{model_fname}.pth',
        run_path=f"brunompac/{project}/{run_id}",
        replace=True,
    )
    net.load_state_dict(torch.load(best_model_file.name))

    return net

def debugger_is_active() -> bool:
    """Returns True if the debugger is currently active.

    From https://stackoverflow.com/a/67065084
    """
    gettrace = getattr(sys, 'gettrace', lambda : None) 
    return gettrace() is not None

def normalize_curve(curve, T=120, timestep=1e-3):
    standard_range = pd.to_timedelta(np.linspace(0,T,int(T/timestep) +1), 's')

    if np.isnan(curve).any():
        return np.nan
    else:
        c = pd.Series(curve[0], index=pd.to_timedelta(curve[1], 's'))
        c = c.reindex(standard_range, method='ffill')
        return c.values

def compute_integral(curve, T=120, timestep=1e-3):
    if np.isnan(curve).any():
        return np.nan
    else:
        return T - timestep * curve.sum()

def get_first_feasible(curve):
    try:
        return np.where(curve > 0)[0][0] / 1000
    except IndexError:
        return np.nan

def load_all_results(shortname: str, results_dir: Path, opts_dir: Path, T=125,
                     TIME_BUDGET=120, test=False):
    results = list()

    if test:
        results_fpaths = [fp for fp in results_dir.glob(shortname+f'_{T}_2*.json') if int(fp.name.split('_')[-1][:-len('.json')]) >= 20]
    else:
        results_fpaths = [fp for fp in results_dir.glob(shortname+f'_{T}_2*.json') if int(fp.name.split('_')[-1][:-len('.json')]) < 20]

    for result_fpath in results_fpaths:
        size = int(result_fpath.name.split('_')[-2])
        size_id = int(result_fpath.name.split('_')[-1][:-len('.json')])

        with open(result_fpath) as f:
            result = json.load(f)

        solution_fpath = opts_dir/f"{T}_{size}_{size_id}_opt.npz"
        solution_npz = np.load(solution_fpath)
        quasi_optimal_objective = solution_npz['arr_0'].astype('uint32')

        results.append(dict(
            size=size,
            size_id=size_id,
            opt_obj=quasi_optimal_objective,
            **result
        ))

    if len(results) == 0:
        return None
    else:
        df = pd.DataFrame(results)
        df['primal_curve'] = df['primal_curve'].apply(normalize_curve, T=TIME_BUDGET)
        df['primal_curve'] = df['primal_curve'] / df['opt_obj']

        df['time_to_feasible'] = df['primal_curve'].map(get_first_feasible)
        df['time_to_feasible'].fillna(TIME_BUDGET, inplace=True)

        return df

def boxplot_bound(df, goal, column, axs, TIME_BUDGET=120):
    df_ = df[df['goal'] == goal]

    labels = list()
    valuess = list()
    for m, h in df_[['model', 'heuristic']].drop_duplicates().values:
        label = f"{m} + {h}"
        labels.append(label)
        valuess.append(
        df_[(df_['model'] == m) & (df_['heuristic'] == h)][column].values
    )

        curve = df_[(df_['model'] == m) & (df_['heuristic'] == h)]['primal_curve'].mean()
        dt = TIME_BUDGET / (len(curve)-1)
        auc = np.sum(curve * dt)
        x = np.arange(len(curve)) * dt

        if m == 'Baseline':
            axs[1].plot(x, curve, color='black', label=f"Baseline ({auc:.2f})")
        else:
            axs[1].plot(x, curve, label=f"{label} ({auc:.2f})")
    axs[1].set_ylabel('Lower bound')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_xlim(0,TIME_BUDGET)
    axs[1].set_ylim(0,1)
    axs[1].grid()
    axs[1].legend()

    axs[0].boxplot(valuess, labels=labels, vert=False, showmeans=True)

    axs[0].vlines(df_[(df_['model'] == 'Baseline')][column].mean(), linestyles='--', *axs[0].get_ylim())
    axs[0].grid()

    return axs
