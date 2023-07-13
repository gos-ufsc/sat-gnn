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
