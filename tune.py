from pathlib import Path
from time import time
import optuna
import pickle
import numpy as np
import torch
import torch.nn

from optuna.trial import Trial

from src.net import SatGNN
from src.trainer import EarlyFixingTrainer
from src.tuning import get_objective
from src.utils import debugger_is_active


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        wandb_project = None
    else:
        wandb_project = 'sat-gnn'
    wandb_group = 'EarlyFixingInstance-validation'

    # instances_fpaths = list(Path('data/raw/').glob('97_9*.jl'))
    instances_fpaths = [fp for fp in Path('data/raw/').glob('97_9*.jl') if fp.name not in ['97_9_9.jl', '97_9_6.jl']]
    print(instances_fpaths)
    with open('97_9_opts.pkl', 'rb') as f:
        opts = pickle.load(f)

    study_fpath = Path('study_val.pkl')

    if study_fpath.exists():
        with open(study_fpath, 'rb') as f:
            study = pickle.load(f)
    else:
        study = optuna.create_study(study_name='early fixing')

    for _ in range(100):
        study.optimize(get_objective(instances_fpaths, opts, wandb_project=wandb_project, wandb_group=wandb_group), n_trials=1)

        with open(study_fpath, 'wb') as f:
            pickle.dump(study, f)
