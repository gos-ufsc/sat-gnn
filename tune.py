from pathlib import Path
from time import time
import optuna
import pickle
import numpy as np
import torch
import torch.nn

from optuna.trial import Trial

from src.net import InstanceGCN
from src.trainer import EarlyFixingInstanceTrainer
from src.tuning import get_objective
from src.utils import debugger_is_active


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        wandb_project = None
    else:
        wandb_project = 'sat-gnn'
    wandb_group = 'EarlyFixingInstance-tuning'

    # instances_fpaths = list(Path('data/raw/').glob('97_9*.jl'))
    instances_fpaths = [fp for fp in Path('data/raw/').glob('97_9*.jl') if (fp.name != '97_9_2.jl') and (fp != '97_9_10.jl')]
    with open('97_9_opts.pkl', 'rb') as f:
        opts = pickle.load(f)

    study_fpath = Path('study.pkl')

    for _ in range(100):
        if study_fpath.exists():
            with open('study.pkl', 'rb') as f:
                study = pickle.load(f)
        else:
            study = optuna.create_study(study_name='early fixing')

        study.optimize(get_objective(instances_fpaths, opts, wandb_project=wandb_project), n_trials=2)

        with open('study.pkl', 'wb') as f:
            pickle.dump(study, f)
