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
from src.utils import debugger_is_active


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        wandb_project = None
    else:
        wandb_project = None
        # wandb_project = 'sat-gnn'

    instances_fpaths = list(Path('data/raw/').glob('97_9*.jl'))
    with open('97_9_opts.pkl', 'rb') as f:
        opts = pickle.load(f)

    def objective(trial: Trial):
        batch_size = trial.suggest_int('batch_size', 2**2, 2**6, log=True)

        trainer = EarlyFixingInstanceTrainer(
            InstanceGCN(2, readout_op=None),
            instances_fpaths=instances_fpaths,
            optimals=opts,
            batch_size=batch_size,
            epochs=5,
            wandb_project=wandb_project,
            wandb_group='EarlyFixingInstance-optuna-test',
            device=device,
        )
        trainer.run()

        return trainer.val_scores[-1]

    with open('test3_study.pkl', 'rb') as f:
        study = pickle.load(f)

    study.optimize(objective, n_trials=5)

    with open('test3_study.pkl', 'wb') as f:
        pickle.dump(study, f)
