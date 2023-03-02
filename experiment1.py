from pathlib import Path

import numpy as np
import torch

from src.net import JobGCN
from src.trainer import JobFeasibilityTrainer
from src.utils import debugger_is_active

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        wandb_project = None  # avoid logging run
        torch.autograd.set_detect_anomaly(True)
    else:
        wandb_project = 'sat-gnn'

    # Classification of the graph+candidate solution into feasible or not
    JobFeasibilityTrainer(
        JobGCN(2, 1),
        instance_fpath=Path('data/raw/97_9.jl'),
        batch_size=2**5,
        epochs=100,
        wandb_project=wandb_project,
        wandb_group='Experiment1 - Job',
        # lr_scheduler='MultiplicativeLR',
        # lr_scheduler_params={'lr_lambda': lambda e: 1 - np.exp(10*(e/E - 1))},
        random_seed=None,
        device=device,
    ).run()
