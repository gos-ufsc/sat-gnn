from pathlib import Path

import numpy as np
import torch

from src.net import InstanceGCN, JobGCN
from src.trainer import JobFeasibilityTrainer, SatelliteFeasibilityTrainer
from src.utils import debugger_is_active

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        wandb_project = None  # avoid logging run
        torch.autograd.set_detect_anomaly(True)
    else:
        wandb_project = 'sat-gnn'

    for _ in range(5):
        JobFeasibilityTrainer(
            InstanceGCN(2),
            instance_fpath=Path('data/raw/97_9.jl'),
            batch_size=2**5,
            epochs=70,
            wandb_project=wandb_project,
            wandb_group='Experiment1 - Job',
            random_seed=None,
            device=device,
        ).run()

    for _ in range(5):
        SatelliteFeasibilityTrainer(
            InstanceGCN(2),
            instances_fpaths=Path('data/raw/').glob('97_9*.jl'),
            batch_size=2**5,
            epochs=70,
            wandb_project=wandb_project,
            wandb_group='Experiment1 - Instance',
            random_seed=None,
            device=device,
        ).run()
