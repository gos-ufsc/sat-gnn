import numpy as np
import torch
import torch.nn

from src.net import GCN
from src.trainer import FactibilityClassificationTrainer, EarlyFixingTrainer
from src.utils import debugger_is_active


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        import random
        seed = 33
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)

        wandb_project = None  # avoid logging run

        torch.autograd.set_detect_anomaly(True)
    else:
        seed = None
        wandb_project = 'sat-gnn'

    # Classification of the graph+candidate solution into feasible or not
    FactibilityClassificationTrainer(
        GCN(2, 1),
        batch_size=2**5,
        epochs=1000,
        wandb_project=wandb_project,
        random_seed=seed,
        device=device,
    ).run()

    # Classification of each dimension of candidate solution into optimal or not
    EarlyFixingTrainer(
        GCN(2, 1, readout_op=None),
        batch_size=2**8,
        epochs=100,
        samples_per_problem=1000,
        wandb_project=wandb_project,
        random_seed=seed,
        device=device,
    ).run()
