from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn

from src.net import InstanceGCN
from src.trainer import EarlyFixingTrainer, PhiEarlyFixingTrainer
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

    # Early fixing the solution to all jobs, including coupling constraints
    for _ in range(1):
        net = InstanceGCN(
            conv1='SAGEConv',
            conv1_kwargs={'aggregator_type': 'pool'},
            # conv2=None,
            conv2='SAGEConv',
            conv2_kwargs={'aggregator_type': 'pool'},
            # conv3='SAGEConv',
            # conv3_kwargs={'aggregator_type': 'pool'},
            n_h_feats=64,
            n_passes=1,
            readout_op=None,
        )

        instances_fpaths = list(Path('data/raw/').glob('97_9*.json'))

        PhiEarlyFixingTrainer(
            net.double(),
            instances_fpaths=instances_fpaths,
            sols_dir='/home/bruno/sat-gnn/data/interim',
            epochs=100,
            wandb_project=wandb_project,
            wandb_group='NEW_TEST',
            random_seed=seed,
            device=device,
        ).run()

    # # Early fixing the solution to all jobs, including coupling constraints
    # instances_fpaths = list(Path('data/raw/').glob('97_9*.jl'))
    # # instances_fpaths = sorted(instances_fpaths)
    # with open('97_9_opts.pkl', 'rb') as f:
    #     opts = pickle.load(f)
    # OnlyXEarlyFixingInstanceTrainer(
    #     InstanceGCN(2, readout_op=None),
    #     instances_fpaths=instances_fpaths,
    #     optimals=opts,
    #     batch_size=2**4,
    #     epochs=100,
    #     wandb_project=wandb_project,
    #     wandb_group='OnlyX-EarlyFixingInstance-test',
    #     random_seed=seed,
    #     device=device,
    # ).run()
