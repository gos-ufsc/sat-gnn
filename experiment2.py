from pathlib import Path

import pickle
import torch

from src.net import InstanceGCN
from src.trainer import EarlyFixingTrainer
from src.utils import debugger_is_active


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if debugger_is_active():
        wandb_project = None  # avoid logging run
        torch.autograd.set_detect_anomaly(True)
    else:
        wandb_project = 'sat-gnn'

    instances_fpaths = list(Path('data/raw/').glob('97_9*.jl'))[:-2]

    with open('97_9_opts.pkl', 'rb') as f:
        opts = pickle.load(f)

    net = InstanceGCN(
        2,
        n_passes=1,
        single_conv_for_both_passes=False,
        n_h_feats=19,
        conv1='GraphConv',
        conv2='SAGEConv',
        conv2_kwargs={
            'aggregator_type': 'pool',
            'feat_drop': .09088,
        },
        conv3='GraphConv',
        readout_op=None,
    )

    EarlyFixingTrainer(
        net,
        instances_fpaths=instances_fpaths,
        optimals=opts,
        batch_size=2**2,
        samples_per_problem=2**9,
        epochs=100,
        wandb_project=wandb_project,
        wandb_group='Experiment2',
        random_seed=None,
        device=device,
    ).run()
