from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn

from src.net import InstanceGCN, JobGCN
from src.trainer import EarlyFixingTrainer, JobFeasibilityTrainer, OnlyXEarlyFixingInstanceTrainer, VariableResourceTrainer
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

    # # Classification of the graph+candidate solution into feasible or not
    # FactibilityClassificationTrainer(
    #     GCN(2, 1),
    #     batch_size=2**5,
    #     epochs=5,
    #     wandb_project=wandb_project,
    #     wandb_group='GraphClassification-test',
    #     random_seed=seed,
    #     device=device,
    # ).run()

    # # Classification of each dimension of candidate solution into optimal or not
    # EarlyFixingTrainer(
    #     GCN(2, 1, readout_op=None),
    #     batch_size=2**5,
    #     epochs=5,
    #     samples_per_problem=1000,
    #     wandb_project=wandb_project,
    #     wandb_group='NodeClassification-test',
    #     random_seed=seed,
    #     device=device,
    # ).run()

    # # Generation of feasible solutions from different resource vectors
    # VariableResourceTrainer(
    #     InstanceGCN(1, 1, readout_op=None),
    #     batch_size=2**4,
    #     epochs=100000,
    #     # optimizer='SGD',
    #     # optimizer_params={'momentum': 0.99},
    #     # lr=.1,
    #     samples_per_problem=100,
    #     wandb_project=wandb_project,
    #     wandb_group='VariableResourceGenerator-test-no_integrality',
    #     random_seed=seed,
    #     device=device,
    # ).run()

    # Early fixing the solution to all jobs, including coupling constraints
    for _ in range(1):
        net = InstanceGCN(
            2,
            n_h_feats=4,
            single_conv_for_both_passes=False,
            n_passes=1,
            conv1='SAGEConv',
            conv1_kwargs={
                'aggregator_type': 'lstm',
                'feat_drop': 0.1,
            },
            conv2='SAGEConv',
            conv2_kwargs={
                'aggregator_type': 'lstm',
                'feat_drop': 0.1,
            },
            conv3='SAGEConv',
            conv3_kwargs={
                'aggregator_type': 'lstm',
                'feat_drop': 0.1,
            },
            readout_op=None,
        )

        instances_fpaths = list(Path('data/raw/').glob('97_9*.jl'))
        instances_fpaths = sorted(instances_fpaths)[:2]
        with open('97_9_opts.pkl', 'rb') as f:
            opts = pickle.load(f)

        EarlyFixingTrainer(
            InstanceGCN(2, readout_op=None),
            instances_fpaths=instances_fpaths,
            optimals=opts,
            batch_size=2**4,
            samples_per_problem=2**7,
            epochs=100,
            wandb_project=wandb_project,
            wandb_group='EarlyFixingInstance-hypertuning-best',
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
