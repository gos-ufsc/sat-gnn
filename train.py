from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn
from src.dataset import MultiTargetDataset, OptimalsDataset, OptimalsWithZetaDataset, VarOptimalityDataset

from src.net import AttentionInstanceGCN, InstanceGCN, VarInstanceGCN
from src.trainer import MultiTargetTrainer, OptimalsTrainer, PhiMultiTargetTrainer, VarOptimalityTrainer
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
            n_h_feats=64,
            readout_op=None,
            conv1='GATv2Conv',
            # conv1_kwargs={'num_heads': 1, 'allow_zero_in_degree': True,},
            conv1_kwargs={'num_heads': 1,},
            conv2='GATv2Conv',
            # conv2_kwargs={'num_heads': 1, 'allow_zero_in_degree': True,},
            conv2_kwargs={'num_heads': 1,},
        )

        instances_fpaths = list(Path('data/raw/').glob('97_*.json'))

        # MultiTargetTrainer(
        #     net.double(),
        #     MultiTargetDataset(instances_fpaths=instances_fpaths,
        #                        sols_dir='/home/bruno/sat-gnn/data/interim',
        #                        split='all'),
        #     epochs=100,
        #     wandb_project=wandb_project,
        #     wandb_group='Multi-target',
        #     random_seed=seed,
        #     device=device,
        # ).run()

        OptimalsTrainer(
            net.double(),
            OptimalsDataset.from_file_lazy('data/processed/optimals_97_all.hdf5'),
            # OptimalsDataset(instances_fpaths=instances_fpaths,
            #                 sols_dir='/home/bruno/sat-gnn/data/interim'),
            epochs=50,
            wandb_project=wandb_project,
            wandb_group='Optimals + Attention',
            random_seed=seed,
            device=device,
        ).run()

        # net = VarInstanceGCN(
        #     n_h_feats=64,
        #     readout_op=None,
        # )
        # VarOptimalityTrainer(
        #     net.double(),
        #     VarOptimalityDataset(instances_fpaths=instances_fpaths,
        #                          sols_dir='/home/bruno/sat-gnn/data/interim'),
        #     epochs=300,
        #     wandb_project=wandb_project,
        #     wandb_group='Var Optimality',
        #     random_seed=seed,
        #     device=device,
        # ).run()

        # OptimalsTrainer(
        #     net.double(),
        #     OptimalsWithZetaDataset.from_file_lazy('data/processed/optimalszeta_97_all.hdf5'),
        #     # OptimalsWithZetaDataset(instances_fpaths=instances_fpaths,
        #     #                         sols_dir='/home/bruno/sat-gnn/data/interim'),
        #     epochs=50,
        #     wandb_project=wandb_project,
        #     wandb_group='Optimals + Zeta',
        #     random_seed=seed,
        #     device=device,
        # ).run()
