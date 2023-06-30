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
            readout_op=None,
            # conv1='GraphConv',
            # conv1_kwargs={'norm': 'both'},
            # conv2='GraphConv',
            # conv2_kwargs={'norm': 'both'},
            # conv1='GATv2Conv',
            # # conv1_kwargs={'num_heads': 1, 'allow_zero_in_degree': True,},
            # conv1_kwargs={'num_heads': 1,},
            # conv2='GATv2Conv',
            # # conv2_kwargs={'num_heads': 1, 'allow_zero_in_degree': True,},
            # conv2_kwargs={'num_heads': 1,},
        )

        instances_fpaths = list(Path('data/raw/').glob('97_*.json'))

        train_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_97_all.hdf5').get_split('train')
        val_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_97_all.hdf5').get_split('val')
        test_dataset = MultiTargetDataset(
            instances_fpaths=instances_fpaths,
            sols_dir='/home/bruno/sat-gnn/data/interim',
            split='val',
            return_model=True,
        )
        MultiTargetTrainer(
            net.double(),
            training_dataset=train_dataset,
            validation_dataset=val_dataset,
            test_dataset=test_dataset,
            epochs=100,
            wandb_project=wandb_project,
            wandb_group='Multi-target + SAGE + PreNorm',
            random_seed=seed,
            ef_time_budget=10*60,  # visibility window
            device=device,
        ).run()

        # # train_dataset = OptimalsDataset(instances_fpaths=instances_fpaths,
        # #                                 sols_dir='/home/bruno/sat-gnn/data/interim',
        # #                                 split='train')
        # train_dataset = OptimalsDataset.from_file_lazy('data/processed/optimals_97_all.hdf5').get_split('train')
        # # val_dataset = OptimalsDataset(instances_fpaths=instances_fpaths,
        # #                               sols_dir='/home/bruno/sat-gnn/data/interim',
        # #                               split='val')
        # val_dataset = OptimalsDataset.from_file_lazy('data/processed/optimals_97_all.hdf5').get_split('val')
        # test_dataset = OptimalsDataset(instances_fpaths=instances_fpaths,
        #                                sols_dir='/home/bruno/sat-gnn/data/interim',
        #                                split='val',
        #                                return_model=True)
        # OptimalsTrainer(
        #     net.double(),
        #     training_dataset=train_dataset,
        #     validation_dataset=val_dataset,
        #     test_dataset=test_dataset,
        #     epochs=100,
        #     wandb_project=wandb_project,
        #     wandb_group='Optimals + GATv2 + PreNorm',
        #     random_seed=seed,
        #     device=device,
        # ).run()

        # train_dataset = VarOptimalityDataset.from_file_lazy('data/processed/varoptimality_97_all.hdf5').get_split('train')
        # train_dataset.samples_per_instance = 20
        # val_dataset = VarOptimalityDataset.from_file_lazy('data/processed/varoptimality_97_all.hdf5').get_split('val')
        # val_dataset.samples_per_instance = 10
        # test_dataset = VarOptimalityDataset(
        #     instances_fpaths=instances_fpaths,
        #     sols_dir='/home/bruno/sat-gnn/data/interim',
        #     split='val',
        #     return_model=True,
        # )
        # test_dataset.samples_per_instance = 1
        # net = VarInstanceGCN(
        #     readout_op=None,
        # )
        # VarOptimalityTrainer(
        #     net.double(),
        #     training_dataset=train_dataset,
        #     validation_dataset=val_dataset,
        #     test_dataset=test_dataset,
        #     epochs=15,
        #     wandb_project=wandb_project,
        #     wandb_group='Var Optimality + SAGE',
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
