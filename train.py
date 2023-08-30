from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn
from src.dataset import MultiTargetDataset, OptimalsDataset, OptimalsWithZetaDataset, VarOptimalityDataset, SolutionFeasibilityDataset
from time import sleep

from src.net import OptSatGNN
from src.trainer import MultiTargetTrainer, OptimalsTrainer, PhiMultiTargetTrainer, VarOptimalityTrainer, FeasibilityClassificationTrainer
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
        # net = InstanceGCN(
        #     readout_op=None,
        # )

        if debugger_is_active():
            train_dataset = MultiTargetDataset(
                [fp for fp in Path('data/raw/').glob('125_*.json')
                 if (int(fp.name.split('_')[1]) < 20) and
                    (int(fp.name.split('_')[2].replace('.json', '')) < 10)]
            )
            val_dataset = train_dataset
        else:
            # train_dataset = MultiTargetDataset(
            #     [fp for fp in Path('data/raw/').glob('125_*.json')
            #      if (int(fp.name.split('_')[1]) < 20) and
            #         (int(fp.name.split('_')[2].replace('.json', '')) < 130)]
            # )
            # val_dataset = MultiTargetDataset(
            #     [fp for fp in Path('data/raw/').glob('125_*.json')
            #      if (int(fp.name.split('_')[1]) < 20) and
            #         (int(fp.name.split('_')[2].replace('.json', '')) >= 130) and
            #         (int(fp.name.split('_')[2].replace('.json', '')) < 160)]
            # )
            train_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_125_train.hdf5')
            val_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_125_val.hdf5')

        net = OptSatGNN(
            n_h_feats=256,
            single_conv_for_both_passes=True,
            n_passes=3,
            conv1='SAGEConv',
            conv1_kwargs={ 'aggregator_type': 'pool' },
            conv2=None,
            conv2_kwargs=dict(),
        )

        # train_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_125_97_small_all_200.hdf5').get_split('all')
        # val_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_125_large_all.hdf5').get_split('val')
        MultiTargetTrainer(
            net.double(),
            training_dataset=train_dataset,
            validation_dataset=val_dataset,
            epochs=150,
            get_best_model=True,
            wandb_project=wandb_project,
            wandb_group='Best-MultiTarget',
            random_seed=seed,
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
        # OptimalsTrainer(
        #     net.double(),
        #     training_dataset=train_dataset,
        #     validation_dataset=val_dataset,
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
        # net = VarInstanceGCN(
        #     readout_op=None,
        # )
        # VarOptimalityTrainer(
        #     net.double(),
        #     training_dataset=train_dataset,
        #     validation_dataset=val_dataset,
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
