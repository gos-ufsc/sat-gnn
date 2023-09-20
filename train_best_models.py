from pathlib import Path

import numpy as np
import torch
import torch.nn

from src.dataset import MultiTargetDataset, OptimalsDataset
from src.net import OptSatGNN
from src.trainer import MultiTargetTrainer, OptimalsTrainer
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

    for _ in range(1):
        ### SatGNN Trained with Multiple Solutions

        if debugger_is_active():
            train_dataset = MultiTargetDataset(
                [fp for fp in Path('data/raw/').glob('125_*.json')
                 if (int(fp.name.split('_')[1]) < 20) and  # small instances
                    (int(fp.name.split('_')[2].replace('.json', '')) < 10)]  # small sample for debugging
            )
            val_dataset = train_dataset
        else:
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

        ### SatGNN Trained with Best Solution

        if debugger_is_active():
            train_dataset = OptimalsDataset(
                [fp for fp in Path('data/raw/').glob('125_*.json')
                 if (int(fp.name.split('_')[1]) < 20) and  # small instances
                    (int(fp.name.split('_')[2].replace('.json', '')) < 10)]  # small sample for debugging
            )
            val_dataset = train_dataset
        else:
            train_dataset = OptimalsDataset.from_file_lazy('data/processed/optimals_125_train.hdf5')
            val_dataset = OptimalsDataset.from_file_lazy('data/processed/optimals_125_val.hdf5')


        net = OptSatGNN(
            n_h_feats=64,
            single_conv_for_both_passes=True,
            n_passes=2,
            conv1='SAGEConv',
            conv1_kwargs={ 'aggregator_type': 'pool' },
            conv2=None,
            conv2_kwargs=dict(),
        )

        OptimalsTrainer(
            net.double(),
            training_dataset=train_dataset,
            validation_dataset=val_dataset,
            lr=1e-2,
            epochs=150,
            get_best_model=True,
            wandb_project=wandb_project,
            wandb_group='Best-Optimals',
            random_seed=seed,
            device=device,
        ).run()
