from pathlib import Path

import numpy as np
import torch

from src.dataset import SolutionFeasibilityDataset
from src.net import FeasSatGNN
from src.trainer import FeasibilityClassificationTrainer, InstanceFeasibilityClassificationTrainer


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_project = 'sat-gnn'

    # ##### Experiment 1

    # wandb_group = 'FeasClass-SingleInstance'
    # for size in [9, 13, 18, 20, 22, 24]:
    #     instances_fpaths = list(Path('data/raw/').glob(f'125_{size}_*.json'))

    #     # randomly pick 5 instances from each size
    #     instances_fpaths = np.random.choice(instances_fpaths, 5, replace=False)

    #     for instance_fpath in instances_fpaths:
    #         dataset = SolutionFeasibilityDataset([instance_fpath,])

    #         InstanceFeasibilityClassificationTrainer(
    #             FeasSatGNN(),
    #             dataset,
    #             epochs=200,
    #             min_train_loss=1e-2,
    #             wandb_project=wandb_project,
    #             wandb_group=wandb_group+f'-{size}',
    #             device=device,
    #         ).run()

    # ##### Experiment 2

    # wandb_group = 'FeasClass-AcrossInstances'
    # for size in [9, 13, 18, 20, 22, 24]:
    #     train_dataset = SolutionFeasibilityDataset(
    #         [fp for fp in Path('data/raw/').glob(f'125_{size}_*.json')
    #          if int(fp.name.split('_')[2].replace('.json', '')) in np.arange(0,20)],
    #     )
    #     val_dataset = SolutionFeasibilityDataset(
    #         [fp for fp in Path('data/raw/').glob(f'125_{size}_*.json')
    #          if int(fp.name.split('_')[2].replace('.json', '')) in np.arange(20,30)],
    #     )

    #     FeasibilityClassificationTrainer(
    #         FeasSatGNN(),
    #         train_dataset,
    #         val_dataset,
    #         epochs=25,
    #         batch_size=2**6,
    #         wandb_project=wandb_project,
    #         wandb_group=wandb_group+f'-{size}',
    #         device=device,
    #     ).run()

    ##### Experiment 3

    wandb_group = 'FeasClass-AcrossSizes'

    train_dataset = SolutionFeasibilityDataset(
        [fp for fp in Path('data/raw/').glob('125_*.json')
         if (int(fp.name.split('_')[1]) <= 18) and
            (int(fp.name.split('_')[2].replace('.json', '')) in np.arange(0,20))],
    )
    val_dataset = SolutionFeasibilityDataset(
        [fp for fp in Path('data/raw/').glob('125_*.json')
         if (int(fp.name.split('_')[1]) >= 20) and
            (int(fp.name.split('_')[2].replace('.json', '')) in np.arange(20,30))],
    )

    FeasibilityClassificationTrainer(
        FeasSatGNN(),
        train_dataset,
        val_dataset,
        epochs=5,
        batch_size=2**5,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        device=device,
    ).run()
