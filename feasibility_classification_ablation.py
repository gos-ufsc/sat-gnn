from pathlib import Path

import numpy as np
import torch

from src.dataset import SolutionFeasibilityDataset
from src.net import FeasSatGNN
from src.trainer import (FeasibilityClassificationTrainer,
                         InstanceFeasibilityClassificationTrainer)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_project = 'sat-gnn'

    wandb_group = 'FeasClass-Ablation'

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

    for random_seed in [42,43,44,45,46,47,48,49,50,51]:
        for n_output_layers in [1,2,3,4]:
            FeasibilityClassificationTrainer(
                FeasSatGNN(n_output_layers=n_output_layers),
                train_dataset,
                val_dataset,
                epochs=5,
                batch_size=2**5,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
                device=device,
                random_seed=random_seed,
            ).run()
