from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from src.dataset import SolutionFeasibilityDataset
from src.net import FeasSatGNN
from src.utils import load_from_wandb


if __name__ == '__main__':
    test_data = SolutionFeasibilityDataset(
        [fp for fp in Path('data/raw/').glob('125_*.json')
            if (int(fp.name.split('_')[1]) >= 20) and
            (int(fp.name.split('_')[2].replace('.json', '')) in np.arange(20,30))],
    )
    test_data.maybe_initialize()

    net = load_from_wandb(FeasSatGNN(), 'g1xgtzc0', 'sat-gnn')

    Y = list()
    P_hat = list()
    for i in tqdm(list(range(len(test_data)))):
        g, y = test_data[i]

        with torch.no_grad():
            p_hat = torch.sigmoid(net(g)).item()
        
        Y.append(y)
        P_hat.append(p_hat)
    P_hat = np.array(P_hat)
    Y = np.array(Y)

    np.savez('feasibility_preds.npz', P_hat=P_hat, Y=Y)
