from pathlib import Path
from time import time
import optuna
import pickle
import numpy as np
import torch
import torch.nn
from tqdm import tqdm

from itertools import product
from random import shuffle

from src.net import OptSatGNN
from src.trainer import MultiTargetTrainer
# from src.tuning import get_objective
from src.dataset import MultiTargetDataset


def product_dict(**kwargs):
    """From https://stackoverflow.com/a/5228294/7964333."""
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_project = 'sat-gnn'
    wandb_group = 'GridSearch-MultiTarget'

    train_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_125_small_train.hdf5')
    val_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_125_small_val.hdf5')

    hp_ranges = {
        'lr': [1e-2, 1e-3, 1e-4],
        'n_h_feats': [2**5, 2**6, 2**7, 2**8],
        'single_conv_for_both_passes': [True, False],
        'n_passes': [1, 2, 3],
        'conv1': ['GraphConv', 'SAGEConv'],
        # 'conv2': ['GraphConv', 'SAGEConv'],
        # 'conv3': ['GraphConv', 'SAGEConv'],
    }

    candidate_hps = product_dict(**hp_ranges)
    candidate_hps = list(candidate_hps)
    shuffle(candidate_hps)

    for hps in tqdm(candidate_hps):
        lr = hps.pop('lr')

        for c in ['conv1', 'conv2', 'conv3']:
            if hps[c] == 'SAGEConv':
                hps[c+'_kwargs'] = {'aggregator_type': 'pool'}

        net = OptSatGNN(**hps)
        trainer = MultiTargetTrainer(
            net,
            train_dataset,
            val_dataset,
            lr=lr,
            epochs=10,
            get_best_model=True,
        )
        trainer.run()

        trainer.best_val
