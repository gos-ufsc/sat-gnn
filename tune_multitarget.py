import os
from itertools import product
from pathlib import Path
from random import shuffle

import torch
import torch.nn
from tqdm import tqdm

from src.dataset import MultiTargetDataset
from src.net import OptSatGNN
from src.trainer import MultiTargetTrainer
from src.utils import debugger_is_active


def product_dict(**kwargs):
    """From https://stackoverflow.com/a/5228294/7964333."""
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb_project = 'sat-gnn'
    wandb_group = 'GridSearch-MultiTarget'

    memory_size_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    if (memory_size_gb < 30) or debugger_is_active():
        train_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_125_small_train.hdf5')
        val_dataset = MultiTargetDataset.from_file_lazy('data/processed/multitarget_125_small_val.hdf5')
    else:
        train_dataset = MultiTargetDataset(
            [fp for fp in Path('data/raw/').glob('125_*.json')
                if (int(fp.name.split('_')[1]) < 20) and
                (int(fp.name.split('_')[2].replace('.json', '')) < 200)]
        )
        val_dataset = MultiTargetDataset(
            [fp for fp in Path('data/raw/').glob('125_*.json')
                if (int(fp.name.split('_')[1]) >= 20) and
                (int(fp.name.split('_')[2].replace('.json', '')) < 20)]
        )
        train_dataset.maybe_initialize()
        val_dataset.maybe_initialize()

    hp_ranges = {
        'lr': [1e-2, 1e-3, 1e-4],
        'n_h_feats': [2**5, 2**6, 2**7, 2**8],
        'single_conv_for_both_passes': [True, False],
        'n_passes': [1, 2, 3],
        'conv1': ['GraphConv', 'SAGEConv'],
        'conv2': ['GraphConv', 'SAGEConv', None],
        # 'conv3': ['GraphConv', 'SAGEConv'],
    }

    candidate_hps = product_dict(**hp_ranges)
    candidate_hps = list(candidate_hps)
    shuffle(candidate_hps)

    for hps in tqdm(candidate_hps):
        lr = hps.pop('lr')

        for c in ['conv1', 'conv2']:
            # try:
            if hps[c] == 'SAGEConv':
                hps[c+'_kwargs'] = {'aggregator_type': 'pool'}
            else:
                hps[c+'_kwargs'] = dict()
            # except:
            #     pass

        for _ in range(1):  # number of runs
            MultiTargetTrainer(
                OptSatGNN(**hps),
                train_dataset,
                val_dataset,
                lr=lr,
                epochs=10,
                get_best_model=True,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
            ).run()
