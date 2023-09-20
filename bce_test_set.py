import sys
from pathlib import Path

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

import wandb
from src.dataset import MultiTargetDataset, OptimalsDataset
from src.net import OptSatGNN


if __name__ == '__main__':
    net_run_id = sys.argv[1]

    net_run = wandb.Api().run('brunompac/sat-gnn/'+net_run_id)
    net_config = net_run.config
    net_group = net_run.group
    net_file = wandb.restore('model_best.pth', run_path='/'.join(net_run.path),
                             replace=True)

    try:
        net = OptSatGNN(
            n_h_feats=net_config['n_h_feats'],
            single_conv_for_both_passes=net_config['single_conv'],
            n_passes=net_config['n_passes'],
        )
        net.load_state_dict(torch.load(net_file.name))
    except RuntimeError:
        net = OptSatGNN(
            conv1='GraphConv', conv1_kwargs=dict(),
            n_h_feats=net_config['n_h_feats'],
            single_conv_for_both_passes=net_config['single_conv'],
            n_passes=net_config['n_passes'],
        )
        net.load_state_dict(torch.load(net_file.name))
    net.eval()

    instances_fpaths = [fp for fp in Path('data/raw/').glob('125_*.json')
                        if (int(fp.name.split('_')[1]) >= 20) and
                           (int(fp.name.split('_')[2].replace('.json', '')) >= 20) and
                           (int(fp.name.split('_')[2].replace('.json', '')) < 40)]
    if 'MultiTarget' in net_group:
        dataset = MultiTargetDataset(instances_fpaths=instances_fpaths)
        dl = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataset = OptimalsDataset(instances_fpaths=instances_fpaths)
        dl = GraphDataLoader(dataset, batch_size=4, shuffle=False)

    loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    losses = list()
    for g in tqdm(dl):
        with torch.set_grad_enabled(False):
            y_hat = net(g)

        y = g.ndata['y']['var'].to(y_hat)

        if 'MultiTarget' in net_group:
            y_hat = y_hat.repeat(*(np.array(y.shape) // np.array(y_hat.shape)))

        loss = loss_func(y_hat.view_as(y), y)

        if 'MultiTarget' in net_group:
            w = g.ndata['w']['var'].to(y_hat)
            weight = torch.softmax(w / w.max(-1)[0].unsqueeze(-1), -1)
            loss = (weight * loss).mean(0).sum()

        losses.append(loss.mean().item() * g.batch_size)
    average_loss = sum(losses) / len(dataset)
    print(average_loss)

    # with open(net_run_id+'_preds_val.pkl', 'wb') as f:
    #     pickle.dump(preds, f)
