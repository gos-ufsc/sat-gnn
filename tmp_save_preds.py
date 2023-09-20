import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

import wandb
from src.net import OptSatGNN
from src.problem import Instance


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

    preds = list()
    instances_fpaths = [fp for fp in Path('data/raw/').glob('125_*.json')
                        if (int(fp.name.split('_')[1]) >= 20) and
                           (int(fp.name.split('_')[2].replace('.json', '')) >= 20) and
                           (int(fp.name.split('_')[2].replace('.json', '')) < 40)]
    for instance_fpath in tqdm(instances_fpaths):
        instance = Instance.from_file(instance_fpath)

        graph = instance.to_graph()
        with torch.set_grad_enabled(False):
            x_hat = net.get_candidate(graph).flatten().cpu()
            x_hat = x_hat[:len(instance.vars_names)]  # drop zetas

        preds.append({
            'fp': instance_fpath,
            'size': instance.jobs,
            'x_hat': x_hat.numpy(),
            'vars_names': instance.vars_names,
        })

    with open(net_run_id+'_preds_test.pkl', 'wb') as f:
        pickle.dump(preds, f)
