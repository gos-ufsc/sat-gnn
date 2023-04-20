import pickle
import sys

from pathlib import Path

from gurobipy import GRB
from tqdm import tqdm
import numpy as np
import torch

from src.dataset import InstanceDataset
from src.net import InstanceGCN
from src.utils import load_from_wandb


def get_ht_performance(graph, model, net, timeout=60):
    vars_names = np.core.defchararray.array([v.getAttr(GRB.Attr.VarName) for v in model.getVars()])
    vars_names = vars_names[(vars_names.find('x(') >= 0) | (vars_names.find('phi(') >= 0)]

    # baseline results
    model_ = model.copy()
    model_.setParam('TimeLimit', timeout)
    model_.update()
    model_.optimize()
    baseline_runtime = model_.Runtime
    baseline_obj = model_.ObjVal
    baseline_gap = model_.MIPGap

    with torch.no_grad():
        x_hat = torch.sigmoid(net(graph)).squeeze(0)

    most_certain_idx  = (x_hat - 0.5).abs().sort(descending=True).indices

    runtimes = list()
    objs = list()
    gaps = list()
    ns = list()
    for n in [0, len(x_hat)]:
    # for n in [0, 50, 100, 150]:
        if n == 0:
            runtimes.append(baseline_runtime)
            objs.append(baseline_obj)
            gaps.append(baseline_gap)
            ns.append(n)
            continue

        fixed_x_hat = (x_hat[most_certain_idx[:n]] > .5).to(x_hat)
        certainty_x_hat = (x_hat[most_certain_idx[:n]] - 0.5).abs() * 2
        fixed_vars_names = vars_names[most_certain_idx[:n]]

        # fix variables
        model_ = model.copy()
        for fixed_var_name, fixed_var_X, var_certainty in zip(fixed_vars_names, fixed_x_hat, certainty_x_hat):
            model_.getVarByName(fixed_var_name).VarHintVal = fixed_var_X
            model_.getVarByName(fixed_var_name).VarHintPri = int(var_certainty * 100)

        model_.setParam('TimeLimit', timeout)
        model_.update()
        model_.optimize()

        if model_.status not in [2, 9]:
            # print('early fixing with n=',n,' made the optimizatio terminate with status ',model_.status)
            break

        runtimes.append(model_.Runtime)
        objs.append(model_.ObjVal)
        gaps.append(model_.MIPGap)
        ns.append(n)

    objs = [100 * o / max(objs) for o in objs]

    return ns, runtimes, objs, gaps

if __name__ == '__main__':
    wandb_run_id = sys.argv[-1]

    assert len(wandb_run_id) == 8, 'a proper wandb run id was not provided'

    instances_dir = Path('/home/bruno/sat-gnn/data/raw')
    instances_fpaths = list(instances_dir.glob('97_9_*.json'))

    net = InstanceGCN(1, readout_op=None)
    net = load_from_wandb(net, wandb_run_id, 'sat-gnn', 'model_last')
    net.eval()

    ds = InstanceDataset(instances_fpaths, split='val', return_model=True)

    performances = list()
    for graph, _, model in tqdm(ds):
        # get T and J
        x_vars = [v.varname for v in model.getVars() if 'x(' in v.varname]
        js = {int(var[len('x('):].split(',')[0]) for var in x_vars}
        ts = {int(var.split(',')[1][:-1]) for var in x_vars}
        T, J = len(ts), len(js)

        ns, runtimes, objs, gaps = get_ht_performance(graph, model, net, timeout=10)

        performances.append(dict(
            T=T, J=J,
            ns=ns, runtimes=runtimes,
            objs=objs, gaps=gaps
        ))

    with open(f'ht_performance_{wandb_run_id}.pkl', 'wb') as f:
        pickle.dump(performances, f)
