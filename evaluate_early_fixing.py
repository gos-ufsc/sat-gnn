import pickle
import sys

from pathlib import Path

from gurobipy import GRB
from tqdm import tqdm
import numpy as np
import torch

from pyscipopt import Model

from src.problem import ModelWithPrimalDualIntegral
from src.dataset import MultiTargetDataset, OptimalsWithZetaDataset
from src.net import InstanceGCN, VarInstanceGCN
from src.utils import load_from_wandb


def get_ef_performance(graph, model, net, timeout=60):
    vars_names = np.core.defchararray.array([v.name for v in model.getVars()])
    # vars_names = vars_names[(vars_names.find('x(') >= 0) | (vars_names.find('phi(') >= 0)]
    vars_names = vars_names[(vars_names.find('x(') >= 0) | (vars_names.find('phi(') >= 0)]

    # baseline results
    model_ = ModelWithPrimalDualIntegral(sourceModel=model)
    model_.setParam('limits/time', timeout)

    model_.optimize()
    baseline_runtime = model_.getSolvingTime()
    baseline_obj = model_.getObjVal()
    baseline_gap = model_.getGap()
    baseline_pd_integral = model_.get_primal_dual_integral()

    with torch.no_grad():
        # x_hat = torch.sigmoid(net(graph)).squeeze(0)
        x_hat = net.get_candidate(graph).flatten()
    # phi_filter = torch.ones_like(x_hat) == 0  # only False
    # phi_filter = phi_filter.view(-1, 2*97)
    # phi_filter[:,97:] = True
    # phi_filter = phi_filter.flatten()
    # x_hat = x_hat[phi_filter]

    # drop zetas
    x_hat = x_hat[:len(vars_names)]

    most_certain_idx  = (x_hat - 0.5).abs().sort(descending=True).indices

    runtimes = list()
    objs = list()
    gaps = list()
    pd_integrals = list()
    ns = list()
    for n in [0, 50, 100, 150, 200, 300, 500, len(x_hat)]:
    # for n in [0, 50, 100, 150]:
        if n == 0:
            runtimes.append(baseline_runtime)
            objs.append(baseline_obj)
            gaps.append(baseline_gap)
            pd_integrals.append(baseline_pd_integral)
            ns.append(n)
            continue

        fixed_x_hat = (x_hat[most_certain_idx[:n]] > .5).to(x_hat)
        fixed_vars_names = vars_names[most_certain_idx[:n]]

        # fix variables
        model_ = ModelWithPrimalDualIntegral(sourceModel=model)
        for fixed_var_name, fixed_var_X in zip(fixed_vars_names, fixed_x_hat):
            for var in model_.getVars():
                if var.name == fixed_var_name:
                    break
            model_.fixVar(var, fixed_var_X)

        model_.setParam('limits/time', timeout)
        model_.optimize()

        if model_.getStatus().lower() not in ['optimal', 'timelimit']:
            print('early fixing with n=',n,' made the optimizatio terminate with status ',model_.getStatus())
            break

        runtimes.append(model_.getSolvingTime())
        objs.append(model_.getObjVal())
        gaps.append(model_.getGap())
        pd_integrals.append(model_.get_primal_dual_integral())
        ns.append(n)

    objs = [100 * o / baseline_obj for o in objs]

    return ns, runtimes, objs, gaps, pd_integrals


if __name__ == '__main__':
    wandb_run_id = sys.argv[-1]

    assert len(wandb_run_id) == 8, 'a proper wandb run id was not provided'

    instances_dir = Path('/home/bruno/sat-gnn/data/raw')
    instances_fpaths = list(instances_dir.glob('97_*.json'))

    net = InstanceGCN(
        readout_op=None,
        # conv1='GATv2Conv',
        # conv1_kwargs={'num_heads': 1,},
        # conv2='GATv2Conv',
        # conv2_kwargs={'num_heads': 1,},
    )
    # net = VarInstanceGCN(readout_op=None)
    net = load_from_wandb(net, wandb_run_id, 'sat-gnn', 'model_last')
    net.eval()

    ds = MultiTargetDataset(instances_fpaths, split='val', return_model=True)
    # ds = OptimalsWithZetaDataset(instances_fpaths, split='val', return_model=True)

    dst_fpath = f'new_scip_ef_performance_{wandb_run_id}.pkl'
    if Path(dst_fpath).exists():
        with open(dst_fpath, 'rb') as f:
            performances = pickle.load(f)
    else:
        performances = list()

    for i in tqdm(range(len(performances), len(ds))):
        graph, _, model = ds[i]

        # get T and J
        x_vars = [v.name for v in model.getVars() if 'x(' in v.name]
        js = {int(var[len('x('):].split(',')[0]) for var in x_vars}
        ts = {int(var.split(',')[1][:-1]) for var in x_vars}
        T, J = len(ts), len(js)

        ns, runtimes, objs, gaps, pd_integrals = get_ef_performance(graph, model, net, timeout=10)

        performances.append(dict(
            T=T, J=J,
            ns=ns, runtimes=runtimes,
            objs=objs, gaps=gaps,
            pd_integrals=pd_integrals,
        ))

        with open(dst_fpath, 'wb') as f:
            pickle.dump(performances, f)
