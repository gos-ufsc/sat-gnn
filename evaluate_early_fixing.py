import pickle
import sys

from pathlib import Path

from gurobipy import GRB
from tqdm import tqdm
import numpy as np
import torch

from pyscipopt import Model
import wandb

from src.problem import ModelWithPrimalDualIntegral
from src.dataset import MultiTargetDataset, OptimalsWithZetaDataset
from src.net import InstanceGCN, VarInstanceGCN
from src.utils import load_from_wandb


def evaluate_early_fixing(model, reference_objective, fixed_vars: dict = None,
                          timeout=10, hide_output=True):
    model_ = ModelWithPrimalDualIntegral(sourceModel=model)
    model_.setParam('limits/time', timeout)
    model_.hideOutput(hide_output)

    if fixed_vars is not None:
        for var in model_.getVars():
            try:
                fixed_var_X = fixed_vars[var.name]
                model_.fixVar(var, fixed_var_X)
            except KeyError:
                pass

    model_.optimize()

    if model_.getStatus().lower() not in ['optimal', 'timelimit']:
        infeasible = True
        runtime = model_.getSolvingTime()
        objective = 0
        gap = -1
        primal_dual_integral = -1
    else:
        infeasible = False
        runtime = model_.getSolvingTime()
        try:
            objective = model_.getObjVal()
            gap = model_.getGap()
            primal_dual_integral = model_.get_primal_dual_integral()
            relative_primal_integral = model_.get_relative_primal_integral(reference_objective)
        except:
            # in case the problem is not infeasible but not solution was
            # found during the time limit
            objective = np.nan
            gap = np.nan
            primal_dual_integral = np.nan
            relative_primal_integral = np.nan

    return infeasible, runtime, objective, gap, primal_dual_integral, relative_primal_integral

if __name__ == '__main__':
    model_run_id = sys.argv[-1]
    N = [0, 50, 200, 1000]
    time_budget = 10*60  # 10 min
    # time_budget = 30  # 10 min

    assert len(model_run_id) == 8, 'a proper wandb run id was not provided'

    instances_dir = Path('/home/bruno/sat-gnn/data/raw')
    instances_fpaths = list(instances_dir.glob('97_24_*.json'))

    # restore model
    net = InstanceGCN(readout_op=None)  # TODO: initialize net following run's config

    model_run = wandb.init(project='sat-gnn', id=model_run_id)
    model_config = model_run.config
    model_group = model_run.group
    model_file = model_run.restore('model_last.pth', replace=True)
    model_run.finish()

    net.load_state_dict(torch.load(model_file.name))
    net.eval()

    run = wandb.init(project='sat-gnn', group=model_group, job_type='early-fixing-eval', config=dict(
        model_run_id=model_run_id,
        **model_config
    ))

    run.config['test_instances'] = [fp.name for fp in instances_fpaths]
    run.config['N'] = N
    run.config['ef_time_budget'] = time_budget

    dataset = MultiTargetDataset(
        instances_fpaths=instances_fpaths,
        sols_dir='/home/bruno/sat-gnn/data/interim',
        split='val',
        return_model=True,
    )
    # ds = OptimalsWithZetaDataset(instances_fpaths, split='val', return_model=True)

    for graph, model in dataset:
        quasi_optimal_objective = graph.ndata['w']['var'][0].max().item()

        # TODO: device!

        vars_names = np.core.defchararray.array([v.name for v in model.getVars()])
        vars_names = vars_names[(vars_names.find('x(') >= 0) | (vars_names.find('phi(') >= 0)]

        # baseline results
        (
            baseline_infeasible,
            baseline_runtime,
            baseline_obj,
            baseline_gap,
            baseline_pd_integral,
            baseline_rel_primal_integral
        ) = evaluate_early_fixing(model, quasi_optimal_objective, None,
                                  time_budget)

        if baseline_infeasible:
            print('INFEASIBLE PROBLEM')
            print(model)
            continue

        with torch.set_grad_enabled(False):
            x_hat = net.get_candidate(graph).flatten().cpu()
            x_hat = x_hat[:len(vars_names)]  # drop zetas

        most_certain_idx  = (x_hat - 0.5).abs().sort(descending=True).indices

        for i in range(len(N)):
            n = N[i] if N[i] >= 0 else len(x_hat)

            if n == 0:
                result = {
                    'infeasible': 0,
                    'runtime': baseline_runtime,
                    'obj': baseline_obj,
                    'relative_obj': 1.,
                    'gap': baseline_gap,
                    'pd_integral': baseline_pd_integral,
                    'rel_primal_integral': baseline_rel_primal_integral,
                }
            else:
                fixed_x_hat = (x_hat[most_certain_idx[:n]] > .5).to(x_hat)
                fixed_vars_names = vars_names[most_certain_idx[:n]]
                fixed_vars = dict(zip(fixed_vars_names, fixed_x_hat))

                (
                    infeasible, runtime, obj, gap, pd_integral, rel_primal_integral
                ) = evaluate_early_fixing(model, quasi_optimal_objective, fixed_vars,
                                        time_budget)

                if infeasible:
                    result = {
                        'runtime': baseline_runtime + runtime,
                        'obj': baseline_obj,
                        'relative_obj': 1.,
                        'gap': baseline_gap,
                        'pd_integral': baseline_pd_integral,
                        'rel_primal_integral': baseline_rel_primal_integral,
                    }
                else:
                    result = {
                        'runtime': runtime,
                        'obj': obj,
                        'relative_obj': obj / baseline_obj,
                        'gap': gap,
                        'pd_integral': pd_integral,
                        'rel_primal_integral': rel_primal_integral,
                    }
                result['infeasible'] = int(infeasible)

            run.log(
                dict(n_fixed=n, **result)
            )
    wandb.finish()