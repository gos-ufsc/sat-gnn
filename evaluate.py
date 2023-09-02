import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import wandb
from src.problem import Instance
from src.solver import (ConfEarlyFixingSolver, EarlyFixingSolver, SCIPSolver, TrustRegionSolver,
                        WarmStartingSolver)

TIME_BUDGET = 2 * 60  # 2 minutes

if __name__ == '__main__':
    evaluation = sys.argv[1]

    try:
        net_run_id = sys.argv[2]
    except IndexError:
        assert evaluation == 'bs'
        net_run_id = 'baseline'

    try:
        n = int(sys.argv[3])
    except ValueError:
        n = float(sys.argv[3])
        assert n < 1.0
    except IndexError:
        n = 0

    assert len(net_run_id) == 8, 'a wandb run id was not provided: '+net_run_id

    if evaluation == 'ef':
        evaluation_name = 'early-fixing'
        if n < 1 and n > 0:
            solver = ConfEarlyFixingSolver(net_run_id, n, timeout=TIME_BUDGET)
        else:
            solver = EarlyFixingSolver(net_run_id, n, timeout=TIME_BUDGET)
    elif evaluation == 'ws':
        evaluation_name = 'warms-starting'
        solver = WarmStartingSolver(net_run_id, n, timeout=TIME_BUDGET)
    elif evaluation == 'tr':
        evaluation_name = 'trust-region'
        solver = TrustRegionSolver(net_run_id, n, timeout=TIME_BUDGET)
    elif evaluation == 'bs':
        evaluation_name = 'baseline'
        solver = SCIPSolver(timeout=TIME_BUDGET)
    else:
        raise NotImplementedError

    solutions_dir = Path('/home/bruno/sat-gnn/data/interim')
    results_dir = Path('/home/bruno/sat-gnn/data/results')
    instances_dir = Path('/home/bruno/sat-gnn/data/raw')
    instances_fpaths = list()
    for i in range(20):  # VALIDATION
        instances_fpaths += sorted(list(instances_dir.glob('125_2*_'+str(i)+'.json')))

    instances_fpaths = [fp for fp in instances_fpaths if not (results_dir/(f"{net_run_id}_{evaluation}_{n}_"+fp.name)).exists()]

    if len(instances_fpaths) > 0:
        run = wandb.init(project='sat-gnn', job_type=evaluation_name+'-eval')
        run.config['model_run_id'] = net_run_id
        run.config['test_instances'] = [fp.name for fp in instances_fpaths]
        run.config['n'] = n if evaluation != 'bs' else 0
        run.config['time_budget'] = TIME_BUDGET

    for instance_fpath in tqdm(instances_fpaths):
        # load (quasi-)optimal objective
        solution_fpath = solutions_dir/instance_fpath.name.replace('.json', '_opt.npz')
        solution_npz = np.load(solution_fpath)
        quasi_optimal_objective = solution_npz['arr_0'].astype('uint32')

        instance_size = int(instance_fpath.name.split('_')[1])
        instance_i = int(instance_fpath.name.split('_')[-1][:-len('.json')])

        instance = Instance.from_file(instance_fpath)

        result = solver.solve(instance)

        result_fpath = results_dir/(f"{net_run_id}_{evaluation}_{n}_"+instance_fpath.name)
        with open(result_fpath, 'w') as f:
            json.dump(asdict(result), f)

        run.log(dict(
            instance_size=instance_size,
            instance_i=instance_i,
            quasi_optimal_objective=quasi_optimal_objective,
            **asdict(result),
        ))
