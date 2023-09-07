import json
import sys
from dataclasses import asdict
from pathlib import Path
from multiprocessing import Pool, Queue
from joblib import cpu_count
from functools import partial

import click
import numpy as np
from tqdm import tqdm

import wandb
from src.problem import Instance
from src.solver import (ConfEarlyFixingSolver, ConfidenceRegionSolver,
                        EarlyFixingSolver, SCIPSolver, TrustRegionSolver,
                        WarmStartingSolver)

def _evaluate_instance(solver, evaluation, net_run_id, n, delta, solutions_dir, results_dir, instance_fpath, hide_output=True):
    # load (quasi-)optimal objective
    solution_fpath = solutions_dir/instance_fpath.name.replace('.json', '_opt.npz')
    solution_npz = np.load(solution_fpath)
    quasi_optimal_objective = solution_npz['arr_0'].astype('uint32')

    instance_size = int(instance_fpath.name.split('_')[1])
    instance_i = int(instance_fpath.name.split('_')[-1][:-len('.json')])

    instance = Instance.from_file(instance_fpath)

    result = solver.solve(instance, hide_output=hide_output)

    if evaluation == 'tr':
        result_fpath = results_dir/(f"{net_run_id}_{evaluation}_{n}_{delta}_"+instance_fpath.name)
    else:
        result_fpath = results_dir/(f"{net_run_id}_{evaluation}_{n}_"+instance_fpath.name)
    with open(result_fpath, 'w') as f:
        json.dump(asdict(result), f)

@click.command()
@click.argument('evaluation', type=click.STRING)
@click.argument('net_run_id', type=click.STRING, required=False)
@click.option('--n', default=0.0, type=click.FLOAT)
@click.option('--delta', default=0.001, type=click.FLOAT)
@click.option('--k', default=1.0, type=click.FLOAT)
@click.option('--threshold', default=0.0, type=click.FLOAT)
@click.option('--T', default=10*60, type=click.INT)
def evaluate(evaluation, net_run_id, n, delta, k, threshold, t):
    T = t

    if net_run_id is None:
        net_run_id = 'baseline'

    assert len(net_run_id) == 8, 'a wandb run id was not provided: '+net_run_id

    if evaluation == 'ef':
        evaluation_name = 'early-fixing'
        if threshold > 0:
            solver = ConfEarlyFixingSolver(net_run_id, threshold, timeout=T)
        else:
            solver = EarlyFixingSolver(net_run_id, n, timeout=T)
    elif evaluation == 'ws':
        evaluation_name = 'warms-starting'
        solver = WarmStartingSolver(net_run_id, n, timeout=T)
    elif evaluation == 'tr':
        evaluation_name = 'trust-region'
        solver = TrustRegionSolver(net_run_id, n, timeout=T, Delta=delta)
    elif evaluation == 'cr':
        evaluation_name = 'confidence-region'
        solver = ConfidenceRegionSolver(net_run_id, timeout=T, k=k)
    elif evaluation == 'bs':
        evaluation_name = 'baseline'
        solver = SCIPSolver(timeout=T)
    else:
        raise NotImplementedError

    solutions_dir = Path('/home/bruno/sat-gnn/data/interim')
    results_dir = Path('/home/bruno/sat-gnn/data/results/parallel')
    instances_dir = Path('/home/bruno/sat-gnn/data/raw')
    instances_fpaths = list()
    for i in range(20):  # VALIDATION
        instances_fpaths += sorted(list(instances_dir.glob('125_2*_'+str(i)+'.json')))

    if evaluation == 'tr':
        instances_fpaths = [fp for fp in instances_fpaths if not (results_dir/(f"{net_run_id}_{evaluation}_{n}_{delta}_"+fp.name)).exists()]
    else:
        instances_fpaths = [fp for fp in instances_fpaths if not (results_dir/(f"{net_run_id}_{evaluation}_{n}_"+fp.name)).exists()]

    # if len(instances_fpaths) > 0:
    #     run = wandb.init(project='sat-gnn', job_type=evaluation_name+'-eval')
    #     run.config['model_run_id'] = net_run_id
    #     run.config['test_instances'] = [fp.name for fp in instances_fpaths]
    #     run.config['n'] = n if evaluation != 'bs' else 0
    #     run.config['time_budget'] = T

        # run.log(dict(
        #     instance_size=instance_size,
        #     instance_i=instance_i,
        #     quasi_optimal_objective=quasi_optimal_objective,
        #     **asdict(result),
        # ))
    from time import time
    start = time()
    # if len(instances_fpaths) > 0:
    #     with Pool(min(len(instances_fpaths), cpu_count()-1)) as pool:
    #         pool.map(partial(_evaluate_instance, solver, evaluation, net_run_id, n, delta, solutions_dir, results_dir), instances_fpaths)

    def worker(args_queue: Queue):
        while True:
            args = args_queue.get(True)
            if args is None:
                args_queue.put(None)
                break
        
            _evaluate_instance(*args)

    queue = Queue()
    pool = Pool(30, worker, (queue,))

    for instance_fpath in instances_fpaths:
        queue.put((solver, evaluation, net_run_id, n, delta, solutions_dir, results_dir, instance_fpath))
    queue.put(None)

    pool.close()
    pool.join()

    # with Pool(20) as pool:
    #     apply_results = list()
    #     for instance_fpath in instances_fpaths:
    #         apply_results.append(
    #             pool.apply_async(
    #                 _evaluate_instance,
    #                 (solver, evaluation, net_run_id, n, delta, solutions_dir, results_dir, instance_fpath)
    #                 # partial(_evaluate_instance, solver, evaluation, net_run_id, n, delta, solutions_dir, results_dir),
    #                 # instance_fpath
    #             )
    #         )
        # for r in tqdm(apply_results):
        #     r.get()

    print(time() - start)

if __name__ == '__main__':
    evaluate()
