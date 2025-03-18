import json
import random
import multiprocessing
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

import wandb
from src.problem import Instance
from src.solver import SCIPSolver


@click.command()
@click.option('--T', default=2*60, type=click.INT)
def evaluate(t):
    T = t

    net_run_id = 'baseline'
    evaluation = 'bs'
    evaluation_name = 'baseline'
    n = 0

    solver = SCIPSolver(timeout=T)

    solutions_dir = Path('/home/bruno/sat-gnn/data/interim')
    results_dir = Path('/home/bruno/sat-gnn/data/results')
    instances_dir = Path('/home/bruno/sat-gnn/data/raw')
    instances_fpaths = list()
    
    instances_fpaths += sorted(list(instances_dir.glob('125_9_*.json')))
    instances_fpaths += sorted(list(instances_dir.glob('125_13_*.json')))
    instances_fpaths += sorted(list(instances_dir.glob('125_18_*.json')))

    instances_fpaths = [fp for fp in instances_fpaths if not (results_dir/(f"{net_run_id}_{evaluation}_{n}_"+fp.name)).exists()]

    random.shuffle(instances_fpaths)

    if len(instances_fpaths) > 0:
        run = wandb.init(project='sat-gnn', job_type=evaluation_name+'-eval')
        run.config['model_run_id'] = net_run_id
        run.config['test_instances'] = [fp.name for fp in instances_fpaths]
        run.config['n'] = 0
        run.config['time_budget'] = T

    def solve_and_save(instance_fpath, result_ptr):
        instance = Instance.from_file(instance_fpath)

        result = solver.solve(instance)
        result_ptr[0] = result

        result_fpath = results_dir/(f"{net_run_id}_{evaluation}_{n}_"+instance_fpath.name)
        with open(result_fpath, 'w') as f:
            json.dump(asdict(result), f)

        return result

    for instance_fpath in tqdm(instances_fpaths):
        print(f"Running {instance_fpath.name}")
        # load (quasi-)optimal objective
        solution_fpath = solutions_dir/instance_fpath.name.replace('.json', '_opt.npz')
        solution_npz = np.load(solution_fpath)
        quasi_optimal_objective = solution_npz['arr_0'].astype('uint32')

        instance_size = int(instance_fpath.name.split('_')[1])
        instance_i = int(instance_fpath.name.split('_')[-1][:-len('.json')])

        result_ptr = [None,]
        p = multiprocessing.Process(target=solve_and_save, args=(instance_fpath, result_ptr))

        p.start()

        p.join(2*T)

        if p.is_alive():
            print(f"Instance {instance_fpath.name} failed")
            p.kill()
            p.join()
        else:
            # result = result_ptr[0]
            run.log(dict(
                instance_size=instance_size,
                instance_i=instance_i,
                quasi_optimal_objective=quasi_optimal_objective,
                # **asdict(result),
            ))

if __name__ == '__main__':
    evaluate()
