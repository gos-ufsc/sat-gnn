from pathlib import Path
from tqdm import tqdm

import json
import pickle
import numpy as np

from src.problem import get_model


if __name__ == '__main__':
    instances_fps = list(Path('data/raw').glob('*.json'))

    dst_fpath = Path('data/interim')/'solutions.pkl'

    try:
        with open(dst_fpath, 'rb') as f:
            solutions = pickle.load(f)

        instances_fps = [f for f in instances_fps if f[:-len('.json')] not in solutions.keys()]
    except FileNotFoundError:
        solutions = dict()

    for fpath in tqdm(instances_fps):
        instance_name = fpath.name[:-len('.json')]

        with open(fpath) as f:
            instance = json.load(f)

        jobs = list(range(instance['jobs']))

        model = get_model(jobs, instance, coupling=True, new_ineq=False, timeout=60)
        model.setParam('PoolSearchMode', 2)
        model.setParam('PoolSolutions', 500)
        model.update()

        model.optimize()

        objs = list()
        sols = list()
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            objs.append(model.PoolObjVal)
            sol = np.array([v.xn for v in model.getVars() if 'x(' in v.VarName])
            sols.append(sol)
        model.Params.SolutionNumber = 0

        solutions[instance_name] = {
            'sols': np.array(sols),
            'objs': np.array(objs),
        }

    with open(dst_fpath, 'wb') as f:
        pickle.dump(solutions, f)
