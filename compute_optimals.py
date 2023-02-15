from pathlib import Path
from tqdm import tqdm

from gurobipy import GRB

import pickle
import numpy as np

from src.data import get_model, load_instance


if __name__ == '__main__':
    instances_fps = list(Path('data/raw').glob('97_9*.jl'))

    solutions = dict()
    for fpath in tqdm(instances_fps):
        instance_name = fpath.name

        instance = load_instance(str(fpath))

        jobs = list(range(instance['jobs'][0]))
        model = get_model(jobs, fpath, coupling=True)

        model.optimize()

        X = np.array([v.X for v in model.getVars()])
        soc_vars_mask = np.array(['soc' in v.getAttr(GRB.Attr.VarName) for v in model.getVars()])
        X = X[~soc_vars_mask]

        solutions[instance_name] = {
            'obj': model.ObjVal,
            'sol': X,
        }
    
    with open('97_9_opts.pkl', 'wb') as f:
        pickle.dump(solutions, f)
