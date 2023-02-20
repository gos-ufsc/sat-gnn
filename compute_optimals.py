from pathlib import Path
from tqdm import tqdm

from gurobipy import GRB

import pickle
import numpy as np

from src.data import get_model, load_instance


if __name__ == '__main__':
    instances_fps = list(Path('data/raw').glob('97_9*.jl'))

    dst_fpath = 'new_97_9_opts.pkl'

    with open(dst_fpath, 'rb') as f:
        solutions = pickle.load(f)

    instances_fps = [f for f in instances_fps if f not in solutions.keys()]
    for fpath in tqdm(instances_fps):
        instance_name = fpath.name
        print(instance_name)

        instance = load_instance(str(fpath))

        jobs = list(range(instance['jobs'][0]))
        model = get_model(jobs, fpath, coupling=True)

        model.Params.LogToConsole = 1
        model.update()

        model.optimize()

        # if model.status != 2:
        #     print('WARNING ', instance_name, ' WAS NOT SOLVED')
        #     print('SOLVER STATUS ', model.status)
        #     continue

        X = np.array([v.X for v in model.getVars()])
        model_vars = np.core.defchararray.array([v.getAttr(GRB.Attr.VarName) for v in model.getVars()])
        X = X[(model_vars.find('x') >= 0) | (model_vars.find('phi') >= 0)]

        solutions[instance_name] = {
            'obj': model.ObjVal,
            'sol': X,
        }
    
    with open(dst_fpath, 'wb') as f:
        pickle.dump(solutions, f)
