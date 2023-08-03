from pathlib import Path
from tqdm import tqdm

from gurobipy import GRB

import numpy as np

from src.problem import Instance


if __name__ == '__main__':
    instances_fps = list(Path('data/raw').glob('120_*.json'))

    dst_dir = Path('data/interim')

    try:
        solutions_fpaths = dst_dir.glob('*_opt.npz')
        solutions = [fp.name[:-len('_opt.npz')] for fp in solutions_fpaths]

        instances_fps = [f for f in instances_fps if f.name[:-len('.json')] not in solutions]
    except FileNotFoundError:
        pass

    for fpath in tqdm(instances_fps):
        instance_name = fpath.name[:-len('.json')]

        instance = Instance.from_file(fpath)

        model = instance.to_gurobipy(coupling=True, new_inequalities=True, timeout=300)

        model.update()
        model.optimize()

        if model.Status == 3:
            print('INFEASIBLE ', fpath)
            continue

        X = np.array([v.X for v in model.getVars()])
        model_vars = np.core.defchararray.array([v.getAttr(GRB.Attr.VarName) for v in model.getVars()])
        X = X[(model_vars.find('x') >= 0) | (model_vars.find('phi') >= 0)]

        obj = model.ObjVal
        gap = model.MIPGap
        runtime = model.Runtime

        np.savez(dst_dir/(instance_name + '_opt.npz'), obj, gap, runtime, X)
