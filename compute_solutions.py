import sys

from pathlib import Path
from tqdm import tqdm

import numpy as np

from src.problem import Instance


if __name__ == '__main__':
    if len(sys.argv) < 2:
        glob_str = '*'  # default
    else:
        glob_str = sys.argv[1]

    instances_fps = list(Path('data/raw/').glob(glob_str+'.json'))

    dst_dir = Path('data/interim/')

    try:
        solutions_fpaths = dst_dir.glob('*_sols.npz')
        solutions = [fp.name[:-len('_sols.npz')] for fp in solutions_fpaths]

        instances_fps = [f for f in instances_fps if f.name[:-len('.json')] not in solutions]
    except FileNotFoundError:
        pass

    for fpath in tqdm(instances_fps):
        instance_name = fpath.name[:-len('.json')]

        instance = Instance.from_file(fpath)

        jobs = list(range(instance.jobs))

        model = instance.to_gurobipy(coupling=True, new_inequalities=True, timeout=300)
        model.setParam('PoolSearchMode', 2)
        model.setParam('PoolSolutions', 500)
        model.update()

        model.optimize()

        # save best solution found
        X = np.array([v.X for v in model.getVars()])
        model_vars = np.core.defchararray.array([v.VarName for v in model.getVars()])
        X = X[(model_vars.find('x') >= 0) | (model_vars.find('phi') >= 0)]

        obj = model.ObjVal
        gap = model.MIPGap
        runtime = model.Runtime

        np.savez(dst_dir/(instance_name + '_opt.npz'), obj, gap, runtime, X)

        objs = list()
        sols = list()
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            objs.append(model.PoolObjVal)
            # sol = np.array([v.xn for v in model.getVars() if 'x(' in v.VarName])
            sol = np.array([v.xn for v in model.getVars() if ('x(' in v.VarName) or ('phi(' in v.VarName)])
            # sol = np.array([v.xn for v in model.getVars()])
            sols.append(sol)
        model.Params.SolutionNumber = 0

        sols = np.array(sols)
        objs = np.array(objs)

        np.savez(dst_dir/(instance_name + '_sols.npz'), sols, objs)
