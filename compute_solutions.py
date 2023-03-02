from pathlib import Path
from tqdm import tqdm

import pickle
import numpy as np

from src.problem import get_model, load_instance


if __name__ == '__main__':
    instances_fps = list(Path('data/raw').glob('97_9*.jl'))

    dst_fpath = f'97_9_sols.pkl'

    try:
        with open(dst_fpath, 'rb') as f:
            solutions = pickle.load(f)

        instances_fps = [f for f in instances_fps if f not in solutions.keys()]
    except FileNotFoundError:
        solutions = dict()
        pass

    for fpath in tqdm(instances_fps):
        instance_name = fpath.name

        instance = load_instance(str(fpath))

        jobs = list(range(instance['jobs'][0]))

        model = get_model(jobs, fpath, coupling=True, new_ineq=True, timeout=30)
        model.setParam('PoolSearchMode', 1)
        model.setParam('PoolSolutions', 5000)
        model.update()

        model.optimize()

        objs = list()
        sols = list()
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            objs.append(model.PoolObjVal)
            sol = np.array([v.xn for v in model.getVars() if ('x(' in v.VarName) or ('phi(' in v.VarName)])
            sols.append(sol)
        model.Params.SolutionNumber = 0

        objs = np.array(objs)
        sols = np.array(sols)

        # small perturbations to the feasible solutions
        perturbation = np.random.choice(2, sols.shape, p=[0.95, 0.05])
        candidate_solutions = np.abs(sols - perturbation)

        infeas_solutions = list()
        for candidate_solution in candidate_solutions:
            model_ = model.copy()
            vars_ = [v for v in model_.getVars() if ('x(' in v.VarName) or ('phi(' in v.VarName)]

            # fix variables
            for v, c_solution in zip(vars_, candidate_solution):
                v.ub = c_solution
                v.lb = c_solution

            model_.setParam('TimeLimit', 1)
            model_.update()
            model_.optimize()

            if model_.status == 3:
                infeas_solutions.append(candidate_solution)
            
            if len(infeas_solutions) >= 500:
                break

        infeas_solutions = np.array(infeas_solutions)

        infeas_solutions = infeas_solutions[
            np.random.choice(len(infeas_solutions), 500, replace=False)
        ]
        feas_solutions = sols[
            np.random.choice(len(sols), 500, replace=False)
        ]
        label = np.zeros(feas_solutions.shape[0] + infeas_solutions.shape[0])
        label[infeas_solutions.shape[0]:] = 1

        solutions[instance_name] = {
            'sol': np.vstack((feas_solutions, infeas_solutions)),
            'label': label
        }

    with open(dst_fpath, 'wb') as f:
        pickle.dump(solutions, f)
