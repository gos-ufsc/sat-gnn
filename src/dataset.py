import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import List

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from joblib import Parallel, cpu_count, delayed
from pyscipopt import Model
from tqdm import tqdm

from src.problem import Instance


class LazyGraphs:
    def __init__(self, fpath):
        assert Path(fpath).exists()

        self._fpath = fpath

    def __getitem__(self, idx):
        return dgl.load_graphs(self._fpath, [idx])[0][0]

    def __len__(self):
        with open(self._fpath+'.pkl', 'rb') as f:
            m = pickle.load(f)
        return len(m['targets'])

class GraphDataset(DGLDataset,ABC):
    def __init__(self, instances_fpaths,
                 sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Graph Dataset', return_model=False,
                 **kwargs):
        super().__init__(name, **kwargs)

        sols_dir = Path(sols_dir)
        assert sols_dir.exists()

        # necessary for re-splitting
        self._own_kwargs = dict(instances_fpaths=sorted(instances_fpaths),
                                sols_dir=sols_dir, return_model=return_model,
                                **kwargs)

        self._is_initialized = False
        self._lazy = False

    def initialize(self, instances_fpaths, sols_dir, return_model=False):
        """Populates whatever is necessary for getitem and len."""
        models = list()
        self.targets = list()
        self.gs = list()
        for instance_fp in tqdm(sorted(instances_fpaths)):
            try:
                target = self.load_target(instance_fp, sols_dir)
            except AssertionError:
                print('optimum was not computed for ', instance_fp)
                continue

            self.targets.append(target)

            instance = Instance.from_file(instance_fp)

            self.gs.append(self.load_graph(instance))

            if return_model:
                models.append(instance.to_scip())

        if return_model:
            self.models = models
        else:
            del models

        self._is_initialized = True
        self._lazy = False

    @abstractmethod
    def load_target(self, instance_fp, sols_dir):
        pass

    def load_graph(self, instance: Instance):
        return instance.to_graph()

    def maybe_initialize(self):
        if not self._is_initialized:
            self.initialize(**self._own_kwargs)

    def len(self):
        return len(self.gs)

    def getitem(self, idx):
        # g = deepcopy(self.gs[idx])
        g = self.gs[idx]

        y = self.targets[idx]

        g.nodes['var'].data['y'] = torch.from_numpy(y).to(g.ndata['x']['var'])

        try:
            m = self.models[idx]
            return g, m
        except AttributeError:
            return g

    def __getitem__(self, idx):
        self.maybe_initialize()
        return self.getitem(idx)

    def __len__(self):
        self.maybe_initialize()
        return self.len()

    def save_dataset(self, fpath):
        self.maybe_initialize()
        dgl.save_graphs(fpath, self.gs)

        meta = self._get_metadata_for_saving()

        try:
            meta['models'] = self.models
        except AttributeError:
            pass

        meta_fpath = str(fpath) + '.pkl'
        with open(meta_fpath, 'wb') as f:
            pickle.dump(meta, f)

    def _get_metadata_for_saving(self):
        meta = {
            'targets': self.targets,
            'kwargs': self._own_kwargs,
        }
        
        return meta
    
    @classmethod
    def _make_self_from_meta(cls, meta):
        self = cls(**meta['kwargs'])

        for k, v in meta.items():
            if k != 'kwargs':
                setattr(self, k, v)

        return self

    @classmethod
    def from_file_lazy(cls, fpath):
        meta_fpath = str(fpath) + '.pkl'
        with open(meta_fpath, 'rb') as f:
            meta = pickle.load(f)

        self = cls._make_self_from_meta(meta)

        self.gs = LazyGraphs(fpath)

        self._is_initialized = True
        self._lazy = True

        return self

class SolutionFeasibilityDataset(GraphDataset):
    def __init__(self, instances_fpaths,
                 sols_dir='/home/bruno/sat-gnn/data/interim', n_random=250,
                 n_dirty=250, noise_size=2, name='Solution Feasibility Dataset',
                 return_model=False, skip_feasibility_check=False, **kwargs):
        super().__init__(instances_fpaths, sols_dir, name, return_model, **kwargs)

        self.n_random = n_random
        self.n_dirty = n_dirty
        self.noise_size = noise_size
        self.skip_feasibility_check = skip_feasibility_check

    def len(self):
        return len(self.idx_gs)

    def load_target(self, instance_fp, sols_dir):
        sol_fp = sols_dir/instance_fp.name.replace('.json', '_sols.npz')

        sol_npz = np.load(sol_fp)
        sols = np.rint(sol_npz['arr_0']).astype('uint8')

        return sols

    def _dirty_candidates_from_sols(self, X_sols: np.ndarray, instance: Instance):
        X = X_sols[np.random.choice(np.arange(X_sols.shape[0]), self.n_dirty, replace=True)]

        X[:,instance.vars_names.find('x(') >= 0] # add noise only to `x_jt` variables

        if self.noise_size == 1:
            noise = np.eye(X.shape[1])[np.random.choice(np.arange(X.shape[1]))]
        else:
            p = self.noise_size / X.shape[1]
            noise = np.random.choice([False, True], size=X.shape, p=[1-p, p])

        X = np.where(noise, 1 - X, X)  # dirty X

        def get_candidate_from_x(x, instance):
            candidate = dict(
                zip(instance.vars_names[instance.vars_names.find('x(') >= 0], x)
            )
            return instance.add_phi_to_candidate(candidate)
        candidates = Parallel(n_jobs=cpu_count()-1)(delayed(get_candidate_from_x)(x, instance) for x in X)
        candidates = list(candidates)

        return candidates

    def _check_feasibility_get_y(self, candidates: List[dict],
                                  instance: Instance):
        src_model = instance.to_scip(coupling=True, new_inequalities=True,
                                     enable_primal_dual_integral=False)
        src_model.setObjective(1, "maximize")
        src_model.hideOutput()

        model_fpath = 'model.cip'
        src_model.writeProblem(model_fpath)

        del src_model

        def is_candidate_feasible(candidate):
            model = Model()
            model.hideOutput()
            model.readProblem('model.cip')

            for var in model.getVars():
                try:
                    value = candidate[var.name]
                    model.fixVar(var, value)
                except KeyError:
                    pass

            model.optimize()

            return int(model.getStatus().lower() == 'optimal')

        y = Parallel(n_jobs=cpu_count()-1)(delayed(is_candidate_feasible)(candidate) for candidate in candidates)
        y = list(y)

        return np.array(y)

    def initialize(self, instances_fpaths, sols_dir, return_model=False):
        """Populates whatever is necessary for getitem and len."""
        models = list()
        self.targets = list()
        self.inputs = list()
        self.idx_gs = list()
        self.gs = list()
        for instance_fp in tqdm(sorted(instances_fpaths)):
            instance = Instance.from_file(instance_fp)

            try:
                X_sols = self.load_target(instance_fp, sols_dir)
            except AssertionError:
                print('solutions were not computed for ', instance_fp)
                continue
            y_sols = np.ones(len(X_sols), dtype='uint8')

            dirty_candidates = self._dirty_candidates_from_sols(
                X_sols,
                instance
            )
            X_dirty = np.stack([np.array([candidate[v] for v in instance.vars_names])
                                for candidate in dirty_candidates])
            if self.skip_feasibility_check:
                y_dirty = np.zeros(X_dirty.shape[0])
            else:
                y_dirty = self._check_feasibility_get_y(
                    dirty_candidates,
                    instance,
                )

            X_random = np.random.choice([0,1], (self.n_random, X_sols.shape[1]))
            random_candidates = [dict(zip(instance.vars_names, x))
                                 for x in X_random]
            if self.skip_feasibility_check:
                y_random = np.zeros(X_random.shape[0])
            else:
                y_random = self._check_feasibility_get_y(
                    random_candidates,
                    instance,
                )

            X = np.vstack([X_sols, X_dirty, X_random])
            y = np.hstack([y_sols, y_dirty, y_random])

            # multiple references to the same graph
            self.idx_gs += [len(self.gs),] * X.shape[0]
            # for _ in range(X.shape[0]):
            #     self.idx_gs.append(len(self.gs))
            self.gs.append(self.load_graph(instance))

            self.targets += list(y)
            self.inputs += list(X)

            if return_model:
                models.append(instance.to_scip())

        if return_model:
            self.models = models
        else:
            del models

        self._is_initialized = True
        self._lazy = False

    def getitem(self, idx):
        g = self.gs[self.idx_gs[idx]]
        x = self.inputs[idx]
        y = self.targets[idx]

        g = self.add_input_to_graph(g, x)

        try:
            m = self.models[idx]
            return g, y, m
        except AttributeError:
            return g, y

    def add_input_to_graph(self, g, x):
        if not self._lazy:
            g = deepcopy(g)

        g.nodes['var'].data['x'] = torch.hstack([
            g.ndata['x']['var'],
            torch.from_numpy(x).to(g.ndata['x']['var']).unsqueeze(1)
        ])

        return g

    def _get_metadata_for_saving(self):
        meta = {
            'targets': self.targets,
            'inputs': self.inputs,
            'idx_gs': self.idx_gs,
            'kwargs': self._own_kwargs,
        }

        return meta

class MultiTargetDataset(GraphDataset):
    def __init__(self, instances_fpaths,
                 sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Instance + Multiple targets', return_model=False,
                 **kwargs):
        super().__init__(instances_fpaths, sols_dir, name, return_model,
                         **kwargs)

    def load_target(self, instance_fp, sols_dir):
        sol_fp = sols_dir/instance_fp.name.replace('.json', '_sols.npz')
        assert sol_fp.exists()

        sol_npz = np.load(sol_fp)
        sols_objs = np.rint(sol_npz['arr_0']).astype('uint8'), sol_npz['arr_1'].astype('uint32')

        return sols_objs

    def getitem(self, idx):
        # g = deepcopy(self.gs[idx])
        g = self.gs[idx]

        y, w = self.targets[idx]

        y = torch.from_numpy(y).T
        w = torch.from_numpy(w.astype(int)).unsqueeze(0)

        g.nodes['var'].data['y'] = y.to(g.ndata['x']['var'])
        g.nodes['var'].data['w'] = w.repeat(*(np.array(y.shape) // np.array(w.shape)))

        try:
            m = self.models[idx]
            return g, m
        except AttributeError:
            return g

class OptimalsDataset(GraphDataset):
    def __init__(self, instances_fpaths,
                 sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Instance + (quasi-)Optimal', return_model=False,
                 **kwargs):
        super().__init__(instances_fpaths, sols_dir, name, return_model, **kwargs)

    def load_target(self, instance_fp, sols_dir):
        sol_fp = sols_dir/instance_fp.name.replace('.json', '_opt.npz')
        assert sol_fp.exists()

        sol_npz = np.load(sol_fp)
        obj, gap, runtime, sol = sol_npz['arr_0'], sol_npz['arr_1'], sol_npz['arr_2'], sol_npz['arr_3']

        return sol
