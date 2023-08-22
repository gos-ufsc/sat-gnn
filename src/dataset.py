from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
import pickle
from typing import List
from joblib import Parallel, delayed, cpu_count

import dgl
import gurobipy
import numpy as np
import torch
from tqdm import tqdm
from pyscipopt import Model
from dgl.data import DGLDataset

from src.problem import Instance


def make_graph_from_matrix(A, b, c):
    # create graph
    n_var = c.shape[0]
    n_con = b.shape[0]

    edges = np.indices(A.shape)  # cons -> vars

    # get only real (non-null) edges
    A_ = A.flatten()
    edges = edges.reshape(edges.shape[0],-1)
    edges = edges[:,A_ != 0]
    edges = torch.from_numpy(edges)

    edge_weights = A_[A_ != 0]

    g = dgl.heterograph({('var', 'v2c', 'con'): (edges[1], edges[0]),
                            ('con', 'c2v', 'var'): (edges[0], edges[1]),},
                            num_nodes_dict={'var': n_var, 'con': n_con,})

    g.edges['v2c'].data['A'] = torch.from_numpy(edge_weights)
    g.edges['c2v'].data['A'] = torch.from_numpy(edge_weights)

    g.nodes['var'].data['x'] = torch.from_numpy(c)
    g.nodes['con'].data['x'] = torch.from_numpy(b)

    return g

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
        for instance_fp in sorted(instances_fpaths):
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

        candidates = list()
        for x in X:
            candidate = dict(
                zip(instance.vars_names[instance.vars_names.find('x(') >= 0], x)
            )
            candidates.append(instance.add_phi_to_candidate(candidate))

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
            y_dirty = self._check_feasibility_get_y(
                dirty_candidates,
                instance,
            )

            X_random = np.random.choice([0,1], (self.n_random, X_sols.shape[1]))
            random_candidates = [dict(zip(instance.vars_names, x))
                                 for x in X_random]
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

        g.ndata['x']['var'] = torch.hstack([
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
                 name='Instance + (quasi-)Optimal', split='train',
                 return_model=False, **kwargs):
        super().__init__(instances_fpaths, sols_dir, name, split, return_model, **kwargs)

    def load_target(self, instance_fp, sols_dir):
        sol_fp = sols_dir/instance_fp.name.replace('.json', '_opt.npz')
        assert sol_fp.exists()

        sol_npz = np.load(sol_fp)
        obj, gap, runtime, sol = sol_npz['arr_0'], sol_npz['arr_1'], sol_npz['arr_2'], sol_npz['arr_3']

        return sol

class OptimalsWithZetaDataset(OptimalsDataset):
    def __init__(self, instances_fpaths,
                 sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Instance + (quasi-)Optimal + Zeta variables',
                 split='train', return_model=False, **kwargs):
        super().__init__(instances_fpaths, sols_dir, name, split, return_model,
                         **kwargs)

    def load_graph(self, instance):
        model = instance.to_gurobipy()

        # add zeta variables
        zeta = dict()
        phi = dict()
        x = dict()
        for j in range(instance['jobs']):
            for t in range(instance['T']):
                phi[j,t] = model.getVarByName("phi(%d,%d)" % (j, t))
                x[j,t] = model.getVarByName("x(%d,%d)" % (j, t))
                zeta[j,t] = model.addVar(name="zeta(%s,%s)" % (j, t), lb=0, ub=1,
                                         vtype=gurobipy.GRB.BINARY)
                if t >= 1:
                    model.addConstr(phi[j,t] - zeta[j,t] - x[j,t] + x[j,t-1] == 0)

        model.update()

        return instance.to_graph(model=model)

    def load_target(self, instance_fp, sols_dir):
        sol_fp = sols_dir/instance_fp.name.replace('.json', '_opt.npz')
        assert sol_fp.exists()

        sol_npz = np.load(sol_fp)
        _, _, _, sol = sol_npz['arr_0'], sol_npz['arr_1'], sol_npz['arr_2'], sol_npz['arr_3']

        # compute zeta target
        instance = Instance.from_file(instance_fp)

        T = instance.T
        J = instance.jobs

        s = sol.reshape((J, 2*T))
        x = s[:,:T]
        phi = s[:,T:]

        zeta = np.zeros_like(phi)
        zeta[:,1:] = phi[:,1:] - x[:,1:] + x[:,:-1]

        sol = np.hstack((sol, zeta.flatten())).astype('uint8')

        return sol

class VarOptimalityDataset(OptimalsDataset):
    def __init__(self, instances_fpaths, sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Optimality of Dimensions - Instance', split='train',
                 samples_per_instance=100, return_model=False, **kwargs):
        super().__init__(instances_fpaths, sols_dir, name, split,
                         return_model, **kwargs)

        self.samples_per_instance = int(samples_per_instance)

    def __len__(self):
        return super().__len__() * self.samples_per_instance

    def getitem(self, idx):
        i = idx // self.samples_per_instance

        # g = deepcopy(self.gs[i])
        g = self.gs[i]

        opt = torch.from_numpy(self.targets[i])

        x = torch.randint(0, 2, opt.shape)  # generate random candidate
        y = (x == opt).type(g.nodes['var'].data['x'].type())

        curr_feats = g.nodes['var'].data['x']
        g.nodes['var'].data['x'] = torch.hstack((
            # unsqueeze features dimension, if necessary
            curr_feats.view(curr_feats.shape[0],-1),
            x.view(x.shape[-1],-1),
        ))

        g.nodes['var'].data['y'] = y.to(g.ndata['x']['var'])

        try:
            m = self.models[i]
            return g, m
        except AttributeError:
            return g
