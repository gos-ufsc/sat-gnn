from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
import pickle

import dgl
import gurobipy
import numpy as np
import torch
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
    def __init__(self, fpath, idxs):
        assert Path(fpath).exists()

        self._fpath = fpath
        self.idxs = idxs

    def __getitem__(self, idx):
        i = int(self.idxs[idx])
        return dgl.load_graphs(self._fpath, [i])[0][0]
    
    def __len__(self):
        return len(self.idxs)

class GraphDataset(DGLDataset,ABC):
    def __init__(self, instances_fpaths,
                 sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Graph Dataset', split='train', return_model=False,
                 **kwargs):
        super().__init__(name, **kwargs)

        sols_dir = Path(sols_dir)
        assert sols_dir.exists()

        # necessary for re-splitting
        self._own_kwargs = dict(instances_fpaths=sorted(instances_fpaths), sols_dir=sols_dir, return_model=return_model, **kwargs)
        self.split = split

        self._is_initialized = False
        self._lazy = False

    def initialize(self, instances_fpaths, sols_dir, return_model=False):
        """Populates whatever is necessary for getitem and len."""
        i_range = torch.arange(150)
        if self.split.lower() == 'train':
            i_range = i_range[:60]
        elif self.split.lower() == 'val':
            i_range = i_range[60:80]
        elif self.split.lower() == 'test':
            i_range = i_range[80:]

        models = list()
        self.targets = list()
        self.gs = list()
        for instance_fp in sorted(instances_fpaths):
            i = int(instance_fp.name[:-len('.json')].split('_')[-1])
            if i not in i_range:  # instance is not part of the split
                continue

            try:
                target = self.load_target(instance_fp, sols_dir)
            except AssertionError:
                print('optimum was not computed for ', instance_fp)
                continue

            self.targets.append(target)

            model, graph = self.load_instance(instance_fp)

            if return_model:
                models.append(model)

            self.gs.append(graph)

        if return_model:
            self.models = models
        else:
            del models

        self._is_initialized = True
        self._lazy = False

    @abstractmethod
    def load_target(self, instance_fp, sols_dir):
        pass

    def load_instance(self, instance_fp):
        instance = Instance.from_file(instance_fp)

        return instance.to_scip(), instance.to_graph()

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

    def get_split(self, split):
        if self._lazy:
            new_self = deepcopy(self)

            i_range = torch.arange(150)
            if split.lower() == 'train':
                i_range = i_range[:60]
            elif split.lower() == 'val':
                i_range = i_range[60:80]
            elif split.lower() == 'test':
                i_range = i_range[80:]

            # filter
            instances_fpaths = self._own_kwargs['instances_fpaths']
            split_mask = list()
            for ifp in instances_fpaths:
                i = int(ifp.name[:-len('.json')].split('_')[-1])
                split_mask.append(i in i_range)
            split_idxs = np.where(split_mask)[0]

            new_self.gs.idxs = split_idxs
            new_self.targets = [target for target, m in zip(self.targets, split_mask) if m]
            try:
                new_self.models = [model for model, m in zip(self.models, split_mask) if m]
            except AttributeError:
                pass

            new_self.split = split

            return new_self
        else:
            return type(self)(**self._own_kwargs, split=split)

    def save_dataset(self, fpath):
        assert self.split == 'all'

        dgl.save_graphs(fpath, self.gs)

        meta = {
            'targets': self.targets,
            'kwargs': self._own_kwargs,
            'split': self.split,
        }

        try:
            meta['models'] = self.models
        except AttributeError:
            pass

        meta_fpath = str(fpath) + '.pkl'
        with open(meta_fpath, 'wb') as f:
            pickle.dump(meta, f)

    @classmethod
    def from_file_lazy(cls, fpath, split=None):
        meta_fpath = str(fpath) + '.pkl'
        with open(meta_fpath, 'rb') as f:
            meta = pickle.load(f)

        # re-split keeping lazyness
        if split is not None:
            i_range = torch.arange(150)
            if self.split.lower() == 'train':
                i_range = i_range[:60]
            elif self.split.lower() == 'val':
                i_range = i_range[60:80]
            elif self.split.lower() == 'test':
                i_range = i_range[80:]
            
            # filter
            instances_fpaths = meta['kwargs']['instances_fpaths']
            split_mask = list()
            for ifp in instances_fpaths:
                i = int(ifp.name[:-len('.json')].split('_')[-1])
                split_mask.append(i in i_range)
            split_idxs = np.where(split_mask)[0]
        else:
            split_mask = np.ones(len(meta['targets'])).astype(bool)
            split_idxs = np.arange(len(meta['targets']))

        self = cls(**meta['kwargs'], split=meta['kwargs'])
        self.targets = [target for target, m in zip(meta['targets'], split_mask) if m]

        try:
            self.models = [model for model, m in zip(meta['models'], split_mask) if m]
        except KeyError:
            pass

        self.gs = LazyGraphs(fpath, idxs=split_idxs)

        self._is_initialized = True
        self._lazy = True

        return self

class MultiTargetDataset(GraphDataset):
    def __init__(self, instances_fpaths,
                 sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Instance + Multiple targets', split='train', return_model=False,
                 **kwargs):
        super().__init__(instances_fpaths, sols_dir, name, split, return_model,
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

    def load_instance(self, instance_fp):
        instance = Instance.from_file(instance_fp)

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

        graph = instance.to_graph(model=model)

        m = instance.to_scip()

        return m, graph

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
