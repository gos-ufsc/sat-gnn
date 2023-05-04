from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
import pickle
import json

import dgl
import gurobipy
import numpy as np
import torch
from dgl.data import DGLDataset

from src.problem import get_model, get_model_scip


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

def make_graph_from_model(model):
    # TODO: include variable bounds (not present in getA())
    A = model.getA().toarray()
    # TODO: include sos variable constraints
    b = np.array(model.getAttr('rhs'))
    c = np.array(model.getAttr('obj'))

    # get only real (non-null) edges
    A_ = A.flatten()
    edges = np.indices(A.shape)  # cons -> vars
    edges = edges.reshape(edges.shape[0],-1)
    edges = edges[:,A_ != 0]
    # edges = torch.from_numpy(edges)

    edge_weights = A_[A_ != 0]

    constraints_sense = np.array([ci.sense for ci in model.getConstrs()])
    constraints_sense = np.array(list(map({'>': 1, '=': 0, '<': -1}.__getitem__, constraints_sense)))

    vars_names = [v.getAttr(gurobipy.GRB.Attr.VarName) for v in model.getVars()]
    # grab all non-decision variables (everything that is not `x` or `phi`)
    soc_vars_mask = np.array([('x(' not in v) and ('phi(' not in v) for v in vars_names])
    soc_vars = np.arange(soc_vars_mask.shape[0])[soc_vars_mask]
    var_vars = np.arange(soc_vars_mask.shape[0])[~soc_vars_mask]
    soc_edges_mask = np.isin(edges.T[:,1], soc_vars)

    var_edges = edges[:,~soc_edges_mask]
    soc_edges = edges[:,soc_edges_mask]

    # translate soc/var nodes index to 0-based
    soc_edges[1] = np.array(list(map(
        dict(zip(soc_vars, np.arange(soc_vars.shape[0]))).get,
        soc_edges[1]
    )))
    var_edges[1] = np.array(list(map(
        dict(zip(var_vars, np.arange(var_vars.shape[0]))).get,
        var_edges[1]
    )))

    g = dgl.heterograph({
        ('var', 'v2c', 'con'): (var_edges[1], var_edges[0]),
        ('con', 'c2v', 'var'): (var_edges[0], var_edges[1]),
        ('soc', 's2c', 'con'): (soc_edges[1], soc_edges[0]),
        ('con', 'c2s', 'soc'): (soc_edges[0], soc_edges[1]),
    })

    soc_edge_weights = edge_weights[soc_edges_mask]
    g.edges['s2c'].data['A'] = torch.from_numpy(soc_edge_weights)
    g.edges['c2s'].data['A'] = torch.from_numpy(soc_edge_weights)

    var_edge_weights = edge_weights[~soc_edges_mask]
    g.edges['v2c'].data['A'] = torch.from_numpy(var_edge_weights)
    g.edges['c2v'].data['A'] = torch.from_numpy(var_edge_weights)

    g.nodes['con'].data['x'] = torch.from_numpy(np.stack((
        b,  # rhs
        A.mean(1),  # c_coeff
        g.in_degrees(etype='v2c').numpy() + \
        g.in_degrees(etype='s2c').numpy(),  # Nc_coeff
        constraints_sense,  # sense
    ), -1))

    g.nodes['var'].data['x'] = torch.from_numpy(np.stack((
        c[~soc_vars_mask],  # obj
        A.mean(0)[~soc_vars_mask],  # v_coeff
        g.in_degrees(etype='c2v').numpy(),  # Nv_coeff
        A.max(0)[~soc_vars_mask],  # max_coeff
        A.min(0)[~soc_vars_mask],  # min_coeff
        np.ones_like(c[~soc_vars_mask]),  # int
        np.array([float(v.rstrip(')').split(',')[-1]) / 97 for v in vars_names[:len(var_vars)]]),  # pos_emb (kind of)
    ), -1))

    g.nodes['soc'].data['x'] = torch.from_numpy(np.stack((
        c[soc_vars_mask],  # obj
        A.mean(0)[soc_vars_mask],  # v_coeff
        g.in_degrees(etype='c2s').numpy(),  # Nv_coeff
        A.max(0)[soc_vars_mask],  # max_coeff
        A.min(0)[soc_vars_mask],  # min_coeff
        np.zeros_like(c[soc_vars_mask]),  # int
    ), -1))

    return g

class LazyGraphs:
    def __init__(self, fpath):
        assert Path(fpath).exists()

        self._fpath = fpath

    def __getitem__(self, idx):
        return dgl.load_graphs(self._fpath, [idx])[0][0]

class GraphDataset(DGLDataset,ABC):
    def __init__(self, instances_fpaths,
                 sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Graph Dataset', split='train', return_model=False,
                 **kwargs):
        super().__init__(name, **kwargs)

        sols_dir = Path(sols_dir)
        assert sols_dir.exists()

        # necessary for re-splitting
        self._own_kwargs = dict(instances_fpaths=instances_fpaths, sols_dir=sols_dir, return_model=return_model, **kwargs)
        self.split = split

        self._is_initialized = False

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

            models.append(model)
            self.gs.append(graph)

        if return_model:
            self.models = models
        else:
            del models

    @abstractmethod
    def load_target(self, instance_fp, sols_dir):
        pass

    def load_instance(self, instance_fp):
        with open(instance_fp) as f:
            instance = json.load(f)

        m = get_model(instance, coupling=True, new_ineq=False)
        graph = make_graph_from_model(m)
        return m,graph
    
    def maybe_initialize(self):
        if not self._is_initialized:
            self.initialize(**self._own_kwargs)

            self._is_initialized = True

    def len(self):
        return len(self.gs)

    def getitem(self, idx):
        # g = deepcopy(self.gs[idx])
        g = self.gs[idx]

        y = self.targets[idx]

        try:
            m = self.models[idx]
            return g, y, m
        except AttributeError:
            return g, y

    def __getitem__(self, idx):
        self.maybe_initialize()
        return self.getitem(idx)

    def __len__(self):
        self.maybe_initialize()
        return self.len()

    def get_split(self, split):
        return type(self)(**self._own_kwargs, split=split)

    def save_dataset(self, fpath):
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
    def lazy_from_file(cls, fpath):
        meta_fpath = str(fpath) + '.pkl'
        with open(meta_fpath, 'rb') as f:
            meta = pickle.load(f)
        
        self = cls(**meta['kwargs'], split=meta['kwargs'])
        self.targets = meta['targets']

        try:
            self.models = meta['models']
        except KeyError:
            pass

        self.gs = LazyGraphs(fpath)

        self._is_initialized = True

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
        sols_objs = sol_npz['arr_0'], sol_npz['arr_1']

        return sols_objs

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

        g = deepcopy(self.gs[i])
        # g = self.gs[i]

        opt = torch.from_numpy(self.targets[i])

        x = torch.randint(0, 2, opt.shape)  # generate random candidate
        y = (x == opt).type(g.nodes['var'].data['x'].type())

        curr_feats = g.nodes['var'].data['x']
        g.nodes['var'].data['x'] = torch.hstack((
            # unsqueeze features dimension, if necessary
            curr_feats.view(curr_feats.shape[0],-1),
            x.view(x.shape[-1],-1),
        ))

        try:
            m = self.models[i]
            return g, y, m
        except AttributeError:
            return g, y
