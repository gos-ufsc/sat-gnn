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

class MultiTargetDataset(DGLDataset):
    def __init__(self, instances_fpaths, sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Optimality of Dimensions - Instance', split='train',
                 return_model=False, **kwargs):
        super().__init__(name, **kwargs)

        sols_dir = Path(sols_dir)
        assert sols_dir.exists()

        i_range = torch.arange(150)
        if split.lower() == 'train':
            i_range = i_range[:60]
        elif split.lower() == 'val':
            i_range = i_range[60:80]
        elif split.lower() == 'test':
            i_range = i_range[80:]

        models = list()
        self.targets = list()
        self.gs = list()
        for instance_fp in instances_fpaths:
            i = int(instance_fp.name[:-len('.json')].split('_')[-1])
            if i not in i_range:  # instance is not part of the split
                continue

            with open(instance_fp) as f:
                instance = json.load(f)

            sol_fp = sols_dir/instance_fp.name.replace('.json', '_sols.npz')
            if not sol_fp.exists():
                print('solutions were not computed for ', instance_fp)
                continue
            sol_npz = np.load(sol_fp)
            sols_objs = sol_npz['arr_0'], sol_npz['arr_1']

            m = get_model(instance, coupling=True, new_ineq=False)

            self.gs.append(make_graph_from_model(m))
            self.targets.append(sols_objs)

            models.append(get_model_scip(instance, coupling=True, new_ineq=False))
        
        if return_model:
            self.models = models
        else:
            del models

    def __len__(self):
        return len(self.gs)

    def __getitem__(self, idx):
        # g = deepcopy(self.gs[idx])
        g = self.gs[idx]

        ys = self.targets[idx]

        try:
            m = self.models[idx]
            return g, ys, m
        except AttributeError:
            return g, ys

class OptimalsDataset(DGLDataset):
    def __init__(self, instances_fpaths, sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Optimality of Dimensions - Instance', split='train',
                 return_model=False, **kwargs):
        super().__init__(name, **kwargs)

        sols_dir = Path(sols_dir)
        assert sols_dir.exists()

        i_range = torch.arange(150)
        if split.lower() == 'train':
            i_range = i_range[:60]
        elif split.lower() == 'val':
            i_range = i_range[60:80]
        elif split.lower() == 'test':
            i_range = i_range[80:]

        models = list()
        self.targets = list()
        self.gs = list()
        for instance_fp in instances_fpaths:
            i = int(instance_fp.name[:-len('.json')].split('_')[-1])
            if i not in i_range:  # instance is not part of the split
                continue

            with open(instance_fp) as f:
                instance = json.load(f)

            sol_fp = sols_dir/instance_fp.name.replace('.json', '_opt.npz')
            if not sol_fp.exists():
                print('optimum was not computed for ', instance_fp)
                continue
            sol_npz = np.load(sol_fp)
            obj, gap, runtime, sol = sol_npz['arr_0'], sol_npz['arr_1'], sol_npz['arr_2'], sol_npz['arr_3']

            m = get_model(instance, coupling=True, new_ineq=False)

            self.gs.append(make_graph_from_model(m))
            self.targets.append(sol)

            models.append(m)

        if return_model:
            self.models = models
        else:
            del models

    def __len__(self):
        return len(self.gs)

    def __getitem__(self, idx):
        # g = deepcopy(self.gs[idx])
        g = self.gs[idx]

        y = self.targets[idx]

        try:
            m = self.models[idx]
            return g, y, m
        except AttributeError:
            return g, y

class VarOptimalityDataset(OptimalsDataset):
    def __init__(self, instances_fpaths, sols_dir='/home/bruno/sat-gnn/data/interim',
                 name='Optimality of Dimensions - Instance', split='train',
                 samples_per_instance=100, return_model=False, **kwargs):
        super().__init__(instances_fpaths, sols_dir, name, split,
                         return_model, **kwargs)

        self.samples_per_instance = int(samples_per_instance)

    def __len__(self):
        return super().__len__() * self.samples_per_instance

    def __getitem__(self, idx):
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
