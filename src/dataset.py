from copy import deepcopy

import dgl
import gurobipy
import numpy as np
import torch
from dgl.data import DGLDataset

from src.data import get_model


class GraphDataset(DGLDataset):
    def __init__(self, As, bs, cs, name='Graphs', **kwargs):
        super().__init__(name, **kwargs)

        assert len(As) == len(bs) == len(cs)

        self.gs = [self.make_graph(A, b, c) for A, b, c in zip(As, bs, cs)]

    @staticmethod
    def make_graph(A, b, c):
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

    def __len__(self):
        return len(self.gs)

    def __getitem__(self, idx):
        return deepcopy(self.gs[idx])

class SatsDataset(GraphDataset):
    """Classification of valid solutions for IP problems.
    """
    def __init__(self, As, bs, cs, resultados, objetivos,
                 name='Satisfiability of Solutions', **kwargs) -> None:
        super().__init__(As, bs, cs, name=name, **kwargs)

        assert len(As) == len(resultados) == len(objetivos)

        # parse X and y
        self._Xs = torch.from_numpy(resultados).double()
        self._ys =torch.from_numpy((objetivos >= 0).astype(float)).double()

    def __len__(self):
        return self._ys.shape[0] * self._ys.shape[1]

    def __getitem__(self, i):
        i_ = i // self._Xs.shape[1]
        j_ = i % self._Xs.shape[1]

        x = self._Xs[i_][j_]
        y = self._ys[i_][j_]

        g = deepcopy(self.gs[i_])
        curr_feats = g.nodes['var'].data['x']
        g.nodes['var'].data['x'] = torch.vstack((
            # unsqueeze batch dimension, if necessary
            curr_feats.view(-1,curr_feats.shape[-1]),
            x.view(-1,x.shape[-1]),
        ))

        return g, y

class VarClassDataset(GraphDataset):
    """Classification of variable within solution.
    
    Provides the problem (A,b,c encoded as a graph) and a random candidate
    solution. The label is 1 for each dimension that is equal to the optimal
    solution.
    """
    def __init__(self, As, bs, cs, optimals, samples_per_problem=1e3, name='Optimality of Dimensions',
                 **kwargs):
        super().__init__(As, bs, cs, name=name, **kwargs)

        assert len(optimals) == len(As)

        self._optimals = torch.from_numpy(np.array(optimals))

        self.samples_per_problem = int(samples_per_problem)

    def __len__(self):
        return super().__len__() * self.samples_per_problem

    def __getitem__(self, idx):
        i = idx // self.samples_per_problem

        opt = self._optimals[i]

        x = torch.randint(0, 2, opt.shape)  # generate random candidate
        y = (x == opt).type(x.type())

        g = super().__getitem__(i)
        curr_feats = g.nodes['var'].data['x']
        g.nodes['var'].data['x'] = torch.vstack((
            # unsqueeze batch dimension, if necessary
            curr_feats.view(-1,curr_feats.shape[-1]),
            x.view(-1,x.shape[-1]),
        ))

        return g, y

class ResourceDataset(DGLDataset):
    def __init__(self, instance, r_std=.1, n_samples=1000, name='Variable Resource', **kwargs):
        super().__init__(name, **kwargs)

        self.n_samples = n_samples

        self._T = instance['tamanho'][0]
        self._J = instance['jobs'][0]
        self.recurso_p = torch.Tensor(instance['recurso_p'][:self._T])

        r = self.recurso_p.unsqueeze(0).repeat(self.n_samples-1, 1)
        r = torch.normal(r, r_std)

        self.rs = torch.vstack((r, self.recurso_p.unsqueeze(0)))

        self.models = list()
        self.gs = list()
        for recurso in self.rs.numpy():
            m = get_model(list(range(self._J)), instance, coupling=True,
                          recurso=recurso)

            self.gs.append(self.make_graph(m))
            self.models.append(m)

    @staticmethod
    def make_graph(model):
        A = model.getA().toarray()
        b = np.array(model.getAttr('rhs'))
        c = np.array(model.getAttr('obj'))

        # get only real (non-null) edges
        A_ = A.flatten()
        edges = np.indices(A.shape)  # cons -> vars
        edges = edges.reshape(edges.shape[0],-1)
        edges = edges[:,A_ != 0]
        # edges = torch.from_numpy(edges)

        edge_weights = A_[A_ != 0]

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

        g.nodes['con'].data['x'] = torch.from_numpy(b)

        g.nodes['var'].data['x'] = torch.from_numpy(c[~soc_vars_mask])
        g.nodes['soc'].data['x'] = torch.from_numpy(c[soc_vars_mask])

        return g

    def __len__(self):
        return len(self.gs)

    def __getitem__(self, idx):
        return self.gs[idx], self.rs[idx]

class InstanceEarlyFixingDataset(DGLDataset):
    def __init__(self, instances, optimals, samples_per_problem=1000, name='Optimality of Dimensions - Instance', **kwargs):
        super().__init__(name, **kwargs)

        assert len(optimals) == len(instances)

        self._optimals = torch.from_numpy(np.array(optimals))

        self.samples_per_problem = int(samples_per_problem)

        self.models = list()
        self.gs = list()
        for instance in instances:
            jobs = list(range(instance['jobs'][0]))
            m = get_model(jobs, instance, coupling=True)

            self.gs.append(ResourceDataset.make_graph(m))
            self.models.append(m)

    def __len__(self):
        return len(self.gs) * self.samples_per_problem

    def __getitem__(self, idx):
        i = idx // self.samples_per_problem

        opt = self._optimals[i]
        g = deepcopy(self.gs[i])

        x = torch.randint(0, 2, opt.shape)  # generate random candidate
        y = (x == opt).type(x.type())

        curr_feats = g.nodes['var'].data['x']
        g.nodes['var'].data['x'] = torch.hstack((
            # unsqueeze features dimension, if necessary
            curr_feats.view(curr_feats.shape[0],-1),
            x.view(x.shape[-1],-1),
        ))

        return g, y
