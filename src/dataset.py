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

        g.nodes['var'].data['c'] = torch.from_numpy(c)
        g.nodes['con'].data['b'] = torch.from_numpy(b)

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
        g.nodes['var'].data['x'] = x

        return g, y

class VarClassDataset(GraphDataset):
    """Classification of variable within solution.
    
    Provides the problem (A,b,c) and a random candidate solution. The label is 1
    for each dimension that is equal to the optimal solution.
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
        g.nodes['var'].data['x'] = x

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

        self.gs = list()
        for recurso in self.rs.numpy():
            m = get_model(list(range(self._J)), instance, coupling=True,
                          recurso=recurso)

            self.gs.append(self.make_graph(m))

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

        soc_vars_mask = np.array(['soc' in v.getAttr(gurobipy.GRB.Attr.VarName) for v in model.getVars()])
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

        g.nodes['con'].data['b'] = torch.from_numpy(b)

        g.nodes['var'].data['c'] = torch.from_numpy(c[~soc_vars_mask])
        g.nodes['soc'].data['c'] = torch.from_numpy(c[soc_vars_mask])

        return g

    def __len__(self):
        return len(self.gs)

    def __getitem__(self, idx):
        return self.gs[idx], self.rs[idx]

