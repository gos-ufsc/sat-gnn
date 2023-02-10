from copy import deepcopy

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset


class SatsDataset(DGLDataset):
    def __init__(self, As, bs, cs, resultados, objetivos) -> None:
        super().__init__('test')

        assert len(As) == len(bs) == len(cs) == len(resultados) == len(objetivos)

        self.gs = [self.make_graph(A, b, c) for A, b, c in zip(As, bs, cs)]

        # parse X and y
        self._Xs = torch.from_numpy(resultados).double()
        self._ys =torch.from_numpy((objetivos >= 0).astype(float)).double()

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
        return self._ys.shape[0] * self._ys.shape[1]

    def __getitem__(self, i):
        i_ = i // 1000
        j_ = i % 1000

        x = self._Xs[i_][j_]
        y = self._ys[i_][j_]

        g = deepcopy(self.gs[i_])
        g.nodes['var'].data['x'] = x

        return g, y
