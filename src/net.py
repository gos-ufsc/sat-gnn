import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv


class JobGCN(nn.Module):
    def __init__(self, n_var_feats, n_con_feats, n_h_feats=10, readout_op='mean'):
        super(JobGCN, self).__init__()
        self.n_var_feats = n_var_feats
        self.n_con_feats = n_con_feats

        self.var_emb = torch.nn.Sequential(
            torch.nn.Linear(n_var_feats, n_h_feats),
            torch.nn.ReLU(),
        ).double()
        self.con_emb = torch.nn.Sequential(
            torch.nn.Linear(n_con_feats, n_h_feats),
            torch.nn.ReLU(),
        ).double()

        c1 = GraphConv(n_h_feats, n_h_feats)
        self.conv1 = HeteroGraphConv({
            'v2c': c1,  # same conv for both passes
            'c2v': c1,
        }).double()

        c2 = GraphConv(n_h_feats, n_h_feats)
        self.conv2 = HeteroGraphConv({
            'v2c': c2,  # same conv for both passes
            'c2v': c2,
        }).double()

        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_h_feats, n_h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(n_h_feats, n_h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(n_h_feats, 1),
        ).double()

        self.readout_op = readout_op

    def forward(self, g):
        var_features = torch.stack((
            g.nodes['var'].data['c'],
            g.nodes['var'].data['x'],
        )).T
        con_features = g.nodes['con'].data['b'].view(-1,self.n_con_feats)

        edge_weights = g.edata['A']

        # embbed features
        X_var = self.var_emb(var_features)
        X_con = self.con_emb(con_features)

        # var -> con
        # TODO: figure out a way to avoid applying the convs to the whole graph,
        # i.e., to ignore 'c2v' edges, for example, in this pass.
        h_con = self.conv1(g, {'con': X_con, 'var': X_var}, mod_kwargs={'edge_weights':edge_weights})['con']
        h_con = F.relu(h_con)
        h_con = self.conv2(g, {'con': h_con, 'var': X_var}, mod_kwargs={'edge_weights':edge_weights})['con']
        h_con = F.relu(h_con)

        # # reverse graph (preserving batch info)
        # batched_num_nodes = {
        #     'var': g.batch_num_nodes('var'),
        #     'con': g.batch_num_nodes('con'),
        # }
        # batched_num_edges = {
        #     'link': g.batch_num_edges()
        # }
        # g = g.reverse(copy_edata=True)
        # g.set_batch_num_nodes(batched_num_nodes)
        # g.set_batch_num_edges(batched_num_edges)

        # con -> var
        h_var = self.conv1(g, {'con': h_con, 'var': X_var}, mod_kwargs={'edge_weights':edge_weights})['var']
        h_var = F.relu(h_var)
        h_var = self.conv2(g, {'con': h_con, 'var': h_var}, mod_kwargs={'edge_weights':edge_weights})['var']
        h_var = F.relu(h_var)

        # per-node logits
        g.nodes['var'].data['logit'] = self.output(h_var)

        if self.readout_op is not None:
            return dgl.readout_nodes(g, 'logit', op=self.readout_op, ntype='var')
        else:
            return torch.stack([g_.nodes['var'].data['logit'] for g_ in dgl.unbatch(g)]).squeeze(-1)

class InstanceGCN(nn.Module):
    def __init__(self, n_var_feats, n_con_feats, n_h_feats=10, readout_op='mean'):
        super().__init__()
        self.n_var_feats = n_var_feats
        self.n_con_feats = n_con_feats

        self.soc_emb = torch.nn.Sequential(
            torch.nn.Linear(n_var_feats, n_h_feats),
            torch.nn.ReLU(),
        ).double()
        self.var_emb = torch.nn.Sequential(
            torch.nn.Linear(n_var_feats, n_h_feats),
            torch.nn.ReLU(),
        ).double()
        self.con_emb = torch.nn.Sequential(
            torch.nn.Linear(n_con_feats, n_h_feats),
            torch.nn.ReLU(),
        ).double()

        c1 = GraphConv(n_h_feats, n_h_feats)
        self.conv1 = HeteroGraphConv({
            'v2c': c1,  # same conv for both passes
            'c2v': c1,
            's2c': c1,
            'c2s': c1,
        }).double()

        c2 = GraphConv(n_h_feats, n_h_feats)
        self.conv2 = HeteroGraphConv({
            'v2c': c2,  # same conv for both passes
            'c2v': c2,
            's2c': c2,
            'c2s': c2,
        }).double()

        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_h_feats, n_h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(n_h_feats, n_h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(n_h_feats, 1),
        ).double()

        self.readout_op = readout_op

    def forward(self, g):
        var_features = g.nodes['var'].data['c'].view(-1,self.n_var_feats)
        soc_features = g.nodes['soc'].data['c'].view(-1,self.n_var_feats)
        # var_features = torch.stack((
        #     g.nodes['var'].data['c'],
        #     g.nodes['var'].data['x'],
        # )).T
        con_features = g.nodes['con'].data['b'].view(-1,self.n_con_feats)

        edge_weights = g.edata['A']

        # embbed features
        X_var = self.var_emb(var_features)
        X_soc = self.soc_emb(soc_features)
        X_con = self.con_emb(con_features)

        # var -> con
        # TODO: figure out a way to avoid applying the convs to the whole graph,
        # i.e., to ignore 'c2v' edges, for example, in this pass.
        h_con = self.conv1(g, {'con': X_con, 'var': X_var, 'soc': X_soc}, mod_kwargs={'edge_weights':edge_weights})['con']
        h_con = F.relu(h_con)
        h_con = self.conv2(g, {'con': h_con, 'var': X_var, 'soc': X_soc}, mod_kwargs={'edge_weights':edge_weights})['con']
        h_con = F.relu(h_con)

        # con -> var
        h_var = self.conv1(g, {'con': h_con, 'var': X_var, 'soc': X_soc}, mod_kwargs={'edge_weights':edge_weights})['var']
        h_var = F.relu(h_var)
        h_var = self.conv2(g, {'con': h_con, 'var': h_var, 'soc': X_soc}, mod_kwargs={'edge_weights':edge_weights})['var']
        h_var = F.relu(h_var)

        # per-node logits
        g.nodes['var'].data['logit'] = self.output(h_var)

        if self.readout_op is not None:
            return dgl.readout_nodes(g, 'logit', op=self.readout_op, ntype='var')
        else:
            return torch.stack([g_.nodes['var'].data['logit'] for g_ in dgl.unbatch(g)]).squeeze(-1)
