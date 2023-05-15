import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv, EGATConv, SAGEConv, GATv2Conv
from copy import deepcopy


def create_batch(g, xs):
    g_batch = list()
    for x in xs:
        g_ = deepcopy(g)
        curr_feats = g_.nodes['var'].data['x']
        g_.nodes['var'].data['x'] = torch.hstack((
            # unsqueeze features dimension, if necessary
            curr_feats.view(curr_feats.shape[0],-1),
            x.view(x.shape[-1],-1),
        ))
        g_batch.append(g_)
    return dgl.batch(g_batch)

class JobGCN(nn.Module):
    """Expects all features to be on the `x` data.
    """
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

        # downscale all weights
        def downscale_weights(module):
            if isinstance(module, torch.nn.Linear):
                module.weight.data /= 10
        self.apply(downscale_weights)

    def forward(self, g):
        var_features = g.nodes['var'].data['x'].view(-1,self.n_var_feats)
        con_features = g.nodes['con'].data['x'].view(-1,self.n_con_feats)

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
    """Expects all features to be on the `x` data.
    """
    def __init__(self, n_var_feats=7, n_con_feats=4, n_soc_feats=6, n_h_feats=64,
                 single_conv_for_both_passes=False, n_passes=1, conv1='SAGEConv',
                 conv1_kwargs={'aggregator_type': 'pool'}, conv2='SAGEConv',
                 conv2_kwargs={'aggregator_type': 'pool'}, conv3=None,
                 conv3_kwargs=dict(), readout_op=None):
        super().__init__()

        self.n_passes = n_passes
        self.n_h_feats = n_h_feats
        self.single_conv_for_both_passes = single_conv_for_both_passes

        self.n_var_feats = n_var_feats
        self.n_con_feats = n_con_feats
        self.n_soc_feats = n_soc_feats

        self.soc_emb = nn.Sequential(
            nn.Linear(n_soc_feats, n_h_feats),
            nn.ReLU(),
        ).double()
        self.var_emb = nn.Sequential(
            nn.Linear(n_var_feats, n_h_feats),
            nn.ReLU(),
        ).double()
        self.con_emb = nn.Sequential(
            nn.Linear(n_con_feats, n_h_feats),
            nn.ReLU(),
        ).double()

        self.convs = list()

        if conv1 == 'GraphConv':
            c1_forward = GraphConv(n_h_feats, n_h_feats, **conv1_kwargs)
            c1_backward = GraphConv(n_h_feats, n_h_feats, **conv1_kwargs)
        elif conv1 == 'EGATConv':
            c1_forward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                  out_node_feats=n_h_feats, out_edge_feats=1,
                                  **conv1_kwargs)
            c1_backward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                   out_node_feats=n_h_feats, out_edge_feats=1,
                                   **conv1_kwargs)
        elif conv1 == 'SAGEConv':
            c1_forward = SAGEConv(n_h_feats, n_h_feats, **conv1_kwargs)
            c1_backward = SAGEConv(n_h_feats, n_h_feats, **conv1_kwargs)
        elif conv1 == 'GATv2Conv':
            c1_forward = GATv2Conv(n_h_feats, n_h_feats, **conv1_kwargs)
            c1_backward = GATv2Conv(n_h_feats, n_h_feats, **conv1_kwargs)

        if single_conv_for_both_passes:
            c1_backward = c1_forward

        self.convs.append(HeteroGraphConv({
            'v2c': c1_forward,
            's2c': c1_forward,
            # 'c2c': c1_forward,
            'c2v': c1_backward,
            'c2s': c1_backward,
            # 'v2v': c1_backward,
            # 's2s': c1_backward,
        }).double())

        if conv2 is not None:
            if conv2 == 'GraphConv':
                c2_forward = GraphConv(n_h_feats, n_h_feats, **conv2_kwargs)
                c2_backward = GraphConv(n_h_feats, n_h_feats, **conv2_kwargs)
            elif conv2 == 'EGATConv':
                c2_forward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                    out_node_feats=n_h_feats, out_edge_feats=1,
                                    **conv2_kwargs)
                c2_backward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                    out_node_feats=n_h_feats, out_edge_feats=1,
                                    **conv2_kwargs)
            elif conv2 == 'SAGEConv':
                c2_forward = SAGEConv(n_h_feats, n_h_feats, **conv2_kwargs)
                c2_backward = SAGEConv(n_h_feats, n_h_feats, **conv2_kwargs)
            elif conv2 == 'GATv2Conv':
                c2_forward = GATv2Conv(n_h_feats, n_h_feats, **conv1_kwargs)
                c2_backward = GATv2Conv(n_h_feats, n_h_feats, **conv1_kwargs)

            if single_conv_for_both_passes:
                c2_backward = c2_forward

            self.convs.append(HeteroGraphConv({
                'v2c': c2_forward,
                's2c': c2_forward,
                # 'c2c': c2_forward,
                'c2v': c2_backward,
                'c2s': c2_backward,
                # 'v2v': c2_backward,
                # 's2s': c2_backward,
            }).double())

        if conv3 is not None:
            if conv3 == 'GraphConv':
                c3_forward = GraphConv(n_h_feats, n_h_feats, **conv3_kwargs)
                c3_backward = GraphConv(n_h_feats, n_h_feats, **conv3_kwargs)
            elif conv3 == 'EGATConv':
                c3_forward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                      out_node_feats=n_h_feats, out_edge_feats=1,
                                      **conv3_kwargs)
                c3_backward = EGATConv(in_node_feats=n_h_feats, in_edge_feats=1,
                                       out_node_feats=n_h_feats, out_edge_feats=1,
                                       **conv3_kwargs)
            elif conv3 == 'SAGEConv':
                c3_forward = SAGEConv(n_h_feats, n_h_feats, **conv3_kwargs)
                c3_backward = SAGEConv(n_h_feats, n_h_feats, **conv3_kwargs)
            elif conv3 == 'GATv2Conv':
                c3_forward = GATv2Conv(n_h_feats, n_h_feats, **conv1_kwargs)
                c3_backward = GATv2Conv(n_h_feats, n_h_feats, **conv1_kwargs)

            if single_conv_for_both_passes:
                c3_backward = c3_forward

            self.convs.append(HeteroGraphConv({
                'v2c': c3_forward,
                's2c': c3_forward,
                # 'c2c': c3_forward,
                'c2v': c3_backward,
                'c2s': c3_backward,
                # 'v2v': c3_backward,
                # 's2s': c3_backward,
            }).double())

        self.convs = nn.Sequential(*self.convs)

        self.output = nn.Sequential(
            nn.Linear(n_h_feats, n_h_feats),
            nn.ReLU(),
            nn.Linear(n_h_feats, n_h_feats),
            nn.ReLU(),
            nn.Linear(n_h_feats, 1),
        ).double()

        self.readout_op = readout_op

        # downscale all weights
        def downscale_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data /= 10
        self.apply(downscale_weights)

    def forward(self, g):
        var_features = g.nodes['var'].data['x'].view(-1,self.n_var_feats)
        soc_features = g.nodes['soc'].data['x'].view(-1,self.n_soc_feats)
        con_features = g.nodes['con'].data['x'].view(-1,self.n_con_feats)

        # edges = ['v2c', 'c2v', 's2c', 'c2s']
        # edge_weights = dict()
        # for e in edges:
        #     edge_weights[e] = (g.edges[e].data['A'].unsqueeze(-1),)
        edge_weights = g.edata['A']

        # embbed features
        h_var = self.var_emb(var_features)
        h_soc = self.soc_emb(soc_features)
        h_con = self.con_emb(con_features)

        for _ in range(self.n_passes):
            # var -> con
            for conv in self.convs:
                # TODO: figure out a way to avoid applying the convs to the
                # whole graph, i.e., to ignore 'c2v' edges, for example, in
                # this pass.
                h_con = conv(g, {'con': h_con, 'var': h_var, 'soc': h_soc},
                            #  mod_args=edge_weights)['con']
                            #  mod_args={edge_weights_key: edge_weights})['con']
                             mod_kwargs={'edge_weights': edge_weights,
                                         'efeats': edge_weights})['con']
                h_con = F.relu(h_con).view(-1, self.n_h_feats)

            # con -> var
            for conv in self.convs:
                edge_weights_key = 'edge_weights' if not isinstance(conv.mods.v2c, EGATConv) else 'efeats'
                hs = conv(g, {'con': h_con, 'var': h_var, 'soc': h_soc},
                        #   mod_args=edge_weights)
                        #   mod_args={edge_weights_key: edge_weights})
                          mod_kwargs={'edge_weights': edge_weights,
                                      'efeats': edge_weights})
                h_var = F.relu(hs['var']).view(-1, self.n_h_feats)
                h_soc = F.relu(hs['soc']).view(-1, self.n_h_feats)

        # per-node logits
        g.nodes['var'].data['logit'] = self.output(h_var)

        if self.readout_op is not None:
            return dgl.readout_nodes(g, 'logit', op=self.readout_op, ntype='var')
        else:
            return torch.stack([g_.nodes['var'].data['logit'] for g_ in dgl.unbatch(g)]).squeeze(-1)

    def get_candidate(self, g):
        return self(g)

class AttentionInstanceGCN(InstanceGCN):
    def __init__(self, n_var_feats=7, n_con_feats=4, n_soc_feats=6, n_h_feats=64, single_conv_for_both_passes=False, n_passes=1, conv1='SAGEConv', conv1_kwargs={ 'aggregator_type': 'pool' }, conv2='SAGEConv', conv2_kwargs={ 'aggregator_type': 'pool' }, conv3=None, conv3_kwargs=dict(), readout_op=None):
        super().__init__(n_var_feats, n_con_feats, n_soc_feats, n_h_feats, single_conv_for_both_passes, n_passes, conv1, conv1_kwargs, conv2, conv2_kwargs, conv3, conv3_kwargs, readout_op)

        self.in_att = nn.Sequential(
            nn.Linear(self.n_h_feats, self.n_h_feats),
            nn.ReLU(),
            nn.Linear(self.n_h_feats, self.n_h_feats),
            nn.Softmax(-1),
        )

    def forward(self, g):
        var_features = g.nodes['var'].data['x'].view(-1,self.n_var_feats)
        soc_features = g.nodes['soc'].data['x'].view(-1,self.n_soc_feats)
        con_features = g.nodes['con'].data['x'].view(-1,self.n_con_feats)

        # edges = ['v2c', 'c2v', 's2c', 'c2s']
        # edge_weights = dict()
        # for e in edges:
        #     edge_weights[e] = (g.edges[e].data['A'].unsqueeze(-1),)
        edge_weights = g.edata['A']

        # embbed features
        h_var = self.var_emb(var_features)
        h_soc = self.soc_emb(soc_features)
        h_con = self.con_emb(con_features)

        var_att = self.in_att(h_var)

        for _ in range(self.n_passes):
            # var -> con
            for conv in self.convs:
                # TODO: figure out a way to avoid applying the convs to the
                # whole graph, i.e., to ignore 'c2v' edges, for example, in
                # this pass.
                h_con = conv(g, {'con': h_con, 'var': h_var, 'soc': h_soc},
                            #  mod_args=edge_weights)['con']
                            #  mod_args={edge_weights_key: edge_weights})['con']
                             mod_kwargs={'edge_weights': edge_weights,
                                         'efeats': edge_weights})['con']
                h_con = F.relu(h_con)

            # con -> var
            for conv in self.convs:
                edge_weights_key = 'edge_weights' if not isinstance(conv.mods.v2c, EGATConv) else 'efeats'
                hs = conv(g, {'con': h_con, 'var': h_var, 'soc': h_soc},
                        #   mod_args=edge_weights)
                        #   mod_args={edge_weights_key: edge_weights})
                          mod_kwargs={'edge_weights': edge_weights,
                                      'efeats': edge_weights})
                h_var = F.relu(hs['var'])
                h_soc = F.relu(hs['soc'])

        # per-node logits
        g.nodes['var'].data['logit'] = self.output(h_var * var_att)
        # g.nodes['var'].data['logit'] = torch.bmm(var_att.unsqueeze(1), h_var.unsqueeze(2)).squeeze(2)

        if self.readout_op is not None:
            return dgl.readout_nodes(g, 'logit', op=self.readout_op, ntype='var')
        else:
            return torch.stack([g_.nodes['var'].data['logit'] for g_ in dgl.unbatch(g)]).squeeze(-1)

class VarInstanceGCN(InstanceGCN):
    def __init__(self, n_var_feats=8, n_con_feats=4, n_soc_feats=6,
                 n_h_feats=64, single_conv_for_both_passes=False, n_passes=1,
                 conv1='SAGEConv', conv1_kwargs={ 'aggregator_type': 'pool' },
                 conv2='SAGEConv', conv2_kwargs={ 'aggregator_type': 'pool' },
                 conv3=None, conv3_kwargs=dict(), readout_op=None):
        super().__init__(n_var_feats, n_con_feats, n_soc_feats, n_h_feats,
                         single_conv_for_both_passes, n_passes, conv1,
                         conv1_kwargs, conv2, conv2_kwargs, conv3, conv3_kwargs,
                         readout_op)

    def get_candidate(self, g, n=100):
        xs = torch.randint(0, 2, (n, g.num_nodes('var')))
        gs = create_batch(g, xs)

        y_hat = torch.sigmoid(self(gs))

        y_flip = 1 - y_hat
        x_hat = (xs - y_flip).abs().mean(0)  # prob. of the predicted optimal solution

        return x_hat
