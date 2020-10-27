import torch
from torch import nn
from torch_geometric.utils import to_dense_adj

from models.common_blocks import act_map, Pool, Unpool, GCN, RefinedGraph


class NLGCN(nn.Module):
    def __init__(self, args):
        # def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(NLGCN, self).__init__()
        self.alpha = args.alpha
        self.ks = args.ks
        self.n_att = args.n_att
        self.l_n = len(self.ks)
        self.dim_hidden = args.dim_hidden
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.activation = args.activation
        self.dropout_c = args.dropout_c
        self.dropout_n = args.dropout_n
        self.num_layers = args.num_layers
        # activation function
        self.n_act = act_map(self.activation)
        self.c_act = act_map(self.activation)
        # source gcn
        self.s_gcn = GCN(self.num_feats, self.dim_hidden, self.n_act, self.dropout_c)
        # graph U net structure
        self.bottom_gcn = GCN(self.dim_hidden, self.dim_hidden, act_map(self.activation), self.dropout_c)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.refined_pooling_graphs = nn.ModuleList()
        self.refined_unpooling_graphs = nn.ModuleList()

        self.down_gcns.append(GCN(self.num_feats, self.dim_hidden, act_map(self.activation), self.dropout_c))
        for i in range(self.l_n - 1):
            self.down_gcns.append(GCN(self.dim_hidden, self.dim_hidden, act_map(self.activation), self.dropout_c))
            self.up_gcns.append(GCN(self.dim_hidden, self.dim_hidden, act_map(self.activation), self.dropout_c))
        self.up_gcns.append(GCN(self.dim_hidden, self.num_classes, act_map('linear'), self.dropout_c))

        for i in range(self.l_n):
            self.pools.append(Pool(self.ks[i], self.dim_hidden, self.dropout_c, self.n_att))
            self.unpools.append(Unpool(self.dim_hidden, self.dim_hidden, self.dropout_c))
            self.refined_pooling_graphs.append(RefinedGraph())
            self.refined_unpooling_graphs.append(RefinedGraph())
        # out GCN
        # self.out_l_1 = nn.Linear(self.dim_hidden, self.dim_hidden)
        # self.out_l_2 = nn.Linear(self.dim_hidden, self.num_classes)
        # self.out_gcn = GCN(self.dim_hidden, self.num_classes, act_map('linear'), p=0.0)
        # self.out_drop = nn.Dropout(self.dropout_n)

        # self.gunet = GraphUNet(self.num_feats, self.dim_hidden, self.num_classes, 4)

    def forward(self, h, g):
        g = to_dense_adj(g)
        g = torch.squeeze(g)
        # h = self.s_gcn(g, h)
        h = torch.squeeze(h)

        h = self.g_unet_forward(g, h)
        # classify
        # h = self.out_drop(h)
        # h = self.out_l_1(h)
        # h = self.c_act(h)
        # h = self.out_drop(h)
        # h = self.out_gcn(g, h)
        # h = self.gunet(h, g)

        return h

    def g_unet_forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        new_gs = []
        new_hs = []

        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)

            # test
            # degrees = torch.sum(g, 1)
            # print(torch.max(degrees), torch.mean(degrees))
            # !!! The pooled graph is 1 degree

            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
            g, new_g = self.refined_pooling_graphs[i](g, h)
            new_gs.append(new_g)
            new_hs.append(h)

        h = self.bottom_gcn(g, h)

        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, idx)
            h = h.add(down_outs[up_idx])  # residual connection
            # g, new_g = self.refined_unpooling_graphs[i](g, h)
            h = self.up_gcns[i](g, h)

        return h, new_gs, new_hs, adj_ms
