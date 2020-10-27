import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import DenseGCNConv


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.gcn = DenseGCNConv(in_dim, out_dim, improved=True)

    def forward(self, g, h):
        h = self.drop(h)
        h = self.gcn(h, g)
        h = self.act(h)
        h = torch.squeeze(h, dim=0)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p, n_att):
        super(Pool, self).__init__()
        self.k = k
        self.n_att = n_att
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)

        self.projs = nn.ModuleList()
        for i in range(self.n_att):
            self.projs.append(nn.Linear(in_dim, 1))

        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        # TODO : denormalize
        # graph augmentation A = (A+I)^2
        num_nodes, _ = g.size()
        idx = torch.arange(num_nodes, dtype=torch.long, device=g.device)
        g[idx, idx] = 1
        g = torch.matmul(g, g)

        scores = []
        for i in range(self.n_att):
            Z = self.drop(h)
            weights = self.projs[i](Z).squeeze()
            scores.append(self.sigmoid(weights))
        score = torch.stack(scores, dim=0).sum(dim=0)

        return top_k_graph(score, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    # values, idx = torch.topk(scores, max(2, k))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    degrees = torch.unsqueeze(degrees, dim=-1)
    degrees = F.threshold(degrees, 0, 1)
    g = g / degrees
    return g


class RefinedGraph(torch.nn.Module):
    def __init__(self):
        super(RefinedGraph, self).__init__()
        self.act = act_map('softmax')

    def forward(self, g, h):
        h = F.normalize(h)
        g = norm_g(g)
        new_g = torch.matmul(h, h.t())
        # new_g = new_g - I
        num_nodes, _ = new_g.size()
        idx = torch.arange(num_nodes, dtype=torch.long, device=new_g.device)
        new_g[idx, idx] = 0
        # topk
        values, indices = torch.topk(new_g, k=5, dim=1)
        new_g = torch.zeros_like(new_g).scatter_(1, indices, values)
        new_g = norm_g(new_g)

        g = g.add(new_g)
        g = norm_g(g)
        return g, new_g


class act_map(torch.nn.Module):
    def __init__(self, act_type):
        super(act_map, self).__init__()
        if act_type == "linear":
            self.f = lambda x: x
        elif act_type == "elu":
            self.f = torch.nn.functional.elu
        elif act_type == "sigmoid":
            self.f = torch.sigmoid
        elif act_type == "tanh":
            self.f = torch.tanh
        elif act_type == "relu":
            self.f = torch.nn.functional.relu
        elif act_type == "relu6":
            self.f = torch.nn.functional.relu6
        elif act_type == "softplus":
            self.f = torch.nn.functional.softplus
        elif act_type == "leaky_relu":
            self.f = torch.nn.functional.leaky_relu
        elif act_type == 'softmax':
            self.f = torch.nn.functional.softmax
        else:
            raise Exception("wrong activate function")

    def forward(self, x):
        return self.f(x)
