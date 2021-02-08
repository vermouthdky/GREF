from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.common_blocks import act_map
from torch_geometric.nn.inits import glorot, zeros
import torch
import ipdb
class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = 2
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.activation = args.activation
        # self.dropout = args.dropout
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.alpha = args.alpha
        if self.num_layers == 1:
            self.layers_GCN.append(GCNConv(self.num_feats, self.num_classes, bias=False))
        elif self.num_layers == 2:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, bias=False))
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, bias=False))
        else:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, bias=False))
            for _ in range(self.num_layers-2):
                self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, bias=False))
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, bias=False))


        for i in range(self.num_layers):
            self.layers_activation.append(act_map(self.activation))

    def forward(self, x, edge_index):

        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers-1:
                x = F.dropout(x, p=0.0, training=self.training)
            x_conv = self.layers_GCN[i](x, edge_index)
            x_conv = self.layers_activation[i](x_conv)
            x = x_conv
        x = torch.squeeze(x)

        return x







