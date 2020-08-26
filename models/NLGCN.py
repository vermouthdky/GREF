from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, GCNConv
from models.common_blocks import act_map
from torch_geometric.nn.inits import glorot, zeros
from models.trainable_adj import TAdj
import torch

class NLGCN(nn.Module):
    def __init__(self, args):
        super(NLGCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.activation = args.activation
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.threshold = args.threshold
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_DenseGCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layers_residual_dense = nn.ModuleList([])
        
        self.TAdj = TAdj(self.num_feats, self.dim_hidden, self.alpha, self.threshold)
        if self.num_layers == 1:
            self.layers_DenseGCN.append(DenseGCNConv(self.num_feats, self.num_classes, improved=True, bias=False))
        elif self.num_layers == 2:
            self.layers_DenseGCN.append(DenseGCNConv(self.num_feats, self.dim_hidden, improved=True, bias=False))
            self.layers_DenseGCN.append(DenseGCNConv(self.dim_hidden, self.num_classes, improved=True, bias=False))
        else:
            self.layers_DenseGCN.append(DenseGCNConv(self.num_feats, self.dim_hidden, improved=True, bias=False))
            for _ in range(self.num_layers-2):
                self.layers_DenseGCN.append(DenseGCNConv(self.dim_hidden, self.dim_hidden, improved=True, bias=False))
            self.layers_DenseGCN.append(DenseGCNConv(self.dim_hidden, self.num_classes, improved=True, bias=False))

        # if self.num_layers == 1:
        #     self.layers_GCN.append(GCNConv(self.num_feats, self.num_classes, cached=self.cached, bias=False))
        # elif self.num_layers == 2:
        #     self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
        #     self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))
        # else:
        #     self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
        #     for _ in range(self.num_layers-2):
        #         self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached, bias=False))
        #     self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))

        
        for i in range(self.num_layers):
            self.layers_activation.append(act_map(self.activation))

    def forward(self, x, edge_index):

        adj = self.TAdj(x, edge_index)
        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers-1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            # x_conv = self.layers_GCN[i](x, edge_index)
            # x = self.layers_activation[i](x_conv)
            x_nonlocal = self.layers_DenseGCN[i](x, adj)
            x = self.layers_activation[i](x_nonlocal)
        x = torch.squeeze(x)
        return x







