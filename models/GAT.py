from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.common_blocks import act_map, batch_norm

class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()

        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.batch_normal = args.batch_normal
        self.residual = args.residual
        self.activation = args.activation
        self.dropout = args.dropout
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.layers_residual = nn.ModuleList([])
        self.type_norm = args.type_norm
        self.skip_weight = args.skip_weight
        self.num_groups = args.num_groups
        # self.num_groups = []
        # for i in range(self.num_layers):
        #     if i < 5:
        #         self.num_groups.append(20)
        #     elif i < 10:
        #         self.num_groups.append(self.num_classes)
        #     else:
        #         self.num_groups.append(3)

        if self.num_layers == 1:
            self.layers_GCN.append(GATConv(self.num_feats, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
        elif self.num_layers == 2:
            self.layers_GCN.append(GATConv(self.num_feats, self.dim_hidden, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
            self.layers_GCN.append(GATConv(self.dim_hidden, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
        else:
            self.layers_GCN.append(GATConv(self.num_feats, self.dim_hidden,heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
            for _ in range(self.num_layers - 2):
                self.layers_GCN.append(GATConv(self.dim_hidden, self.dim_hidden, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))
            self.layers_GCN.append(GATConv(self.dim_hidden, self.num_classes, heads=1, concat=True, dropout=self.dropout,
                                           bias=False))

        for i in range(self.num_layers):
            self.layers_activation.append(act_map(self.activation))
            dim_in = self.layers_GCN[i].in_channels
            dim_out = self.layers_GCN[i].out_channels
            if self.type_norm in ['None', 'batch', 'pair', 'unSkipGroup']:
                skip_connect = False
            else:
                skip_connect = True
            self.layers_bn.append(batch_norm(dim_out, self.type_norm, skip_connect, self.num_groups, self.skip_weight))
            if self.residual:
                self.layers_residual.append(nn.Linear(dim_in, dim_out, bias=False))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers-1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x_conv = self.layers_GCN[i](x, edge_index)
            x_conv = self.layers_bn[i](x_conv)
            x_conv = self.layers_activation[i](x_conv)
            if self.residual:
                x = x_conv + self.layers_residual[i](x)
            else:
                x = x_conv
            # if i == self.num_layers - 1:
            #     print(self.layers_bn[i].bn.running_var.view(-1, 7))
        return x