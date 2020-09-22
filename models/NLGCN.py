from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, GCNConv
from models.common_blocks import act_map
from torch_geometric.nn.inits import glorot, zeros
from models.trainable_adj import TAdj
import torch
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse, dropout_adj
import torch.nn.functional as F

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
        self.temperature = args.temperature
        self.threshold = args.threshold
        self.freezed = args.freezed
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_DenseGCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layer_linear = nn.Linear(self.num_feats, self.dim_hidden)
        self.layers_TAdj = nn.ModuleList([])

        self.TAdj = TAdj(self.num_feats, self.dim_hidden, self.alpha, self.temperature, self.threshold, self.training)
        
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
                
        for i in range(self.num_layers):
            self.layers_activation.append(act_map(self.activation))
            
    def forward(self, x, edge_index):

        adj = to_dense_adj(edge_index)
        adj = torch.squeeze(adj, dim=0)

        for i in range(self.num_layers):   

            if i == 0 or i == self.num_layers-1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.layers_DenseGCN[i](x, adj)
            x = self.layers_activation[i](x)
            x = torch.squeeze(x, dim=0)

        adj_new = F.tanh(torch.matmul(x, x.t()))

        # get the relations between supernodes and nodes
        # adj_super = F.softmax(x, dim=1) # N x num_classes
        # adj_new_upper = torch.cat((adj, adj_super), dim=1)
        # adj_new_lower = torch.cat((adj_super.t(), torch.zeros(self.num_classes, self.num_classes)), dim=1)
        # adj_new = torch.cat((adj_new_upper, adj_new_lower), dim=0) 
            
        return x, adj_new







