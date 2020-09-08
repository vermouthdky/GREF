import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse
from models.common_blocks import act_map
# trainable adjacent matrix

class TAdj(nn.Module):
    def __init__(self, num_feats, dim_hidden, alpha, temperature, threshold):
        super(TAdj, self).__init__()
        self.num_feats = num_feats
        self.dim_hidden = dim_hidden
        self.alpha = alpha
        self.temperature = temperature
        self.threshold = threshold

        self.W_theta = nn.Linear(self.num_feats, self.dim_hidden, bias=False)
        self.W_phi = nn.Linear(self.num_feats, self.dim_hidden, bias=False)
        self.activation = act_map('relu')
        
    def forward(self, X, adj):

        X_theta = self.W_theta(X)
        X_theta = self.activation(X_theta)

        X_phi = self.W_phi(X)
        X_phi = self.activation(X_phi)

        A = torch.matmul(X_theta, X_phi.t())

        # self.num_nodes = X_theta.size(dim=0)
        # A = X_theta + X_phi.t()

        A = A/self.temperature

        # A = F.tanh(A/self.temperature)
        # A = F.threshold(A, self.threshold, 0.)
        
        # A = F.softmax(A/self.temperature, dim=0)

        A_orig = adj
        A_orig = torch.squeeze(A_orig, dim=0)
        P = A_orig + self.alpha * A

        return P, A