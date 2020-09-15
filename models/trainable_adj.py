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

        self.W_theta = nn.Linear(self.num_feats, self.dim_hidden, bias=True)
        self.W_phi = nn.Linear(self.num_feats, self.dim_hidden, bias=True)
        self.activation = act_map('leaky_relu')
        
    def forward(self, X, adj):

        X_theta = self.W_theta(X)
        X_theta = self.activation(X_theta)

        # X_phi = self.W_phi(X)
        # X_phi = self.activation(X_phi)

        A = torch.matmul(X_theta, X_theta.t())
        # A = X_theta + X_phi.t()

        A = F.tanh(A)
        A_new = F.threshold(A, self.threshold, 0.)

        # random selection with a mask
        # num_nodes = A.size(0)
        # mask = torch.cuda.FloatTensor(num_nodes, num_nodes).uniform_() > 0.6
        # A_new = A*mask

        A_orig = adj
        A_orig = torch.squeeze(A_orig, dim=0)
        P = A_orig + self.alpha * A_new

        return P, A