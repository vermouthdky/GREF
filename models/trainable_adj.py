import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse
    
# trainable adjacent matrix
class TAdj(torch.nn.Module):
    def __init__(self, num_feats, dim_hidden, alpha, temperature):
        super(TAdj, self).__init__()
        self.num_feats = num_feats
        self.dim_hidden = dim_hidden
        self.alpha = alpha
        self.temperature = temperature
        # self.num_nodes = 
        self.W_theta = torch.nn.Linear(self.num_feats, self.dim_hidden, bias=False)
        self.W_phi = torch.nn.Linear(self.num_feats, self.dim_hidden, bias=False)
        # self.W = torch.nn.Linear(self.)
        
    def forward(self, X, adj):
        X_theta = self.W_theta(X)
        X_phi = self.W_phi(X)
        A = torch.matmul(X_theta, X_phi.t())
        A = F.softmax(A, dim=1)
        A_orig = adj
        A_orig = torch.squeeze(A_orig, dim=0)
        A = A_orig + self.alpha * A
        return A