import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCNSVD, GCN, RGCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.global_attack import Random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.0,  help='pertubation rate')
parser.add_argument('--k', type=int, default=100, help='Truncated Components.')
parser.add_argument('--ptb_type', type=str, default='add', choices=['add', 'remove', 'meta'])
parser.add_argument('--cuda_num', type=int, default=0)
parser.add_argument('--model_type', type=str, default='GCN_SVD', choices=['GCN_SVD', 'RGCN', 'GCN'])

args = parser.parse_args()
device = torch.device(f"cuda:{args.cuda_num}" if args.cuda_num else "cpu")

# make sure you use the same data splits as you generated attacks

for seed in [5, 15, 20, 25, 35]:
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    # load original dataset (to get clean features and labels)
    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack', seed=15, require_mask=True)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # num_edge = adj.sum(axis=None)/2
    # attacker = Random()
    # attacker.attack(adj, n_perturbations=int(args.ptb_rate*num_edge), type=args.ptb_type)
    # perturbed_adj = attacker.modified_adj
    if args.ptb_rate > 0:
        perturbed_data = PrePtbDataset(root='/tmp/', name=args.dataset, attack_method='meta', ptb_rate=args.ptb_rate)
        perturbed_adj = perturbed_data.adj
    else: perturbed_adj = adj
    # Setup Defense Model
    if args.model_type == 'GCN_SVD':
        model = GCNSVD(nfeat=features.shape[1], nclass=labels.max()+1,
                    nhid=16, device=device)
    elif args.model_type == 'RGCN':
        model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1,
                    nhid=32, device=device)
    elif args.model_type == 'GCN':
        model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)

    # model = model.to(device)

    model.fit(features, perturbed_adj, labels, idx_train, idx_val, k=args.k, verbose=False)
    model.eval()
    output = model.test(idx_test)
    print(args.dataset, args.ptb_type, args.ptb_rate)
    if args.model_type != 'RGCN':
        print("{:.2f}".format(100*output.item()))