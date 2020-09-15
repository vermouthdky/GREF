import torch
import os
from models.GCN import GCN
from models.GAT import GAT
from models.NLGCN import NLGCN
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
from torch_geometric.datasets import Coauthor

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
import glob
from torch_geometric.utils import remove_self_loops, add_self_loops, dense_to_sparse
import numpy as np
from torch_geometric.utils import to_dense_adj, contains_isolated_nodes
from torch_sparse import spspmm, coalesce, to_scipy, from_scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.sparse
import matplotlib.gridspec as gridspec
from options.base_options import reset_weight
from sklearn import metrics
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import sklearn

import seaborn as sns

# from entropy_loss import EntropyLoss
def load_data(dataset="Cora"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, split='public', transform=T.NormalizeFeatures())[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0] #!!! 2*N 新版可能有改变
        else:
            data.edge_index = edge_index
        return data
    elif dataset in ['CoauthorCS']:
        data = Coauthor(path, 'cs', transform=T.NormalizeFeatures())[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index

        # devide training validation and testing set
        train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        val_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        test_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        train_num = 40
        val_num = 150
        for i in range(15): # number of labels
            index = (data.y == i).nonzero()[:,0]
            perm = torch.randperm(index.size(0))
            # print(index[perm[:train_num]])
            # print(perm[train_num:(train_num+val_num)])
            # print(index[perm[(train_num+val_num):]])
            train_mask[index[perm[:train_num]]] = 1
            val_mask[index[perm[train_num:(train_num+val_num)]]] = 1
            test_mask[index[perm[(train_num+val_num):]]] = 1
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data
    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')

def load_ppi_data():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset = PPI(path, split='train', transform=T.NormalizeFeatures())
    val_dataset = PPI(path, split='val', transform=T.NormalizeFeatures())
    test_dataset = PPI(path, split='test', transform=T.NormalizeFeatures())
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return [train_loader, val_loader, test_loader]

def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

class trainer(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        if self.dataset in ["Cora", "Citeseer", "Pubmed", 'CoauthorCS']:
            self.data = load_data(self.dataset)
            self.loss_fn = torch.nn.functional.nll_loss
        elif self.dataset in ['PPI']:
            self.data = load_ppi_data()
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')

        self.entropy_loss = torch.nn.functional.binary_cross_entropy_with_logits

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.grad_clip = args.grad_clip
        self.weight_decay = args.weight_decay
        self.alpha = args.alpha
        self.lamb = args.lamb
        self.num_classes = args.num_classes

        if self.type_model == 'GCN':
            self.model = GCN(args)
        elif self.type_model == 'GAT':
            self.model = GAT(args)
        elif self.type_model == 'NLGCN':
            self.model = NLGCN(args)
        else:
            raise Exception(f'the model of {self.type_model} has not been implemented')


        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            self.data.to(self.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.loss_weight = args.loss_weight  # 0.0001

    def train_net(self, epoch):
        # try:
        loss_train = self.run_trainSet(epoch)
        acc_train, acc_valid, acc_test = self.run_testSet()
        return loss_train, acc_train, acc_valid, acc_test
        

    def train(self):
        best_acc = 0
        for epoch in range(self.epochs):
            loss_train, acc_train, acc_valid, acc_test = self.train_net(epoch)
            print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, loss_train,
                                                                                         acc_valid, acc_test))
            if best_acc < acc_valid:
                best_acc = acc_valid
                self.model.cpu()
                self.save_model(self.type_model, self.dataset)
                self.model.to(self.device)

        self.log = self.load_log(type_model=self.type_model, dataset=self.dataset, load=True)
        state_dict = self.load_model(self.type_model, self.dataset)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        acc_train, acc_valid, acc_test = self.run_testSet()
        print('acc_train: {:.4f}, acc_valid: {:.4f}, acc_test:{:.4f}'.format(acc_train, acc_valid, acc_test))
        self.log['acc_train'][-1].append(acc_train)
        self.log['acc_valid'][-1].append(acc_valid)
        self.log['acc_test'][-1].append(acc_test)

        self.log['alpha'][-1].append(self.alpha)

        self.save_log(self.log, self.type_model, self.dataset)

    def run_trainSet(self, epoch):
        self.model.train()
        loss = 0.
        if self.dataset in ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS']:
            logits, adj_new= self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(logits[self.data.train_mask], 1)
            loss = self.loss_fn(logits, self.data.y[self.data.train_mask])

            # L1 regularization for matrix sparsification
            # loss += 1e-6*torch.norm(adj_new, 1)

            # label guiding loss
            mask = self.data.train_mask
            adj_new = adj_new[mask, :][:, mask]
            label = torch.zeros(len(self.data.y[mask]), self.num_classes).to(self.device)
            label = label.scatter_(1, torch.unsqueeze(self.data.y[mask], dim=1), 1)
            adj_label = torch.matmul(label, label.t())
            loss += self.lamb*self.entropy_loss(adj_new, adj_label)

            if epoch % 200 == 0:
                value_max = torch.max(adj_new).cpu()
                value_min = torch.min(adj_new).cpu()
                print(f'value_max : {value_max}')
                print(f'value_min : {value_min}')

                heat_map = sns.heatmap(adj_new.cpu().detach().numpy())
                fig = heat_map.get_figure()
                fig.savefig(self.figurename(f'adj_new{epoch}.png'))
                plt.clf()
            
            if epoch % 1000 == 0:
                heat_map = sns.heatmap(adj_label.cpu().detach().numpy())
                fig = heat_map.get_figure()
                fig.savefig(self.figurename(f'adj_label.png'))
                plt.clf()

        elif self.dataset in ['PPI']:
            for data in self.data[0]:
                num_nodes = data.x.size(0)
                # edge_index, _ = remove_self_loops(data.edge_index)
                data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)
                if isinstance(data.edge_index, tuple):
                    data.edge_index = data.edge_index[0]
                logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                loss += self.loss_fn(logits, data.y.to(self.device))
            raise Exception(f'the dataset of {self.dataset} has not been implemented')

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

    def run_testSet(self):
        self.model.eval()
        # torch.cuda.empty_cache()
        if self.dataset in ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS']:
            with torch.no_grad():
                logits, _ = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(logits, 1)
            acc_train = evaluate(logits, self.data.y, self.data.train_mask)
            acc_valid = evaluate(logits, self.data.y, self.data.val_mask)
            acc_test = evaluate(logits, self.data.y, self.data.test_mask)
            return acc_train, acc_valid, acc_test
        elif self.dataset in ['PPI']:
            accs = [0., 0., 0.]
            for i in range(1, 3):
                total_micro_f1 = 0.
                for data in self.data[i]:
                    num_nodes = data.x.size(0)
                    # edge_index, _ = remove_self_loops(data.edge_index)
                    data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)
                    if isinstance(data.edge_index, tuple):
                        data.edge_index = data.edge_index[0]
                    with torch.no_grad():
                        logits, _ = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                    pred = (logits > 0).float().cpu()
                    micro_f1 = metrics.f1_score(data.y, pred, average='micro')
                    total_micro_f1 += micro_f1
                total_micro_f1 /= len(self.data[i].dataset)
                accs[i] = total_micro_f1
            return accs[0], accs[1], accs[2]
        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')


    def load_model(self, type_model='GCN', dataset='PPI'):
        filename = self.filename(filetype='params', type_model=type_model, dataset=dataset)
        if os.path.exists(filename):
            print('load model: ', type_model, filename)
            return torch.load(filename)
        else:
            return None

    def save_model(self, type_model='GCN', dataset='PPI'):
        filename = self.filename(filetype='params', type_model=type_model, dataset=dataset)
        state = self.model.state_dict()
        torch.save(state, filename)
        print('save model to', filename)

    def filename(self, filetype='logs', type_model='GCN', dataset='PPI'):
        filedir = f'./{filetype}/{dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        alpha = str(self.model.alpha)
        num_layers = int(self.model.num_layers)

        filename = f'{filetype}_{type_model}' \
                    f'L{num_layers}Alpha{alpha}.pth.tar'

        filename = os.path.join(filedir, filename)
        return filename

    def figurename(self, figure, filetype='figures', dataset='Cora'):
        filedir = f'./{filetype}/{dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        filename = figure
        return os.path.join(filedir, filename)
    
    def load_log(self, type_model='GCN',  dataset='PPI', load=True):
        log = {}
        filename = self.filename(filetype='logs', type_model=type_model, dataset=dataset)
        if load and os.path.exists(filename) and os.path.getsize(filename):
            log = torch.load(filename)
            print('load the log from ', filename)

        if len(log) == 0:
            log['acc_train'], log['acc_valid'], log['acc_test'] = [], [], []
            log['alpha'] = []
            log['L'] = []

        for key in log.keys():
            if len(log[key]) == 0:
                log[key].append([])
            elif len(log[key][-1]) > 0:
                log[key].append([])
            else:
                continue
        return log
    
    def save_log(self, log, type_model='GCN',  dataset='PPI'):
        filename = self.filename(filetype='logs', type_model=type_model, dataset=dataset)
        torch.save(log, filename)
        print('save log to', filename)

    def plot_test_accuracy(self):
        pass
