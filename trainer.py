import os
#
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn import metrics
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import PPI
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops, add_self_loops, dense_to_sparse

from models.GAT import GAT
from models.GCN import GCN
from models.NLGCN import NLGCN
from models.g_U_Net import gunet
from models.JKNet import JKNetMaxpool
from models.APPNP import APPNP
from models.simpleGCN import simpleGCN

import wandb
# dataset attacking defense
from deeprobust.graph.data import PrePtbDataset, Dataset
from deeprobust.graph import utils
from deeprobust.graph.global_attack import Random
import numpy as np

import ipdb

def load_perterbued_data(dataset, ptb_rate, ptb_type="meta"):
    if ptb_type == 'meta':
        data = Dataset(root='/tmp/', name=dataset.lower(), setting='nettack', seed=15, require_mask=True)
        data.x, data.y = data.features, data.labels
        if ptb_rate > 0:
            perturbed_data = PrePtbDataset(root='/tmp/', name=dataset.lower(), attack_method='meta', ptb_rate=ptb_rate)
            data.edge_index = perturbed_data.adj
        else:
            data.edge_index = data.adj
        return data

    elif ptb_type == 'random_add':
        data = Dataset(root='/tmp/', name=dataset.lower(), setting='nettack', seed=15, require_mask=True)
        data.x, data.y = data.features, data.labels
        num_edge = data.adj.sum(axis=None)/2
        attacker = Random()
        attacker.attack(data.adj, n_perturbations=int(ptb_rate*num_edge), type='add')
        data.edge_index = attacker.modified_adj
        return data

    elif ptb_type == 'random_remove':
        data = Dataset(root='/tmp/', name=dataset.lower(), setting='nettack', seed=15, require_mask=True)
        data.x, data.y = data.features, data.labels
        num_edge = data.adj.sum(axis=None)/2
        attacker = Random()
        attacker.attack(data.adj, n_perturbations=int(ptb_rate*num_edge), type='remove')
        data.edge_index = attacker.modified_adj
        return data
    
    raise Exception(f"the ptb_type of {ptb_type} has not been implemented")

def load_data(dataset="Cora"):
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "data", dataset
    )
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(
            path, dataset, split="public", transform=T.NormalizeFeatures()
        )[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]  # !!! 2*N 新版可能有改变
        else:
            data.edge_index = edge_index
        return data
    elif dataset in ["CoauthorCS"]:
        data = Coauthor(path, "cs", transform=T.NormalizeFeatures())[0]
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
        for i in range(15):  # number of labels
            index = (data.y == i).nonzero()[:, 0]
            perm = torch.randperm(index.size(0))
            train_mask[index[perm[:train_num]]] = 1
            val_mask[index[perm[train_num: (train_num + val_num)]]] = 1
            test_mask[index[perm[(train_num + val_num):]]] = 1
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data
    else:
        raise Exception(f"the dataset of {dataset} has not been implemented")


def load_ppi_data():
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "data", "PPI"
    )
    train_dataset = PPI(path, split="train", transform=T.NormalizeFeatures())
    val_dataset = PPI(path, split="val", transform=T.NormalizeFeatures())
    test_dataset = PPI(path, split="test", transform=T.NormalizeFeatures())
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
        self.device = torch.device(
            f"cuda:{args.cuda_num}" if args.cuda else "cpu")
        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            if args.ptb:
                self.data = load_perterbued_data(self.dataset, args.ptb_rate, args.ptb_type)
                self.loss_fn = torch.nn.functional.nll_loss
            else:
                self.data = load_data(self.dataset)
                self.loss_fn = torch.nn.functional.nll_loss
        elif self.dataset in ["PPI"]:
            self.data = load_ppi_data()
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise Exception(
                f"the dataset of {self.dataset} has not been implemented")

        self.entropy_loss = torch.nn.functional.binary_cross_entropy_with_logits

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.beta = args.beta
        self.lamb = args.lamb
        self.num_classes = args.num_classes
        self.ptb_rate = args.ptb_rate
        self.ptb_type = args.ptb_type
        self.metric = args.metric
        self.num_layers = args.num_layers

        if self.type_model == "GCN":
            self.model = GCN(args)
        elif self.type_model == "GAT":
            self.model = GAT(args)
        elif self.type_model == "NLGCN":
            self.model = NLGCN(args)
        elif self.type_model == "g_U_Net":
            self.model = gunet(args)
        elif self.type_model == "JKNet":
            self.model = JKNetMaxpool(args)
        elif self.type_model == "SGC":
            self.model = simpleGCN(args)
        elif self.type_model == "APPNP":
            self.model = APPNP(args)
        else:
            raise Exception(
                f"the model of {self.type_model} has not been implemented")

        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            if args.ptb:
                self.data.edge_index, self.data.x, self.data.y = utils.preprocess(self.data.edge_index, self.data.x, self.data.y, preprocess_adj=False, sparse=False, device=self.device)

            else:
                self.data.to(self.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        wandb.init(project="Gref", config=args)
        wandb.watch(self.model)
    def train_net(self, epoch):
        # try:
        loss_train = self.run_trainSet(epoch)
        acc_train, acc_valid, acc_test = self.run_testSet()
        return loss_train, acc_train, acc_valid, acc_test

    def train(self):
        best_acc = 0
        the_loss = 999
        for epoch in range(self.epochs):
            loss_train, acc_train, acc_valid, acc_test = self.train_net(epoch)
            print("Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}".format(
                epoch, loss_train, acc_valid, acc_test))

            wandb.log({"epoch": epoch, 'loss':loss_train, 'val_acc':acc_valid, 'test_acc':acc_test})

            # if self.dataset == 'Pubmed':
            #     if best_acc < acc_valid and loss_train < 0.2:
            #         best_acc = acc_valid
            #         the_loss = loss_train
            #         self.model.cpu()
            #         self.save_model(self.type_model, self.dataset)
            #         self.model.to(self.device)
            # else:
            if best_acc < acc_valid:
                best_acc = acc_valid
                the_loss = loss_train
                self.model.cpu()
                self.save_model(self.type_model, self.dataset)
                self.model.to(self.device)

        self.log = self.load_log(
            type_model=self.type_model, dataset=self.dataset, load=True)
        state_dict = self.load_model(self.type_model, self.dataset)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        acc_train, acc_valid, acc_test = self.run_testSet()
        print(
            "acc_train: {:.4f}, acc_valid: {:.4f}, acc_test:{:.4f}".format(
                acc_train, acc_valid, acc_test
            )
        )
        self.log["acc_train"][-1].append(acc_train)
        self.log["acc_valid"][-1].append(acc_valid)
        self.log["acc_test"][-1].append(acc_test)

        self.log["ptb"][-1].append(self.ptb_rate)

        self.save_log(self.log, self.type_model, self.dataset)

    def run_trainSet(self, epoch):
        self.model.train()
        loss = 0.0
        # ipdb.set_trace()
        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            if self.type_model == "NLGCN":
                logits, new_gs, new_hs, gs = self.model(self.data.x, self.data.edge_index)
                logits = F.log_softmax(logits[self.data.train_mask], 1)
                
                loss = self.loss_fn(logits, self.data.y[self.data.train_mask])

                # graph regularization
                for new_h, new_g in zip(new_hs, new_gs):
                    n = new_h.size()[0]
                    d = new_h.size()[1]
                    D = torch.diag(torch.sum(new_g, dim=1))
                    # deg_inv_sqrt = new_g.sum(dim=-1).clamp(min=1).pow(-0.5)
                    # L = deg_inv_sqrt.unsqueeze(-1) * (D-new_g) * deg_inv_sqrt.unsqueeze(-2)
                    L = D - new_g
                    loss += self.lamb / (d ^ 2) * torch.trace(
                        torch.chain_matmul(new_h.t(), L, new_h))
                    # loss += -self.beta * torch.sum(torch.log(torch.sum(new_g, dim=1))) / n
                    loss += self.gamma * torch.norm(new_g) / (n ^ 2)
                    # loss += self.beta * torch.norm(new_g, p=1) / (n ^ 2)

            else:
                logits = self.model(self.data.x, self.data.edge_index)
                logits = F.log_softmax(logits[self.data.train_mask], 1)
                loss = self.loss_fn(logits, self.data.y[self.data.train_mask])

        elif self.dataset in ["PPI"]:
            for data in self.data[0]:
                num_nodes = data.x.size(0)
                # edge_index, _ = remove_self_loops(data.edge_index)
                data.edge_index = add_self_loops(
                    data.edge_index, num_nodes=num_nodes)
                if isinstance(data.edge_index, tuple):
                    data.edge_index = data.edge_index[0]
                logits, new_gs, new_hs, gs = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                loss += self.loss_fn(logits, data.y.to(self.device))

                # graph regularization
                for new_h, new_g in zip(new_hs, new_gs):
                    n = new_h.size()[0]
                    d = new_h.size()[1]
                    D = torch.diag(torch.sum(new_g, dim=1))
                    # deg_inv_sqrt = new_g.sum(dim=-1).clamp(min=1).pow(-0.5)
                    # L = deg_inv_sqrt.unsqueeze(-1) * (D-new_g) * deg_inv_sqrt.unsqueeze(-2)
                    L = D - new_g
                    loss += self.lamb / (d ^ 2) * torch.trace(
                        torch.chain_matmul(new_h.t(), L, new_h))
                    # loss += -self.beta * torch.sum(torch.log(torch.sum(new_g, dim=1))) / n
                    loss += self.gamma * torch.norm(new_g) / (n ^ 2)
        # raise Exception(f"the dataset of {self.dataset} has not been implemented")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_testSet(self):
        self.model.eval()
        # torch.cuda.empty_cache()
        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            if self.type_model == 'NLGCN':
                with torch.no_grad():
                    logits, _, _, _ = self.model(self.data.x, self.data.edge_index)
                logits = F.log_softmax(logits, 1)
                acc_train = evaluate(logits, self.data.y, self.data.train_mask)
                acc_valid = evaluate(logits, self.data.y, self.data.val_mask)
                acc_test = evaluate(logits, self.data.y, self.data.test_mask)
            else:
                with torch.no_grad():
                    logits = self.model(self.data.x, self.data.edge_index)
                logits = F.log_softmax(logits, 1)
                acc_train = evaluate(logits, self.data.y, self.data.train_mask)
                acc_valid = evaluate(logits, self.data.y, self.data.val_mask)
                acc_test = evaluate(logits, self.data.y, self.data.test_mask)
            return acc_train, acc_valid, acc_test
        elif self.dataset in ["PPI"]:
            accs = [0.0, 0.0, 0.0]
            for i in range(1, 3):
                total_micro_f1 = 0.0
                for data in self.data[i]:
                    num_nodes = data.x.size(0)
                    # edge_index, _ = remove_self_loops(data.edge_index)
                    data.edge_index = add_self_loops(
                        data.edge_index, num_nodes=num_nodes
                    )
                    if isinstance(data.edge_index, tuple):
                        data.edge_index = data.edge_index[0]
                    with torch.no_grad():
                        logits, _, _, _ = self.model(
                            data.x.to(self.device), data.edge_index.to(
                                self.device)
                        )
                    pred = (logits > 0).float().cpu()
                    micro_f1 = metrics.f1_score(data.y, pred, average="micro")
                    total_micro_f1 += micro_f1
                total_micro_f1 /= len(self.data[i].dataset)
                accs[i] = total_micro_f1
            return accs[0], accs[1], accs[2]
        else:
            raise Exception(
                f"the dataset of {self.dataset} has not been implemented")

    def load_model(self, type_model="GCN", dataset="PPI"):
        filename = self.filename(
            filetype="params", type_model=type_model, dataset=dataset
        )
        if os.path.exists(filename):
            print("load model: ", type_model, filename)
            return torch.load(filename)
        else:
            return None

    def save_model(self, type_model="GCN", dataset="PPI"):
        filename = self.filename(
            filetype="params", type_model=type_model, dataset=dataset
        )
        state = self.model.state_dict()
        torch.save(state, filename)
        print("save model to", filename)

    def filename(self, filetype="logs", type_model="GCN", dataset="PPI"):
        filedir = f"./{filetype}/{dataset}"
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        layers = self.num_layers

        filename = f"{filetype}_{type_model}" f"layers{layers}.pth.tar"

        filename = os.path.join(filedir, filename)
        return filename

    def figurename(self, figure, filetype="figures", dataset="Cora"):
        filedir = f"./{filetype}/{dataset}"
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        filename = figure
        return os.path.join(filedir, filename)

    def load_log(self, type_model="GCN", dataset="PPI", load=True):
        log = {}
        filename = self.filename(
            filetype="logs", type_model=type_model, dataset=dataset
        )
        if load and os.path.exists(filename) and os.path.getsize(filename):
            log = torch.load(filename)
            print("load the log from ", filename)

        if len(log) == 0:
            log["acc_train"], log["acc_valid"], log["acc_test"] = [], [], []
            log["ptb"] = []

        for key in log.keys():
            if len(log[key]) == 0:
                log[key].append([])
            elif len(log[key][-1]) > 0:
                log[key].append([])
            else:
                continue
        return log

    def save_log(self, log, type_model="GCN", dataset="PPI"):
        filename = self.filename(
            filetype="logs", type_model=type_model, dataset=dataset)
        torch.save(log, filename)
        print("save log to", filename)


    def plot_test_accuracy(self):
        pass
