import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import PPI
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops, add_self_loops

from models.GAT import GAT
from models.GCN import GCN
from models.NLGCN import NLGCN


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
            # print(index[perm[:train_num]])
            # print(perm[train_num:(train_num+val_num)])
            # print(index[perm[(train_num+val_num):]])
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

        if self.type_model == "GCN":
            self.model = GCN(args)
        elif self.type_model == "GAT":
            self.model = GAT(args)
        elif self.type_model == "NLGCN":
            self.model = NLGCN(args)
        else:
            raise Exception(
                f"the model of {self.type_model} has not been implemented")

        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            self.data.to(self.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.writer = SummaryWriter("runs/NLGCN")

    def train_net(self, epoch):
        # try:
        loss_train = self.run_trainSet(epoch)
        acc_train, acc_valid, acc_test = self.run_testSet()
        return loss_train, acc_train, acc_valid, acc_test

    def train(self):
        best_acc = 0
        for epoch in range(self.epochs):
            loss_train, acc_train, acc_valid, acc_test = self.train_net(epoch)
            print("Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}".format(
                epoch, loss_train, acc_valid, acc_test))

            if epoch % 5 == 4:
                self.writer.add_scalar("Loss/train", loss_train, epoch)
                self.writer.add_scalar("Accuracy/train", acc_train, epoch)
                self.writer.add_scalar("Accuracy/test", acc_test, epoch)

            if best_acc < acc_valid:
                best_acc = acc_valid
                self.model.cpu()
                self.save_model(self.type_model, self.dataset)
                self.model.to(self.device)

        self.writer.close()
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

        self.log["alpha"][-1].append(self.alpha)

        self.save_log(self.log, self.type_model, self.dataset)

    def run_trainSet(self, epoch):
        self.model.train()
        loss = 0.0
        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            if self.type_model == "NLGCN":
                logits, new_gs, new_hs, gs = self.model(self.data.x, self.data.edge_index)
                logits = F.log_softmax(logits[self.data.train_mask], 1)
                loss = self.loss_fn(logits, self.data.y[self.data.train_mask])

                # graph regularization
                for new_h, new_g in zip(new_hs, new_gs):
                    n = new_h.size()[0]
                    d = new_h.size()[1]
                    loss += self.lamb / (d ^ 2) * torch.trace(
                        torch.chain_matmul(new_h.t(), torch.diag(torch.sum(new_g, dim=1)) - new_g, new_h))
                    # loss += -self.beta * torch.sum(torch.log(torch.sum(new_g, dim=1))) / n
                    loss += self.gamma * torch.norm(new_g) / (n ^ 2)

                if epoch % 200 == 199:
                    for i, adj_new in enumerate(new_gs):
                        heat_map = sns.heatmap(adj_new.cpu().detach().numpy())
                        fig = heat_map.get_figure()
                        fig.savefig(self.figurename(f"adj_new{epoch}_{i}.png"))
                        plt.clf()

                    for i, g in enumerate(gs):
                        heat_map = sns.heatmap(g.cpu().detach().numpy())
                        fig = heat_map.get_figure()
                        fig.savefig(self.figurename(f"adj{epoch}_{i}.png"))
                        plt.clf()

                if epoch == 399:
                    pass
            else:
                logits, adj = self.model(self.data.x, self.data.edge_index)
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
                logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                loss += self.loss_fn(logits, data.y.to(self.device))
            raise Exception(f"the dataset of {self.dataset} has not been implemented")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_testSet(self):
        self.model.eval()
        # torch.cuda.empty_cache()
        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            with torch.no_grad():
                logits, _, _, _ = self.model(self.data.x, self.data.edge_index)
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
                        logits = self.model(
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

        alpha = str(self.model.alpha)
        num_layers = int(self.model.num_layers)

        filename = f"{filetype}_{type_model}" f"L{num_layers}Alpha{alpha}.pth.tar"

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
            log["alpha"] = []
            log["L"] = []

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
