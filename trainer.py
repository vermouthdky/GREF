import torch
import os
from models.GCN import GCN
from models.simpleGCN import simpleGCN
from models.GAT import GAT
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
from torch_geometric.datasets import Coauthor

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
import glob
from torch_geometric.utils import remove_self_loops, add_self_loops, dense_to_sparse
import numpy as np
from MI.kde import mi_kde, Kget_dists, entropy_estimator_bd
from torch_geometric.utils import scatter_, to_dense_adj, contains_isolated_nodes
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
import umap


def load_data(dataset="Cora"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, T.NormalizeFeatures())[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0] #!!! 2*N 新版可能有改变
        else:
            data.edge_index = edge_index
        return data
    elif dataset in ['CoauthorCS']:
        data = Coauthor(path, 'cs', T.NormalizeFeatures())[0]
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


def remove_feature(data, miss_rate):
    num_nodes = data.x.size(0)
    erasing_pool = torch.arange(num_nodes)[~data.train_mask]
    size = int(len(erasing_pool) * miss_rate)
    idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
    x = data.x
    x[idx_erased] = 0.
    return x


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

        self.miss_rate = args.miss_rate
        if self.miss_rate > 0.:
            self.data.x = remove_feature(self.data, self.miss_rate)

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.grad_clip = args.grad_clip
        self.weight_decay = args.weight_decay
        if self.type_model == 'GCN':
            self.model = GCN(args)
        elif self.type_model == 'simpleGCN':
            self.model = simpleGCN(args)
        elif self.type_model == 'GAT':
            self.model = GAT(args)
        else:
            raise Exception(f'the model of {self.type_model} has not been implemented')


        if self.dataset in ["Cora", "Citeseer", "Pubmed", "CoauthorCS"]:
            self.data.to(self.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # self.var_noise = args.var_noise
        # self.seed = args.random_seed
        # self.type_norm = args.type_norm
        # self.skip_weight = args.skip_weight

        self.loss_weight = args.loss_weight  # 0.0001
        row_index = [[int(i)] * (self.model.num_groups - i - 1) for i in range(self.model.num_groups)]
        self.row_index = np.concatenate(row_index, axis=0)
        col_index = [list(range(i + 1, self.model.num_groups)) for i in range(self.model.num_groups - 1)]
        self.col_index = np.concatenate(col_index, axis=0)


    def train_net(self):
        try:
            loss_train = self.run_trainSet()
            acc_train, acc_valid, acc_test = self.run_testSet()
            return loss_train, acc_train, acc_valid, acc_test
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e



    # def compute_MI(self):
    #     self.model.eval()
    #     data_x = self.data.x.data.cpu().numpy()
    #     data_y = self.data.y.data.cpu().numpy()
    #     probs_node= None
    #     with torch.no_grad():
    #         layers_self = self.model(self.data.x, self.data.edge_index)
    #     layer_self = layers_self.data.cpu().numpy()
    #     MI_XiX, MI_XiY = mi_kde(layer_self, data_x, data_y, self.model.num_classes, self.var_noise,
    #                             'lower', entro_which='second', probs_node=probs_node)
    #     h_norm = np.sum(np.square(layer_self), axis=1, keepdims=True)
    #     h_norm[h_norm == 0.] = 1e-3
    #     layer_self = layer_self / np.sqrt(h_norm)
    #     dists_h = np.mean(Kget_dists(layer_self))
    #     return MI_XiX, MI_XiY, dists_h

    # def edge_index_list(self, hops):
    #     edge_index_list = []
    #     edge_index_neighbor, _ = remove_self_loops(self.data.edge_index)
    #     num_nodes = self.data.x.size(0)
    #     if isinstance(edge_index_neighbor, tuple):
    #         edge_index_neighbor = edge_index_neighbor[0]
    #     edge_index_list.append(edge_index_neighbor)

    #     edge_index_current = edge_index_neighbor
    #     for i in range(1, hops):
    #         value_int8 = edge_index_current.new_ones((edge_index_current.size(1),), dtype=torch.int8)
    #         numpy_last = to_scipy(edge_index_current.cpu(), value_int8.cpu(), num_nodes, num_nodes).toarray()

    #         value_neighbor = edge_index_neighbor.new_ones((edge_index_neighbor.size(1),), dtype=torch.float)
    #         value_current = edge_index_current.new_ones((edge_index_current.size(1),), dtype=torch.float)
    #         index, value = spspmm(edge_index_current, value_current, edge_index_neighbor, value_neighbor,
    #                               num_nodes, num_nodes, num_nodes)
    #         value.fill_(0)
    #         index, value = remove_self_loops(index, value)
    #         edge_index_current = torch.cat([edge_index_current, index], dim=1)
    #         edge_index_current, _ = coalesce(edge_index_current, None, num_nodes, num_nodes)

    #         value_current = edge_index_current.new_ones((edge_index_current.size(1),), dtype=torch.int8)
    #         numpy_current = to_scipy(edge_index_current.cpu(), value_current.cpu(), num_nodes, num_nodes).toarray()
    #         residual = scipy.sparse.coo_matrix(numpy_current - numpy_last, (num_nodes, num_nodes))
    #         residual_index, _ = from_scipy(residual)
    #         # print(residual_index, residual_index.size())
    #         # print(residual_index, residual_index.size())

    #         # edge_index_list.append(edge_index_current)
    #         edge_index_list.append(residual_index.to(self.device))
    #     return edge_index_list



    def train_compute_MI(self):
        best_acc = 0
        for epoch in range(self.epochs):
            loss_train, acc_train, acc_valid, acc_test = self.train_net()
            print('Epoch: {:02d}, Loss: {:.4f}, val_acc: {:.4f}, test_acc:{:.4f}'.format(epoch, loss_train,
                                                                                         acc_valid, acc_test))
            if best_acc < acc_valid:
                best_acc = acc_valid
                self.model.cpu()
                self.save_model(self.type_model, self.dataset)
                self.model.to(self.device)

        # # # # # compute MI and dis
        self.log = self.load_log(type_model=self.type_model, dataset=self.dataset, load=True)
        state_dict = self.load_model(self.type_model, self.dataset)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        acc_train, acc_valid, acc_test = self.run_testSet()
        # print('val_acc: {:.4f}, test_acc:{:.4f}'.format(acc_valid, acc_test))
        # self.log['acc_train'][-1].append(acc_train)
        # self.log['acc_valid'][-1].append(acc_valid)
        # self.log['acc_test'][-1].append(acc_test)
        # # #
        # MI_XiX, MI_XiY, dists_h = self.compute_MI()
        # self.log['MI_XiX'][-1].append(MI_XiX), self.log['MI_XiY'][-1].append(MI_XiY)
        # self.log['dis'][-1].append(dists_h)



        #
        # # hops = 5
        # # edge_index_list = self.edge_index_list(hops)
        # # MI_NihopXi_series, MI_NihopY_series = self.MI_neighbors_hops(edge_index_list)
        # # # self.log['MI_NihopXi'][-1].append(MI_NihopXi_series), self.log['MI_NihopY'][-1].append(MI_NihopY_series)
        # # # dis_series = self.dis_neighbors_hops(edge_index_list)
        # # # self.log['dis_hop'][-1].append(dis_series)
        # # print(MI_NihopXi_series)
        #
        # # hops = 3
        # # # edge_index_neighbor = self.edge_index_list(hops)[-1]
        # # edge_index_neighbor = torch.cat(self.edge_index_list(hops), dim=1)
        # # num_nodes = self.data.x.size(0)
        # # adj_tensor = torch.ones((num_nodes, num_nodes), dtype=torch.long) - \
        # #              torch.diag(torch.ones((num_nodes,), dtype=torch.long))
        # # edge_index_remote = dense_to_sparse(adj_tensor)[0]
        # # value_neighbor = edge_index_neighbor.new_ones((edge_index_neighbor.size(1),), dtype=torch.int8)
        # # numpy_neighbor = to_scipy(edge_index_neighbor.cpu(), value_neighbor.cpu(), num_nodes, num_nodes).toarray()
        # # value_remote = edge_index_remote.new_ones((edge_index_remote.size(1),), dtype=torch.int8)
        # # numpy_remote = to_scipy(edge_index_remote.cpu(), value_remote.cpu(), num_nodes, num_nodes).toarray()
        # # residual = scipy.sparse.coo_matrix(numpy_remote - numpy_neighbor, (num_nodes, num_nodes))
        # # residual_index, _ = from_scipy(residual)
        # # edge_index_list_community = [edge_index_neighbor, residual_index.to(self.device)]
        # # # edge_index_list_community = [edge_index_neighbor, edge_index_remote.to(self.device)]
        # # MI_NicomXi_series, MI_NicomY_series = self.MI_neighbors_hops(edge_index_list_community,
        # #                                                              MI_X='MI_NicomXi', MI_Y='MI_NicomY')
        # # self.log['MI_NicomXi'][-1].append(MI_NicomXi_series), self.log['MI_NicomY'][-1].append(MI_NicomY_series)
        # # dis_series = self.dis_neighbors_hops(edge_index_list_community, dis_name='dis_com')
        # # self.log['dis_com'][-1].append(dis_series)



        # dis_cluster = self.dis_cluster()
        # self.log['dis_cluster'][-1].append(dis_cluster)

        # CosDis_cluster = self.CosDis_cluster()
        # self.log['CosDis_cluster'][-1].append(CosDis_cluster)
        # # MI_cluster = self.MI_cluster()
        # # self.log['MI_cluster'][-1].append(MI_cluster)


        #
        # self.visualize_nodes()
        #
        # self.save_log(self.log, self.type_model, self.dataset)


        # compute the distant node pairs
        adj = to_dense_adj(self.data.edge_index).data.cpu().numpy().squeeze(axis=0)
        print(adj.shape)
        rows, cols = adj.shape
        num_distant = 0.
        num_disLabel = 0.
        for i in range(rows):
            if i % 100 == 0:
                print(i)
            for j in range(cols):
                if adj[i ,j] > 0.:
                    pass
                else:
                    num_distant += 1.
                    label_i = self.data.y[i]
                    label_j = self.data.y[j]
                    if label_i != label_j:
                        num_disLabel += 1.

        print(num_distant, num_disLabel, num_disLabel/num_distant)


    def visualize_nodes(self):
        self.model.eval()
        with torch.no_grad():
            _, X = self.model.visualize_forward(self.data.x, self.data.edge_index)
        num_nodes = X.size(0)
        if self.type_norm == 'group':
            running_mean = self.model.layers_bn[-1].bn.running_mean.view(-1, self.model.num_classes)
            X = torch.cat([X, running_mean], dim=0)

        X = X.data.cpu().numpy()
        X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        # scaler = sklearn.preprocessing.StandardScaler()
        # X = scaler.fit_transform(X)

        label = self.data.y.data.cpu().numpy()
        color = plt.cm.Set1(label / float(self.model.num_classes))


        trans_func = TSNE(n_components=2, init='pca', random_state=0, early_exaggeration=1)
        # trans_func = PCA(n_components=2)
        # trans_func = Isomap(n_components=2)
        # trans_func = umap.UMAP(n_components=2)
        print('TSNE processing')
        X_trans = trans_func.fit_transform(X)
        # X_trans = X
        print('TSNE processing end')
        x_min, x_max = np.min(X_trans, 0), np.max(X_trans, 0)
        X_trans = (X_trans - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = plt.subplot(111)
        plt.scatter(X_trans[:num_nodes, 0], X_trans[:num_nodes, 1], c=color, marker='o', s=10)
        if self.type_norm == 'group':
            plt.scatter(X_trans[num_nodes:, 0], X_trans[num_nodes:, 1], c='k', marker='^', s=200)
            # print(X[0,])
            # print(X_trans[num_nodes:,])
            # print(X[num_nodes:])

        plt.xticks([])
        plt.yticks([])
        # plt.xlim(-10,10)
        # plt.ylim(-10,10)
        plt.tight_layout()

        savefile = self.filename_fig(self.type_norm+'_scatter')
        plt.savefig(savefile)


    def plot_hypers(self, layer, hypers_group, hypers_skip):
        self.model.num_layers = layer
        Hypers_group, Hypers_skip = np.meshgrid(hypers_group, hypers_skip)
        Z_accs = np.zeros((len(hypers_skip), len(hypers_group)))
        print_name = 'acc_test'
        print_seeds = [100, 200, 300, 400, 500]



        for i in range(len(hypers_skip)):
            for j in range(len(hypers_group)):
                group = Hypers_group[i, j]
                skip = Hypers_skip[i, j]
                self.model.skip_weight = skip
                self.model.num_groups = group
                pfc_seeds = []
                for seed in print_seeds:
                    self.seed = seed
                    path = self.filename(filetype='logs', type_model=self.type_model, dataset=self.dataset)
                    # print(path)
                    log = torch.load(path)
                    # print(log)
                    if len(log[print_name][-1]) == 0:
                        pfc = log[print_name][-2][0]
                    else:
                        pfc = log[print_name][-1][0]
                    pfc_seeds.append(pfc)
                pfc_mean = np.mean(pfc_seeds)
                Z_accs[i, j] = pfc_mean
                print(group, skip, pfc_mean)


        print(Z_accs)
        norm =  plt.Normalize(np.min(Z_accs), np.max(Z_accs))
        colors = cm.viridis(norm(Z_accs))
        rcount, ccount, _ = colors.shape

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(Hypers_group, Hypers_skip, Z_accs, rcount=rcount, ccount=ccount,
                               facecolors=colors, linewidths=2.) # shade=False
        # surf = ax.plot_surface(Hypers_group, Hypers_skip, Z_accs)
        # surf.set_facecolor((0, 0, 0, 0))

        savefile = self.filename_fig('hyper')
        ax.set_xlabel('Group', fontsize=12)
        ax.set_ylabel('Balancing factor', fontsize=12)
        ax.grid(alpha=0.001, lw=0.001)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        print(savefile)
        plt.savefig(savefile)





    def CosDis_cluster(self):
        self.model.eval()
        with torch.no_grad():
            X = self.model(self.data.x, self.data.edge_index)
        X_labels = []
        for i in range(self.model.num_classes):
            X_label = X[self.data.y == i].data.cpu().numpy()
            h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
            h_norm[h_norm < 1e-3] = 1e-3
            X_label = X_label / np.sqrt(h_norm)
            X_labels.append(X_label)


        # print(X_labels[0][0,:], X_labels[1][0,:], X_labels[2][0,:], X_labels[3][0,:])

        dis_intra = 0.
        for i in range(self.model.num_classes):
            dis_intra += np.mean(np.dot(X_labels[i], X_labels[i].T))
        dis_intra /= self.model.num_classes

        dis_inter = 0.
        for i in range(self.model.num_classes-1):
            for j in range(i+1, self.model.num_classes):
                dis_inter += np.mean(np.dot(X_labels[i], X_labels[j].T))
        num_inter =  float(self.model.num_classes * (self.model.num_classes-1) / 2)
        dis_inter /= num_inter



        print('dis_intra: ', dis_intra)
        print('dis_inter: ', dis_inter)
        return [dis_intra, dis_inter]



    def dis_cluster(self):
        self.model.eval()
        with torch.no_grad():
            X = self.model(self.data.x, self.data.edge_index)
        X_labels = []
        for i in range(self.model.num_classes):
            X_label = X[self.data.y == i].data.cpu().numpy()
            h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
            h_norm[h_norm == 0.] = 1e-3
            X_label = X_label / np.sqrt(h_norm)
            X_labels.append(X_label)

        dis_intra = 0.
        for i in range(self.model.num_classes):
            x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
            dis_intra += np.mean(dists)
        dis_intra /= self.model.num_classes

        dis_inter = 0.
        for i in range(self.model.num_classes-1):
            for j in range(i+1, self.model.num_classes):
                x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
                x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
                dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
                dis_inter += np.mean(dists)
        num_inter =  float(self.model.num_classes * (self.model.num_classes-1) / 2)

        dis_inter /= num_inter
        print('dis_intra: ', dis_intra)
        print('dis_inter: ', dis_inter)
        return [dis_intra, dis_inter]


    def MI_cluster(self):
        self.model.eval()
        X = self.model(self.data.x, self.data.edge_index)
        X_labels = []

        entropy_intra = 0.
        for i in range(self.model.num_classes):
            X_label = X[self.data.y == i]
            X_labels.append(X_label)

            X_label = X_label.data.cpu().numpy()
            h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
            h_norm[h_norm == 0.] = 1e-3
            X_label = X_label / np.sqrt(h_norm)

            entropy_X = entropy_estimator_bd(X_label, self.var_noise, None)
            entropy_intra += entropy_X
        entropy_intra /= self.model.num_classes

        # sample_size = 2000
        # MIs_intra = 0.
        # for i in range(self.model.num_classes):
        #     len_i = X_labels[i].size(0)
        #     # index_i = np.random.randint(0, len_i, size=sample_size)
        #     # index_j = np.random.randint(0, len_i, size=sample_size)
        #     index_i = torch.randint(0, len_i, size=(sample_size,), dtype=torch.long, device=self.device)
        #     index_j = torch.randint(0, len_i, size=(sample_size,), dtype=torch.long, device=self.device)
        #     X_i = X_labels[i].index_select(0, index_i).data.cpu().numpy()
        #     X_j = X_labels[i].index_select(0, index_j).data.cpu().numpy()
        #     data_y = None
        #     MI_intra, _ = mi_kde(X_i, X_j, data_y, self.model.num_classes,
        #                                    self.var_noise, 'lower', entro_which='both')
        #     MIs_intra += MI_intra
        # MIs_intra /=  self.model.num_classes

        sample_size = 2000
        MIs_inter = 0.
        for i in range(self.model.num_classes-1):
            len_i = X_labels[i].size(0)
            for j in range(i+1, self.model.num_classes):
                len_j = X_labels[j].size(0)
                # index_i = np.random.randint(0, len_i, size=sample_size)
                # index_j = np.random.randint(0, len_j, size=sample_size)
                index_i = torch.randint(0, len_i, size=(sample_size,), dtype=torch.long, device=self.device)
                index_j = torch.randint(0, len_j, size=(sample_size,), dtype=torch.long, device=self.device)
                X_i = X_labels[i].index_select(0, index_i).data.cpu().numpy()
                X_j = X_labels[j].index_select(0, index_j).data.cpu().numpy()
                data_y = None
                MI_inter, _ = mi_kde(X_i, X_j, data_y, self.model.num_classes,
                                     self.var_noise, 'lower', entro_which='both')
                MIs_inter += MI_inter
        num_inter = float(self.model.num_classes * (self.model.num_classes - 1) / 2)

        MIs_inter /= num_inter
        print('entropy_intra: ', entropy_intra)
        print('MIs_inter: ', MIs_inter)
        return [entropy_intra, MIs_inter]



    def dis_neighbors_hops(self, edge_index_list, dis_name='dis_hop'):
        self.model.eval()
        X = self.model(self.data.x, self.data.edge_index)

        dis_series = []
        dis_previous = 0.
        size_previous = 0.
        for i in range(len(edge_index_list)):
            X_j = X.index_select(0, edge_index_list[i][0])
            X_j = F.normalize(X_j, p=2, dim=-1)
            X_i = X.index_select(0, edge_index_list[i][1])
            X_i = F.normalize(X_i, p=2, dim=-1)
            dis = torch.sum(torch.norm(X_i - X_j, p=2, dim=-1))
            size = X_i.size(0)
            # dis = torch.sum(torch.norm(X_i-X_j, p=2, dim=-1)) - dis_previous
            # size = X_i.size(0) - size_previous
            # dis_previous += dis
            # size_previous += size
            dis_series.append(dis.data.cpu().numpy().sum() / size)
        print(dis_name, dis_series)
        return dis_series


    def MI_neighbors_hops(self, edge_index_list, MI_X='MI_NihopXi', MI_Y='MI_NihopY'):
        self.model.eval()
        X = self.model(self.data.x, self.data.edge_index)
        X_self = X.data.cpu().numpy()
        data_y = self.data.y.data.cpu().numpy()
        num_nodes = X_self.shape[0]

        MI_NihopXi_series = []
        MI_NihopY_series = []
        X_previous = 0
        for i in range(len(edge_index_list)):
            X_j = X.index_select(0, edge_index_list[i][0])
            X_j = scatter_('add', X_j, edge_index_list[i][1], 0, num_nodes)
            X_current = X_j.data.cpu().numpy()
            # X_current = (X_j - X_previous).data.cpu().numpy()
            # X_previous = X_j
            MI_NihopXi, MI_NihopY = mi_kde(X_current, X_self, data_y, self.model.num_classes,
                                       self.var_noise, 'lower', entro_which='first')
            MI_NihopXi_series.append(MI_NihopXi)
            MI_NihopY_series.append(MI_NihopY)
        print(MI_X, ':', MI_NihopXi_series)
        print(MI_Y, ':', MI_NihopY_series)
        return MI_NihopXi_series, MI_NihopY_series


    def run_trainSet(self):
        self.model.train()
        loss = 0.
        if self.dataset in ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS']:
            logits = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(logits[self.data.train_mask], 1)
            loss = self.loss_fn(logits, self.data.y[self.data.train_mask])
        elif self.dataset in ['PPI']:
            for data in self.data[0]:
                num_nodes = data.x.size(0)
                # edge_index, _ = remove_self_loops(data.edge_index)
                data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)
                if isinstance(data.edge_index, tuple):
                    data.edge_index = data.edge_index[0]
                logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                loss += self.loss_fn(logits, data.y.to(self.device))
        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')


        if self.type_norm in ['group'] and self.loss_weight > 0.:
            loss_cluster = 0.
            for i in range(self.model.num_layers):
                running_mean = self.model.layers_bn[i].bn.weight.view(self.model.num_groups, -1)
                running_var = self.model.layers_bn[i].bn.bias.view(self.model.num_groups, -1)

                dis_all = torch.mm(running_mean, running_mean.t())
                norm_var = running_var.norm(dim=1)
                loss_cluster += torch.sum(dis_all[self.row_index, self.col_index]) + torch.sum(norm_var)
            loss += self.loss_weight * loss_cluster
            # loss += 0.0001 * loss_cluster

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
                logits = self.model(self.data.x, self.data.edge_index)
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
                        logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                    pred = (logits > 0).float().cpu()
                    micro_f1 = metrics.f1_score(data.y, pred, average='micro')
                    total_micro_f1 += micro_f1
                total_micro_f1 /= len(self.data[i].dataset)
                accs[i] = total_micro_f1
            return accs[0], accs[1], accs[2]
        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')


    def plot_appendix(self, args):
        print_seeds = [100, 200, 300, 400, 500]
        print_norms = ['None', 'batch', 'pair', 'group']
        print_models = ['GCN', 'GAT', 'simpleGCN']
        # print_names = ['acc_test', 'MI_XiX', 'dis_cluster']
        print_names = ['acc_test', 'dis_cluster', 'dis_cluster']

        figure_all = plt.figure(figsize=(9, 6))
        spec = gridspec.GridSpec(ncols=3, nrows=3, figure=figure_all, wspace=0.35, left=0.1, right=0.95)


        for idx_name, print_model in enumerate(print_models):
            self.type_model = print_model
            if self.type_model in ['GCN', 'GAT']:
                print_layers = list(range(1, 10, 1)) + list(range(10, 31, 5))
            else:
                print_layers = [1, 5] + list(range(10, 121, 10))
            for idx_print_name, print_name in enumerate(print_names):
                pfc_mean_norms = []
                for name_norm in print_norms:
                    pfc_mean_norms.append([])
                    self.type_norm = name_norm
                    for layer in print_layers:
                        self.model.num_layers = layer
                        if name_norm == 'group':
                            args.type_model = self.type_model
                            args.num_layers = layer
                            args = reset_weight(args)
                            self.model.skip_weight = args.skip_weight
                            # print(self.model.skip_weight)
                        pfc_seeds = []
                        for seed in print_seeds:
                            self.seed = seed
                            path = self.filename(filetype='logs', type_model=self.type_model, dataset=self.dataset)
                            log = torch.load(path)
                            if len(log[print_name][-1]) == 0:
                                pfc = log[print_name][-2][0]
                            else:
                                pfc = log[print_name][-1][0]
                            # if name_norm =='None':
                            #     print(pfc)
                            pfc_seeds.append(pfc)

                        if type(pfc_seeds[0]) in [float, np.float64]:
                            pass
                        elif len(pfc_seeds[0]) == 2 and print_name == 'dis_cluster':
                            pfc_seeds = np.array(pfc_seeds)
                            if idx_print_name == 1:
                                if self.dataset == 'CoauthorCS' and self.type_model == 'simpleGCN':
                                    tmp_seeds = pfc_seeds[:, 1] - pfc_seeds[:, 0]
                                    pfc_seeds[tmp_seeds < 0.35] = [1., 1.]
                                print(pfc_seeds)
                                pfc_seeds = pfc_seeds[:, 1] / pfc_seeds[:, 0]
                            else:
                                pfc_seeds = pfc_seeds[:, 0]
                            pfc_seeds[np.isnan(pfc_seeds)] = 1.
                        elif len(pfc_seeds[0]) == 2 and print_name == 'MI_cluster':
                            pfc_seeds = np.array(pfc_seeds)
                            # a large number means the MI +infinity,
                            # since the embedding distance close to 0
                            pfc_seeds[pfc_seeds < 1.5e-3] = 100
                            pfc_seeds = 1. / (pfc_seeds[:, 1] + pfc_seeds[:, 0])
                        else:
                            pass
                        pfc_mean = np.mean(pfc_seeds)
                        pfc_mean_norms[-1].append(pfc_mean)

                # ax = plt.subplot(idx_subplots[idx_name])
                ax = figure_all.add_subplot(spec[idx_name, idx_print_name])
                colors = ['r', 'b', 'k', 'm', 'g']
                markers = ['s', 'o', 'v', '^', '1', '2', '+']
                LineStyles = ['-', '--', ':', '-.']
                handles = []
                for i, name_norm in enumerate(print_norms):
                    handle, = plt.plot(print_layers, pfc_mean_norms[i], LineStyles[i], color=colors[i])
                    handles.append(handle)

                if idx_name == 0:
                    if print_name == 'acc_test':
                        title = 'Test accuracy'
                    elif print_name == 'MI_XiX':
                        title = 'Instance gain'
                    elif print_name == 'dis_cluster':
                        if idx_print_name == 1:
                            title = 'Group ratio'
                        else:
                            title = 'Intra-group distance'
                    ax.set_title(title, fontsize=15)
                if idx_name == 2:
                    ax.set_xlabel('Layers', fontsize=15)
                if idx_print_name == 0:
                    if print_model == 'simpleGCN':
                        ax.set_ylabel('SGC', fontsize=15)
                    else:
                        ax.set_ylabel(print_model, fontsize=15)
                if idx_name == 0 and idx_print_name == 1:
                    plt.legend(handles, print_norms, bbox_to_anchor=(0.5, 1.5), loc='upper center', borderaxespad=0.,
                               fontsize=15, ncol=4)


                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)




        # figure_all.tight_layout()
        # spec.tight_layout(figure_all)

        self.type_model = 'all'
        print_name = 'all'
        savefile = self.filename_fig(print_name)
        plt.savefig(savefile)



    def plot_all(self, args):
        print_seeds = [100, 200, 300, 400, 500]
        print_norms = ['None', 'batch', 'pair', 'group']
        print_models = ['simpleGCN', 'GCN', 'GAT']
        print_names = ['acc_test', 'MI_XiX', 'dis_cluster']

        figure_all = plt.figure(figsize=(12, 2.5), constrained_layout=True)
        spec = gridspec.GridSpec(ncols=3, nrows=1, figure=figure_all)


        # for idx_name, print_name in enumerate(print_names):
        for idx_name, print_model in enumerate(print_models):
            self.type_model = print_model
            print(self.type_model)
            print_name = print_names[2]
            if self.type_model in ['GCN', 'GAT']:
                print_layers = list(range(1, 10, 1)) + list(range(10, 31, 5))
            else:
                print_layers = [1, 5] + list(range(10, 121, 10))

            pfc_mean_norms = []
            for name_norm in print_norms:
                pfc_mean_norms.append([])
                self.type_norm = name_norm
                for layer in print_layers:
                    self.model.num_layers = layer
                    if name_norm == 'group':
                        args.type_model = self.type_model
                        args.num_layers = layer
                        args = reset_weight(args)
                        self.model.skip_weight = args.skip_weight
                        print(self.model.skip_weight)
                    pfc_seeds = []
                    for seed in print_seeds:
                        self.seed = seed
                        path = self.filename(filetype='logs', type_model=self.type_model, dataset=self.dataset)
                        log = torch.load(path)
                        if name_norm == 'None':
                            print(log )
                        if len(log[print_name][-1]) == 0:
                            pfc = log[print_name][-2][0]
                        else:
                            pfc = log[print_name][-1][0]
                        # if name_norm =='None':
                        #     print(pfc)
                        pfc_seeds.append(pfc)

                    if type(pfc_seeds[0]) in [float, np.float64]:
                        pass
                    elif len(pfc_seeds[0]) == 2 and print_name == 'dis_cluster':
                        pfc_seeds = np.array(pfc_seeds)
                        pfc_seeds[pfc_seeds < 1.e-4] = 1.e-4
                        pfc_seeds = pfc_seeds[:, 1] / pfc_seeds[:, 0]
                    elif len(pfc_seeds[0]) == 2 and print_name == 'MI_cluster':
                        pfc_seeds = np.array(pfc_seeds)
                        # a large number means the MI +infinity,
                        # since the embedding distance close to 0
                        pfc_seeds[pfc_seeds < 1.5e-3] = 100
                        pfc_seeds = 1. / (pfc_seeds[:, 1] + pfc_seeds[:, 0])
                    else:
                        pass
                    pfc_mean = np.mean(pfc_seeds)
                    pfc_mean_norms[-1].append(pfc_mean)

            # ax = plt.subplot(idx_subplots[idx_name])
            ax = figure_all.add_subplot(spec[0, idx_name])
            colors = ['r', 'b', 'k', 'm', 'g']
            markers = ['s', 'o', 'v', '^', '1', '2', '+']
            LineStyles = ['-', '--', ':', '-.']
            handles = []
            for i, name_norm in enumerate(print_norms):
                handle, = plt.plot(print_layers, pfc_mean_norms[i], LineStyles[i], color=colors[i])
                handles.append(handle)
            if print_name in ['acc_test']:
                label_y = 'Accuracy'
            elif print_name in ['MI_XiX']:
                label_y = 'Instance gain'
            elif print_name in ['MI_cluster']:
                label_y = 'Group divergence'
            elif print_name in ['dis_cluster']:
                label_y = 'Group distance'
            else:
                label_y = ''

            ax.set_xlabel('Layers', fontsize=15)
            ax.set_ylabel(label_y, fontsize=15)
            if self.type_model == 'simpleGCN':
                ax.set_title('SGC')
            else:
                ax.set_title(self.type_model)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            if idx_name == 2:
                plt.legend(handles, print_norms, bbox_to_anchor=(1.55, 0.9), loc='upper right', borderaxespad=0.,
                           fontsize=15)
            # plt.subplots_adjust(wspace=0.05)


        # figure_all.tight_layout()

        self.type_model = 'all'
        # print_name = 'all'
        savefile = self.filename_fig(print_name)
        plt.savefig(savefile)


    def plot_acc(self, args):
        if self.type_model in ['GCN', 'GAT']:
            print_layers = list(range(1, 10, 1)) + list(range(10, 31, 5))
            # print_layers = [5, 6]
        else:
            print_layers = [1, 5] + list(range(10, 121, 10))
            # print_layers = [120]
        print_seeds = [100, 200, 300, 400, 500]
        print_norms = ['None', 'batch', 'pair', 'group']
        pfc_mean_norms = []
        pfc_std_norms = []
        print_name = 'acc_test' # 'MI_cluster' # 'dis_cluster' # 'MI_XiX' # 'acc_test' # 'CosDis_cluster'

        for name_norm in print_norms:
            pfc_mean_norms.append([])
            pfc_std_norms.append([])
            self.type_norm = name_norm
            print(self.type_norm)

            for layer in print_layers:
                self.model.num_layers = layer
                if name_norm == 'group':
                    args.num_layers = layer
                    args = reset_weight(args)
                    self.model.skip_weight = args.skip_weight

                pfc_seeds = []
                for seed in print_seeds:
                    self.seed = seed
                    path = self.filename(filetype='logs', type_model=self.type_model, dataset=self.dataset)
                    log = torch.load(path)
                    # print(log)
                    if len(log[print_name][-1]) == 0:
                        pfc = log[print_name][-2][0]
                    else:
                        pfc = log[print_name][-1][0]
                    pfc_seeds.append(pfc)

                if type(pfc_seeds[0]) in [float, np.float64]:
                    pass
                elif len(pfc_seeds[0]) == 2 and print_name == 'dis_cluster':
                    pfc_seeds = np.array(pfc_seeds)
                    # if name_norm in ['pair', 'group']:
                    #     print(pfc_seeds)
                    # if self.dataset == 'Cora' and self.type_model == 'simpleGCN':
                    #     pfc_seeds[pfc_seeds < 1.e-3] = 1.e-3 # Cora
                    if self.dataset == 'CoauthorCS' and self.type_model == 'simpleGCN':
                        tmp_seeds = pfc_seeds[:, 1] - pfc_seeds[:, 0]
                        pfc_seeds[tmp_seeds < 0.35] = [1., 1.]

                    pfc_seeds = pfc_seeds[:,1] / pfc_seeds[:,0]
                    # pfc_seeds = pfc_seeds[:, 0]
                    pfc_seeds[np.isnan(pfc_seeds)] = 1.
                elif len(pfc_seeds[0]) == 2 and print_name == 'MI_cluster':
                    pfc_seeds = np.array(pfc_seeds)
                    # a large number means the MI +infinity,
                    # since the embedding distance close to 0
                    pfc_seeds[pfc_seeds < 5e-4] = 100
                    pfc_seeds = 1. / (pfc_seeds[:, 1] + pfc_seeds[:, 0])
                elif len(pfc_seeds[0]) == 2 and print_name == 'CosDis_cluster':
                    pfc_seeds = np.array(pfc_seeds)
                    # print(pfc_seeds)
                    # a large number means the MI +infinity,
                    # since the embedding distance close to 0
                    pfc_seeds = pfc_seeds[:, 0] - pfc_seeds[:, 1]
                else:
                    pass
                pfc_mean = np.mean(pfc_seeds)
                # if name_norm in ['pair','group']:
                #     print(layer, pfc_mean)
                pfc_std = np.std(pfc_seeds)
                pfc_mean_norms[-1].append(pfc_mean)
                pfc_std_norms[-1].append(pfc_std)
        # print(pfc_mean_norms)

        savefile = self.filename_fig(print_name)
        fig, ax1 = plt.subplots()
        colors = ['r', 'b', 'k', 'm', 'g']
        markers = ['s', 'o', 'v', '^', '1', '2', '+']
        LineStyles = ['-', '--', ':', '-.']
        max_group = np.max(pfc_mean_norms[-1])
        for i, name_norm  in enumerate(print_norms):
            print(name_norm)
            print('layer', print_layers[1],  pfc_mean_norms[i][1])
            print('layer', print_layers[10], pfc_mean_norms[i][10])
            print('layer', print_layers[-1], pfc_mean_norms[i][-1])
            print(pfc_mean_norms[i])
            index_max = np.argmax(pfc_mean_norms[i])
            print('max layer', print_layers[index_max], pfc_mean_norms[i][index_max])
            print('improvement by group', (max_group - pfc_mean_norms[i][index_max]) / pfc_mean_norms[i][index_max])


            plt.plot(print_layers, pfc_mean_norms[i], LineStyles[i],
                     marker=markers[i], markersize=5, color=colors[i])


        if print_name in ['acc_test']:
            label_y = 'Accuracy'
        elif print_name in ['MI_XiX', 'MI_cluster']:
            label_y = 'Mutual information'
        elif print_name in ['dis_cluster', 'CosDis_cluster']:
            label_y = 'Group Distance'
        else:
            label_y = ''


        ax1.set_xlabel('Layers', fontsize=18)
        ax1.set_ylabel(label_y, fontsize=18)
        # plt.legend(print_norms, loc='upper right', bbox_to_anchor=(1, 0.45), fontsize=15) # 'lower left'
        plt.legend(print_norms, loc='lower left', fontsize=15)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(savefile)



    def filename_fig(self, print_name):
        filedir = f'./figs/{self.dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        residual = 1 if self.model.residual else 0
        miss_rate = int(self.miss_rate * 10)
        filename = f'{self.dataset}_{self.type_model}_{print_name}_' \
                   f'R{residual}M{miss_rate}.pdf'
        filename = os.path.join(filedir, filename)
        return filename


    def load_log(self, type_model='GCN',  dataset='PPI', load=True):
        log = {}
        filename = self.filename(filetype='logs', type_model=type_model, dataset=dataset)
        if load and os.path.exists(filename) and os.path.getsize(filename):
            log = torch.load(filename)
            print('load the log from ', filename)

        if len(log) == 0:
            log['acc_train'], log['acc_valid'], log['acc_test'] = [], [], []
            log['MI_XiX'], log['MI_XiY'] = [], []
            log['MI_NihopXi'], log['MI_NihopY'] = [], []
            log['MI_NicomXi'], log['MI_NicomY'] = [], []
            log['dis_hop'], log['dis_com'] = [], []
            log['dis'] = []
            log['dis_cluster'] = []
            log['MI_cluster'] = []
            log['dis_cluster'] = []
            log['CosDis_cluster'] = []

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

    def filename_ori(self, filetype='logs', type_model='GCN', dataset='PPI'):
        filedir = f'./{filetype}/{dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        num_layers = int(self.model.num_layers)
        residual = 1 if self.model.residual else 0
        type_norm = self.type_norm
        miss_rate = int(self.miss_rate * 10)
        seed = int(self.seed)
        filename = f'{filetype}_{type_model}_{type_norm}' \
            f'L{num_layers}R{residual}M{miss_rate}S{seed}.pth.tar'

        filename = os.path.join(filedir, filename)
        return filename


    def filename(self, filetype='logs', type_model='GCN', dataset='PPI'):
        filedir = f'./{filetype}/{dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        num_layers = int(self.model.num_layers)
        residual = 1 if self.model.residual else 0
        type_norm = self.type_norm
        miss_rate = int(self.miss_rate * 10)
        seed = int(self.seed)

        if type_norm == 'group':
            group = self.model.num_groups
            skip_weight = int(self.model.skip_weight * 1e3)
            loss_weight = int(self.loss_weight * 1e4)

            filename = f'{filetype}_{type_model}_bias_{type_norm}' \
                f'L{num_layers}R{residual}M{miss_rate}S{seed}G{group}S{skip_weight}L{loss_weight}.pth.tar'
        else:

            filename = f'{filetype}_{type_model}_{type_norm}' \
                       f'L{num_layers}R{residual}M{miss_rate}S{seed}.pth.tar'

        filename = os.path.join(filedir, filename)
        return filename

    def get_saved_info(self, path=None):
        paths = glob.glob(path)
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 2)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 2, 'epoch')
        epochs.sort()
        return epochs

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

