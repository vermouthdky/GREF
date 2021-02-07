import argparse


def reset_weight(args):

    if args.dataset == 'Citeseer' and args.miss_rate == 0.:
        if args.type_model in ['GAT', 'GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.005
        elif args.type_model in ['simpleGCN']:
            # args.skip_weight = 0.0005 if args.num_layers < 70 else 0.001 # for original group norm
            # 0.002 > 0.003 > 0.001 > 0.005
            args.skip_weight = 0.0005 if args.num_layers < 60 else 0.002

    elif args.dataset == 'Citeseer' and args.miss_rate == 1.:
        if args.type_model in ['GCN']:
            args.skip_weight = 0.005
        elif args.type_model in ['GAT']:
            # args.skip_weight = 0.005 # for original group norm
            args.skip_weight = 0.01
        elif args.type_model in ['simpleGCN']:
            # args.epochs = 500 # for original group norm
            # args.skip_weight = 0.001 # for original group norm
            # 0.0005  > 0.001                                  Leave for tuning
            args.skip_weight = 0.0005

    elif args.dataset == 'Pubmed' and args.miss_rate == 0.:
        if args.type_model in ['GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.01
        elif args.type_model in ['GAT']:
            args.skip_weight = 0.005 if args.num_layers < 6 else 0.01  #
        elif args.type_model in ['simpleGCN']:
            args.skip_weight = 0.05  # 0.05 > 0.1
            # args.skip_weight = 0.05 # for original group norm

    elif args.dataset == 'Pubmed' and args.miss_rate == 1.:
        if args.type_model in ['GCN']:
            # args.skip_weight = 0.02
            args.skip_weight = 0.02
        elif args.type_model in ['GAT']:
            args.skip_weight = 0.03
        elif args.type_model in ['simpleGCN']:
            args.skip_weight = 0.05
        # args.skip_weight = 0.03# for original group norm

    elif args.dataset == 'Cora' and args.miss_rate == 0.:
        if args.type_model in ['GCN']:
            # args.skip_weight = 0.001 if args.num_layers < 6 else 0.005
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.03  # 0.03
        elif args.type_model in ['GAT']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.01  # 0.03
        elif args.type_model in ['simpleGCN']:
            args.skip_weight = 0.01 if args.num_layers < 60 else 0.005  # 0.005
            # args.skip_weight = 0.01 # for original group norm

    elif args.dataset == 'Cora' and args.miss_rate == 1.:
        if args.type_model in ['GCN', 'GAT']:
            args.skip_weight = 0.01
        elif args.type_model in ['simpleGCN']:
            args.skip_weight = 0.005 if args.num_layers < 70 else 0.03  # 0.01 first part
            # args.skip_weight = 0.001

        # args.skip_weight = 0.01 #  for original group norm, GCN, GAT and simpleGCN

    elif args.dataset == 'CoauthorCS' and args.miss_rate == 0.:
        if args.type_model in ['GAT', 'GCN']:
            args.skip_weight = 0.001 if args.num_layers < 6 else 0.03
        elif args.type_model in ['simpleGCN']:
            # args.skip_weight = 0.001 if args.num_layers < 60 else 0.03 # 0.03, for original group norm
            # args.skip_weight = 0.001 if args.num_layers < 60 else .3 # 0.3 > 0.1 > 0.05 > 0.03
            args.epochs = 500
            args.skip_weight = 0.001 if args.num_layers < 10 else .5  # 0.5
    elif args.dataset == 'CoauthorCS' and args.miss_rate == 1.:
        if args.type_model in ['GCN']:
            args.skip_weight = 0.03
        elif args.type_model in ['GAT']:
            args.skip_weight = 0.03
        elif args.type_model in ['simpleGCN']:
            # args.skip_weight = 1. # 0.003 > 0.005 > 0.01 > > 0.001 > 0.03
            args.epochs = 500
            args.skip_weight = 0.003 if args.num_layers < 30 else 0.5  # 0.5

        # args.skip_weight = 0.03 # 0.03, for original group norm

    return args


class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self):
        parser = argparse.ArgumentParser(description='GNN-Mutual Information')

        # build up the common parameter
        parser.add_argument('--random_seed', type=int, default=123)
        parser.add_argument("--cuda", type=bool, default=True, required=False,
                            help="run in cuda mode")
        parser.add_argument('--cuda_num', type=int,
                            default=0, help="GPU number")
        parser.add_argument("--dataset", type=str, default="Cora", required=False,
                            help="The input dataset.")
        parser.add_argument('--save_interval', type=int, default=50,
                            help='time interval to save model parameter')

        # build up the supernet hyperparameter
        parser.add_argument('--type_model', type=str, default="NLGCN")
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument("--epochs", type=int, default=500, help="number of training the one shot model")
        parser.add_argument("--multi_label", type=bool, default=False, help="multi_label or single_label task")
        parser.add_argument("--dropout_c", type=float, default=0.6, help="adj matrix dropout rate")
        parser.add_argument("--dropout_n", type=float, default=0.08, help='dense layer dropout rate')
        parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
        parser.add_argument('--weight_decay', type=float, default=5e-4)  # 5e-4
        parser.add_argument('--grad_clip', type=float, default=0.0)
        parser.add_argument('--dim_hidden', type=int, default=64)
        parser.add_argument('--auto_continue', type=bool, default=False)
        parser.add_argument('--transductive', type=bool, default=True)
        parser.add_argument('--activation', type=str, default="relu", required=False)
        parser.add_argument('--residual', default=False, action="store_true")
        parser.add_argument('--batch_normal', default=False, action="store_true")
        parser.add_argument('--early_stop', default=True, action="store_true")
        parser.add_argument('--num_neighbors', type=float, default=1.0)
        parser.add_argument('--threshold', type=float, default=0.6)
        parser.add_argument('--temperature', type=float, default=1e-6,
                            help='temperature for sigmoid in adj matrix sparsification')
        parser.add_argument('--ks', type=float, nargs='+', default=[0.5, 0.5, 0.5, 0.5])
        parser.add_argument('--n_att', type=int, default=1, help='the number of multi attention strategy')

        parser.add_argument('--alpha', type=float, default=2)
        parser.add_argument('--lamb', type=float, default=0.0005)
        parser.add_argument('--gamma', type=float, default=0.001)
        parser.add_argument('--beta', type=float, default=0)

        parser.add_argument('--ptb_rate', type=float, default=0)
        parser.add_argument('--ptb', type=bool, default=False)
        parser.add_argument('--ptb_type', type=str, default='meta')

        parser.add_argument('--metric', type=str, default='attention')

        parser.add_argument('--adj_dropout', type=float, default=0.5,
                            help="dropout rate in APPNP")  # 5e-4
        parser.add_argument('--dropout', type=float, default=0.6)
        parser.add_argument('--type_norm', type=str, default="None")
        args = parser.parse_args()
        args = self.reset_model_parameter(args)
        return args

    def reset_model_parameter(self, args):
        if args.dataset == 'PPI':
            args.num_feats = 50
            args.num_classes = 121
            args.dropout_c = 0.9
            args.weight_decay = 0
            args.lr = 0.05
            args.residual = False
            args.batch_normal = False
            args.multi_label = True
            args.transductive = False

            args.alpha = 1
            args.lamb = 0.0005
            args.gamma = 0.001
            args.beta = 0

        elif args.dataset == 'Cora':
            args.num_feats = 1433
            args.num_classes = 7
            args.dropout_c = 0.9  # 0.5
            args.lr = 0.005  # 0.005
            args.multi_label = False

            args.alpha = 1
            args.lamb = 0.02
            args.gamma = 0.04
            args.beta = 0.0

            args.ks = [0.5, 0.5, 0.5, 0.5]

        elif args.dataset == 'Citeseer':
            args.num_feats = 3703
            args.num_classes = 6
            args.dropout_c = 0.9
            args.weight_decay = 5e-4 #5e-4
            args.lr = 0.005
            args.multi_label = False

            args.alpha = 0.6
            args.lamb = 0.0001
            args.gamma = 0.0
            args.beta = 0.0

            args.ks = [0.6, 0.5, 0.5, 0.4]

        elif args.dataset == 'Pubmed':
            args.num_feats = 500
            args.num_classes = 3
            args.dropout_c = 0.5
            args.weight_decay = 1e-3
            args.lr = 0.01
            args.multi_label = False

            args.alpha = 5
            args.lamb = 0.4
            args.gamma = 0.0
            args.beta = 0.0

            args.ks = [0.1, 0.5, 0.5, 0.4]

        elif args.dataset == 'Reddit':
            args.num_feats = 602
            args.num_classes = 41
            args.dropout_c = 0.
            args.weight_decay = 0.
            args.lr = 0.005
            args.multi_label = False

        elif args.dataset == 'CoauthorCS':
            args.num_feats = 6805
            args.num_classes = 15
            args.dropout_c = 0.6
            args.weight_decay = 5e-5
            args.lr = 0.005
            args.multi_label = False
        return args
