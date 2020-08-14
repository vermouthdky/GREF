from options.base_options import BaseOptions, reset_weight
from trainer import trainer
import torch
import os
import numpy as np
import random

def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

# seeds = [100, 200, 300, 400, 500] #  + [123, 50, 150, 250, 350, 450]
seeds = [200]
# layers_GCN = list(range(1, 10, 1)) + list(range(10, 31, 5))
layers_GCN = [20] # userd for study GCN hyperparameters: 20 ; and visulization
# layers_GCN = [1, 2, 3, 4, 7, 8, 9] + list(range(10, 31, 5))

layers_SGCN = [1, 5] + list(range(10, 121, 10))
# layers_SGCN =  [1, 5] + list(range(10, 51, 10)) + [90, 100, 110, 120] # userd for study hyperparameters: 40 #

hypers_group = [1, 5, 10, 15, 20, 30]
hypers_skip = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

def main(args):

    # set random seed
    if args.train:
        if args.type_model in ['GCN', 'GAT']:
            layers = layers_GCN
        else:
            layers = layers_SGCN

        for layer in layers:
            args.num_layers = layer
            if args.type_norm == 'group':
                args = reset_weight(args)
            for seed in seeds:
                args.random_seed = seed
                set_seed(args)
                trnr = trainer(args)
                trnr.train_compute_MI()



if __name__ == "__main__":
    # args = build_controller_args()
    args = BaseOptions().initialize()
    main(args)