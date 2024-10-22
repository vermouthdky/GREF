import os
import random

import numpy as np
import torch

from options.base_options import BaseOptions
from trainer import trainer

# import ray
# from ray import tune
# from ray.tune.schedulers import AsyncHyperBandScheduler

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

# seeds = [15, 100, 200, 300, 400]  #+ [123, 50, 150, 250, 350]
seeds = [5, 15, 20, 25, 35]
# seeds = [5]
# layers_GCN = list(range(1, 10, 1)) + list(range(10, 31, 5))
# layers_GCN = [4] # userd for study GCN hyperparameters: 20 ; and visulization
# layers_GCN = [1, 2, 3, 4, 7, 8, 9] + list(range(10, 31, 5))

layers_SGCN = [1, 5] + list(range(10, 121, 10))
# layers_SGCN =  [1, 5] + list(range(10, 51, 10)) + [90, 100, 110, 120] # userd for study hyperparameters: 40 #

hypers_group = [1, 5, 10, 15, 20, 30]
hypers_skip = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

# thresholds = [0.5, 0.1, 0.005, 0.002, 0.0005]


def main(args, num_samples=10, max_num_samples=500, gpus_per_trial=4):

    for seed in seeds:
        args.random_seed = seed
        set_seed(args)
        trnr = trainer(args)
        trnr.train()



if __name__ == "__main__":
    # args = build_controller_args()
    args = BaseOptions().initialize()
    main(args)