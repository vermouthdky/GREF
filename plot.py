
import os
import torch
import numpy as np
import ipdb
def load_log(filename, dataset="Cora"):
        log = {}
        filename = f"./logs/{dataset}/"+filename

        if os.path.exists(filename) and os.path.getsize(filename):
            log = torch.load(filename)
            print("load the log from ", filename)
        log = np.array(log["acc_test"])
        return log

def print_summary(log):
    mean = np.mean(log, axis=0)
    std = np.std(log, axis=0)
    print(
        "{:.2f}±{:.2f}".format(100*mean[0], 100*std[0])
    )

def print_value(log):
    for i in range(5):
        print("{:.2f}".format(100*log[i][0]))

if __name__ == "__main__":

    # for model in ['NLGCN']:
    #     for metric in ['attention', 'identity']:
    #         for ptb_rate in ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25']:
    #             file = 'logs_'+model+metric+'ptb'+ptb_rate+'.pth.tar'
    #             log = load_log(file, dataset='Cora')
    #             # print_value(log)
    #             print_summary(log)
    file = 'logs_'+'GAT'+'layers4'+'.pth.tar'
    log = load_log(file, dataset='Cora')
    print_value(log)
    print_summary(log)

    # parameter study


    # alpha_results = np.zeros([30, 9])
    # lamb_results = np.zeros([30, 9])
    # gamma_results = np.zeros([30, 9])
    # for j, ptb_rate in enumerate(['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4']):
    #     # for i, alpha in enumerate(['0.0', '0.5', '1.0', '2.0', '4.0', '8.0']):
    #     #     file = 'logs_NLGCNalpha'+alpha+'_ptb'+ptb_rate+'.pth.tar'
    #     #     log = load_log(file, dataset='Cora')
    #     #     alpha_results[5*i:5*i+5, j] = log[:, 0]
    #     for i, lamb in enumerate(['0.0', '5e-05', '0.0005', '0.005', '0.05', '0.5']):
    #         file = 'logs_NLGCNlamb'+lamb+'_ptb'+ptb_rate+'.pth.tar'
    #         log = load_log(file, dataset='Cora')
    #         lamb_results[5*i:5*i+5, j] = log[:, 0]
    #     for i, gamma in enumerate(['0.0', '1e-05', '0.0001', '0.001', '0.01', '0.1']):
    #         file = 'logs_NLGCNgamma'+gamma+'_ptb'+ptb_rate+'.pth.tar'
    #         log = load_log(file, dataset='Cora')
    #         gamma_results[5*i:5*i+5, j] = log[:, 0]
    # np.savetxt("alpha.csv", alpha_results, delimiter=',')
    # np.savetxt("lamb.csv", lamb_results, delimiter=',')
    # np.savetxt("gamma.csv", gamma_results, delimiter=',')
