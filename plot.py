from options.base_options import BaseOptions
from trainer import trainer


def load_log(args, type_model='NLGCN', dataset='Cora'):
    trnr = trainer(args)
    logs = trnr.load_log(type_model=type_model, dataset=dataset, load=True)
    return logs


def plot_accuracy(args, type_model='NLGCN', dataset='Cora'):
    logs = load_log(args, type_model=type_model, dataset=dataset)


if __name__ == "__main__":
    # args = build_controller_args()
    args = BaseOptions().initialize()
    print(load_log(args))
