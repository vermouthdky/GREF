import torch
import torch.nn.functional as F


class act_map(torch.nn.Module):
    def __init__(self, act_type):
        super(act_map, self).__init__()
        if act_type == "linear":
            self.f = lambda x: x
        elif act_type == "elu":
            self.f = torch.nn.functional.elu
        elif act_type == "sigmoid":
            self.f = torch.sigmoid
        elif act_type == "tanh":
            self.f = torch.tanh
        elif act_type == "relu":
            self.f = torch.nn.functional.relu
        elif act_type == "relu6":
            self.f = torch.nn.functional.relu6
        elif act_type == "softplus":
            self.f = torch.nn.functional.softplus
        elif act_type == "leaky_relu":
            self.f = torch.nn.functional.leaky_relu
        else:
            raise Exception("wrong activate function")

    def forward(self, x):
        return self.f(x)

# class batch_norm(torch.nn.Module):
#     def __init__(self, dim_hidden):
#         super(batch_norm, self).__init__()
#         self.bn = torch.nn.BatchNorm1d(dim_hidden, momentum=0.5)
#
#     def forward(self, x):
#         return self.bn(x)

class batch_norm(torch.nn.Module):
    def __init__(self, dim_hidden, type_norm, skip_connect=False, num_groups=1, skip_weight=0.005):
        super(batch_norm, self).__init__()
        self.type_norm = type_norm
        self.skip_connect = skip_connect
        self.num_groups = num_groups
        self.skip_weight = skip_weight
        self.dim_hidden = dim_hidden
        if self.type_norm == 'batch':
            self.bn = torch.nn.BatchNorm1d(dim_hidden, momentum=1.)
        elif self.type_norm == 'group':
            # self.bn = torch.nn.ModuleList([])
            # for group in range(self.num_groups):
            #     self.bn.append(torch.nn.BatchNorm1d(dim_hidden, momentum=0.5)) # 0.5
            self.bn = torch.nn.BatchNorm1d(dim_hidden*self.num_groups, momentum=0.3)
            # self.bn.running_mean += torch.randn(self.bn.running_mean.size(), requires_grad=False)
            # self.bn.running_var += torch.randn(self.bn.running_var.size(), requires_grad=False)
            # self.bn.weight.data += torch.randn(self.bn.weight.size(), requires_grad=False)
            # self.bn.bias.data += torch.randn(self.bn.bias.size(), requires_grad=False)
            self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)
        else:
            pass

    def forward(self, x):
        if self.type_norm == 'None':
            return x
        elif self.type_norm == 'batch':
            # print(self.bn.running_mean.size())
            return self.bn(x)
        elif self.type_norm == 'pair':
            col_mean = x.mean(dim=0)
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = x / rownorm_mean
            return x
        elif self.type_norm in ['group', 'unSkipGroup']:
            # running_mean = torch.stack([self.bn[group].running_mean
            #                             for group in range(self.num_groups)], dim=0).t()

            if self.num_groups == 1:
                x_temp = self.bn(x)
            else:
                # running_mean = self.bn.running_mean.view(-1, self.dim_hidden).t()
                # score_cluster = F.softmax(torch.mm(x, running_mean), dim=1)
                score_cluster = F.softmax(self.group_func(x), dim=1)
                # print(score_cluster[123,])
                # x_temp = []
                # for group in range(self.num_groups):
                #     tmp = score_cluster[:, group].unsqueeze(dim=1) * x
                #     tmp = self.bn[group](tmp)
                #     x_temp.append(tmp)
                # x_temp = torch.sum(torch.stack(x_temp, dim=2), dim=2)
                # x_temp = 0.
                # for group in range(self.num_groups):
                #     x_temp += self.bn[group](score_cluster[:, group].unsqueeze(dim=1) * x)
                x_temp = torch.cat([score_cluster[:, group].unsqueeze(dim=1) * x for group in range(self.num_groups)], dim=1)
                x_temp = self.bn(x_temp).view(-1, self.num_groups, self.dim_hidden).sum(dim=1)

            if self.skip_connect:
                x = x + x_temp * self.skip_weight
                # x = x * self.skip_weight + x_temp
                # x = x_temp
            else:
                x = x_temp
            return x
        else:
            raise Exception(f'the normalization has not been implemented')