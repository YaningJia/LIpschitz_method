import math
import torch
import torch_scatter
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
import torch.nn as nn


class GraphSageConv(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        super(GraphSageConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_t = Parameter(torch.FloatTensor(self.in_channels, self.in_channels))
        self.lin_l = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        self.lin_r = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        if bias:
            self.bias = Parameter(torch.FloatTensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.lin_t.size(1))
        self.lin_t.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.lin_l.size(1))
        self.lin_l.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.lin_r.size(1))
        self.lin_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_1 = torch.matmul(x, self.lin_t)
        if self.bias is not None:
            x_1 = x_1 + self.bias
        x_relu = F.relu(x_1)
        x_prop = torch.matmul(adj, x_relu)
        x = torch.matmul(x, self.lin_l) + torch.matmul(x_prop, self.lin_r)

        if self.normalize:
            x = F.normalize(x)

        out = x
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

# class GraphSageConv(MessagePassing):
#
#     def __init__(self, in_channels, out_channels, normalize=True):
#         super(GraphSageConv, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.normalize = normalize
#
#         self.lin_l = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
#         self.lin_r = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.lin_l.size(1))
#         self.lin_l.data.uniform_(-stdv, stdv)
#
#         stdv = 1. / math.sqrt(self.lin_r.size(1))
#         self.lin_r.data.uniform_(-stdv, stdv)
#
#     def forward(self, x, adj):
#
#         x_prop = torch.matmul(adj, x)
#         x = torch.matmul(x, self.lin_l) + torch.matmul(x_prop, self.lin_r)
#
#         if self.normalize:
#             x = F.normalize(x)
#
#         out = x
#         return out
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_channels) + ' -> ' \
#                + str(self.out_channels) + ')'

# class GraphSageConv(MessagePassing):
#
#     def __init__(self, in_channels, out_channels, normalize=True):
#         super(GraphSageConv, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.normalize = normalize
#
#         self.lin_l = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
#         self.lin_r = Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.lin_l.size(1))
#         self.lin_l.data.uniform_(-stdv, stdv)
#
#         stdv = 1. / math.sqrt(self.lin_r.size(1))
#         self.lin_r.data.uniform_(-stdv, stdv)
#
#     def forward(self, x, adj):
#
#         x_prop = torch.matmul(adj, x)
#         x = torch.matmul(x, self.lin_l) + torch.matmul(x_prop, self.lin_r)
#         out1 = x
#
#         if self.normalize:
#             x = F.normalize(x)
#
#         out = x
#         return out, out1
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_channels) + ' -> ' \
#                + str(self.out_channels) + ')'