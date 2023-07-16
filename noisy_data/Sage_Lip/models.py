import torch
import torch.nn as nn
import torch.nn.functional as F
from Sage_Lip.layers import GraphSageConv
from torch_geometric.nn import SAGEConv


class GraphSage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage, self).__init__()
        self.conv1 = GraphSageConv(nfeat, nhid)
        self.conv2 = GraphSageConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x, adj)
        return x

class GraphSage_layer_4(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage_layer_4, self).__init__()
        self.conv1 = GraphSageConv(nfeat, nhid)
        self.conv2 = GraphSageConv(nhid, nhid)
        self.conv3 = GraphSageConv(nhid, nhid)
        self.conv4 = GraphSageConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv4(x, adj)
        return x

class GraphSage_layer_6(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage_layer_6, self).__init__()
        self.conv1 = GraphSageConv(nfeat, nhid)
        self.conv2 = GraphSageConv(nhid, nhid)
        self.conv3 = GraphSageConv(nhid, nhid)
        self.conv4 = GraphSageConv(nhid, nhid)
        self.conv5 = GraphSageConv(nhid, nhid)
        self.conv6 = GraphSageConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv4(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv5(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv6(x, adj)

        return x


class GraphSage_layer_8(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage_layer_8, self).__init__()
        self.conv1 = GraphSageConv(nfeat, nhid)
        self.conv2 = GraphSageConv(nhid, nhid)
        self.conv3 = GraphSageConv(nhid, nhid)
        self.conv4 = GraphSageConv(nhid, nhid)
        self.conv5 = GraphSageConv(nhid, nhid)
        self.conv6 = GraphSageConv(nhid, nhid)
        self.conv7 = GraphSageConv(nhid, nhid)
        self.conv8 = GraphSageConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv4(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv5(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv6(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv7(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv8(x, adj)

        return x

class GraphSage_layer_10(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage_layer_10, self).__init__()
        self.conv1 = GraphSageConv(nfeat, nhid)
        self.conv2 = GraphSageConv(nhid, nhid)
        self.conv3 = GraphSageConv(nhid, nhid)
        self.conv4 = GraphSageConv(nhid, nhid)
        self.conv5 = GraphSageConv(nhid, nhid)
        self.conv6 = GraphSageConv(nhid, nhid)
        self.conv7 = GraphSageConv(nhid, nhid)
        self.conv8 = GraphSageConv(nhid, nhid)
        self.conv9 = GraphSageConv(nhid, nhid)
        self.conv10 = GraphSageConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv4(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv5(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv6(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv7(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv8(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv9(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv10(x, adj)

        return x

class GraphSage_layer_12(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage_layer_12, self).__init__()
        self.conv1 = GraphSageConv(nfeat, nhid)
        self.conv2 = GraphSageConv(nhid, nhid)
        self.conv3 = GraphSageConv(nhid, nhid)
        self.conv4 = GraphSageConv(nhid, nhid)
        self.conv5 = GraphSageConv(nhid, nhid)
        self.conv6 = GraphSageConv(nhid, nhid)
        self.conv7 = GraphSageConv(nhid, nhid)
        self.conv8 = GraphSageConv(nhid, nhid)
        self.conv9 = GraphSageConv(nhid, nhid)
        self.conv10 = GraphSageConv(nhid, nhid)
        self.conv11 = GraphSageConv(nhid, nhid)
        self.conv12 = GraphSageConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv4(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv5(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv6(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv7(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv8(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv9(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv10(x, adj)
        x = F.relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv11(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv12(x, adj)

        return x
# class GraphSage(nn.Module):
#
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GraphSage, self).__init__()
#         self.conv1 = GraphSageConv(nfeat, nhid)
#         self.conv2 = GraphSageConv(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         lip_x = []
#         x, x_1 = self.conv1(x, adj)
#         lip_x.append(x_1)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout,training=self.training)
#         x, x_2 = self.conv2(x, adj)
#         lip_x.append(x_2)
#         return x, lip_x

