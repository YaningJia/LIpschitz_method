import torch.nn as nn
import torch.nn.functional as F
from GCN_Lip.layers import GraphConvolution

# def my_relu(x):
#     return torch.maximum(x, torch.zeros_like(x))

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class GCN_layer_3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return x

class GCN_layer_4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_4, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        return x

class GCN_layer_5(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_5, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        return x

class GCN_layer_6(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_6, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)

        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        return x

class GCN_layer_7(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_7, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)

        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = self.gc7(x, adj)
        return x

class GCN_layer_8(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_8, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc8(x, adj)
        return x

class GCN_layer_9(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_9, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc9(x, adj))
        return x

class GCN_layer_10(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_10, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc10(x, adj)

        return x

class GCN_layer_11(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_11, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.relu(self.gc10(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc11(x, adj)
        return x

class GCN_layer_12(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_12, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc10(x, adj))
        x = F.relu(self.gc11(x, adj))
        x = self.gc12(x, adj)
        return x

class GCN_layer_14(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_14, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc10(x, adj))
        x = F.relu(self.gc11(x, adj))
        x = F.relu(self.gc12(x, adj))
        x = F.relu(self.gc13(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc14(x, adj)
        return x

class GCN_layer_16(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_layer_16, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nhid)
        self.gc8 = GraphConvolution(nhid, nhid)
        self.gc9 = GraphConvolution(nhid, nhid)
        self.gc10 = GraphConvolution(nhid, nhid)
        self.gc11 = GraphConvolution(nhid, nhid)
        self.gc12 = GraphConvolution(nhid, nhid)
        self.gc13 = GraphConvolution(nhid, nhid)
        self.gc14 = GraphConvolution(nhid, nhid)
        self.gc15 = GraphConvolution(nhid, nhid)
        self.gc16 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc10(x, adj))
        x = F.relu(self.gc11(x, adj))
        x = F.relu(self.gc12(x, adj))
        x = F.relu(self.gc13(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc14(x, adj))
        x = F.relu(self.gc15(x, adj))
        x = self.gc16(x, adj)
        return x

