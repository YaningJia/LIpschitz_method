import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT_Lip.layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)    # 最后 cancat =False
        # 输入维度：(nhid * nheads)各个 attentions 的输出 在 dim=1 连接在一起

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x_list = []
        attention_list = []
        for att in self.attentions:
            layer_out, attention = att(x, adj)
            layer_out = F.elu(layer_out)
            x_list.append(layer_out)
            attention_list.append(attention)
        x = torch.cat(x_list, dim=1)    # 将各个自注意力层的输入连接在一起
        x = F.dropout(x, self.dropout, training=self.training)
        x, attention = self.out_att(x, adj)
        attention_list.append(attention)
        x = F.elu(x)
        return x , attention_list

# class GAT_MUL(nn.Module):
#     def __init__(self, nfeat, hids, nclass, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(GAT_MUL, self).__init__()
#         self.dropout = dropout
#
#         self.input_att = [GraphAttention_Mul_Layer(nfeat, hids[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.input_att):
#             self.add_module('input_att_{}'.format(i), attention)   # attentions 为list类型，不会自动注册， 不同层的名字不能重复
#
#         self.layers = []   # 用来记录中间层，除去输入层与输出层
#         for i in range(1, len(hids)):
#             self.attentions = [GraphAttention_Mul_Layer(nheads * hids[i-1], hids[i], dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#             self.layers.append(self.attentions)
#             for j, attention in enumerate(self.attentions):
#                 self.add_module('attentions_layer_{}_{}'.format(i, j), attention)  # attentions 为list类型，不会自动注册， 不同层的名字不能重复
#
#         self.out_att = GraphAttention_Mul_Layer(nheads * hids[-1], nclass, dropout=dropout, alpha=alpha, concat=False) # module的子类，会自动注册
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.input_att], dim=1)
#         x_hin = x
#         for i in range(len(self.layers)):
#             attentions = self.layers[i]  # 开始中间层的输出
#             x_hid = torch.cat([att(x_hin, adj) for att in  attentions], dim=1)
#             x_hin = x_hid
#         x_hout = x_hin
#         x = F.dropout(x_hout, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return x

