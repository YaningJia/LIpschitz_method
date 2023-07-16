"""
分层 节点Lipschitz常数相乘后取最大值
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import time
import random
import argparse

from utils import  accuracy
from dataset_process import load_data
from model import GAT
from utils import args

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data('citeseer')

model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)

# optimizer = optim.Adam(model.parameters(),
#                        lr=args.lr,
#                        weight_decay=args.weight_decay)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr)

model = model.to(device)
features = features.to(device)
adj = adj.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

att_input = []
att_output = []

class Gauss_noise():
    def __init__(self, mean=0, std=0.1):
        super(Gauss_noise, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(loc=self.mean, scale=self.std, size=data.shape)
        noise = torch.tensor(noise).float().to(device)
        data_noise = data + noise
        return data_noise.to(device)


noise = Gauss_noise(0, 0.05)

def get_model_param(model):
    a_list = []
    W_list = []
    for name, param in model.named_parameters():
        if name.find('W') != -1:
            # print(name)
            W_list.append(param)
        if name.find('.a') != -1:
            # print(name)
            a_list.append(param)

    return a_list, W_list


# def get_attention_softmax():
#     _, attention_soft = model(features, adj)
#
#     return attention_soft


def hook_function(module, input, output):
    # print(input[0])
    att_input.append(input[0])  # 输入特征与自主注意系数 保存输入至数组中
    att_output.append(output[0])  # 同时包括输出与对应的自注意力系数，网络层的输出层保存至数组中


# def get_layer_input_output():
#     handle1 = model.attentions[0].register_forward_hook(hook_function)  # 多头自注意力层注册前向hook函数,记录其输出
#     handle2 = model.out_att.register_forward_hook(hook_function)  # 最后一层自注意力层注册hook函数，记录连接后的输出
#     model(features, adj)
#     handle1.remove()
#     handle2.remove()
#
#     layers_input = att_input
#     layer_output = att_output
#
#     return layers_input, layer_output

def get_exp_mask(output):

    output_grad_exp = torch.exp(output).detach().clone()  #不这样写会造成计算图相关的问题
    output_grad_ones = torch.ones_like(output)
    mask = torch.where(output > 0, output_grad_ones, output_grad_exp)
    return mask

def Lipschitz_compute_layer(W, a, X, attention):
    a_2 = a[int(a.shape[0] / 2):, ]
    v = torch.mm(a_2.T, W.T)
    V_norm = torch.norm(v, p=2)
    S_1 = torch.zeros((attention.shape[0], 1)).to(device)
    for i in range(attention.shape[0]):
        S_1[i, 0] = attention[i, i]

    S_2 = S_1.repeat(1, W.shape[1])
    W_norm = torch.norm(W, dim=0).unsqueeze(dim=0)
    P_1 = torch.mm(S_1, W_norm)  #
    P_2 = S_2 * torch.mm(X, W)
    S_sum = torch.mm(attention, X)
    P_3 = S_2 * torch.mm(S_sum, W)
    lip_mat = (P_2 - P_3) * V_norm + P_1

    # lip_layer_con = torch.norm(P_4, dim=1).unsqueeze(dim=1)

    return lip_mat

def lip_regulation(model, att_list):

    lip_layer = []
    a_List, W_List = get_model_param(model)
    # input, output = get_layer_input_output()
    print("len(att_input)", len(att_input))
    print("len(att_output)", len(att_output))
    # att_list = get_attention_softmax()
    param_len = len(a_List) - 1  # 考虑多头自注意力层
    for i in range(param_len):
        a = a_List[i]
        W = W_List[i]
        lip_constant_mat = Lipschitz_compute_layer(W, a, att_input[i], att_list[i])
        mask = get_exp_mask(att_output[i])
        lip_constant_mat = lip_constant_mat * mask
        lip_constant = torch.norm(lip_constant_mat, dim=1).unsqueeze(dim=1)
        # print("multi_layer_{}_lip_constant:".format(i), lip_constant)
        # print(lip_constant.shape)
        lip_layer.append(lip_constant)

    lip_multi_layer = torch.cat(lip_layer, dim=1)
    lip_multi_layer_norm = torch.norm(lip_multi_layer, dim=1).unsqueeze(dim=1)
    # lip_multi_layer_max = torch.max(lip_multi_layer, dim=1)[0]
    # lip_multi_layer_max = torch.unsqueeze(lip_multi_layer_max, dim=1)

    a = a_List[param_len]
    W = W_List[param_len]

    lip_final_constant_mat = Lipschitz_compute_layer(W, a, att_input[param_len], att_list[param_len])
    mask = get_exp_mask(att_output[param_len])   # 针对最后一层是elu激活函数
    lip_final_constant_mat = lip_final_constant_mat * mask
    lip_final_constant = torch.norm(lip_final_constant_mat, dim=1).unsqueeze(dim=1)
    # print("final_lip_layer_constant:", lip_final_constant)
    # print(lip_final_constant.shape)
    lip_model_con = lip_multi_layer_norm * lip_final_constant
    # print("lip_model_con:", lip_model_con)

    return max(lip_model_con)


def train(epoch, u=0.0, add_noise=False, lip_train=False):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    handle0 = model.attentions[0].register_forward_hook(hook_function)  # 多头自注意力层注册前向hook函数,记录其输出
    handle1 = model.attentions[1].register_forward_hook(hook_function)
    handle2 = model.attentions[2].register_forward_hook(hook_function)
    handle3 = model.attentions[3].register_forward_hook(hook_function)
    handle4 = model.attentions[4].register_forward_hook(hook_function)
    handle5 = model.attentions[5].register_forward_hook(hook_function)
    handle6 = model.attentions[6].register_forward_hook(hook_function)
    handle7 = model.attentions[7].register_forward_hook(hook_function)

    handle_final = model.out_att.register_forward_hook(hook_function)  # 最后一层自注意力层注册hook函数，记录连接后的输出

    output, attention_list = model(features, adj)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    handle0.remove()
    handle1.remove()
    handle2.remove()
    handle3.remove()
    handle4.remove()
    handle5.remove()
    handle6.remove()
    handle7.remove()
    handle_final.remove()

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    if lip_train is True:
        loss_lip = lip_regulation(model, attention_list)
        print('loss_lip:', loss_lip)
        loss = loss_train + u * loss_lip
        print('loss_train:', loss_train)
        # loss = loss_lip
    else:
        loss = loss_train

    loss.backward()
    optimizer.step()

    if not args.fastmode:

        model.eval()
        if add_noise:
            features_noise = noise(features)
            output, _ = model(features_noise, adj)
        else:
            output, _ = model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def test(add_noise=False):
    model.eval()
    if add_noise:
        features_noise = noise(features)
        output, _ = model(features_noise, adj)
    else:
        output, _ = model(features, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

    return acc_test.item()
# Testing
if __name__ == '__main__':

    lip_constant = []
    for epoch in range(400):

        print('___________________________________________________________________________')
        att_input = []
        att_output = []
        train(epoch, u=0.0008, add_noise=False, lip_train=False)
        if (epoch + 1) % 5 == 0:
            test_acc = test(add_noise=True)
            lip_constant.append(float(format(test_acc, '.4f')))

    test(add_noise=True)

