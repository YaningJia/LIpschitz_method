from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import time
import torch.nn.functional as F
import torch.optim as optim

from Sage_Lip.dataset_process import load_data
from Sage_Lip.utils import accuracy, args
from Sage_Lip.models import GraphSage, GraphSage_layer_4, GraphSage_layer_6, GraphSage_layer_8

# from utils import args

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# model_parameter_path = 'model_parameter/GCN_model_no_l2.pth'
# model_parameter_path_1 = 'model_parameter/GCN_model_untrain_no_l2.pth'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data('citeseer')

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

# Model and optimizer
model = GraphSage_layer_4(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

# optimizer = optim.Adam(model.parameters(),
#                        lr=args.lr, weight_decay=args.weight_decay)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr)

if args.cuda:
    model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

noise = Gauss_noise(0, 0.00)

def get_model_param(model):

    lin_l = []
    lin_r = []
    lin_t = []
    for layer_name, layer_param in model.named_parameters():
        if layer_name.find('lin_l') != -1:
            lin_l.append(layer_param)
        elif layer_name.find('lin_r') != -1:
            lin_r.append(layer_param)
        elif layer_name.find('lin_t') != -1:
            lin_t.append(layer_param)

    return lin_l, lin_r, lin_t, len(lin_l)

def sup_lip_constant(a, W1, W2, W):

    ones_mat = torch.ones_like(a)
    W1_norm = torch.norm(W1, dim=0).unsqueeze(dim=0)
    W1_mat = torch.matmul(ones_mat, W1_norm)
    W_mul = torch.mm(W, W2)
    W_mul_norm = torch.norm(W_mul, dim=0).unsqueeze(dim=0)
    W_mul_mat = torch.mm(a, W_mul_norm)
    lip_mat = W1_mat + W_mul_mat
    return lip_mat

def lip_regulation(model, adj):

    W_L, W_R, W, layer_len = get_model_param(model)

    a_diag = torch.zeros((adj.shape[0], 1)).to(device)    # 用来记录Adj对角元素的值
    lip_con = torch.ones((features.shape[0], 1)).to(device)

    for i in range (adj.shape[0]):
        a_diag[i, 0] = adj[i,i] + 1

    # for len in range(layer_len -1):   #只用少量的层往往效果会更好
    for i in range(layer_len):
        lip_mat = sup_lip_constant(a_diag, W_L[i], W_R[i], W[i])
        lip_norm = torch.norm(lip_mat, dim=1).unsqueeze(dim=1)  # 对于每行节点取L2_Norm从而取得其Lipschitz常数，之后求取节点的最大值
        lip_con = lip_con * lip_norm

    lip_constant = torch.max(lip_con)
    return lip_constant

def train(epoch, u=0.0, lip_train=True):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    if lip_train == True:
        loss_lip = lip_regulation(model, adj)
        print("loss_lip:", loss_lip)
        loss = loss_train + u * loss_lip
    else:
        loss = loss_train

    if epoch + 1 == 200:
        loss_lip = lip_regulation(model, adj)
        print("loss_lip:", loss_lip)
        loss = loss_train + u * loss_lip


    loss.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test(add_noise = False):
    model.eval()
    if add_noise == True:
        features_noise = noise(features)
        output = model(features_noise, adj)
    else:
        output = model(features, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

if __name__ == '__main__':

    # torch.save(model.state_dict(), model_parameter_path_1)
    test_acc = []
    for epoch in range(200):
        train(epoch, u=0.008, lip_train=False)    # cora: 0.05/0.01 citeseer:0.05(单层)   pubmed:0.005
    # torch.save(model.state_dict(), model_parameter_path)
    test()
