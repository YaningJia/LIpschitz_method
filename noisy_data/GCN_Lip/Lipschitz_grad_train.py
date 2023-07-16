from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import autograd
from GCN_Lip.utils import accuracy, args
from GCN_Lip.models import GCN_layer_10
from GCN_Lip.dataset_process import load_data
from torchviz import make_dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_parameter_path = 'model_parameter/GCN_model_no_l2.pth'
# model_parameter_path_1 = 'model_parameter/GCN_model_untrain_no_l2.pth'



np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')

# Model and optimizer
model = GCN_layer_10(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

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

noise = Gauss_noise(0, 0.00)
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


def grad_regulation(features, model):  #用于计算对于梯度的值

    input = features.detach()
    input.requires_grad_(True)
    out = model(input, adj)

    lip_constant = []
    for i in range(out.shape[1]):
        v = torch.zeros_like(out).to(device)
        v[:,i] = 1

        gradients = autograd.grad(outputs=out, inputs=input, grad_outputs=v,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_norm = torch.norm(gradients, p=2, dim=1)
        print("grad_norm:", grad_norm)
        lip_constant.append(torch.max(grad_norm))
        gradients.data.zero_()

    print("lip_constant:", lip_constant)
    lip_con = max(lip_constant)
    return lip_con

# def train(epoch, u = 0.0, lip_grad_train=False):
#
#     model.train()
#     optimizer.zero_grad()
#     output = model(features, adj)
#
#     loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
#     acc_train = accuracy(output[idx_train], labels[idx_train])
#
#     if lip_grad_train is True:
#
#         lip_grad_loss = grad_regulation(features, model)
#         print("lip_grad_loss:", lip_grad_loss)
#         print("loss_train:", loss_train)
#         loss = loss_train + u * lip_grad_loss
#         # loss = lip_grad_loss
#     else:
#         loss = loss_train
#
#     print("features.requires_grad:", features.requires_grad)
#     # make_dot(loss, params=dict(list(model.named_parameters()))).view() # 计算图可视化操作
#
#     loss.backward()
#     # loss.detach_().requires_grad_(True)
#     optimizer.step()
#
#     if not args.fastmode:
#         model.eval()
#         output = model(features, adj)
#
#     loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
#     acc_val = accuracy(output[idx_val], labels[idx_val])
#     print('Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss.item()),
#           'acc_train: {:.4f}'.format(acc_train.item()),
#           'loss_val: {:.4f}'.format(loss_val.item()),
#           'acc_val: {:.4f}'.format(acc_val.item()))


def train(epoch, u = 0.0, lip_grad_train=False):

    # features.requires_grad_(True)
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    if lip_grad_train == True:
        features_grad = features.detach().clone()
        features_grad.requires_grad_(True)
        out = model(features_grad, adj)
        v = torch.ones_like(output)
        gradients = autograd.grad(outputs=out, inputs=features_grad, grad_outputs=v,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients[idx_train]
        features_grad.requires_grad_(False)
        # print("gradients:", gradients)
        grad_norm = ((torch.norm(gradients, dim=1))).mean()
        print("loss_lip:", grad_norm)
        loss = loss_train + u * grad_norm
    else:
        loss = loss_train

    # print("features.requires_grad:", features.requires_grad)
    # make_dot(loss, params=dict(list(model.named_parameters()))).view() # 计算图可视化操作
    # features.requires_grad_(False)
    # print("features.requires_grad:", features.requires_grad)
    loss.backward()
    # loss.detach_().requires_grad_(True)
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
          'acc_val: {:.4f}'.format(acc_val.item()))

def test(test_noise = False):
    model.eval()
    if test_noise == True:
         features_noise = noise(features)
    else:
        features_noise = features

    output = model(features_noise, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

if __name__ == '__main__':

    # torch.save(model.state_dict(), model_parameter_path_1)
    test_acc = []
    for epoch in range(300):
        train(epoch, u=0.01, lip_grad_train=False)  #cora:0.01
    test()