import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import time
import random
import argparse

from utils import accuracy
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


noise = Gauss_noise(0, 0.01)

def train(epoch, u=0.0, lip_grad_train = False):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, _ = model(features, adj)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    if lip_grad_train == True:
        lip_mat = []
        features_grad = features.detach().clone()
        features_grad.requires_grad_(True)
        out, _ = model(features_grad, adj)
        for i in range(out.shape[1]):
            v = torch.zeros_like(out)
            v[:, i] = 1
            gradients = autograd.grad(outputs=out, inputs=features_grad, grad_outputs=v,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients[idx_train]
            grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
            lip_mat.append(grad_norm)

        features_grad.requires_grad_(False)
        lip_concat = torch.cat(lip_mat, dim=1)
        lip_con_norm = torch.norm(lip_concat, dim=1)
        lip_con = torch.max(lip_con_norm)

        print("loss_lip:", lip_con)
        loss = loss_train + u * lip_con

    else:
        loss = loss_train

    features.requires_grad_(False)
    loss.backward()
    optimizer.step()

    if not args.fastmode:

        model.eval()
        output, _ = model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()

# def train(epoch, u=0.0, lip_grad_train = False):
#     t = time.time()
#     features.requires_grad_(True)
#     model.train()
#     optimizer.zero_grad()
#     output, _ = model(features, adj)
#     # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#     loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
#     acc_train = accuracy(output[idx_train], labels[idx_train])
#     if lip_grad_train == True:
#         lip_mat = []
#         # features_grad = features.detach().clone()
#         # features_grad.requires_grad_(True)
#         # out, _ = model(features, adj)
#         for i in range(output.shape[1]):
#             v = torch.zeros_like(output)
#             v[:, i] = 1
#             gradients = autograd.grad(outputs=output, inputs=features, grad_outputs=v,
#                                       create_graph=True, retain_graph=True, only_inputs=True)[0]
#             gradients = gradients[idx_train]
#             grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
#             lip_mat.append(grad_norm)
#
#         # features_grad.requires_grad_(False)
#         lip_concat = torch.cat(lip_mat, dim=1)
#         lip_con_norm = torch.norm(lip_concat, dim=1)
#         lip_con = torch.max(lip_con_norm)
#
#         print("loss_lip:", lip_con)
#         loss = loss_train + u * lip_con
#
#     else:
#         loss = loss_train
#
#     features.requires_grad_(False)
#     loss.backward()
#     optimizer.step()
#
#     if not args.fastmode:
#
#         model.eval()
#         output, _ = model(features, adj)
#
#     loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
#     acc_val = accuracy(output[idx_val], labels[idx_val])
#     print('Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss_train.data.item()),
#           'acc_train: {:.4f}'.format(acc_train.data.item()),
#           'loss_val: {:.4f}'.format(loss_val.data.item()),
#           'acc_val: {:.4f}'.format(acc_val.data.item()),
#           'time: {:.4f}s'.format(time.time() - t))
#
#     return loss_val.data.item()

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

    test_acc = []
    for epoch in range(400):
        print('___________________________________________________________________________')
        train(epoch, u=0.0015, lip_grad_train=True)  # u = 0.0006
        if (epoch + 1) % 5 == 0:
            acc_test = test(add_noise=True)
            test_acc.append(float(format(acc_test, '.4f')))

    test()