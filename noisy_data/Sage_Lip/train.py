from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# from dataset_process import load_data
from utils import accuracy, args, load_data
from models import GraphSage
# from utils import args

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

adj, features, labels, idx_train, idx_val, idx_test = load_data('citeseer')

# model_parameter_path_1 = 'Lipschitz_parameter/GraphSage_model_untrained_citeseer.pth'
# model_parameter_path_2 = 'Lipschitz_parameter/GraphSage_model_train_normal_citeseer.pth'
# model_parameter_path_3 = 'Lipschitz_parameter/GraphSage_model_train_l2_citeseer.pth'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Model and optimizer
model = GraphSage(nfeat=features.shape[1],
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

def train(epoch):

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))

def test():
    model.eval()
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
    for epoch in range(400):
        train(epoch)

        if (epoch + 1) % 5 == 0:
            acc_test = test()
            test_acc.append(float(format(acc_test, '.4f')))

    # torch.save(model.state_dict(), model_parameter_path_2)
    print(test_acc)
    print(max(test_acc))
