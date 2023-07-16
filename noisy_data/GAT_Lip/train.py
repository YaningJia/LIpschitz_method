import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import time
import random
import argparse

from utils import accuracy
from dataset_process import load_data
from model import GAT
from utils import args
from torch.cuda.amp import autocast as autocast, GradScaler
# from apex import amp
scaler = GradScaler()

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')
# model_parameter_path_1 = 'Lipschitz_parameter/GAT_model_untrained_citeseer.pth'
# model_parameter_path_2 = 'Lipschitz_parameter/GAT_model_train_normal_citeseer.pth'
# model_parameter_path_3 = 'Lipschitz_parameter/GAT_model_train_l2_citeseer.pth'

model = GAT(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)
#
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

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)

# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
def train(epoch):
    t = time.time()

    model.train()

    optimizer.zero_grad()
    with autocast():
       output, _ = model(features, adj)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
       loss_train = F.cross_entropy(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])

    scaler.scale(loss_train).backward()
    # loss_train.backward()
    scaler.step(optimizer)
    # optimizer.step()
    scaler.update()

    if not args.fastmode:

        model.eval()
        with autocast():
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

def test():
    model.eval()
    with autocast():
       output, _ = model(features, adj)
       loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    return acc_test.item()

# Testing
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