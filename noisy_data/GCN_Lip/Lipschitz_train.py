"""
每层节点相乘后取最大值 ,引入了relu激活函数的处理
"""

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import time

from GCN_Lip.utils import accuracy, args
from GCN_Lip.models import GCN, GCN_layer_3, GCN_layer_4, GCN_layer_5
from GCN_Lip.dataset_process import load_data
import time

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# model_parameter_path_1 = '../model_parameter/GCN_model_layer_16_cora_lip_relu.pth'
# model_parameter_path_2 = '../model_parameter/GCN_model_layer_16_citeseer_lip_relu.pth'
# model_parameter_path_3 = '../model_parameter/GCN_model_layer_16_pubmed_lip_relu.pth'
# model_parameter_path_1 = '../model_parameter_train/GCN_model_layer_4_pubmed_Normal.pth'
# model_parameter_path_2 = '../model_parameter_train/GCN_model_layer_4_pubmed_L2_Reg.pth'
# model_parameter_path_3 = '../model_parameter_train/GCN_model_layer_4_pubmed_Lip_Node.pth'
# model_parameter_path_4 = '../model_parameter_train/GCN_model_layer_4_pubmed_Lip_ReLU.pth'
# model_parameter_path_5 = '../model_parameter_train/GCN_model_layer_4_pubmed_Lip_Grad.pth'

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

noise = Gauss_noise(0, 0.0)

layer_output = []
layer_next_relu = [0, 1, 2, 3]      #当前层后面如果接relu层的话记录层数

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data('citeseer')

# Model and optimizer
model = GCN_layer_5(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
# #
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


def hook_function(module, input, output):
    layer_output.append(output)


def get_out_mask(output):
    mask = torch.where(output > 0.0, 1.0, 0.0)
    return mask

def get_model_param(model):

    W = []
    for layer_name, layer_param in model.named_parameters():
        if layer_name.find('weight')!= -1:
            W.append(layer_param)

    layer_len = len(W)
    return W, layer_len

def sup_lip_constant(weight, adj, mask):

    weight_norm = torch.norm(weight, dim=0).unsqueeze(dim=0).to(device)

    lip_mat = torch.mm(adj, weight_norm)

    return lip_mat * mask

def lip_regulation(model, adj):                    #计算lip_loss，即网络的lipschitz常数

    adj_diag = torch.zeros((adj.shape[0], 1)).to(device)    # 用来记录Adj对角元素的值

    for i in range (adj.shape[0]):
        adj_diag[i, 0] = adj[i,i]

    lip_con = torch.ones((features.shape[0], 1)).to(device)

    W_list , layer_len = get_model_param(model)

    for i in range(layer_len):
        if i in layer_next_relu:
            mask = get_out_mask(layer_output[i])
        else:
            mask = torch.ones_like(layer_output[i])

        W = W_list[i]
        lip_mat = sup_lip_constant(W, adj_diag, mask)
        lip_mat_norm = torch.norm(lip_mat, dim=1).unsqueeze(dim=1)
        lip_con = lip_con * lip_mat_norm

    lip_constant = torch.max(lip_con)   #这处不要改成 max(lip_con)

    return lip_constant

def train(epoch, u = 0.0, lip_train=False):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    handle_1 = model.gc1.register_forward_hook(hook_function)
    handle_2 = model.gc2.register_forward_hook(hook_function)
    handle_3 = model.gc3.register_forward_hook(hook_function)
    handle_4 = model.gc4.register_forward_hook(hook_function)
    handle_5 = model.gc5.register_forward_hook(hook_function)
    # handle_6 = model.gc6.register_forward_hook(hook_function)
    # handle_7 = model.gc7.register_forward_hook(hook_function)
    # handle_8 = model.gc8.register_forward_hook(hook_function)
    # handle_9 = model.gc9.register_forward_hook(hook_function)
    # handle_10 = model.gc10.register_forward_hook(hook_function)
    # handle_11 = model.gc11.register_forward_hook(hook_function)
    # handle_12 = model.gc12.register_forward_hook(hook_function)
    # handle_13 = model.gc13.register_forward_hook(hook_function)
    # handle_14 = model.gc14.register_forward_hook(hook_function)
    # handle_15 = model.gc15.register_forward_hook(hook_function)
    # handle_16 = model.gc16.register_forward_hook(hook_function)

    output = model(features, adj)
    handle_1.remove()
    handle_2.remove()
    handle_3.remove()
    handle_4.remove()
    handle_5.remove()
    # handle_6.remove()
    # handle_7.remove()
    # handle_8.remove()
    # handle_9.remove()
    # handle_10.remove()
    # handle_11.remove()
    # handle_12.remove()
    # handle_13.remove()
    # handle_14.remove()
    # handle_15.remove()
    # handle_16.remove()

    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_lip = lip_regulation(model, adj)
    if lip_train == True:

        print('loss_lip:', loss_lip)
        loss = loss_train + u * loss_lip
        # print('loss_train:', loss_train)
        # loss = loss_lip
    else:
        loss = loss_train

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

    return acc_train.item(), loss_lip.item(), loss.item()


def test():
    model.eval()
    features_noise = noise(features)
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
    train_acc = []
    lip_con = []
    loss = []
    acc_test_best = 0.0
    for epoch in range(400):
        layer_output = []
        print('____________________________________________________________________________________')
        acc_train, lip_constant, loss_train = train(epoch, u=0.00001, lip_train=True)   # cora:u=0.01/0.05 citeseer:u=0.001 pubmed: u=0.01/0.05(更好)
        # loss.append(float(format(loss_train, '.4f')))
        if (epoch + 1) % 5 == 0:
            acc_test = test()
            if acc_test > acc_test_best:
                acc_test_best = float(format(acc_test, '.4f'))
                torch.save(model.state_dict(), model_parameter_path_3)
    test()