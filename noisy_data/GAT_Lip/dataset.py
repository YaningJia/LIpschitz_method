import torch
from utils import load_data,  args, accuracy
from model import GAT
import torch.nn as nn
import torch.nn.functional as F
from utils import Gauss_noise
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_parameter_path = 'model_parameter/GAT_model_layer_14.pth'    #训练好的模型参数
model_parameter_path_1 = 'model_parameter/GAT_model_layer_14_untrain.pth'    #初始化的模型参数

noise = Gauss_noise(0, 0.5)

adj, features, labels, idx_train, idx_val, idx_test = load_data()

adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)

print(features)
features_array = np.array(features.cpu())
np.savetxt('grad_norm_1', features_array)
# model = GAT(nfeat=features.shape[1],
#             nhid=args.hidden,
#             nclass=int(labels.max()) + 1,
#             dropout=args.dropout,
#             nheads=args.nb_heads,
#             alpha=args.alpha)
#
# model = model.to(device)
#
# model.load_state_dict(torch.load(model_parameter_path))
#
# def test(features):
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.data.item()),
#           "accuracy= {:.4f}".format(acc_test.data.item()))
#
# features = noise(features)
#
# test(features)
