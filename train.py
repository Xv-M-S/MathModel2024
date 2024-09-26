import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_snapshot', type=int, default=10, help='number of snapshot')
parser.add_argument('--layer', type=int, default=2, help='number of layers')
parser.add_argument('--dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--path', type=str, default='', help='path of datasets')
parser.add_argument('--target', type=str, default='', help='train for target')
parser.add_argument('--save_name', type=str, default='dgnn', help='save name of model')
args = parser.parse_args()



import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import dgl
import pickle
import xgboost
import time
import numpy as np
from minibatch import get_dataloader
from model import NetModel
from datetime import datetime
from sklearn import linear_model

t_inf = 0
valid_targets = {'throughput','mcs_nss','seq_time'}
if args.target not in valid_targets:
    print(f"错误：目标 '{args.target}' 不合法。有效目标包括：{valid_targets}")
    sys.exit(1)
train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args.path, args.batch_size, args.target, all_cuda=True)
model = NetModel(args.layer, args.dim, args.num_snapshot, args.target).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def eval(model, dataloader, target):
    global t_inf
    model.eval()
    rmse = list()
    rmse_tot = 0
    with torch.no_grad():
        for g, idx in dataloader:
            t_s = time.time()
            g = g.to('cuda')
            mask = g.nodes['ap'].data['mask'] > 0
            mask = mask.view(-1)
            pred = model(g)[mask]
            if target == "mcs_nss":
                true = g.nodes['ap'].data['predict'][mask][:,1:3].view(-1,2) # MCS,NSS
            elif target == "seq_time":
                true = g.nodes['ap'].data['predict'][mask][:,7].view(-1,1) # seq_time
            elif target == "throughput":
                true = g.nodes['ap'].data['predict'][mask][:,8].view(-1,1) # 吞吐量
            loss = torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8)
            rmse.append(float(loss) * pred.shape[0])
            rmse_tot += pred.shape[0]
            t_inf += time.time() - t_s
    return np.sum(np.array(rmse)) / rmse_tot

# torch.autograd.set_detect_anomaly(True)
best_e = 0
best_valid_rmse = float('inf')
model_fn = 'models/{}.pkl'.format(args.save_name)
if not os.path.exists('models'):
    os.mkdir('models')
for e in range(args.epoch):
    model.train()
    train_rmse = list()
    rmse_tot = 0
    for g, _ in train_dataloader:
        g = g.to('cuda')
        optimizer.zero_grad()
        mask = g.nodes['ap'].data['mask'] > 0
        mask = mask.view(-1)
        # mask = mask.unsqueeze(1)
        pred = model(g)[mask]
        if args.target == "mcs_nss":
            true = g.nodes['ap'].data['predict'][mask][:,1:3].view(-1,2) # MCS,NSS
        elif args.target == "seq_time":
            true = g.nodes['ap'].data['predict'][mask][:,7].view(-1,1) # seq_time
        elif args.target == "throughput":
            true = g.nodes['ap'].data['predict'][mask][:,8].view(-1,1) # 吞吐量
        # print(pred.shape)
        # print(true.shape)
        loss = torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8)
        train_rmse.append(float(loss) * pred.shape[0])
        rmse_tot += pred.shape[0]
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()
    train_rmse = np.sum(np.array(train_rmse)) / rmse_tot
    valid_rmse = eval(model, valid_dataloader, args.target)
    print('Epoch: {} Training RMSE: {:.4f} Validation RMSE: {:.4f}'.format(e, train_rmse, valid_rmse))
    if valid_rmse < best_valid_rmse:
        best_e = e
        best_valid_rmse = valid_rmse
        torch.save(model.state_dict(), model_fn)

print('Loading model in epoch {}...'.format(best_e))
model.load_state_dict(torch.load(model_fn, weights_only=True))
print('Test RMSE: {:.4f}'.format(eval(model, test_dataloader, args.target)))

print('Inference time per sequence: {:.4f}ms'.format(t_inf * 1000 / test_dataloader.__len__()))