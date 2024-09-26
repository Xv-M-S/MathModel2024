import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_snapshot', type=int, default=1, help='number of snapshot')
parser.add_argument('--layer', type=int, default=2, help='number of layers')
parser.add_argument('--dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--model_path', type=str, default='', help='path of model datasets')
parser.add_argument('--data_path', type=str, default='', help='path of eval datasets')
parser.add_argument('--target', type=str, default='throughput', help='name of predict item')
parser.add_argument('--eval_lever', type=int, default=0, help='0 for predict local spilt_test datasets; 1 for predict local spilt_valid datasets;')
args = parser.parse_args()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

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


train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args.data_path, args.batch_size, args.target, all_cuda=True)
model = NetModel(args.layer, args.dim, args.num_snapshot, args.target).cuda()

def eval(model, dataloader, target):
    global t_inf
    model.eval()
    rmse = list()
    rmse_tot = 0
    with torch.no_grad():
        count = 0
        sum_error = 0
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
            
            if target == "mcs_nss":
                for i in range(pred.size(0)):
                    for j in range(pred.size(1)):
                        if true[i, j] != 0:  # 仅在真实值不为0时计算
                            sum_error += torch.abs(pred[i][j] - true[i][j])/true[i][j] * 100
                            count += 1
            elif target == "throughput":
                for i in range(pred.size(0)):
                    for j in range(pred.size(1)):
                        if true[i, j] != 0:  # 仅在真实值不为0时计算
                            sum_error += torch.abs(pred[i][j] - true[i][j])/true[i][j] * 100
                            count += 1
            elif target == "seq_time":
                absolute_error = torch.abs(pred - true)/true * 100
                sum_error += absolute_error.sum(dim=0)/absolute_error.shape[0]
                count += 1

            rmse.append(float(loss) * pred.shape[0])
            rmse_tot += pred.shape[0]
            t_inf += time.time() - t_s
        error = sum_error.cpu().detach().numpy()/count
        print(error)
        print('Relative Error {}%'.format(error))
    return np.sum(np.array(rmse)) / rmse_tot

print('Loading model from {}'.format(args.model_path))
model.load_state_dict(torch.load(args.model_path, weights_only=True))
if args.eval_lever == 0:
    print('Test RMSE: {:.4f}'.format(eval(model, test_dataloader, args.target)))
elif args.eval_lever == 1:
    print('Valid RMSE: {:.4f}'.format(eval(model, valid_dataloader, args.target)))
