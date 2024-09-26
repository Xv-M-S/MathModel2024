import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_snapshot', type=int, default=1, help='number of snapshot')
parser.add_argument('--layer', type=int, default=2, help='number of layers')
parser.add_argument('--dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--model_path', type=str, default='', help='path of model datasets')
parser.add_argument('--data_path', type=str, default='', help='path of eval datasets')
parser.add_argument('--save_path', type=str, default='', help='save path of results')
parser.add_argument('--target', type=str, default='throughput', help='name of predict item')
parser.add_argument('--eval_lever', type=int, default=0, help='0 for predict official test datasets; 1 for predict official train datasets;')

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
import pandas as pd

DEBUG = False
t_inf = 0

train_dataloader, _, test_dataloader = get_dataloader(args.data_path, args.batch_size, args.target, all_cuda=True)
model = NetModel(args.layer, args.dim, args.num_snapshot, args.target).cuda()

def eval(model, dataloader, pre_item ,save_path):
    global t_inf
    model.eval()
    rmse = list()
    rmse_tot = 0
    with torch.no_grad():
        temp_g = None
        for g, idx in dataloader:
            temp_g = g
            break
        test_ids = temp_g.nodes['ap'].data['mask'].shape[0]
        pre_num = 1
        if pre_item == "mcs_nss": pre_num = 2
        if DEBUG:
            print('test_ids: ' + str(test_ids))
            print('pre_num: ' + str(pre_num))

        cat_pred = torch.zeros(test_ids, pre_num).to("cuda:0")
        cat_true = torch.zeros(test_ids, pre_num).to("cuda:0")
        res = torch.zeros(test_ids, pre_num).to('cuda:0') 

        count = 0
        for g, idx in dataloader:
            count += 1
            t_s = time.time()
            g = g.to('cuda')
            
            mask = g.nodes['ap'].data['mask'] > 0
            mask = mask.view(-1)
            pred = model(g)[mask]
            res += pred
            
            if pre_item == "mcs_nss":
                true = g.nodes['ap'].data['predict'][mask][:,1:3].view(-1,2) # MCS,NSS
            elif pre_item == "seq_time":
                true = g.nodes['ap'].data['predict'][mask][:,7].view(-1,1) # seq_time
            elif pre_item == "throughput":
                true = g.nodes['ap'].data['predict'][mask][:,8].view(-1,1) # 吞吐量
            loss = torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8)
            rmse.append(float(loss) * pred.shape[0])
            rmse_tot += pred.shape[0]
            t_inf += time.time() - t_s
        # 将预测和真实值写入文件
        res /= count
        non_zero_mask = (true != 0)
        absolute_error = torch.abs(pred[non_zero_mask] - true[non_zero_mask])/true[non_zero_mask] * 100
        avg_error = absolute_error.sum(dim=0)/absolute_error.shape[0]
        print('Relative Error {:.4f}%'.format(avg_error.cpu().detach().numpy()))
        with open(save_path, 'w') as f:
            for p,t in zip(res.cpu().detach().numpy(), true.cpu().detach().numpy()):
                if pre_item == "mcs_nss":
                    if t[0] == 0: error_0 = 'invalid'
                    else: error_0 = abs(p[0] - t[0])/t[0] * 100
                    if t[1] == 0: error_1 = 'invalid'
                    else: error_1 = abs(p[1] - t[1])/t[1] * 100
                    f.write(f"{int(np.round(p[0]))}  {int(np.round(p[1]))} {int(np.round(t[0]))} {int(np.round(t[1]))} {error_0}% {error_1}%\n")
                else:
                    if t[0] == 0: error = 'invalid'
                    else: error = abs(p[0] - t[0])/t[0] * 100
                    f.write(f"{p[0]} {t[0]} {error}%\n")  # 将预测值和真实值写入文件

    return np.sum(np.array(rmse)) / rmse_tot

print('Loading model from {}'.format(args.model_path))
model.load_state_dict(torch.load(args.model_path, weights_only=True))
if args.eval_lever == 0:
    print('Test RMSE: {:.4f}'.format(eval(model, test_dataloader, args.target, args.save_path)))
else:
    print('Train RMSE: {:.4f}'.format(eval(model, train_dataloader, args.target, args.save_path)))
