import os
import torch
import dgl

class NetworkData(torch.utils.data.Dataset):

    def __init__(self, fn_dir, target):
        train = list()
        valid = list()
        test = list()
        for fn in os.listdir(fn_dir):
            g = dgl.load_graphs(os.path.join(fn_dir, fn))[0][0]
            # 预测seq,mcs_nss使用节点特征的前十维,预测吞吐量使用特征的前12维度
            if target == "throughput":
                g.nodes['ap'].data['feat'] = g.nodes['ap'].data['feat'].float()
            else:
                g.nodes['ap'].data['feat'] = g.nodes['ap'].data['feat'].float()[:,:10]
            g.nodes['ap'].data['mask'] = g.nodes['ap'].data['mask'].float()
            g.nodes['ap'].data['predict'] = g.nodes['ap'].data['predict'].float()
            if target == "throughput":
                g.nodes['sta'].data['feat'] = g.nodes['sta'].data['feat'].float()
            else:
                g.nodes['sta'].data['feat'] = g.nodes['sta'].data['feat'].float()[:,:10]
            g.nodes['sta'].data['mask'] = g.nodes['sta'].data['mask'].float()
            g.nodes['sta'].data['predict'] = g.nodes['sta'].data['predict'].float()
            g.edges['sta-ap'].data['feat'] = g.edges['sta-ap'].data['feat'].float()
            g.edges['ap-sta'].data['feat'] = g.edges['ap-sta'].data['feat'].float()
            g.edges['ap-ap'].data['feat'] = g.edges['ap-ap'].data['feat'].float()


            # no caa
            # g.nodes['sta'].data['feat'][:, 5] = 0
            # g.nodes['sta'].data['feat'][:, 6] = 0
            # g.nodes['sta'].data['feat'][:, 7] = 0
            # g.nodes['ap'].data['feat'][:, 5] = 0
            # g.nodes['ap'].data['feat'][:, 6] = 0
            # g.nodes['ap'].data['feat'][:, 7] = 0
            # # no loc
            # g.nodes['sta'].data['feat'][:, 1] = 0
            # g.nodes['sta'].data['feat'][:, 2] = 0
            # g.nodes['sta'].data['feat'][:, 4] = 0
            # g.nodes['ap'].data['feat'][:, 1] = 0
            # g.nodes['ap'].data['feat'][:, 2] = 0
            # g.nodes['ap'].data['feat'][:, 4] = 0
            # # no 业务流量
            # g.nodes['ap'].data['feat'][:, 3] = 0
            # g.nodes['sta'].data['feat'][:, 3] = 0
            # # no 节点间RSSI
            # g.edges['ap-ap'].data['feat'][:, 2] = 0
            # g.edges['sta-ap'].data['feat'][:, 2] = 0
            # g.edges['ap-sta'].data['feat'][:, 2] = 0
            # g.edges['ap-ap'].data['feat'][:, 3] = 0
            # g.edges['sta-ap'].data['feat'][:, 3] = 0
            # g.edges['ap-sta'].data['feat'][:, 3] = 0
            # g.edges['ap-ap'].data['feat'][:, 4] = 0
            # g.edges['sta-ap'].data['feat'][:, 4] = 0
            # g.edges['ap-sta'].data['feat'][:, 4] = 0

            if fn.startswith('train'):
                train.append(g)
            elif fn.startswith('valid'):
                valid.append(g)
            else:
                test.append(g)
        self.data = train + valid + test
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(torch.arange(len(train)))
        self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(torch.arange(len(train), len(train) + len(valid)))
        self.test_sampler = torch.utils.data.sampler.SubsetRandomSampler(torch.arange(len(train) + len(valid), len(train) + len(valid) + len(test)))

    def __len__(self):
        return len(Self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], idx

    def to_cuda(self):
        cuda_data = [d.to('cuda') for d in self.data]
        self.data = cuda_data

    def get_sampler(self):
        return self.train_sampler, self.valid_sampler, self.test_sampler

def get_dataloader(fn_dir, batchsize, target, all_cuda=False):
    print('loading data...')
    dataset = NetworkData(fn_dir, target)
    if all_cuda:
        dataset.to_cuda()
    train_sampler, valid_sampler, test_sampler = dataset.get_sampler()

    train_dataloader = dgl.dataloading.GraphDataLoader(dataset, sampler=train_sampler, batch_size=batchsize)
    valid_dataloader = dgl.dataloading.GraphDataLoader(dataset, sampler=valid_sampler, batch_size=batchsize)
    test_dataloader = dgl.dataloading.GraphDataLoader(dataset, sampler=test_sampler, batch_size=batchsize)

    return train_dataloader, valid_dataloader, test_dataloader