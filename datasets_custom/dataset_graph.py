import copy
import os
import time

from tqdm import trange
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch
from datasets_custom.utils_labels import trans_labels

class PairData(Data):
    def __init__(self, x=None, edge_index=None):
        super().__init__(x=x, edge_index=edge_index)

    def __inc__(self, key, value, *args, **kwargs):
        if not hasattr(Data, key):
            if 'edge' in key:  # 边
                return getattr(self, 'x').size(0)
            else:
                return 0
        else:
            return super(PairData, self).__inc__(key, value)


# 数据增强 node feature加噪+随机drop边
def data_transform(data):
    if data is None:
        return None

    x = data.x
    edge = data.edge_index

    # 随机节点特征加噪
    n = len(x)

    # rate = random.uniform(0.0, 0.2)
    rate = random.uniform(0.1, 0.3)
    # rate = 0.1
    rows_to_noise = np.random.choice(n, int(n * rate), replace=False)
    noise = torch.randn_like(x[rows_to_noise]) * 0.1
    x[rows_to_noise] += noise

    # rate = random.uniform(0.0, 0.2)
    rate = random.uniform(0.1, 0.3)
    # rate = 0.2
    rows_to_drop = np.random.choice(n, int(n * rate), replace=False)
    x[rows_to_drop] = 0

    # 随机丢弃边
    # rate = random.uniform(0.0, 0.2)
    rate = random.uniform(0.1, 0.3)
    # rate = 0.2
    m = edge.shape[1]
    k = m - int(rate * m)
    columns = np.random.choice(m, k, replace=False)
    edge = edge[:, columns]

    if hasattr(data, 'edge_latent'):
        edge_latent = data.edge_latent
        rate = random.uniform(0.1, 0.3)
        # rate = 0.2
        m = edge_latent.shape[0]
        k = m - int(rate * m)
        columns = np.random.choice(m, k, replace=False)
        edge_latent = edge_latent[columns, :]
        data.edge_latent = edge_latent

    data.x = x
    data.edge_index = edge

    return data


class Generic_Split_graph(Dataset):
    def __init__(self, slide_data, data_dir=None, label_col=None, transform=None,
                 pre_transform=None, pre_filter=None, mode='train', use_transform=True,
                 in_memory=False, bar=False, case_col='case_id', args=None
                 ):
        super().__init__()

        print('Generic_Split_graph init')

        slide_data = slide_data.drop_duplicates(subset=case_col)
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.label_col = label_col
        self.case_col = case_col
        self.args = args

        self.mode = mode
        self.use_transform = use_transform

        self.in_memory = in_memory  # 是否把数据读到内存里面

        if 'dataset' not in self.slide_data.columns:
            self.slide_data['dataset'] = 'tmp'

        self.censorship = list(self.slide_data['censorship'])
        self.dataset = list(self.slide_data['dataset'])

        self.data_in_memory = []
        if self.in_memory:
            a = time.time()
            print('Load data to memory', flush=True)
            print(len(self.slide_data))
            if bar:
                rg = trange(len(self.slide_data))
            else:
                rg = range(len(self.slide_data))

            self.censorship = []
            self.dataset = []
            for i in rg:
                tmp = self.get_data_from_idx(i)
                if tmp is not None:
                    self.data_in_memory.append(tmp)
                    self.censorship.append(self.slide_data['censorship'][i])
                    self.dataset.append(self.slide_data['dataset'][i])
            print(f"Load data finish, cost {time.time() - a}s", flush=True)
        else:
            print('Load data from paths', flush=True)

    def get_data_from_idx(self, idx):
        case_id = self.slide_data['case_id'][idx]
        try:
            event_time = self.slide_data[self.label_col][idx]
        except:
            print('event_time error')
            return None
        c = self.slide_data['censorship'][idx]
        slide_ids = self.slide_data['slide_id'][idx]

        wsi_path = ''
        for pth in self.data_dir:
            tmp_pth = f"{pth}/{slide_ids.rstrip('.svs').rstrip('.ndpi')}.pt"
            if os.path.exists(tmp_pth):
                wsi_path = tmp_pth
                break
        if wsi_path == '':
            print('error:', idx, slide_ids, 'not exists')
            return None
            # raise NotImplementedError

        data_origin = torch.load(wsi_path)

        data_t = PairData(data_origin.x, data_origin.edge_index)

        data_t.centroid = data_origin.centroid

        event_time = trans_labels(event_time, args=self.args)
        if event_time is None:
            print('label is None')
            return None
        data_t.label = event_time

        data_t.c = c
        data_t.slide_id = slide_ids.split('/')[-1].rstrip('.svs').rstrip('.ndpi')

        data_t.indataset = self.slide_data['dataset'][idx]
        data_t.case_id = case_id

        data_t.eg_wo_add = data_origin.edge_index.transpose(0, -1)

        # data_t.wsi_path = wsi_path

        return data_t

    def get(self, idx):
        if self.in_memory:
            data_t = copy.deepcopy(self.data_in_memory[idx])
        else:
            data_t = self.get_data_from_idx(idx)

        # if self.mode == 'train' and self.use_transform:
        #     data_t = data_transform(data_t)
        return data_t

    def len(self):
        if len(self.data_in_memory) > 0:
            return len(self.data_in_memory)
        else:
            return len(self.slide_data)
