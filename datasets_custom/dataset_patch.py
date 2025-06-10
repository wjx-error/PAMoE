import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import glob
import random
import math
from tqdm import trange
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
from datasets_custom.utils_labels import trans_labels
import h5py

from datasets_custom.dataset_graph import Generic_Split_graph


# 数据增强 node feature加噪+随机drop边
def data_transform(x, coords):
    # 随机节点特征加噪
    n = len(x)

    # 加噪
    rate = random.uniform(0.1, 0.2)
    rows_to_noise = np.random.choice(n, int(n * rate), replace=False)
    # rows_to_noise = rows_to_noise.astype(np.int32)
    # print(x.shape)
    # print(rows_to_noise)
    noise = torch.randn_like(x[rows_to_noise]) * 0.1
    x[rows_to_noise] += noise

    # 随机drop节点
    max_drop_rate = 0.2
    min_drop_rate = 0.1
    rate = random.uniform(1 - max_drop_rate, 1 - min_drop_rate)
    rows_to_not_drop = np.random.choice(n, int(n * rate), replace=False)

    x = x[rows_to_not_drop]
    coords = coords[rows_to_not_drop]

    return x, coords


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False, seed=7, print_info=True,
                 patient_strat=False, label_col='survival',
                 case_col='case_id', train_mode='patch'
                 ):
        r"""
        Generic_WSI_Survival_Dataset

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
        """
        super(Generic_WSI_Survival_Dataset, self).__init__()
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.case_col = case_col
        self.train_mode = train_mode

        print('csv_path', csv_path)

        ## 带有label的pd
        try:
            slide_data = pd.read_csv(csv_path, index_col=0, low_memory=False, encoding='gbk')
        except:
            slide_data = pd.read_csv(csv_path, index_col=0, low_memory=False)

        assert label_col in slide_data.columns
        self.label_col = label_col
        ###shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        # patients_df = slide_data.drop_duplicates([case_col]).copy()  # 删除重复的case
        slide_data.reset_index(drop=True, inplace=True)  # 去了重，刷新index

        self.patient_data = {case_col: slide_data[case_col].values}

        # 取['survival_days', 'disc_label', case_col, 'slide_id', 'label', 'censorship'] 换了个列名顺序
        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        slide_data[case_col] = slide_data[case_col].astype(str)

        self.slide_data = slide_data

        self.filelist = list(self.slide_data['slide_id'])

    def processed_file_names(self):
        return self.filelist

    def len(self):
        return len(self.slide_data)

    def get_split_from_df(self, all_splits, split_key: str = 'train', use_transform=True, combine=False):
        if combine:
            print('combine')
            print(split_key)
            key_list = split_key.split('_')
            ans_list = []
            for kk in key_list:
                ans_list.append(all_splits[kk])
            split = pd.concat(ans_list, axis=0)
            split = split.dropna().reset_index(drop=True)
        else:
            split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        try:
            split = split.astype(int).astype(str)
        except:
            split = split.astype(str)

        self.slide_data[self.case_col] = self.slide_data[self.case_col].astype(str)
        if len(split) == 0:
            return None
        mask = self.slide_data[self.case_col].isin(split.tolist())

        df_slice = self.slide_data[mask].reset_index(drop=True)

        print('self.train_mode', self.train_mode)

        if self.train_mode == 'patch':
            split = Generic_Split(df_slice, data_dir=self.data_dir, label_col=self.label_col, mode=split_key,
                                  use_transform=use_transform, in_memory=self.in_memory, bar=self.bar,
                                  case_col=self.case_col
                                  )
        elif self.train_mode == 'graph':
            split = Generic_Split_graph(df_slice, data_dir=self.data_dir, label_col=self.label_col, mode=split_key,
                                        use_transform=use_transform, in_memory=self.in_memory, bar=self.bar,
                                        case_col=self.case_col, args=args
                                        )
        else:
            raise NotImplementedError
        return split

    def return_splits(self, csv_path=None, split_key='train_val', use_transform=True, combine=False):
        assert csv_path
        all_splits = pd.read_csv(csv_path)

        if combine:
            print('make combine split')
            split = self.get_split_from_df(all_splits=all_splits, split_key=split_key,
                                           use_transform=False, combine=combine)
            return split

        if '_' in split_key:
            ans = []
            if 'train' in split_key:
                print('make train split')
                train_split = self.get_split_from_df(all_splits=all_splits, split_key='train',
                                                     use_transform=use_transform)
                ans.append(train_split)
            if 'val' in split_key:
                print('make val split')
                val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
                ans.append(val_split)
            if 'test' in split_key:
                print('make test split')
                try:
                    test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')
                    ans.append(test_split)
                except Exception:
                    ans.append(None)

            if len(ans) == 1:
                return ans[0]
            return ans

        else:
            print(f'make {split_key} split')
            split = self.get_split_from_df(all_splits=all_splits, split_key=split_key)
            return split


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, args_o, in_memory=False, bar=False, **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]

        self.in_memory = in_memory
        self.bar = bar

        global args
        args = args_o

        # print('self.slide_data',self.slide_data)


class Generic_Split(Dataset):
    def __init__(self, slide_data, data_dir=None, label_col=None, transform=None,
                 pre_transform=None, pre_filter=None, mode='train', use_transform=True,
                 in_memory=False, bar=False,
                 case_col='case_id'
                 ):
        super().__init__()
        slide_data = slide_data.drop_duplicates(subset=case_col)
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.label_col = label_col
        self.case_col = case_col

        self.mode = mode
        self.use_transform = use_transform

        self.in_memory = in_memory  # weather load data to the memory (faster)

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
        try:
            case_id = self.slide_data[self.case_col][idx]
            event_time = self.slide_data[self.label_col][idx]
        except:
            # print('label error')
            return None

        event_time = trans_labels(event_time, args=args)
        if event_time is None:
            print('label is None')
            return None
        c = self.slide_data['censorship'][idx]
        slide_ids = self.slide_data['slide_id'][idx]

        wsi_path = ''
        for pth in self.data_dir:
            # tmp_pth = f"{pth}/{slide_ids.rstrip('.svs').rstrip('.ndpi')}/20x.pkl"
            tmp_pth = f"{pth}/{slide_ids.rstrip('.svs').rstrip('.ndpi')}.h5"
            if os.path.exists(tmp_pth):
                wsi_path = tmp_pth
                break
        if wsi_path == '':
            # print('error:', idx, tmp_pth, 'not exists')
            return None

        try:
            # with open(wsi_path, 'rb') as f:
            #     data_origin = pickle.load(f)
            # data_origin = torch.load(wsi_path)
            data_origin = h5py.File(wsi_path, 'r')

        except:
            print('pickle load error')
            return None

        x = np.asarray(data_origin['features'], dtype=np.float32)
        # coords = np.asarray(data_origin['address'], dtype=np.int32)
        coords = np.asarray(data_origin['coords'], dtype=np.int32)

        coords = torch.tensor(coords, dtype=torch.int32)
        x = torch.tensor(x, dtype=torch.float32)
        return x, coords, event_time, c, case_id

    def __getitem__(self, idx):
        if self.in_memory:
            x, coords, event_time, c, case_id = copy.deepcopy(self.data_in_memory[idx])
        else:
            gt = self.get_data_from_idx(idx)
            if gt is not None:
                x, coords, event_time, c, case_id = gt
            else:
                # print('gt none')
                return None

        if self.mode == 'train' and self.use_transform:
            # print('data_transform')
            x, coords = data_transform(x, coords)

        return x, coords, event_time, c, case_id

    def __len__(self):
        if len(self.data_in_memory) > 0:
            return len(self.data_in_memory)
        else:
            return len(self.slide_data)
