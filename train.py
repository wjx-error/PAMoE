from utils.utils import ordered_yaml, seed_torch, get_args
from argparse import Namespace
import argparse
import torch
import os
from datasets_custom.make_datasets_utils import get_dataset_total, get_dataset_splits
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer
from utils.core_utils import train
from utils.file_utils import save_pkl, load_pkl
import gc
import numpy as np
import copy
import pandas as pd
import warnings
from utils.core_utils_graph import train_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, dataset_all):
    split_list = os.listdir(args.split_dir)
    split_list = [x for x in split_list if '.csv' in x]
    split_list.sort()
    print('split_list', split_list)

    if args.train_mode == 'graph':
        train_func = train_graph
    elif args.train_mode == 'patch':
        train_func = train
    else:
        raise NotImplementedError
    print('train_func', train_func)

    results_dict = {}
    folds = copy.deepcopy(split_list)
    folds = [x.rstrip('.csv') for x in folds]
    folds.append('average')
    folds.append('std')
    results_dict['folds'] = folds

    for i, csv_nm in enumerate(split_list):
        print(f"Start Fold {i}, csv: {csv_nm}")
        start = timer()
        seed_torch(args.seed)

        ### Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset, test_dataset = get_dataset_splits(args, dataset_all, csv_nm)
        print(f'training: {len(train_dataset)}, validation: {len(val_dataset)}', flush=True)
        if test_dataset is not None:
            print(f'testing: {len(test_dataset)}', flush=True)
        datasets = (train_dataset, val_dataset, test_dataset)

        patient_results, metrics_dict = train_func(datasets, args, cur=i, bar=args.bar)
        save_pkl(f"{args.results_dir}/{csv_nm}_val_results.pkl", patient_results)

        for key, value in metrics_dict.items():
            print(key, value)
            if i == 0:
                results_dict[key] = []
            results_dict[key].append(value)

        end = timer()
        print(f'Fold {i} Time: {(end - start):.2f} seconds')
        print()
        gc.collect()

    for key, value in results_dict.items():
        if key == 'folds':
            continue
        tmp_list = copy.deepcopy(results_dict[key])
        results_dict[key].append(np.mean(tmp_list))
        results_dict[key].append(np.std(tmp_list))

    results_latest_df = pd.DataFrame(results_dict)

    results_latest_df.to_csv(f"{args.results_dir}/summary_latest_{str(args.param_code)}_{str(args.exp_code)}.csv",
                             index=False)

    csv_combine_pth = f'./summary_latest_val_in_train/{args.which_splits}/{args.model_type}/'
    os.makedirs(csv_combine_pth, exist_ok=True)
    results_latest_df.to_csv(f"{csv_combine_pth}/summary_{str(args.param_code)}_{str(args.exp_code)}.csv",
                             index=False)
    print()
    # print(results_latest_df, flush=True)
    print(results_latest_df.to_string(float_format='{:.4f}'.format), flush=True)

    print()

    return results_latest_df


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='survival')
parser.add_argument('--project_name', type=str, default='PAAD')
parser.add_argument('--yml_opt_path', type=str, default='transmil_pamoe')
args = parser.parse_args()

if __name__ == "__main__":
    args = get_args(args)
    seed_torch(args.seed)

    # view settings
    settings = {}
    for key, value in args.__dict__.items():
        settings[key] = value
    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("################# Settings Finish ###################")
    print(flush=True)
    os.makedirs(args.results_dir, exist_ok=True)
    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    # init dataset
    dataset_all = get_dataset_total(args)

    start = timer()
    results = main(args, dataset_all)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
