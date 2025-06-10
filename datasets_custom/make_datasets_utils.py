import torch
from datasets_custom.dataset_patch import Generic_MIL_Survival_Dataset


def get_dataset_total(args):
    print('Init Dataset')
    dataset = Generic_MIL_Survival_Dataset(csv_path=args.csv_path,
                                           data_dir=args.data_dir,
                                           shuffle=False,
                                           seed=args.seed,
                                           print_info=False,
                                           patient_strat=False,
                                           label_col=args.label_col,
                                           in_memory=args.in_memory,
                                           bar=args.bar,
                                           args_o=args,
                                           case_col='case_id',
                                           train_mode=args.train_mode
                                           )
    return dataset


def get_dataset_splits(args, dataset_all, csv_nm):
    ### Gets the Train + Val Dataset Loader.
    if args.testing:
        train_dataset, val_dataset, test_dataset = dataset_all.return_splits(
            csv_path=f'{args.split_dir}/{csv_nm}', split_key='train_val_test')
        datasets = (train_dataset, val_dataset, test_dataset)
    else:
        train_dataset, val_dataset = dataset_all.return_splits(csv_path=f'{args.split_dir}/{csv_nm}',
                                                               split_key='train_val')
        datasets = (train_dataset, val_dataset, None)

    return datasets
