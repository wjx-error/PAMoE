# cross val split
from utils_preprocess import *

tcga_name = 'PAAD'
csv_pth = f'../dataset_csv/{tcga_name}.csv'
out_pth = f'../splits/survival/5foldcv/{tcga_name}_41_val_patient/'
floder_num = 5

os.makedirs(out_pth, exist_ok=True)
df = pd.read_csv(csv_pth)
df['case_id'] = df['case_id'].astype(str)
floder = split_cv(df, floder_num=floder_num)

for idx, x in enumerate(floder):
    print(f'flod{idx} num:', len(x))

for i in range(floder_num):
    train_list, val_list = get_train_val_from_split(floder, i=i, floder_num=floder_num)
    print('len(train_list),len(val_list)')
    print(len(train_list), len(val_list))

    train_patient = {}
    for x in train_list:
        train_patient[x[:12]] = 1

    tmp_list = [x for x in val_list]

    for x in tmp_list:
        if x[:12] in train_patient.keys():
            # print('remove', x[:12], x)
            val_list.remove(x)
            continue
        train_patient[x[:12]] = 2

    print(len(train_list), len(val_list))
    print()
    train_list, val_list = pad_nan([train_list, val_list])
    tmp_df = pd.DataFrame.from_dict({'train': train_list, 'val': val_list})
    # print(tmp_df)
    tmp_df.to_csv(f'{out_pth}/splits_{int(i)}.csv')
