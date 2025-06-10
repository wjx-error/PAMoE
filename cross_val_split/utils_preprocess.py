import copy
import os
import pandas as pd
import math
import random
import numpy as np


def pad_nan(in_list):
    lenn = [len(x) for x in in_list]
    maxx = np.max(lenn)

    for x in in_list:
        kk = maxx - len(x)
        for k in range(kk):
            x.append(None)

    return in_list


def split_cv(all_pd, floder_num=5):
    all_censored = all_pd[all_pd['censorship'] == 0]
    all_uncensored = all_pd[all_pd['censorship'] == 1]

    all_censored_nm = all_censored['case_id']
    suffled_censored_nm = all_censored_nm.sample(frac=1)
    ct_censored = math.ceil(len(suffled_censored_nm) / floder_num)
    print('ct_censored', ct_censored)
    print('len(suffled_censored_nm)', len(suffled_censored_nm))

    all_uncensored_nm = all_uncensored['case_id']
    suffled_uncensored_nm = all_uncensored_nm.sample(frac=1)
    ct_uncensored = math.ceil(len(suffled_uncensored_nm) / floder_num)
    print('ct_uncensored', ct_uncensored)
    print('len(suffled_uncensored_nm)', len(suffled_uncensored_nm))

    floder = []
    for i in range(floder_num):
        if i == floder_num - 1:
            inm = list(suffled_censored_nm.iloc[i * ct_censored:]) + \
                  list(suffled_uncensored_nm.iloc[i * ct_uncensored:])
            random.shuffle(inm)
            floder.append(inm)
            continue
        inm = list(suffled_censored_nm.iloc[i * ct_censored:(i + 1) * ct_censored]) + \
              list(suffled_uncensored_nm.iloc[i * ct_uncensored:(i + 1) * ct_uncensored])
        random.shuffle(inm)
        floder.append(inm)
    return floder


def get_train_val_test_from_split(floder, i=0, floder_num=5):
    train_list, val_list, test_list = [], [], []
    for j in range(floder_num):
        if j == i:
            test_list.extend(floder[j])
        elif j == (i + 1) % floder_num:
            val_list.extend(floder[j])
        else:
            train_list.extend(floder[j])
    return train_list, val_list, test_list


def get_train_val_from_split(floder, i=0, floder_num=5):
    train_list, val_list = [], []
    for j in range(floder_num):
        if j == i:
            val_list.extend(floder[j])
        else:
            train_list.extend(floder[j])
    return train_list, val_list
