import copy
import time
from argparse import Namespace
from collections import OrderedDict
import os
import pickle

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
import torch
from datasets_custom.dataset_generic import save_splits
from utils.utils import *
from utils.utils_loss import *
from torch.cuda.amp import GradScaler, autocast
import gc
import tqdm
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
import cv2
from utils.utils_metrics import *
from utils.utils_earlystop import *
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_model(model, data_WSI, args, loss_fn, use_fp16=True):
    data_WSI = [x for x in data_WSI if x is not None]
    if len(data_WSI) == 0:
        return

    # print('data_WSI', data_WSI)
    label = torch.tensor([data.label for data in data_WSI])
    c = torch.tensor([data.c for data in data_WSI])

    case_id = [data.case_id for data in data_WSI]
    coords = [data.centroid for data in data_WSI]

    # x = torch.tensor([data.x for data in data_WSI])
    x = None

    c = c.to(device)
    label = label.to(device)

    c = torch.tensor(c).cuda()

    # st = time.time()
    # print(x.shape, coords.shape)
    with torch.cuda.amp.autocast(enabled=use_fp16):
        _, pred, loss_pamoe = model(data_WSI)
    # ed = time.time()
    # print('model forward cost', ed - st, 's')
    # print(pred.shape)

    if torch.any(torch.isnan(pred)):
        return None, pred, (x, coords, label, c, case_id)

    if loss_fn is not None:
        if args.task == 'survival':
            loss_dict = {'S': pred, 'c': c, 'event_time': label}
            loss = loss_fn(**loss_dict)
        else:
            loss = loss_fn(pred, label)
    else:
        loss = None

    if (loss is not None) and (loss_pamoe is not None) and (args.alpha_pamoe > 0):
        loss_pamoe = loss_pamoe.mean()
        loss = loss + args.alpha_pamoe * loss_pamoe

    return loss, pred, (x, coords, label, c, case_id)


def train_loop_survival(epoch, model, loader, optimizer, args, writer=None,
                        loss_fn=None, gc=16, loss_scaler=None, bar=True):
    model.train()
    train_loss = 0.
    all_pred_scores, all_censorships, all_ground_truth = [], [], []
    if bar == True:
        loader = tqdm.tqdm(loader)

    for batch_idx, datas in enumerate(loader):
        if not datas:
            continue
        loss, pred, dts = forward_model(model, datas, args, loss_fn, use_fp16=args.use_fp16)
        if torch.any(torch.isnan(pred)):
            print('error: train out nan')
            model.zero_grad()
            # optimizer.zero_grad()
            continue
        if torch.isnan(loss):
            # print('loss nan')
            model.zero_grad()
            # optimizer.zero_grad()
            continue
        _, coords, label, c, case_id = dts

        loss_scaler(loss, optimizer, parameters=model.parameters())
        optimizer.zero_grad()

        loss_value = loss.item()
        if pred.shape[1] == 1:
            pred = pred.detach().cpu().numpy()
        else:
            pred = pred.detach()
            # pred = F.softmax(pred, dim=1)
            pred = pred.cpu().numpy()
        train_loss += loss_value

        # print('pred.shape', pred.shape)

        # all_pred_scores.extend(pred.flatten())
        all_pred_scores.extend(pred)
        all_ground_truth.extend(label.cpu().numpy().flatten())
        all_censorships.extend(c.cpu().numpy().flatten())

    # calculate loss and error for epoch
    train_loss /= len(all_pred_scores)

    if args.task == 'survival':
        c_index, c_index_all = cindex_metric_train(all_pred_scores, all_censorships, all_ground_truth, epoch,
                                                   train_loss)
        if writer:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/c_index', c_index, epoch)
    else:
        train_results = acc_metric_train(all_pred_scores, all_censorships, all_ground_truth,
                                         epoch, train_loss, num_class=args.num_classes)
        if writer:
            writer.add_scalar('train/acc', train_results['acc'], epoch)
            writer.add_scalar('train/bacc', train_results['bacc'], epoch)
            writer.add_scalar('train/aucroc', train_results['aucroc'], epoch)


def validate_survival(epoch, model, loader, args, early_stopping=None,
                      writer=None, results_dir=None, cur=None,
                      bar=True, mode='val'):
    _, metrics_dict = summary_survival(model, loader, args, bar=bar, mode=mode,
                                       get_patient_results=False, epoch=epoch, verbose=True)

    if args.task == 'survival':
        c_index = metrics_dict['val_cindex']
        c_index_all = metrics_dict['val_cindex_all']
        all_cindex = [c_index, c_index_all]
        in_met = all_cindex
        if writer:
            writer.add_scalar('val/c-index', c_index, epoch)
    else:
        in_met = metrics_dict['val_bacc']
        # in_met = metrics_dict['val_loss']
        if writer:
            writer.add_scalar('val/ce_loss', metrics_dict['val_loss'], epoch)
            writer.add_scalar('val/acc', metrics_dict['val_acc'], epoch)
            writer.add_scalar('val/precision', metrics_dict['val_precision'], epoch)
            writer.add_scalar('val/aucroc', metrics_dict['val_aucroc'], epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, in_met, model,
                       ckpt_name=os.path.join(results_dir, f"s_{cur}_maxcindex_checkpoint.pt"))

        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False


def summary_survival(model, loader, args, bar=True, mode='val', get_patient_results=True,
                     epoch='summary_survival', verbose=False):
    model.eval()
    all_pred_scores, all_censorships, all_ground_truth = [], [], []
    patient_results = {}
    if bar == True:
        loader = tqdm.tqdm(loader)
    for batch_idx, datas in enumerate(loader):
        if not datas:
            continue
        with torch.no_grad():
            _, pred, dts = forward_model(model, datas, args, loss_fn=None, use_fp16=args.use_fp16)
        if torch.any(torch.isnan(pred)):
            print('error: val out nan')
            continue
        x, coords, label, c, case_id = dts

        # pred = pred.cpu().numpy()
        if pred.shape[1] == 1:
            pred = pred.detach().cpu().numpy()
        else:
            pred = pred.detach()
            # pred = F.softmax(pred, dim=1)
            pred = pred.cpu().numpy()

        # all_pred_scores.extend(pred.flatten())
        all_pred_scores.extend(pred)
        all_censorships.extend(c.cpu().numpy())
        all_ground_truth.extend(label.cpu().numpy())

        if get_patient_results:
            for idx, cd in enumerate(case_id):
                pred_t = pred[idx]
                patient_results.update({cd: {'pred': pred_t, 'label': label[idx], 'censorship': c[idx]}})

    if args.task == 'survival':
        c_index, c_index_all = cindex_metric_val(all_pred_scores, all_censorships, all_ground_truth,
                                                 epoch, mode=mode, verbose=verbose)
        metrics_dict = {
            f'{mode}_cindex': c_index,
            f'{mode}_cindex_all': c_index_all,
        }
    else:
        metrics_dict = acc_metric_val(all_pred_scores, all_censorships, all_ground_truth, epoch,
                                      num_class=args.num_classes, verbose=verbose, mode=mode)
    metrics_dict = OrderedDict(metrics_dict)
    return patient_results, metrics_dict
