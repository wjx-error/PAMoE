from sklearn.metrics import (precision_score, recall_score, f1_score,
                             auc, roc_curve, roc_auc_score,
                             accuracy_score, balanced_accuracy_score
                             )
from sklearn.preprocessing import label_binarize
import numpy as np
from lifelines.utils import concordance_index
import torch
import torch.nn.functional as F


def filtered_roc_auc_score(y_true, y_scores, multi_class='ovo', average='macro'):
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        print("Only one class present in y_true. ROC AUC score is not defined in that case.")
        return None
    y_scores_filtered = y_scores[:, unique_classes]
    y_true_bin = label_binarize(y_true, classes=unique_classes)
    # return roc_auc_score(y_true_bin, y_scores_filtered, multi_class='ovr')
    return roc_auc_score(y_true_bin, y_scores_filtered, multi_class=multi_class, average=average)


def metrics(outputs, targets, labels, average='macro'):  # macro micro binary
    if outputs.shape[1] == 2:
        average = 'binary'
    else:
        average = 'macro'
    preds = outputs.argmax(1)

    acc = accuracy_score(targets, preds)
    bacc = balanced_accuracy_score(targets, preds)

    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    f1 = f1_score(targets, preds, average=average, zero_division=0)
    try:
        if average == 'binary':
            fpr, tpr, thresholds = roc_curve(targets, preds)
            aucroc = auc(fpr, tpr)
        else:
            # targets_onehot = label_binarize(targets, classes=labels)
            # aucroc = roc_auc_score(targets_onehot, outputs, multi_class='ovr', labels=labels)
            aucroc = filtered_roc_auc_score(targets, outputs, multi_class='ovo', average=average)
    except:
        print('aucroc error')
        print('targets', targets)
        print('preds', preds)
        aucroc = -1
    return acc, bacc, precision, recall, f1, aucroc


def cindex_metric_train(all_risk_scores, all_censorships, all_event_times, epoch, train_loss):
    all_risk_scores = np.asarray(all_risk_scores).flatten()
    all_censorships = np.asarray(all_censorships).flatten()
    all_event_times = np.asarray(all_event_times).flatten()
    c_index_all = 1 - concordance_index(all_event_times, all_risk_scores)
    c_index = 1 - concordance_index(all_event_times, all_risk_scores, event_observed=(1 - all_censorships))
    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, '
          f'train_c_index_censored: {c_index:.4f}, train_c_index_all: {c_index_all:.4f}',
          flush=True)
    return c_index, c_index_all


def cindex_metric_val(all_risk_scores, all_censorships, all_event_times, epoch, mode='val', verbose=True):
    all_pred_scores = np.asarray(all_risk_scores).flatten()
    all_censorships = np.asarray(all_censorships).flatten()
    all_ground_truth = np.asarray(all_event_times).flatten()

    # c_index_lifelines = concordance_index(all_event_times, all_risk_scores, event_observed=1 - all_censorships)
    c_index_all = 1 - concordance_index(all_ground_truth, all_pred_scores)
    c_index = 1 - concordance_index(all_ground_truth, all_pred_scores, event_observed=(1 - all_censorships))
    if verbose:
        print(f'{mode} Epoch: {epoch}, val_c_index_censored: {c_index:.4f}, val_c_index_all: {c_index_all:.4f}',
              flush=True)

    return c_index, c_index_all


def acc_metric_train(all_pred_scores, all_censorships, all_ground_truth, epoch, train_loss, num_class):
    all_pred_scores = np.asarray(all_pred_scores)
    all_ground_truth = np.asarray(all_ground_truth).flatten()

    labels = [i for i in range(int(num_class))]
    acc, bacc, precision, recall, f1, aucroc = metrics(all_pred_scores, all_ground_truth,
                                                       average='macro', labels=labels)
    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, '
          f'acc: {acc:.4f}, bacc: {bacc:.4f}, precision: {precision:.4f}, '
          f'recall: {recall:.4f}, f1: {f1:.4f}, aucroc: {aucroc:.4f}',
          flush=True)

    results = {
        'acc': acc,
        'bacc': bacc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'aucroc': aucroc,
    }

    return results


def acc_metric_val(all_pred_scores, all_censorships, all_ground_truth, epoch, num_class, mode='val', verbose=True):
    all_pred_scores = np.asarray(all_pred_scores)
    all_ground_truth = np.asarray(all_ground_truth).flatten()

    with torch.no_grad():
        loss_ce = F.cross_entropy(torch.tensor(all_pred_scores).float(), torch.tensor(all_ground_truth).long())
    loss_ce = float(loss_ce)

    labels = [i for i in range(int(num_class))]
    acc, bacc, precision, recall, f1, aucroc = metrics(all_pred_scores, all_ground_truth,
                                                       average='macro', labels=labels)
    if verbose:
        print(f'{mode} Epoch: {epoch}, loss_ce: {loss_ce:.4f}, '
              f'acc: {acc:.4f}, bacc: {bacc:.4f}, precision: {precision:.4f}, '
              f'recall: {recall:.4f}, f1: {f1:.4f}, aucroc: {aucroc:.4f}',
              flush=True)
    results = {
        f'{mode}_loss': loss_ce,
        f'{mode}_acc': acc,
        f'{mode}_bacc': bacc,
        f'{mode}_precision': precision,
        f'{mode}_recall': recall,
        f'{mode}_f1': f1,
        f'{mode}_aucroc': aucroc,
    }

    return results
