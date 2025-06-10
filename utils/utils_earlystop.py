import numpy as np
import torch

class EarlyStopping_cindex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_cindex = None
        self.best_cindex_num = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, cindex, model, ckpt_name='best_checkpoint.pt'):
        cindex_num = cindex[0]
        if epoch < self.warmup:
            pass
        elif self.best_cindex == None:
            self.best_cindex = cindex
            self.best_cindex_num = cindex_num
            self.save_checkpoint(cindex, cindex_num, model, ckpt_name)

        elif cindex_num <= self.best_cindex_num:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, best val cindex: {self.best_cindex}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.save_checkpoint(cindex, cindex_num, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_cindex, cindex_num, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation c-index insreased ({self.best_cindex[0]:.4f}, {self.best_cindex[1]:.4f} --> {val_cindex[0]:.4f}, {val_cindex[1]:.4f}).  Saving model to {ckpt_name}...')
            # print(f'Validation c-index insreased ({self.best_cindex:.6f} --> {val_cindex:.6f}).  Saving model to {ckpt_name}...')
        torch.save(model.state_dict(), ckpt_name)
        self.best_cindex = val_cindex
        self.best_cindex_num = cindex_num