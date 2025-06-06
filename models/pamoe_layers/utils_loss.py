import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prototype loss
# prototypes (b,Class*K,d)
def get_compatibility_loss_batch(data_WSI, prototypes, num_classes):  # (N,dim), (Class*K,d), train_labels, Class
    loss = []
    for idx, data in enumerate(data_WSI):
        pp = get_compatibility_loss(data.x, prototypes[idx], data.patch_classify_type, num_classes)
        loss.append(pp)
    loss = sum(loss) / len(loss)
    return loss

def get_compatibility_loss(x, prototypes, labels, num_classes):  # (N,dim), (Class*K,d), train_labels, Class
    x = x.to(prototypes.device)
    labels = labels.to(prototypes.device)

    # labels = torch.argmax(labels, dim=1)
    labels = labels.to(torch.int64)
    dim = x.size(-1)  # d
    dots = []
    prototypes = prototypes.reshape(num_classes, -1, dim)  # .max(dim=1)[0] # (Class, K, d)
    for i in range(prototypes.size(1)):  # K
        prototypes_i = prototypes[:, i]  # (Class, d)
        dots_i = torch.cdist(x, prototypes_i, p=2)  # (N,Class)
        dots.append(dots_i.unsqueeze(1))  # (N,1,Class)

    # dots = torch.cat(dots, dim=1).max(dim=1)[0]  # (N,K,Class) -> (N,Class)
    dots = torch.cat(dots, dim=1).mean(dim=1)  # (N,K,Class) -> (N,Class)

    attn = dots  # .softmax(dim=1) # n x c
    positives = torch.gather(input=attn[:, :], dim=1, index=labels[:, None]).squeeze()  # (N)
    negatives = attn[F.one_hot(labels) == 0]
    comp_loss = torch.sum(positives) + torch.logsumexp(-negatives, dim=0)
    comp_loss = comp_loss / (x.size(0) * prototypes.size(0))
    return comp_loss


def get_orthogonal_regularization_loss(s):  # (Class*K,d)
    # Orthogonality regularization.
    s = s.unsqueeze(0) if s.dim() == 2 else s  # (1,Class*K,d)
    k = s.size(-1)  # d
    ss = torch.matmul(s.transpose(1, 2), s)  # (1,d,d)
    i_s = torch.eye(k).type_as(ss)  # (d,d)对角1
    ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)
    return ortho_loss
