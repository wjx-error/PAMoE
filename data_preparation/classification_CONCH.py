import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import copy
import numpy as np

from data_preparation.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP

import json
from conch.open_clip_custom import create_model_from_pretrained
from conch.downstream.zeroshot_path import zero_shot_classifier

import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@torch.no_grad()
def run_zeroshot_wogt(model, classifier, dataloader, device):
    logits_all, preds_all, coords_all = [], [], []
    # dataloader = tqdm(dataloader)
    for batch_idx, (imgs, coords) in enumerate(dataloader):
        imgs = imgs.to(device)
        image_features = model.encode_image(imgs)
        logits = image_features @ classifier
        preds = logits.argmax(dim=1)
        logits_all.append(logits.cpu().numpy())
        preds_all.append(preds.cpu().numpy())
        coords_all.append(coords)

    # Save raw preds & targets
    logits_all = np.concatenate(logits_all, axis=0)
    logits_all = F.softmax(torch.from_numpy(logits_all) * model.logit_scale.exp().item(), dim=1).numpy()
    preds_all = np.concatenate(preds_all, axis=0)

    coords_all = np.concatenate(coords_all, axis=0)

    dump = {}
    dump['logits'] = logits_all
    dump['preds'] = preds_all
    dump['coords'] = coords_all
    if hasattr(model, "logit_scale"):
        dump['temp_scale'] = model.logit_scale.exp().item()
    return dump


def compute_w_loader(output_path, loader, model, verbose=0):
    """
	args:
		output_path: directory to save computed features (.pt file)
		model: pytorch model
		verbose: level of feedback
	"""
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))

    with torch.inference_mode():
        dump = run_zeroshot_wogt(model, zeroshot_weights, loader, device)
    patch_classify_type = copy.deepcopy(dump['preds'])
    patch_classify_type[patch_classify_type >= 4] = 4

    G = {
        'centroid': torch.Tensor(dump['coords']),
        'patch_classify_type': patch_classify_type,
    }
    # print(G)
    print(G['centroid'].shape, G['patch_classify_type'].shape)
    print(np.unique(G['patch_classify_type']))
    torch.save(G, output_path)
    return output_path


parser = argparse.ArgumentParser(description='Classfication')
parser.add_argument('--data_h5_dir', type=str, default='../CLAM-master/data/patches')
parser.add_argument('--data_slide_dir', type=str, default='../TCGA_DATAS/TCGA-PAAD/')
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default='../CLAM-master/data/patches/process_list_autogen.csv')
parser.add_argument('--out_dir', type=str, default='./classfication')
parser.add_argument('--checkpoint_path', type=str, default='./pretrained_weights/CONCH/pytorch_model.bin')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.out_dir))

    # get model
    model_cfg = 'conch_ViT-B-16'
    checkpoint_path = args.checkpoint_path
    force_image_size = 224
    model, img_transforms = create_model_from_pretrained(model_cfg, checkpoint_path, device=device,
                                                         force_image_size=force_image_size)
    model.eval()
    # prompts
    prompt_file = './prompts/cls.json'
    with open(prompt_file) as f:
        prompts = json.load(f)
    classnames = prompts['classnames']
    templates = prompts['templates']
    n_classes = len(classnames)
    idx_to_class = {}
    for idx, k in enumerate(classnames.keys()):
        idx_to_class[idx] = k
    classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
    for class_idx, classname in enumerate(classnames_text):
        print(f'{class_idx} {idx_to_class[class_idx]}: {classname}')
    zeroshot_weights = zero_shot_classifier(model, classnames_text, templates, device=device)

    _ = model.eval()
    model = model.to(device)
    total = len(bags_dataset)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for bag_candidate_idx in (range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        output_path = f"{args.out_dir}/{slide_id}.pt"
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path,
                                     wsi=wsi,
                                     img_transforms=img_transforms)

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
