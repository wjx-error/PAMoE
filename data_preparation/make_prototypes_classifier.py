from __future__ import print_function
import argparse
import torch
import numpy as np
import h5py
import random
import os


def load_datas(root_pth, cls_root, max_patches):
    print(root_pth, flush=True)

    wsi_list = os.listdir(root_pth)
    wsi_list = [x for x in wsi_list if '.h5' in x]
    wsi_list = [x.replace('.h5', '') for x in wsi_list]
    random.shuffle(wsi_list)
    print(len(wsi_list))

    all_features = []
    all_classes = []
    cnt = 0
    print('max_patches', max_patches, flush=True)
    for slide_id in wsi_list:
        feat_path = f"{root_pth}/{slide_id}.h5"
        cls_path = f"{cls_root}/{slide_id}.pt"
        try:
            data_origin = h5py.File(feat_path, 'r')
            cls_origin = torch.load(cls_path)
        except Exception as e:
            print('load error', e)
            continue

        features = np.asarray(data_origin['features'], dtype=np.float32)
        cls = np.asarray(cls_origin['patch_classify_type'], dtype=np.int16)

        coord1 = np.asarray(data_origin['coords'], dtype=np.int32)
        coord2 = np.asarray(cls_origin['centroid'], dtype=np.int32)
        if ~(np.equal(coord1, coord2).all()):
            continue

        all_features.append(features)
        all_classes.append(cls)

        cnt += features.shape[0]
        if cnt > max_patches:
            break
    print('load_datas finish', flush=True)
    all_features = torch.from_numpy(np.concatenate(all_features, axis=0))
    all_classes = torch.from_numpy(np.concatenate(all_classes, axis=0))
    return all_features, all_classes


def main(args):
    n_patches = args.max_patches
    patches, clses = load_datas(args.fearture_source, args.cls_source, max_patches=n_patches)

    prototypes = []
    for i in range(4):
        tmp_arr = patches[clses == i]
        tmp_ft = torch.mean(tmp_arr, dim=0)
        prototypes.append(tmp_ft)

    torch.save(prototypes, f"{args.save_dir}/{args.project}.pt")


# Generic training settings
parser = argparse.ArgumentParser(description='Make prototypes')
parser.add_argument('--project', type=str, default='PAAD')
# model / loss fn args ###
parser.add_argument('--max_patches', type=int, default=500000,
                    help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
# dataset / split args ###
parser.add_argument('--fearture_source', type=str, default='../CLAM-master/datas/feature/(project_name)/h5_files',
                    help='manually specify the data source from CLAM  .h5 format')
parser.add_argument('--cls_source', type=str, default='./classify/(project_name)/',
                    help='manually specify the cls source from classification.py  .pt format')
parser.add_argument('--save_dir', type=str, default='../prototypes_tmp/',
                    help='prototype save folder')

args = parser.parse_args()
if __name__ == "__main__":
    os.makedirs(args.save_dir, exist_ok=True)

    args.fearture_source = args.fearture_source.replace('(project_name)', args.project)
    args.cls_source = args.cls_source.replace('(project_name)', args.project)

    main(args)
