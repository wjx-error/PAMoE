"""
This will perform K-means clustering on the training data

Good reference for clustering
https://github.com/facebookresearch/faiss/wiki/FAQ#questions-about-training
"""

from __future__ import print_function
import argparse
import torch
import pickle
import numpy as np
from sklearn.cluster import KMeans
import time
import glob
import h5py
import random

import os


def load_datas(root_pth, max_patches):
    print(root_pth, flush=True)
    pth_list = glob.glob(f"{root_pth}/**.h5", recursive=True)
    random.shuffle(pth_list)
    # print(pth_list)
    all_features = []
    num = 0
    print('max_patches', max_patches, flush=True)
    for feat_path in pth_list:
        try:
            data_origin = h5py.File(feat_path, 'r')
        except:
            print('pickle load error')
            continue

        features = np.asarray(data_origin['features'], dtype=np.float32)
        all_features.append(features)

        # print(features.shape, num)
        num += features.shape[0]
        if num > max_patches:
            break
    print('load_datas finish', flush=True)
    all_features = torch.from_numpy(np.concatenate(all_features, axis=0))
    return all_features


def main(args):
    n_proto = args.n_proto
    n_iter = args.n_iter
    n_init = args.n_init
    mode = args.mode
    n_proto_patches = args.n_proto_patches

    n_patches = args.n_proto * args.n_proto_patches
    patches = load_datas(args.data_source, max_patches=n_patches)
    print(patches.type, patches.shape)

    print('\nInit Datasets...', end=' ', flush=True)
    os.makedirs(f"../prototypes", exist_ok=True)

    s = time.time()
    if mode == 'kmeans':
        print("\nUsing Kmeans for clustering...")
        print(f"\n\tNum of clusters {n_proto}, num of iter {n_iter}", flush=True)
        kmeans = KMeans(n_clusters=n_proto, max_iter=n_iter)
        kmeans.fit(patches[:n_patches].cpu())
        weight = kmeans.cluster_centers_[np.newaxis, ...]

    elif mode == 'faiss':
        # assert use_cuda, f"FAISS requires access to GPU. Please enable use_cuda"
        try:
            import faiss
        except ImportError:
            print("FAISS not installed. Please use KMeans option!")
            raise

        numOfGPUs = torch.cuda.device_count()
        print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...", flush=True)
        print(f"\tNum of clusters {n_proto}, num of iter {n_iter}", flush=True)

        kmeans = faiss.Kmeans(patches.shape[1],
                              n_proto,
                              niter=n_iter,
                              nredo=n_init,
                              verbose=True,
                              max_points_per_centroid=n_proto_patches,
                              gpu=numOfGPUs)
        kmeans.train(patches.numpy())
        weight = kmeans.centroids[np.newaxis, ...]

    else:
        raise NotImplementedError(f"Clustering not implemented for {mode}!")

    print('weight', weight.shape, flush=True)

    e = time.time()
    print(f"\nClustering took {e - s} seconds!", flush=True)

    save_fpath = f"./prototypes/{args.project}.pt"
    print('save to:', save_fpath)

    weight = weight.squeeze()
    weight = torch.tensor(weight)
    lst = [weight[i] for i in range(len(weight))]

    torch.save(lst, save_fpath)

    # writer = open(save_fpath, 'wb')
    # pickle.dump({'prototypes': weight}, writer)
    # writer.close()


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--project', type=str, default='PAAD')
# model / loss fn args ###
parser.add_argument('--n_proto', type=int, default=16, help='Number of prototypes')
parser.add_argument('--n_proto_patches', type=int, default=10000,
                    help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
parser.add_argument('--n_init', type=int, default=5,
                    help='Number of different KMeans initialization (for FAISS)')
parser.add_argument('--n_iter', type=int, default=50,
                    help='Number of iterations for Kmeans clustering')
parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='kmeans')

# dataset / split args ###
parser.add_argument('--data_source', type=str,
                    default='path/to/data_root_dir/features_TCGA_256/feature_(project_name)_256_uni_dim_1024',
                    help='manually specify the data source')

args = parser.parse_args()

if __name__ == "__main__":
    args.data_source = args.data_source.replace('(project_name)', args.project)
    main(args)
