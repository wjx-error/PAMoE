### System
import sys
import h5py
import nmslib
from tqdm import tqdm
from utils.utils import *


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


def pt2graph(wsi_h5, cls_pth, radius=9):
    from torch_geometric.data import Data as geomData
    from itertools import chain
    coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]

    # node types from data_preparation/classification_CONCH.py
    if cls_pth is not None:
        clss = torch.load(cls_pth)
        coords_cls, cls = clss.centroid, clss.patch_classify_type
        assert np.array_equal(coords, coords_cls)
        cls = torch.Tensor(cls)
    else:
        cls = None

    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius - 1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int)
    edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    G = geomData(x=torch.Tensor(features),
                 edge_index=edge_spatial,
                 edge_latent=edge_latent,
                 centroid=torch.Tensor(coords),
                 patch_classify_type=cls,
                 )
    return G


def createDir_h5toPyG(h5_path, node_type_path, save_path):
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        pbar.set_description('%s - Creating Graph' % (h5_fname[:12]))
        try:
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            h5_fname_p = h5_fname.rstrip('.pt').replace('.h5', '')

            if node_type_path is not None:
                cls_pth = f"{node_type_path}/{h5_fname_p}.pt"
            else:
                cls_pth = None

            G = pt2graph(wsi_h5, cls_pth, radius=5)
            torch.save(G, os.path.join(save_path, h5_fname[:-3] + '.pt'))
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')


if __name__ == "__main__":
    h5_path = 'path/to/data_root_dir/tcga_brca_20x_features/h5_files/'
    save_path = 'path/to/data_root_dir/tcga_brca_20x_features/graph_files/'

    # set node_type_path=None when you want to create graphes without node types
    # node_type_path = None
    node_type_path = 'path/to/data_root_dir/classfication/'

    os.makedirs(save_path, exist_ok=True)

    createDir_h5toPyG(h5_path, node_type_path, save_path)
