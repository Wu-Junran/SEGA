import sys
sys.path.append('/gs/software/anaconda3/lib/python3.7/site-packages/')
import json
import torch
import random
import shutil
import argparse
import numpy as np
import os.path as osp
import torch.nn as nn
from gin import Encoder
from torch import optim
from hrn import HRN, HRNEncoder
import torch.nn.functional as F
from multiprocessing import Pool
from torch.autograd import Variable
from codingTree import get_tree_data
from torch_geometric.data import Data
from torch_geometric.utils import degree
from aug import TUDataset_aug as TUDataset
from torch_geometric.typing import OptTensor
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import Constant
from evaluate_embedding import evaluate_embedding


def add_degree(data, max_degree):
    if data.x is not None:
        deg = degree(data.edge_index[0], data.x.shape[0], dtype=torch.long)
    else:
        deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
    deg = deg.view((-1, 1))
    max_deg = torch.tensor(max_degree, dtype=deg.dtype)
    deg = torch.min(deg, max_deg).view(-1)
    onehot_deg = F.one_hot(deg, num_classes=max_degree+1).to(torch.float)
    if data.x is not None:
        data.deg_x = torch.cat([data.x, onehot_deg.to(data.x.dtype)], dim=-1)
    else:
        data.deg_x = onthot_deg
    return data


class GraphTransform():
    def __init__(self, tree_depth=2, max_degree=10):
        self.tree_depth = tree_depth
        self.max_degree = max_degree

    def transform(self, odata):
        # copy
        data = GTData()
        for key in odata.keys:
            data[key] = odata[key]
        if data.x is None:
            data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        nodeSize, edgeSize, edgeMat = get_tree_data(adj, self.tree_depth)
        data.treeNodeSize = torch.LongTensor(nodeSize).view(1, -1)
        for layer in range(1, self.tree_depth+1):
            data['treePHLayer%s' % layer] = torch.ones([nodeSize[layer], 1])  # place holder
            data['treeEdgeMatLayer%s' % layer] = torch.LongTensor(edgeMat[layer]).T
        data = add_degree(data, self.max_degree)
        return data


class GTData(Data):
    def __init__(self, x: OptTensor=None, edge_index: OptTensor=None,
                 edge_attr: OptTensor=None, y: OptTensor=None,
                 pos: OptTensor=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key.find('treeEdgeMatLayer') >= 0:
            layer = int(key.replace('treeEdgeMatLayer', ''))
            return torch.tensor([[self.treeNodeSize[0][layer]],
                                 [self.treeNodeSize[0][layer-1]]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key.find('treeEdgeMatLayer') >= 0:
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


def run_once(input_):
    dataset, tree_depth = input_
    path = 'data'
    degreeDim = 10 if dataset in ['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'] else 100
    pre_transform = GraphTransform(tree_depth, degreeDim).transform
    processed_filename = 'data_t%s.pt' % tree_depth
    dataset = TUDataset(path, name=dataset, aug='none', pre_transform=pre_transform,
                        processed_filename=processed_filename).shuffle()
    print(input_)


if __name__ == '__main__':
    social_datasets = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'github_stargazers', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    bio_datasets = ['NCI1', 'PROTEINS', 'DD', 'MUTAG', 'ENZYMES', 'PTC_MR']
    tree_depths = [2, 3, 4, 5]
    pool = Pool(32)
    pool.map(run_once, [(d, td) for d in social_datasets + bio_datasets for td in tree_depths])
    pool.close()
    pool.join()
