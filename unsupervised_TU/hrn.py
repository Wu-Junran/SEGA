#!/usr/bin/env python
# encoding: utf-8
import sys
import time
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Callable
from functools import reduce
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp


class HRNConv(MessagePassing):
    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class HRNEncoder(torch.nn.Module):
    def __init__(self, num_convs=2, pooling_type='sum', num_features=-1,
                 hidden_dim=32, output_dim=32, final_dropout=0,
                 link_input=False, drop_root=False, device='cpu'):
        super(HRNEncoder, self).__init__()
        self.num_convs = num_convs
        self.pooling_type = pooling_type
        self.num_features = num_features  # input_dim
        self.nhid = hidden_dim  # hidden dim
        self.output_dim = output_dim # output dim
        self.dropout_ratio = final_dropout
        self.link_input = link_input
        self.drop_root = drop_root
        self.device = device
        self.convs = self.get_convs()
        self.pool = self.get_pool()
        self.dim_align = self.get_linear()

    def get_linear(self):
        init_dim = self.nhid * self.num_convs
        if self.link_input:
            init_dim += self.num_features
        if self.drop_root:
            init_dim -= self.nhid
        if self.pooling_type == 'root':
            init_dim = self.nhid
        return nn.Sequential(
            nn.Linear(init_dim, self.output_dim),
            nn.Dropout(p=self.dropout_ratio),
        )

    def __process_layer_batch(self, data, layer=0):
        if layer == 0:
            return data.batch
        return data['treePHLayer%s_batch' % layer]

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        for _ in range(self.num_convs):
            conv = HRNConv(
                nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.BatchNorm1d(_output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.BatchNorm1d(_output_dim),
                    nn.ReLU(),
                ))
            convs.append(conv)
            _input_dim = _output_dim
        return convs

    def get_pool(self):
        if self.pooling_type == 'sum':
            return gsp
        else:
            return gap

    def forward(self, data):
        x = data.deg_x
        xs = [x] if self.link_input else []
        for _ in range(self.num_convs):
            edge_index = data['treeEdgeMatLayer%s' % (_+1)]
            size = data.treeNodeSize[:, [_, _+1]].sum(dim=0)
            x = self.convs[_](x, edge_index, size=size)
            xs.append(x)

        if self.pooling_type == 'root':
            x = self.dim_align(xs[-1])
            return x

        if self.drop_root:
            xs = xs[:-1]
        pooled_xs = []
        for _, x in enumerate(xs):
            batch = self.__process_layer_batch(data, _ if self.link_input else _+1)
            pooled_x = self.pool(x, batch)
            pooled_xs.append(pooled_x)

        x = torch.cat(pooled_xs, dim=1)
        x = self.dim_align(x)
        return x

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                x = self.forward(data)
                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class HRN(torch.nn.Module):
    def __init__(self, encoder, emb_dim):
        super(HRN, self).__init__()
        self.encoder = encoder
        self.input_dim = self.encoder.output_dim
        self.proj_head = nn.Sequential(nn.Linear(self.input_dim, emb_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(emb_dim, emb_dim))

    def forward(self, data):
        z = self.encoder(data)
        z = self.proj_head(z)
        return z
