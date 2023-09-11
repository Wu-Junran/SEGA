import sys
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


class simclr(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

    def loss_cal(self, x, x_aug, sym=False):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss0 = -torch.log(loss0).mean()
            loss1 = -torch.log(loss1).mean()
            loss = (loss0 + loss1) / 2.0
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = -torch.log(loss).mean()
        return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


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
        data.deg_x = onehot_deg
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


def run_once(args):
    setup_seed(args.seed)
    epochs = 20
    log_interval = 10
    path = 'data'

    degreeDim = 10 if args.DS in ['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'] else 100
    pre_transform = GraphTransform(args.tree_depth, degreeDim).transform
    processed_filename = 'data_t%s.pt' % args.tree_depth
    dataset = TUDataset(path, name=args.DS, aug=args.aug, pre_transform=pre_transform,
                        processed_filename=processed_filename).shuffle()
    dataset_eval = TUDataset(path, name=args.DS, aug='none', pre_transform=pre_transform,
                        processed_filename=processed_filename).shuffle()

    try:
        num_features = dataset.get_num_feature()
    except:
        num_features = 1

    fb_keys = [key for key in dataset[0][0].keys if key.find('treePHLayer')>=0]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, follow_batch=fb_keys)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, follow_batch=fb_keys)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # gnn
    model = simclr(num_features, args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # hrn
    encoder = HRNEncoder(args.tree_depth, args.tree_pooling_type,
                         dataset[0][0].deg_x.shape[1], args.tree_hidden_dim,
                         args.hidden_dim*args.num_gc_layers, args.tree_dropout,
                         args.tree_link_input, args.tree_drop_root,
                         device)
    treeM = HRN(encoder, args.hidden_dim*args.num_gc_layers).to(device)
    treeOpt = torch.optim.Adam(treeM.parameters(), lr=args.tree_learning_rate)

    accuracies = {'G':[], 'T':[], 'GT': []}
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        treeM.train()
        for data in dataloader:
            data, data_aug = data
            optimizer.zero_grad()
            treeOpt.zero_grad()
            # augmentation view
            node_num, _ = data.x.size()
            if args.aug in ['dnodes', 'subgraph', 'random2', 'random3', 'random4']:
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]
                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
            data_aug = data_aug.to(device)
            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            # coding tree view
            data = data.to(device)
            x_hrn = treeM(data)
            # contrastive loss
            loss = model.loss_cal(x_hrn, x_aug, args.loss_sym)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            treeOpt.step()

        if epoch % log_interval == 0:
            model.eval()
            treeM.eval()
            embG, y = model.encoder.get_embeddings(dataloader_eval)
            embT, y = treeM.encoder.get_embeddings(dataloader_eval, device)
            embGT = np.concatenate((embG, embT), axis=1)
            acc_valG, accG = evaluate_embedding(embG, y)
            acc_valT, accT = evaluate_embedding(embT, y)
            acc_valGT, accGT = evaluate_embedding(embGT, y)
            accuracies['G'].append(accG)
            accuracies['T'].append(accT)
            accuracies['GT'].append(accGT)
            print(epoch, args.seed, accG, accT, accGT, flush=True)
            with open('logs/%s.out' % args.DS, 'a+') as fp:
                fp.write('%s %s %s\t%.6f %.6f %.6f\n' % (
                         args_line, epoch, args.seed, accG*100, accT*100, accGT*100))
    return accuracies


def is_run(dataset, args_line):
    with open('logs/%s.out' % dataset) as fp:
        for l in fp:
            if l.find('%s 20 all' % args_line)>=0:
                return True
    return False


def run_tuning(args):
    global args_line
    args_line = '%s %s %s\t%s %s %s %s %s %s %s\t%s' % (
            args.DS,
            args.batch_size,
            'lossS' if args.loss_sym else 'lossNS',
            args.tree_learning_rate,
            'linkIn' if args.tree_link_input else 'jumpIn',
            'dropR' if args.tree_drop_root else 'saveR',
            args.tree_hidden_dim,
            args.tree_depth,
            args.tree_pooling_type,
            args.tree_dropout,
            args.aug)
    if is_run(args.DS, args_line):
        return
    accsG, accsT, accsGT = [], [], []
    for i in [0, 1, 2, 3, 4]:
        args.seed = i
        print(args_line, args.seed)
        accs = run_once(args)
        accsG.append(accs['G'])
        accsT.append(accs['T'])
        accsGT.append(accs['GT'])
    accsG = np.array(accsG)
    accsT = np.array(accsT)
    accsGT = np.array(accsGT)
    with open('logs/%s.out' % args.DS, 'a+') as fp:
        for i in range(accsG.shape[1]):
            acc_std = '%.6f %.4f\t%.6f %.4f\t%.6f %.4f' % (
                       accsG[:,i].mean()*100, accsG[:,i].std()*100,
                       accsT[:,i].mean()*100, accsT[:,i].std()*100,
                       accsGT[:,i].mean()*100, accsGT[:,i].std()*100)
            print(args.aug, (i+1)*10, acc_std)
            fp.write('%s %s all\t%s\n' % (args_line, (i+1)*10, acc_std))


def arg_parse():
    parser = argparse.ArgumentParser(description='tree contrastive learning Arguments.')
    parser.add_argument('-d', '--DS', default='PROTEINS', help='Dataset')
    parser.add_argument('-l', '--local', dest='local', action='store_const',
            const=True, default=False)
    parser.add_argument('-g', '--glob', dest='glob', action='store_const',
            const=True, default=False)
    parser.add_argument('-p', '--prior', dest='prior', action='store_const',
            const=True, default=False)
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--loss_sym', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tree_depth', type=int, default=5)
    parser.add_argument('--tree_pooling_type', type=str, default='sum')
    parser.add_argument('--tree_hidden_dim', type=int, default=32)
    parser.add_argument('--tree_dropout', type=int, default=0)
    parser.add_argument('--tree_link_input', action='store_true')
    parser.add_argument('--tree_drop_root', action='store_true')
    parser.add_argument('--tree_learning_rate', type=float, default=0.01)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    social_datasets = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'github_stargazers', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    bio_datasets = ['NCI1', 'PROTEINS', 'DD', 'MUTAG', 'ENZYMES', 'PTC_MR']
    run_tuning(args)
