import time

import os, sys
import os.path as osp
import copy as cp


import torch
from torch_geometric.utils import dense_to_sparse, to_undirected, to_dense_adj
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_sparse import coalesce

import numpy as np
import scipy.sparse as ssp

from ogb.graphproppred import PygGraphPropPredDataset

from utils import utils
from shutil import copy
from tqdm import tqdm


# Download and process data at '/home/jasmine/Torch_practice/dataset/ogbg_molhiv/'
dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = '/home/jasmine/Torch_practice/dataset/')
 
split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, directed=False):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = utils.get_pos_neg_edges(self.split, self.split_edge, self.data.edge_index, self.data.num_nodes, self.percent)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(self.data.edge_index, self.data.edge_weight, self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])), shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None
        
        # Extract enclosing subgraphs for pos and neg edges
        pos_list = utils.extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label, self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)
        neg_list = utils.extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label, self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc)

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list