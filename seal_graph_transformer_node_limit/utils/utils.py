
import shutil
import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path, floyd_warshall
from queue import PriorityQueue
import torch
import torch.nn.functional as F
from torch_sparse import spspmm
import fastremap

from torch_geometric.data.collate import collate

import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (
    negative_sampling, add_self_loops, train_test_split_edges, to_dense_adj)
import pdb
from self_graph_transformer import CustomData

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})


import utils.algos as algos  # nopep8


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A,
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        num_edges = A[list(fringe)].sum()
        res = set(A[list(fringe)].indices)
    else:
        num_edges = A[:, list(fringe)].sum()
        res = set(A[:, list(fringe)].indices)

    return res, num_edges


def k_hop_subgraph_limited(src, dst, num_hops, A, sample_ratio=1.0,
                            max_nodes_per_hop=None, node_features=None,
                            y=1, directed=False, A_csc=None, max_nodes=512, z_limit=1000):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe, num_edges = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)

        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)

        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
        if len(nodes) > max_nodes:
            break
    subgraph = A[nodes, :][:, nodes]
    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    
    z = drnl_node_labeling(subgraph, 0, 1)
    zero_z_val = z.max()+1
    z[z == 0] = zero_z_val
    z_sort_arg = z.argsort()
    z_sort_arg = z_sort_arg.argsort()
    z_argsort_mask = (z_sort_arg < max_nodes)
    z_limit_mask = ((z <= z_limit) & z_argsort_mask)

    nodes = torch.tensor(nodes, dtype=torch.long)[z_limit_mask].tolist()

    subgraph = A[nodes, :][:, nodes]
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0


    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe, num_edges = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)

        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)

        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    # if len(nodes) != 52:
    #     print("Why????????")
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False,
                             unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False,
                             unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def eig(sym_mat):
    # (sorted) eigenvectors with numpy
    EigVal, EigVec = np.linalg.eigh(sym_mat)

    # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    eigvec = torch.from_numpy(EigVec).float()  # [N, N (channels)]
    eigval = torch.from_numpy(
        np.sort(np.abs(np.real(EigVal)))).float()  # [N (channels),]
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def lap_eig(dense_adj, in_degree):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
    """
    num_nodes = dense_adj.shape[0]
    dense_adj = dense_adj.detach().float().numpy()
    in_degree = in_degree.detach().float().numpy()

    # Laplacian
    A = dense_adj
    N = np.diag(in_degree.clip(1) ** -0.5)
    L = np.eye(num_nodes) - N @ A @ N

    eigvec, eigval = eig(L)
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def get_edge_id(src, dst, z):
    src_z = min(z[src], 10)
    dst_z = min(z[dst], 10)

    if (src_z > dst_z):
        return 10*src_z + dst_z
    else:
        return 10*dst_z + src_z


def graph_tokenizer(adj, edge_index, node_feature, global_degree, edge_feature, src, dst,  max_nodes=512):

    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False,
                             unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)

    assert (src == 0 and dst == 1)
    dist2dst = shortest_path(adj_wo_src, directed=False,
                             unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)

    dist2src = torch.from_numpy(dist2src)
    dist2dst = torch.from_numpy(dist2dst)

    max_dist = 99
    dist2src = dist2src.clamp(0, max_dist)
    dist2dst = dist2dst.clamp(0, max_dist)
    dist2src[torch.isnan(dist2src)] = max_dist
    dist2dst[torch.isnan(dist2dst)] = max_dist

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.
    if (dist2src[dst] != 0) or dist2dst[src] != 0:
        print("Issue!!!!!!!!")

    # Add graphormer spatial data

    # #make mask with zero padding
    # edge_token = torch.zeros(edge_index.shape[1])
    # for i in range(0, edge_index.shape[1]):
    #     if z[edge_index[0][i]] > z[edge_index[1][i]]:
    #         val = z[edge_index[1][i]]*100 + z[edge_index[0][i]]
    #     else:
    #         val = z[edge_index[0][i]]*100 + z[edge_index[1][i]]
    #     edge_token[i] = val

    # token[1:cur_idx] = sorted(token[1:cur_idx])

    # shortest_path_result, _ = floyd_warshall(
    #     adj.toarray())
    # spatial_pos = torch.from_numpy((shortest_path_result)).long()

    node_dist = torch.stack(
        [dist2src, dist2dst]).t().to(torch.long)
    node_dist = torch.min(dist2src, dist2dst).to(torch.long)
    # dist2src1 = shortest_path(adj, indices=[src, dst])
    # global_degree = torch.ceil(torch.log2(global_degree + 1)).to(torch.long)

    # z, adj, node_feature, global_degree, node_dist, edge_index, edge_feature = get_z_restricted_graph(
    #     z, adj, node_feature, global_degree, node_dist, edge_index, edge_feature, max_nodes=max_nodes)

    z = change_z(z)
    local_degree = torch.tensor(adj.sum(1)).to(torch.long).view(-1)
    local_degree = torch.ceil(torch.log2(local_degree + 1)).to(torch.long)

    global_degree = torch.ceil(std_normalize(
        global_degree.to(torch.float), 50, 50)).to(torch.long)
    global_degree = global_degree.clamp(0, 100)

    shortest_path_result = shortest_path(
        adj.toarray(), directed=False, unweighted=True)

    # idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    # adj_wo_src = adj[idx, :][:, idx]

    # idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    # adj_wo_dst = adj[idx, :][:, idx]


    # shortest_path_wo_src = shortest_path(
    #     adj_wo_src, directed=False, unweighted=True)
    # shortest_path_wo_src = np.insert(shortest_path_wo_src, src, 0, axis=0)
    # shortest_path_wo_src = np.insert(shortest_path_wo_src, src, 0, axis=1)

    # shortest_path_wo_src[src] = dist2src
    # shortest_path_wo_src[:, src] = dist2src

    # shortest_path_wo_dst = shortest_path(
    #     adj_wo_dst, directed=False, unweighted=True)

    # shortest_path_wo_dst = np.insert(shortest_path_wo_dst, dst, 0, axis=0)
    # shortest_path_wo_dst = np.insert(shortest_path_wo_dst, dst, 0, axis=1)
    # shortest_path_wo_dst[dst] = dist2dst
    # shortest_path_wo_dst[:, dst] = dist2dst

    # shortest_path_result = (shortest_path_wo_dst + shortest_path_wo_src)//2
    # if (shortest_path_result[0, 1] == 3):
    #     print("far")
    spatial_pos = torch.from_numpy(shortest_path_result).long()
    spatial_pos = spatial_pos.clamp(0, 99) + 1
    spatial_pos[spatial_pos == 100] = 0

    # z_adj = adj.copy()
    # for i in range(len(edge_z)):
    #     z_adj[edge_index[0][i], edge_index[1][i]] = edge_z[i]

    # lap_eigvec, lap_eigval = lap_eig(dense_adj, in_degree)# [N, N], [N,]
    # lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)
    # return z.to(torch.long), edge_data.to(torch.long), lap_eigval.squeeze(0), lap_eigvec.squeeze(0), in_degree.to(torch.long) #torch.Tensor(mask).to(torch.long)
    
    adj = torch.tensor(adj.toarray(), dtype=torch.long)
    # Add 1 to non zero entries of adj, make diagonal entries 1
    adj = pad_till_nodes(adj, max_nodes)
    spatial_pos = pad_till_nodes(spatial_pos, max_nodes)
    graph_edge_mat = torch.ones(
        (spatial_pos.shape[0], spatial_pos.shape[1]), dtype=torch.long)
    graph_edge_mat[src, dst] = 2
    graph_edge_mat[dst, src] = 2
    graph_edge_mat = pad_till_nodes(graph_edge_mat, max_nodes)
    return z.to(torch.long), adj, node_feature, global_degree, local_degree, node_dist, edge_index, edge_feature, spatial_pos, graph_edge_mat


def std_normalize(input, mu, std):
    m1 = input.mean()
    s1 = input.std()
    return mu - (input - m1)*(std/s1)


def pad_till_nodes(x, max_nodes):
    x_new = 0*torch.ones((max_nodes, max_nodes), dtype=torch.long)
    x_new[:x.shape[0], :x.shape[1]] = x
    return x_new


def change_z(z, max_z = 100):
    zero_z_val = z.max()+1
    z[z == 0] = zero_z_val
    z[(z != zero_z_val) & (z > max_z)] = max_z
    z[z == zero_z_val] = max_z + 1
    return z

def get_z_restricted_graph(z, adj, node_feature, global_degree, node_dist, edge_index, edge_feature, z_limit=100, z_edge_limit=100, max_nodes=512, max_node_dist=10):
    zero_z_val = z.max()+1
    z[z == 0] = zero_z_val
    z_sort_arg = z.argsort()
    z_sort_arg = z_sort_arg.argsort()
    z_argsort_mask = (z_sort_arg < max_nodes)
    # Calculate max z based on 2,10 distances
    max_z_replace = 1 + 2 + ((2 + max_node_dist)//2) * \
        ((2 + max_node_dist)//2 + (2 + max_node_dist) % 2 - 1)
    z[(z != zero_z_val) & (z > max_z_replace)] = max_z_replace
    z[z == zero_z_val] = max_z_replace + 1
    orig_z = z.clone()
    z_limit_mask = ((z <= z_limit) & z_argsort_mask)
    z = z[z_limit_mask]
    global_degree = global_degree[z_limit_mask]
    if node_feature is not None:
        node_feature = node_feature[z_limit_mask]
    node_dist = node_dist[z_limit_mask]
    indexes = np.where(z_limit_mask)[0]
    not_indexes = np.where(np.logical_not(z_limit_mask))[0]
    idx_map = torch.zeros(len(indexes) + len(not_indexes), dtype=torch.long)
    idx_map[indexes] = torch.linspace(
        0, len(indexes), len(indexes)).to(torch.long)
    idx_map[not_indexes] = -1 * torch.ones(len(not_indexes), dtype=torch.long)
    adj = adj[indexes, :][:, indexes]
    z_edge_index = orig_z[edge_index]

    z_limit_edge_mask = ((z_edge_index[0] <= z_edge_limit) & (
        z_edge_index[1] <= z_edge_limit))
    edge_index = edge_index.t()[z_limit_edge_mask].t()
    edge_feature = edge_feature[z_limit_edge_mask]

    temp_edge_index = edge_index.flatten()
    temp_edge_index = idx_map[temp_edge_index]
    edge_index = temp_edge_index.reshape(
        (edge_index.shape[0], edge_index.shape[1]))

    edge_outside_max_mask = ((edge_index[0] >= 0) & (edge_index[1] >= 0))
    edge_index = edge_index.t()[edge_outside_max_mask].t()
    edge_feature = edge_feature[edge_outside_max_mask]

    # if z.shape[0] != node_feature.shape[0] or z.shape[0] != node_dist.shape[0] or z.shape[0] != local_degree.shape[0]:
    #     print("Shape mismatched shape")

    return z, adj, node_feature, global_degree, node_dist, edge_index, edge_feature


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False,
                         unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False,
                             unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False,
                             unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def convert_to_single_emb(x, offset: int = 2):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def construct_pyg_graph(node_ids, adj, dists, node_features, y, degrees, node_label='drnl', max_nodes=512):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    num_edges = edge_index.shape[1]
    y = torch.tensor([y])
    if node_label.startswith('drnl'):  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists) == 0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z > 100] = 100  # limit the maximum label to 100
    elif node_label == 'tokenizer':
        z, adj, node_feature, local_degree, global_degree, node_dist, edge_index, edge_feature, spatial_pos, graph_edge_mat = graph_tokenizer(
            adj, edge_index, node_features, degrees, edge_weight, 0, 1, max_nodes=max_nodes)  # return tokens (size : number of edges in subgraph)
        num_nodes = z.shape[0]
        num_edges = edge_index.shape[1]
        # edge_weight.unsqueeze(1).to(torch.long)
        return CustomData(x=node_feature,
                          edge_data=edge_feature.unsqueeze(1).to(torch.long),
                          edge_index=edge_index, node_num=num_nodes,
                          edge_num=num_edges,
                          y=y,
                          global_degree=global_degree, dist=node_dist, z_data=z,
                          local_degree=local_degree,
                          adjacency_matrix=adj.view(-1),
                          spatial_pos=spatial_pos.view(-1),
                          graph_edge_mat=graph_edge_mat.view(-1))

    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight,
                y=y, z=z, node_id=node_ids, num_nodes=num_nodes)
    return data


def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl',
                                ratio_per_hop=1.0, max_nodes_per_hop=None,
                                directed=False, A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                             max_nodes_per_hop, node_features=x, y=y,
                             directed=directed, A_csc=A_csc)
        data = construct_pyg_graph(*tmp, node_label)
        data_list.append(data)

    return data_list


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()

        if 'edge_neg' in split_edge['train']:
            # use presampled  negative training edges for ogbl-vessel
            neg_edge = split_edge[split]['edge_neg'].t()

        else:
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))

        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    # edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def merge_results(self):
        max_epochs = 0
        for run in range(len(self.results)):
            max_epochs = max(max_epochs, len(self.results[run]))
        for run in range(len(self.results)):
            current_epochs = len(self.results[run])
            last_epoch = self.results[run][-1]
            for _ in range(current_epochs, max_epochs):
                self.results[run].append(last_epoch)

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
        else:
            self.merge_results()
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}', file=f)


def get_loss(pred, ans, vocab_size, label_smoothing, pad):
    # took this "normalizing" from tensor2tensor. We subtract it for
    # readability. This makes no difference on learning.
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / float(vocab_size - 1)
    normalizing = -(
        confidence * math.log(confidence) + float(vocab_size - 1) *
        low_confidence * math.log(low_confidence + 1e-20))

    one_hot = torch.zeros_like(pred).scatter_(1, ans.unsqueeze(1), 1)
    one_hot = one_hot * confidence + (1 - one_hot) * low_confidence
    log_prob = F.log_softmax(pred, dim=1)

    xent = -(one_hot * log_prob).sum(dim=1)
    xent = xent.masked_select(ans != pad)
    loss = (xent - normalizing).mean()
    return loss


def get_accuracy(pred, ans, pad):
    pred = pred.max(1)[1]
    n_correct = pred.eq(ans)
    n_correct = n_correct.masked_select(ans != pad)
    return n_correct.sum().item() / n_correct.size(0)


def save_checkpoint(model, filepath, global_step, is_best):
    model_save_path = filepath + '/last_model.pt'
    torch.save(model, model_save_path)
    torch.save(global_step, filepath + '/global_step.pt')
    if is_best:
        best_save_path = filepath + '/best_model.pt'
        shutil.copyfile(model_save_path, best_save_path)


def load_checkpoint(model_path, device, is_eval=True):
    if is_eval:
        model = torch.load(model_path + '/best_model.pt')
        model.eval()
        return model.to(device=device)

    model = torch.load(model_path + '/last_model.pt')
    global_step = torch.load(model_path + '/global_step.pt')
    return model.to(device=device), global_step


def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask


def create_trg_self_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention.
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=device)
    t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)

    return t_self_mask
