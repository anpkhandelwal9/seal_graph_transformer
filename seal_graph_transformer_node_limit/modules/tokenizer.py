import math
from platform import node

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.utils.to_dense_adj import to_dense_adj
from torch_geometric.nn.models import node2vec

from .orf import gaussian_orthogonal_random_matrix_batched


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        module.weight.data[module.padding_idx] = 0


class NodeEmbedding(nn.Module):
    def __init__(self, num_node_features, max_value, hidden_dim):
        super().__init__()
        if num_node_features > 1:
            self.act = nn.ReLU()
            self.dropout = nn.Dropout2d(0.5)
            self.bn = nn.SyncBatchNorm(hidden_dim)
            self.fc1 = nn.Linear(num_node_features, hidden_dim)
        else:
            self.fc1 = nn.Embedding(
                max_value, hidden_dim, padding_idx=max_value-1)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.fc1(x))))


class EdgeEmbedding(nn.Module):
    def __init__(self, num_edge_features, max_value, hidden_dim):
        super().__init__()
        if num_edge_features == 1:
            self.layer = nn.Embedding(
                max_value, hidden_dim, padding_idx=max_value-1)
        else:
            self.layer = nn.Linear(num_edge_features, hidden_dim)

    def forward(self, x):
        return self.layer(x)


class GraphFeatureTokenizer(nn.Module):
    """
    Compute node and edge features for each node and edge in the graph.
    """

    def __init__(
            self,
            num_node_features,
            num_edge_features,
            rand_node_id,
            rand_node_id_dim,
            orf_node_id,
            orf_node_id_dim,
            lap_node_id,
            lap_node_id_k,
            lap_node_id_sign_flip,
            lap_node_id_eig_dropout,
            type_id,
            hidden_dim,
            n_layers,
            num_heads,
            max_z=1000,  # max value of drnl,
            max_edge_z=10000,
            max_nodes_in_graph=10000,
            num_gcn_layers=3,
            max_nodes=512
    ):
        super(GraphFeatureTokenizer, self).__init__()

        self.use_node_features = False
        self.use_edge_features = False
        self.use_edge_encoding_bias = True
        self.use_gcn = False
        self.max_nodes = max_nodes
        if self.use_node_features:
            hidden_channels = hidden_dim//4

        else:
            hidden_channels = hidden_dim//3
        # hidden_channels = hidden_dim
        if self.use_node_features:
            self.node_encoder = NodeEmbedding(
                num_node_features, max_z, hidden_dim - 3*hidden_channels)
        # self.node_dropout = nn.Dropout2d()
        # self.act = nn.ReLU(inplace=True)
        # self.bn = nn.SyncBatchNorm(num_features=hidden_dim-2*hidden_channels)

        if self.use_edge_features:
            self.edge_encoder = EdgeEmbedding(
                num_edge_features, 112, hidden_dim)

        if self.use_gcn:
            self.convs = nn.ModuleList()
            for _ in range(num_gcn_layers):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.local_degree_encoder = nn.Embedding(
            1000, hidden_channels, padding_idx=199)

        self.global_degree_encoder = nn.Embedding(
            200, hidden_channels, padding_idx=199)

        z_embedding_size = hidden_channels
        if not(self.use_node_features):
            z_embedding_size = hidden_dim - 2*hidden_channels
        self.z_encoder = nn.Embedding(
            max_z, z_embedding_size, padding_idx=max_z-1)
        # self.node_id_encoder = nn.Embedding(
        #     max_nodes_in_graph, hidden_channels, padding_idx=0
        # )
        self.dist_encoder = nn.Embedding(
            100, hidden_dim, padding_idx=99)

        self.graph_token = nn.Embedding(1, hidden_dim)
        # self.null_token = nn.Embedding(1, hidden_dim)  # this is optional

        self.rand_node_id = rand_node_id
        self.rand_node_id_dim = rand_node_id_dim
        self.orf_node_id = orf_node_id
        self.orf_node_id_dim = orf_node_id_dim
        self.lap_node_id = lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.lap_node_id_sign_flip = lap_node_id_sign_flip

        self.type_id = type_id

        if self.rand_node_id:
            self.rand_encoder = nn.Linear(
                2 * rand_node_id_dim, hidden_dim, bias=False)

        if self.lap_node_id:
            self.lap_encoder = nn.Linear(
                2 * lap_node_id_k, hidden_dim, bias=False)
            self.lap_eig_dropout = nn.Dropout2d(
                p=lap_node_id_eig_dropout) if lap_node_id_eig_dropout > 0 else None

        if self.orf_node_id:
            self.orf_encoder = nn.Linear(
                2 * orf_node_id_dim, hidden_dim, bias=False)

        if self.type_id:
            self.order_encoder = nn.Embedding(2, hidden_dim)

        self.num_attention_heads = num_heads

        if self.use_edge_encoding_bias:
            # self.src_dst_bias_encoder = nn.Embedding(2, 1, padding_idx=0)
            self.edge_bias_encoder = nn.Embedding(
                100, num_heads, padding_idx=0)

            self.spatial_pos_bias_encoder = nn.Embedding(
                100, num_heads, padding_idx=0)

            self.graph_node_bias_encoder = nn.Embedding(
                3, num_heads, padding_idx=0)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def get_batch(self, node_feature, edge_index, edge_feature, node_num, edge_num, perturb=None):
        """
        :param node_feature: Tensor([sum(node_num), D])
        :param edge_index: LongTensor([2, sum(edge_num)])
        :param edge_feature: Tensor([sum(edge_num), D])
        :param node_num: list
        :param edge_num: list
        :param perturb: Tensor([B, max(node_num), D])
        :return: padded_index: LongTensor([B, T, 2]), padded_feature: Tensor([B, T, D]), padding_mask: BoolTensor([B, T])
        """
        seq_len = [n + e for n, e in zip(node_num, edge_num)]
        b = len(seq_len)
        d = node_feature.size(-1)
        max_len = max(seq_len)
        max_n = max(node_num)
        if not(self.use_edge_features):
            max_len = max_n
        device = edge_index.device

        token_pos = torch.arange(max_len, device=device)[
            None, :].expand(b, max_len)  # [B, T]

        seq_len = torch.tensor(seq_len, device=device, dtype=torch.long)[
            :, None]  # [B, 1]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[
            :, None]  # [B, 1]
        edge_num = torch.tensor(edge_num, device=device, dtype=torch.long)[
            :, None]  # [B, 1]

        node_index = torch.arange(max_n, device=device, dtype=torch.long)[
            None, :].expand(b, max_n)  # [B, max_n]
        node_index = node_index[None, node_index <
                                node_num].repeat(2, 1)  # [2, sum(node_num)]

        padded_node_mask = torch.less(token_pos, node_num)
        if self.use_edge_features:
            padded_edge_mask = torch.logical_and(
                torch.greater_equal(token_pos, node_num),
                torch.less(token_pos, node_num + edge_num)
            )
        else:
            padded_edge_mask = None

        padded_index = torch.zeros(
            b, max_len, 2, device=device, dtype=torch.long)  # [B, T, 2]
        padded_index[padded_node_mask, :] = node_index.t()
        if self.use_edge_features:
            padded_index[padded_edge_mask, :] = edge_index.t()

        if perturb is not None:
            perturb_mask = padded_node_mask[:, :max_n]  # [B, max_n]
            node_feature = node_feature + \
                perturb[perturb_mask].type(
                    node_feature.dtype)  # [sum(node_num), D]

        padded_feature = torch.zeros(
            b, max_len, d, device=device, dtype=node_feature.dtype)  # [B, T, D]
        padded_feature[padded_node_mask, :] = node_feature
        if self.use_edge_features:
            padded_feature[padded_edge_mask, :] = edge_feature

        padding_mask = torch.greater_equal(token_pos, seq_len)  # [B, T]
        return padded_index, padded_feature, padding_mask, padded_node_mask, padded_edge_mask

    @ staticmethod
    @ torch.no_grad()
    def get_node_mask(node_num, device):
        b = len(node_num)
        max_n = max(node_num)
        node_index = torch.arange(max_n, device=device, dtype=torch.long)[
            None, :].expand(b, max_n)  # [B, max_n]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[
            :, None]  # [B, 1]
        node_mask = torch.less(node_index, node_num)  # [B, max_n]
        return node_mask

    @ staticmethod
    @ torch.no_grad()
    def get_random_sign_flip(eigvec, node_mask):
        b, max_n = node_mask.size()
        d = eigvec.size(1)

        sign_flip = torch.rand(b, d, device=eigvec.device, dtype=eigvec.dtype)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        sign_flip = sign_flip[:, None, :].expand(b, max_n, d)
        sign_flip = sign_flip[node_mask]
        return sign_flip

    def handle_eigvec(self, eigvec, node_mask, sign_flip):
        if sign_flip and self.training:
            sign_flip = self.get_random_sign_flip(eigvec, node_mask)
            eigvec = eigvec * sign_flip
        else:
            pass
        return eigvec

    @ staticmethod
    @ torch.no_grad()
    def get_orf_batched(node_mask, dim, device, dtype):
        b, max_n = node_mask.size(0), node_mask.size(1)
        orf = gaussian_orthogonal_random_matrix_batched(
            b, dim, dim, device=device, dtype=dtype)  # [B, D, D]
        orf = orf[:, None, ...].expand(
            b, max_n, dim, dim)  # [B, max(n_node), D, D]
        orf = orf[node_mask]  # [sum(n_node), D, D]
        return orf

    @ staticmethod
    def get_index_embed(node_id, node_mask, padded_index):
        """
        :param node_id: Tensor([sum(node_num), D])
        :param node_mask: BoolTensor([B, max_n])
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, 2D])
        """
        b, max_n = node_mask.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)

        padded_node_id = torch.zeros(
            b, max_n, d, device=node_id.device, dtype=node_id.dtype)  # [B, max_n, D]
        padded_node_id[node_mask] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(b, max_n, 2, d)
        padded_index = padded_index[..., None].expand(b, max_len, 2, d)
        index_embed = padded_node_id.gather(1, padded_index)  # [B, T, 2, D]
        index_embed = index_embed.view(b, max_len, 2 * d)
        return index_embed

    def get_type_embed(self, padded_index):
        """
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, D])
        """
        order = torch.eq(padded_index[..., 0],
                         padded_index[..., 1]).long()  # [B, T]
        order_embed = self.order_encoder(order)
        return order_embed

    def add_special_tokens(self, padded_feature, padding_mask):
        """
        :param padded_feature: Tensor([B, T, D])
        :param padding_mask: BoolTensor([B, T])
        :return: padded_feature: Tensor([B, 2/3 + T, D]), padding_mask: BoolTensor([B, 2/3 + T])
        """
        b, _, d = padded_feature.size()

        num_special_tokens = 1  # 2
        graph_token_feature = self.graph_token.weight.expand(b, 1, d)  # [1, D]
        # null_token_feature = self.null_token.weight.expand(
        #     b, 1, d)  # [1, D], this is optional
        special_token_feature = graph_token_feature
        # torch.cat(
        #     (graph_token_feature, null_token_feature), dim=1)  # [B, 2, D]
        special_token_mask = torch.zeros(
            b, num_special_tokens, dtype=torch.bool, device=padded_feature.device)

        padded_feature = torch.cat(
            (special_token_feature, padded_feature), dim=1)  # [B, 2 + T, D]
        padding_mask = torch.cat(
            (special_token_mask, padding_mask), dim=1)  # [B, 2 + T]
        return padded_feature, padding_mask

    def get_attn_bias(self, adjacency_matrix):
        device = adjacency_matrix.device
        max_n = adjacency_matrix.shape[1]
        b = adjacency_matrix.shape[0]
        # [B X 1+N X 1+N]
        edge_attn_bias = torch.zeros(
            [b, max_n+1, max_n+1], dtype=torch.long).to(device)

        edge_attn_bias[:, 1:, 1:] = adjacency_matrix
        edge_attn_bias[:, 0, :] = 0
        edge_attn_bias[:, :, 0] = 0
        return edge_attn_bias

    def retrieve_2d(self, x, b, max_n):
        x = x.reshape(b, self.max_nodes, self.max_nodes)
        return x[:, :max_n, :max_n]

    def forward(self, batched_data, perturb=None):
        (
            node_data,
            z_data,
            node_num,
            # lap_eigval,
            # lap_eigvec,
            global_degree,
            node_dist,
            edge_index,
            edge_data,
            edge_num,
            # adjacency_matrix,
            local_degree,
            adjacency_matrix,
            spatial_pos,
            graph_edge_mat
        ) = (
            batched_data.x,
            batched_data.z_data,
            batched_data.node_num,
            # batched_data.lap_eigval,
            # batched_data.lap_eigvec,
            batched_data.global_degree,
            batched_data.dist,
            batched_data.edge_index,
            batched_data.edge_data,
            batched_data.edge_num,
            # batched_data.adjacency_matrix,
            batched_data.local_degree,
            batched_data.adjacency_matrix,
            batched_data.spatial_pos,
            batched_data.graph_edge_mat
        )

        b = node_num.shape[0]
        max_n = node_num.max()
        adjacency_matrix = self.retrieve_2d(adjacency_matrix, b, max_n)
        spatial_pos = self.retrieve_2d(spatial_pos, b, max_n)
        graph_edge_mat = self.retrieve_2d(graph_edge_mat, b, max_n)
        # shortest_path = shortest_path.reshape(b, max_n, max_n)
        if z_data.max() >= 1000 or global_degree.max() >= 200 or local_degree.max() >= 1000 or node_dist.max() >= 100 or z_data.min() < 0 or local_degree.min() < 0 or node_dist.min() < 0:
            print("Issue!!!")

        z_feature = self.z_encoder(z_data)  # [sum(n_node), D]
        node_dist_feature = self.dist_encoder(node_dist)
        global_degree_feature = self.global_degree_encoder(global_degree)
        local_degree_feature = self.local_degree_encoder(local_degree)
        # node_id_feature = self.node_id_encoder(node_ids)

        if self.use_node_features:
            node_feature = torch.cat([self.node_encoder(
                node_data), z_feature, global_degree_feature, local_degree_feature], dim=1) + node_dist_feature # + z_feature  # + node_dist_feature
            # node_feature = self.act(self.node_encoder(node_data))
        else:
            node_feature = torch.cat(
                [z_feature, global_degree_feature, local_degree_feature], dim=1) + node_dist_feature

        if self.use_edge_features:
            edge_feature = self.edge_encoder(edge_data).sum(-2)
        else:
            edge_feature = None

        if self.use_gcn:
            for i in range(len(self.convs)-1):
                node_feature = self.convs[i](
                    node_feature, edge_index, edge_data.squeeze().to(torch.float))
                node_feature = F.relu(node_feature)
                node_feature = F.dropout(
                    node_feature, p=0.5, training=self.training)
            node_feature = self.convs[-1](
                node_feature, edge_index, edge_data.squeeze().to(torch.float))

        # node_feature[node_feature!=0] = 0
        # edge_feature[edge_feature!=0] = 0

        device = node_feature.device
        dtype = node_feature.dtype

        padded_index, padded_feature, padding_mask, _, _ = self.get_batch(
            node_feature, edge_index, edge_feature, node_num, edge_num, perturb
        )
        node_mask = self.get_node_mask(
            node_num, node_feature.device)  # [B, max(n_node)]

        if self.rand_node_id:
            rand_node_id = torch.rand(sum(
                node_num), self.rand_node_id_dim, device=device, dtype=dtype)  # [sum(n_node), D]
            rand_node_id = F.normalize(rand_node_id, p=2, dim=1)
            rand_index_embed = self.get_index_embed(
                rand_node_id, node_mask, padded_index)  # [B, T, 2D]
            padded_feature = padded_feature + \
                self.rand_encoder(rand_index_embed)

        if self.orf_node_id:
            b, max_n = len(node_num), max(node_num)
            orf = gaussian_orthogonal_random_matrix_batched(
                b, max_n, max_n, device=device, dtype=dtype
            )  # [b, max(n_node), max(n_node)]
            orf_node_id = orf[node_mask]  # [sum(n_node), max(n_node)]
            if self.orf_node_id_dim > max_n:
                # [sum(n_node), Do]
                orf_node_id = F.pad(
                    orf_node_id, (0, self.orf_node_id_dim - max_n), value=float('0'))
            else:
                # [sum(n_node), Do]
                orf_node_id = orf_node_id[..., :self.orf_node_id_dim]
            orf_node_id = F.normalize(orf_node_id, p=2, dim=1)
            orf_index_embed = self.get_index_embed(
                orf_node_id, node_mask, padded_index)  # [B, T, 2Do]
            padded_feature = padded_feature + self.orf_encoder(orf_index_embed)

        if self.lap_node_id:
            lap_dim = lap_eigvec.size(-1)
            if self.lap_node_id_k > lap_dim:
                # [sum(n_node), Dl]
                eigvec = F.pad(
                    lap_eigvec, (0, self.lap_node_id_k - lap_dim), value=float('0'))
            else:
                # [sum(n_node), Dl]
                eigvec = lap_eigvec[:, :self.lap_node_id_k]
            if self.lap_eig_dropout is not None:
                eigvec = self.lap_eig_dropout(
                    eigvec[..., None, None]).view(eigvec.size())
            lap_node_id = self.handle_eigvec(
                eigvec, node_mask, self.lap_node_id_sign_flip)
            lap_index_embed = self.get_index_embed(
                lap_node_id, node_mask, padded_index)  # [B, T, 2Dl]
            padded_feature = padded_feature + self.lap_encoder(lap_index_embed)

        if self.type_id:
            padded_feature = padded_feature + self.get_type_embed(padded_index)

        padded_feature, padding_mask = self.add_special_tokens(
            padded_feature, padding_mask)  # [B, 1+T, D], [B, 1+T]

        padded_feature = padded_feature.masked_fill(
            padding_mask[..., None], float('0'))

        # Add bias for graph->src, graph->dst
        if self.use_edge_encoding_bias:

            graph_node_bias = self.graph_node_bias_encoder(self.get_attn_bias(
                graph_edge_mat)).permute(3, 0, 1, 2)
            edge_attn_bias = self.edge_bias_encoder(self.get_attn_bias(
                adjacency_matrix)).permute(3, 0, 1, 2)
            spatial_pos_attn_bias = self.spatial_pos_bias_encoder(self.get_attn_bias(
                spatial_pos)).permute(3, 0, 1, 2)
            attn_bias = (edge_attn_bias + spatial_pos_attn_bias + graph_node_bias).contiguous()
        else:
            attn_bias = None
        # [B, 2+T, D], [B, 2+T], [B, T, 2]
        return padded_feature, padding_mask, padded_index, attn_bias
