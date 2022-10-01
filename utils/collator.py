import torch
from torch.nn import functional as F
from self_graph_transformer import CustomData


@torch.no_grad()
def collator(
        items,
        max_node=512,
        max_edge=2048,
        multi_hop_max_dist=20,
        spatial_pos_max=20
):
    items = [item for item in items if
             item is not None and item.node_data.size(0) <= max_node and item.edge_data.size(0) <= max_edge]

    (
        edge_index,
        edge_data,
        node_data,
        z_data,
        in_degree,
        node_dist,
        # lap_eigvec,
        # lap_eigval,
        ys
    ) = zip(*[
        (
            item.edge_index,
            item.edge_data,
            item.node_data,
            item.z_data,
            item.in_degree,
            item.node_dist,
            # item.lap_eigvec,
            # item.lap_eigval,
            item.y
        )
        for item in items
    ])

    node_num = [i.size(0) for i in node_data]
    edge_num = [i.size(0) for i in edge_data]
    max_n = max(node_num)

    y = torch.cat(ys)  # [B,]
    edge_index = torch.cat(edge_index, dim=1)  # [2, sum(edge_num)]
    # [sum(edge_num), De], +1 for nn.Embedding with pad_index=999
    edge_data = torch.cat(edge_data)
    # [sum(node_num), Dn], +1 for nn.Embedding with pad_index=0
    node_data = torch.cat(node_data)
    # [sum(node_num),], +1 for nn.Embedding with pad_index=0
    in_degree = torch.cat(in_degree) + 1
    node_dist = torch.cat(node_dist) + 1

    # [sum(node_num),], +1 for nn.Embedding with pad_index=0
    z_data = torch.cat(z_data) + 1

    # [sum(node_num), Dl] = [sum(node_num), max_n]
    # lap_eigvec = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigvec])
    # lap_eigval = torch.cat([F.pad(i, (0, max_n - i.size(1)), value=float('0')) for i in lap_eigval])

    result = CustomData(
        edge_index=edge_index,
        edge_data=edge_data,
        node_data=node_data,
        # lap_eigvec=lap_eigvec,
        # lap_eigval=lap_eigval,
        z_data=z_data,
        y=y,
        node_num=node_num,
        edge_num=edge_num,
        num_graphs=y.shape[0],
        in_degree=in_degree,
        node_dist=node_dist
    )

    return result
