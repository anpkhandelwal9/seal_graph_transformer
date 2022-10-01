import copy
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data.feature_store import (
    FeatureStore,
    FeatureTensorType,
    TensorAttr,
    _field_status,
)
from torch_geometric.data.graph_store import (
    EDGE_LAYOUT_TO_ATTR_NAME,
    EdgeAttr,
    EdgeLayout,
    GraphStore,
    adj_type_to_edge_tensor_type,
    edge_tensor_type_to_adj_type,
)
from torch_geometric.data.storage import (
    BaseStorage,
    EdgeStorage,
    GlobalStorage,
    NodeStorage,
)
from torch_geometric.deprecation import deprecated
from torch_geometric.typing import (
    EdgeTensorType,
    EdgeType,
    FeatureTensorType,
    NodeType,
    OptTensor,
)
from torch_geometric.utils import subgraph
from torch_geometric.data import Data


class TokenData(Data):
    def __init__(self, z=None, y=None):
        self.z=z
        self.y=y

    def stores_as(self, data: 'TokenData'):
        return self

    r"""
    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        input_ids (Tensor, optional): Token for transformer. (default: :obj:'None')
        attention_mask (Tensor, optional): Mask for transformer (default: :obj:'None')
        **kwargs (optional): Additional attributes.

    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, input_ids: OptTensor = None, attention_mask: OptTensor = None, **kwargs):

        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

        if input_ids is not None:
            self.input_ids = input_ids
        if attention_mask is not None:
            self.attention_mask = attention_mask


