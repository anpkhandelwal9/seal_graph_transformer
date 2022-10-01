
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..utils.tokenizer import GraphTokenizer

class GraphTransformer(nn.module):
    def __init__(
            self,
            num_atoms: int,
            num_in_degree: int,
            num_out_degree: int,
            num_edges: int,
            num_spatial: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,

            rand_node_id: bool = False,
            rand_node_id_dim: int = 64,
            orf_node_id: bool = False,
            orf_node_id_dim: int = 64,
            lap_node_id: bool = False,
            lap_node_id_k: int = 8,
            lap_node_id_sign_flip: bool = False,
            lap_node_id_eig_dropout: float = 0.0,
            type_id: bool = False,

            stochastic_depth: bool = False,

            performer: bool = False,
            performer_finetune: bool = False,
            performer_nb_features: int = None,
            performer_feature_redraw_interval: int = 1000,
            performer_generalized_attention: bool = False,
            performer_auto_check_redraw: bool = True,

            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            encoder_normalize_before: bool = False,
            layernorm_style: str = "postnorm",
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,

            return_attention: bool = False

    ) -> None:

        super().__init__()