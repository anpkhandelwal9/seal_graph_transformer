"""
Modified from https://github.com/microsoft/Graphormer
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.nn import BCEWithLogitsLoss

#from lr import PolynomialDecayLR

from modules import TokenGTGraphEncoder

logger = logging.getLogger(__name__)

class CustomData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __inc__(self, *args, **kwargs):
        return 0

class TokenGTModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, **kwargs):
        super().__init__()
        self.encoder = TokenGTEncoder(encoder_embed_dim = embedding_dim, **kwargs)
        self.linear = nn.Linear(embedding_dim, num_classes)
    
    def max_nodes(self):
        return self.encoder.max_nodes

    def forward(self, batched_data, **kwargs):
        out,_  = self.encoder(batched_data, **kwargs)
        return self.linear(out)


class TokenGTEncoder(nn.Module):
    def __init__(self, 
        num_node_features = 1,
        num_edge_features = 2,
        num_classes = 2, 
        share_encoder_input_output_embed = False,
        encoder_embed_dim = 1024,
        prenorm=True,
        postnorm=False,
        max_nodes = 128,
        encoder_layers = 1,
        encoder_attention_heads = 1,
        return_attention = False,
        performer=False,
        performer_finetune=False,
        performer_nb_features=None,
        performer_feature_redraw_interval=1000,
        performer_generalized_attention=True,
        ):
        super().__init__()
        assert not (prenorm and postnorm)
        assert prenorm or postnorm
        self.max_nodes = max_nodes
        self.encoder_layers = encoder_layers
        self.num_attention_heads = encoder_attention_heads
        self.return_attention = return_attention

        if prenorm:
            layernorm_style = "prenorm"
        elif postnorm:
            layernorm_style = "postnorm"
        else:
            raise NotImplementedError

        self.graph_encoder = TokenGTGraphEncoder(  # Only encodes feature, returns padded_index inside attn_dict
            # <
            num_node_features,
            num_edge_features,
            embedding_dim = encoder_embed_dim,
            num_encoder_layers = self.encoder_layers,
            num_attention_heads = self.num_attention_heads,
            ffn_embedding_dim = encoder_embed_dim,
            return_attention=return_attention,
            performer=performer,
            performer_finetune=performer_finetune,
            performer_nb_features=performer_nb_features,
            performer_feature_redraw_interval=performer_feature_redraw_interval,
            performer_generalized_attention=performer_generalized_attention,
            max_nodes=max_nodes
        )

        self.activation_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(encoder_embed_dim)
        self.share_input_output_embed = share_encoder_input_output_embed
        self.embed_out = None
        

        # Remove head is set to true during fine-tuning
        
        self.lm_head_transform_weight = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        self.load_softmax = False
        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(encoder_embed_dim, num_classes, bias=False)
            else:
                raise NotImplementedError

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep, attn_dict = self.graph_encoder(batched_data, perturb=perturb)

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
                self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        if self.return_attention:
            return x[:, 0, :], attn_dict
        else:
            return x[:, 0, :], None

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes


    