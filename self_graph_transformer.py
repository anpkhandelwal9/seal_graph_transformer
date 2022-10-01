"""
Modified from https://github.com/microsoft/Graphormer
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss

#from lr import PolynomialDecayLR

from seal_graph_transformer.modules import init_graphormer_params, TokenGTGraphEncoder

logger = logging.getLogger(__name__)


class CustomData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __inc__(self, key, val, *args, **kwargs):
        return 0

class TokenGTModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, **kwargs):
        super().__init__()
        self.encoder = TokenGTEncoder(encoder_embed_dim = embedding_dim, **kwargs)
        self.linear = nn.Linear(embedding_dim, num_classes)
    
    def max_nodes(self):
        return self.encoder.max_nodes

    def forward(self, batched_data, **kwargs):
        out, attn  = self.encoder(batched_data, **kwargs)
        return self.linear(out)


class TokenGTEncoder(nn.Module):
    def __init__(self, 
        num_node_features = 1,
        num_edge_features = 1,
        num_classes = 2, 
        share_encoder_input_output_embed = False,
        encoder_embed_dim = 1024,
        prenorm=True,
        postnorm=False,
        max_nodes = 128,
        encoder_layers = 6,
        encoder_attention_heads = 8,
        return_attention = True):
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
            return_attention=return_attention
        )

        self.activation_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(encoder_embed_dim)
        self.share_input_output_embed = share_encoder_input_output_embed
        self.embed_out = None
        

        # Remove head is set to true during fine-tuning
        
        self.masked_lm_pooler = nn.Linear(encoder_embed_dim, encoder_embed_dim)
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
            return x[:, 0, :]

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenGTModel(embedding_dim=768, num_classes = 2, type_id=True, return_attention=True)
    model.to(device)
    batched_data = {}
    # batched_data["node_data"] =  torch.tensor([[101], [102],[201], [202], [203]], dtype=torch.float)
    # batched_data["node_num"] = [2,3]
    # batched_data["lap_eigvec"] = None
    # batched_data["edge_index"] = torch.tensor([[0,0,1], [1,1,2]])
    # batched_data["edge_data"] = torch.tensor([[1],[1], [1]], dtype=torch.float)
    # batched_data["edge_num"] = [1,2]
    # out = model(batched_data)
    data1 = CustomData(node_data=torch.Tensor([[101],[102]]).to(device),z_data=torch.Tensor([1,1]).to(torch.long).to(device), node_num=torch.Tensor([2]).to(torch.long).to(device), edge_index=torch.Tensor([[0], [1]]).to(torch.long).to(device), edge_data=torch.Tensor([[1]]).to(device), edge_num=torch.Tensor([1]).to(torch.long).to(device), y=torch.Tensor([0]).to(torch.long).to(device), lap_eigvec=torch.Tensor([0.1]).to(device))
    data2 = CustomData(node_data=torch.Tensor([[201],[202], [203]]).to(device), z_data=torch.Tensor([1,1,2]).to(torch.long).to(device), node_num=torch.Tensor([3]).to(torch.long).to(device), edge_index=torch.Tensor([[0,1], [1,2]]).to(torch.long).to(device), edge_data=torch.Tensor([[1],[2]]).to(device), edge_num=torch.Tensor([2]).to(torch.long).to(device), y=torch.Tensor([1]).to(torch.long).to(device), lap_eigvec=torch.Tensor([0.1]).to(device))
    dataset = [data1, data2]
    list_data = []
    # for _ in range(10000):
    #     list_data += dataset
    loader = DataLoader(dataset, batch_size=2) 
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)
    # lr_scheduler = PolynomialDecayLR(
    #         optimizer,
    #         warmup_updates=args.warmup_updates,
    #         tot_updates=args.tot_updates,
    #         lr=args.peak_lr,
    #         end_lr=args.end_lr,
    #         power=1.0)

    pbar = tqdm(loader, ncols=70)
    for data in pbar:
        optimizer.zero_grad()
        out = model(data)
        loss = BCEWithLogitsLoss()(out[:,1].view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
    