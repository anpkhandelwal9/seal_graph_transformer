import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
            self,
            embedding_dim,
            ffn_embedding_dim,
            q_noise,
            qn_block_size,
            activation_fn,
            activation_dropout,
            dropout,
            module_name
    ):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout
        )
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.dropout_module = nn.Dropout(
            dropout
        )

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
