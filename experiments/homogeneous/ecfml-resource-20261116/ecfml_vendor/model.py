from __future__ import annotations

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "softsign":
        return nn.Softsign()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError("activation must be one of: softsign, relu, tanh, softplus.")


class MobileViTTabular(nn.Module):
    """MobileViT-inspired backbone adapted to 61-dimensional tabular metrics."""

    def __init__(
        self,
        input_dim: int = 61,
        token_length: int = 18,
        embedding_dim: int = 64,
        conv_channels: int = 64,
        conv_kernel_size: int = 3,
        transformer_heads: int = 4,
        transformer_layers: int = 1,
        transformer_ff_dim: int = 256,
        dropout: float = 0.0,
        classifier_hidden_dim: int = 62,
        num_classes: int = 2,
        activation: str = "softsign",
    ) -> None:
        super().__init__()
        if conv_channels % transformer_heads != 0:
            raise ValueError("conv_channels must be divisible by transformer_heads.")

        self.input_dim = input_dim
        self.token_length = token_length
        self.embedding_dim = embedding_dim
        self.conv_channels = conv_channels

        self.projection = nn.Linear(input_dim, token_length * embedding_dim)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(conv_channels, track_running_stats=False)
        self.activation = make_activation(activation)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels * 2, classifier_hidden_dim),
            make_activation(activation),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(x).view(x.shape[0], self.token_length, self.embedding_dim)
        local = tokens.transpose(1, 2)
        local = self.activation(self.batch_norm(self.conv(local))).transpose(1, 2)
        global_features = self.transformer(local)
        fused = torch.cat([local, global_features], dim=-1).transpose(1, 2)
        pooled = self.pool(fused).squeeze(-1)
        return self.classifier(pooled)


def build_model(config: dict) -> MobileViTTabular:
    return MobileViTTabular(**config["model"])
