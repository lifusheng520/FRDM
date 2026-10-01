"""Reasoning-state descriptor and transition-indicator MLPs."""

from __future__ import annotations

import torch
from torch import nn


class DescriptorMLP(nn.Module):
    """A one-hidden-layer MLP used by equations (4)--(6) and (8)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ReasoningProjectors(nn.Module):
    """Map raw cached observations to z, c, b, a, and m."""

    def __init__(
        self,
        hidden_size: int,
        prediction_input_dim: int,
        query_input_dim: int,
        z_dim: int,
        concept_dim: int,
        prediction_dim: int,
        operation_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent = DescriptorMLP(hidden_size, z_dim, hidden_dim, dropout)
        self.concept = DescriptorMLP(
            hidden_size + query_input_dim, concept_dim, hidden_dim, dropout
        )
        self.prediction = DescriptorMLP(
            prediction_input_dim, prediction_dim, hidden_dim, dropout
        )
        self.attention = DescriptorMLP(
            hidden_size, operation_dim, hidden_dim, dropout
        )
        self.mlp = DescriptorMLP(hidden_size, operation_dim, hidden_dim, dropout)

    def states(
        self,
        hidden: torch.Tensor,
        prediction: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_embedding.ndim != 2 or query_embedding.shape[0] != hidden.shape[0]:
            raise ValueError("query_embedding must have shape [batch, query_input_dim].")
        expanded_query = query_embedding.unsqueeze(1).expand(
            -1, hidden.shape[1], -1
        )
        z = self.latent(hidden)
        concept = self.concept(torch.cat((hidden, expanded_query), dim=-1))
        prediction = prediction.clamp_min(torch.finfo(prediction.dtype).tiny)
        prediction = prediction / prediction.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(prediction.dtype).tiny
        )
        log_prediction = prediction.log()
        compact_prediction = self.prediction(log_prediction)
        uncertainty = -(prediction * log_prediction).sum(dim=-1, keepdim=True)
        state = torch.cat((z, concept, compact_prediction, uncertainty), dim=-1)
        return state, z, concept, compact_prediction, uncertainty

    def operations(
        self, attention: torch.Tensor, mlp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attention(attention), self.mlp(mlp)
