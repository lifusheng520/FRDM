"""The five semantically grounded local reasoning dynamics (9)--(13)."""

from __future__ import annotations

import torch
from torch import nn


class DynamicsMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
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


class LocalReasoningDynamics(nn.Module):
    """Operation-specific F_K, F_I, F_C, F_P, and F_H mappings."""

    def __init__(
        self,
        z_dim: int,
        concept_dim: int,
        prediction_dim: int,
        operation_dim: int,
        dynamics_dim: int,
        hidden_dim: int,
        dropout: float,
        homogeneous: bool = False,
    ) -> None:
        super().__init__()
        self.homogeneous = homogeneous
        if homogeneous:
            common_dim = (
                z_dim + concept_dim + prediction_dim + 1 + 2 * operation_dim
            )
            self.dynamics = nn.ModuleList(
                [
                    DynamicsMLP(common_dim, hidden_dim, dynamics_dim, dropout)
                    for _ in range(5)
                ]
            )
        else:
            self.knowledge_enrichment = DynamicsMLP(
                z_dim + operation_dim, hidden_dim, dynamics_dim, dropout
            )
            self.information_routing = DynamicsMLP(
                z_dim + operation_dim, hidden_dim, dynamics_dim, dropout
            )
            self.concept_composition = DynamicsMLP(
                z_dim + concept_dim + 2 * operation_dim,
                hidden_dim,
                dynamics_dim,
                dropout,
            )
            self.prediction_refinement = DynamicsMLP(
                z_dim + prediction_dim + 1, hidden_dim, dynamics_dim, dropout
            )
            self.hop_transition = DynamicsMLP(
                z_dim + concept_dim + operation_dim,
                hidden_dim,
                dynamics_dim,
                dropout,
            )

    def forward(
        self,
        z: torch.Tensor,
        concept: torch.Tensor,
        prediction: torch.Tensor,
        uncertainty: torch.Tensor,
        attention: torch.Tensor,
        mlp: torch.Tensor,
    ) -> torch.Tensor:
        if self.homogeneous:
            common = torch.cat(
                (z, concept, prediction, uncertainty, attention, mlp), dim=-1
            )
            return torch.stack([dynamic(common) for dynamic in self.dynamics], dim=-2)
        return torch.stack(
            (
                self.knowledge_enrichment(torch.cat((z, mlp), dim=-1)),
                self.information_routing(torch.cat((z, attention), dim=-1)),
                self.concept_composition(
                    torch.cat((z, concept, attention, mlp), dim=-1)
                ),
                self.prediction_refinement(
                    torch.cat((z, prediction, uncertainty), dim=-1)
                ),
                self.hop_transition(torch.cat((z, concept, attention), dim=-1)),
            ),
            dim=-2,
        )


class GlobalReasoningDynamics(nn.Module):
    """Parameter-matched single-dynamics ablation."""

    def __init__(
        self,
        state_dim: int,
        operation_dim: int,
        dynamics_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = DynamicsMLP(
            state_dim + 2 * operation_dim,
            5 * hidden_dim,
            dynamics_dim,
            dropout,
        )

    def forward(
        self, state: torch.Tensor, attention: torch.Tensor, mlp: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((state, attention, mlp), dim=-1))
