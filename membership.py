"""Gaussian fuzzy memberships from FRDM equation (16)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class GaussianMembership(nn.Module):
    """Compare every candidate local dynamic with the low-rank reference."""

    def __init__(
        self,
        num_modes: int = 5,
        sigma_min: float = 1e-4,
        epsilon: float = 1e-8,
        initial_center: float = 0.0,
        initial_spread: float = 1.0,
    ) -> None:
        super().__init__()
        if num_modes <= 0:
            raise ValueError("num_modes must be positive.")
        if sigma_min <= 0 or epsilon <= 0 or initial_spread <= sigma_min:
            raise ValueError(
                "sigma_min and epsilon must be positive and initial_spread must "
                "exceed sigma_min."
            )
        self.num_modes = num_modes
        self.sigma_min = sigma_min
        self.epsilon = epsilon
        self.centers = nn.Parameter(torch.full((num_modes,), float(initial_center)))
        inverse_softplus = math.log(math.expm1(initial_spread - sigma_min))
        self.raw_spreads = nn.Parameter(
            torch.full((num_modes,), inverse_softplus)
        )

    @property
    def spreads(self) -> torch.Tensor:
        return F.softplus(self.raw_spreads) + self.sigma_min

    def forward(
        self, local_dynamics: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (*reference.shape[:-1], self.num_modes, reference.shape[-1])
        if local_dynamics.shape != expected:
            raise ValueError(
                f"Expected local dynamics shape {expected}, got "
                f"{tuple(local_dynamics.shape)}."
            )
        distances = torch.linalg.vector_norm(
            local_dynamics - reference.unsqueeze(-2), dim=-1
        )
        standardized = (distances - self.centers) / self.spreads
        memberships = torch.exp(-0.5 * standardized.square()).clamp_min(
            torch.finfo(local_dynamics.dtype).tiny
        )
        mixing_weights = memberships / (
            memberships.sum(dim=-1, keepdim=True) + self.epsilon
        )
        return memberships, mixing_weights, distances
