"""Explicit semantic scores used by the paper and implementation."""

from __future__ import annotations

import torch


def route_score(attention: torch.Tensor) -> torch.Tensor:
    """Return RouteScore(a) = ||a||_2 over the attention feature dimension."""
    if attention.ndim < 2 or attention.shape[-1] == 0:
        raise ValueError("attention must include a non-empty feature dimension.")
    return attention.norm(dim=-1)


def phi_a(gamma_change: torch.Tensor, uncertainty_drop: torch.Tensor) -> torch.Tensor:
    """Combine confidence gain and uncertainty reduction as a soft conjunction.

    The paper definition is phi_A(x, y) = sqrt([x]_+ [y]_+), where
    [v]_+ = max(v, 0).  The score is positive only when both signals improve.
    """
    if gamma_change.shape != uncertainty_drop.shape:
        raise ValueError("gamma_change and uncertainty_drop must have the same shape.")
    return (
        gamma_change.clamp_min(0.0) * uncertainty_drop.clamp_min(0.0)
    ).sqrt()


def prediction_refinement_score(
    predictions: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Compute Eq. (26): positive reduction in KL distance to final prediction."""
    if predictions.ndim != 3 or predictions.shape[1] < 2:
        raise ValueError("predictions must have shape [queries, layers + 1, vocabulary].")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    probabilities = predictions.clamp_min(epsilon)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    final = probabilities[:, -1:].detach()
    kl = (probabilities * (probabilities.log() - final.log())).sum(dim=-1)
    return (kl[:, :-1] - kl[:, 1:]).clamp_min(0.0)
