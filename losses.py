"""The FRDM learning objective in equation (19)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossConfig


def diversity_loss(local_dynamics: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean ordered-pair squared cosine, equivalent to L_div."""
    normalized = F.normalize(local_dynamics, dim=-1, eps=eps)
    gram = torch.matmul(normalized, normalized.transpose(-1, -2)).square()
    modes = local_dynamics.shape[-2]
    off_diagonal = ~torch.eye(modes, dtype=torch.bool, device=gram.device)
    return gram[..., off_diagonal].mean()


def membership_sparsity_loss(memberships: torch.Tensor) -> torch.Tensor:
    """Mean L1 norm of raw fuzzy memberships, equivalent to L_sp."""
    return memberships.abs().sum(dim=-1).mean()


def fuzzy_dynamics_loss(
    outputs: dict[str, torch.Tensor], config: LossConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted = outputs["predicted_next_hidden"]
    target = outputs["target_next_hidden"]
    memberships = outputs["memberships"]
    local_dynamics = outputs["local_dynamics"]

    dynamics = (predicted - target).square().sum(dim=-1).mean()
    diversity = diversity_loss(local_dynamics)
    sparsity = membership_sparsity_loss(memberships)
    total = (
        config.dynamics_weight * dynamics
        + config.diversity_weight * diversity
        + config.sparsity_weight * sparsity
    )
    metrics = {
        "loss": total.detach(),
        "dynamics": dynamics.detach(),
        "diversity": diversity.detach(),
        "sparsity": sparsity.detach(),
        # A diagnostic only; it is not an objective term.
        "membership_mass": memberships.sum(dim=-1).mean().detach(),
    }
    return total, metrics
