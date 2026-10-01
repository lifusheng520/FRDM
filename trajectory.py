"""Metrics and serializable fuzzy reasoning trajectories."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.fuzzy_dynamics.config import REASONING_MODES
from .metrics import reconstruction_metrics


def build_trajectory_records(
    memberships: torch.Tensor,
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    uncertainty: torch.Tensor,
    metadata: list[dict[str, Any]],
    offset: int = 0,
    semantic_scores: torch.Tensor | None = None,
    mixing_weights: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    records = []
    for batch_index in range(memberships.shape[0]):
        item_metadata = metadata[offset + batch_index] if metadata else {"index": offset + batch_index}
        layers = []
        for layer_index in range(memberships.shape[1]):
            mu = memberships[batch_index, layer_index]
            alpha = (
                mixing_weights[batch_index, layer_index]
                if mixing_weights is not None
                else mu / mu.sum().clamp_min(1e-8)
            )
            dominant_index = int(mu.argmax())
            layer = {
                    "layer": layer_index,
                    "membership": {
                        name: float(mu[mode_index])
                        for mode_index, name in enumerate(REASONING_MODES)
                    },
                    "mixing_weight": {
                        name: float(alpha[mode_index])
                        for mode_index, name in enumerate(REASONING_MODES)
                    },
                    "dominant_mode": REASONING_MODES[dominant_index],
                    "uncertainty": float(uncertainty[batch_index, layer_index, 0]),
                    "predicted_delta_norm": float(
                        predicted_delta[batch_index, layer_index].norm()
                    ),
                    "target_delta_norm": float(target_delta[batch_index, layer_index].norm()),
                }
            if semantic_scores is not None:
                layer["semantic_score"] = {
                    name: float(semantic_scores[batch_index, layer_index, mode_index])
                    for mode_index, name in enumerate(REASONING_MODES)
                }
            layers.append(layer)
        records.append({**item_metadata, "trajectory": layers})
    return records


def aggregate_membership_analysis(
    memberships: torch.Tensor,
    local_deltas: torch.Tensor,
    mixing_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Aggregate layer usage, hard transitions, and weighted mode contributions."""
    alpha = (
        mixing_weights
        if mixing_weights is not None
        else memberships / memberships.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    mean_by_layer = alpha.mean(dim=0)
    usage = alpha.mean(dim=(0, 1))
    raw_usage = memberships.mean(dim=(0, 1))
    dominant = alpha.argmax(dim=-1)
    transition_counts = torch.zeros(5, 5, dtype=torch.float64)
    for left, right in zip(dominant[:, :-1].reshape(-1), dominant[:, 1:].reshape(-1)):
        transition_counts[int(left), int(right)] += 1
    row_totals = transition_counts.sum(dim=1, keepdim=True)
    transition_probabilities = transition_counts / row_totals.clamp_min(1.0)
    weighted_contribution = (
        memberships * local_deltas.norm(dim=-1)
    ).mean(dim=(0, 1))
    membership_entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(-1)
    normalized_deltas = F.normalize(local_deltas, dim=-1, eps=1e-8)
    pairwise_cosines = torch.stack(
        [
            (normalized_deltas[..., left, :] * normalized_deltas[..., right, :]).sum(-1)
            for left in range(local_deltas.shape[-2])
            for right in range(left + 1, local_deltas.shape[-2])
        ],
        dim=-1,
    )
    if memberships.shape[1] > 1:
        membership_left = alpha[:, :-1]
        membership_right = alpha[:, 1:]
        membership_step_total_variation = (
            0.5 * (membership_right - membership_left).abs().sum(dim=-1)
        ).mean()
        membership_step_cosine = F.cosine_similarity(
            membership_left, membership_right, dim=-1, eps=1e-8
        ).mean()
        dominant_switch_rate = (
            dominant[:, 1:] != dominant[:, :-1]
        ).float().mean()
    else:
        membership_step_total_variation = torch.zeros((), device=memberships.device)
        membership_step_cosine = torch.ones((), device=memberships.device)
        dominant_switch_rate = torch.zeros((), device=memberships.device)
    result: dict[str, Any] = {
        "mode_usage": {
            name: float(usage[index]) for index, name in enumerate(REASONING_MODES)
        },
        "raw_membership_mean": {
            name: float(raw_usage[index])
            for index, name in enumerate(REASONING_MODES)
        },
        "mean_total_membership_mass": float(memberships.sum(dim=-1).mean()),
        "mean_membership_sparsity_ratio": float(
            (
                memberships.sum(dim=-1).square()
                / memberships.square().sum(dim=-1).clamp_min(1e-8)
            ).mean()
        ),
        "weighted_mode_contribution": {
            name: float(weighted_contribution[index])
            for index, name in enumerate(REASONING_MODES)
        },
        "mean_membership_by_layer": [
            {name: float(row[index]) for index, name in enumerate(REASONING_MODES)}
            for row in mean_by_layer
        ],
        "dominant_transition_counts": transition_counts.tolist(),
        "dominant_transition_probabilities": transition_probabilities.tolist(),
        "mode_order": list(REASONING_MODES),
        "mean_membership_entropy": float(membership_entropy.mean()),
        "effective_number_of_modes": float(membership_entropy.mean().exp()),
        "mean_membership_step_total_variation": float(
            membership_step_total_variation
        ),
        "mean_membership_step_cosine_similarity": float(membership_step_cosine),
        "dominant_mode_switch_rate": float(dominant_switch_rate),
        "local_dynamics_mean_pairwise_cosine": float(pairwise_cosines.mean()),
        "local_dynamics_mean_pairwise_absolute_cosine": float(
            pairwise_cosines.abs().mean()
        ),
        "local_dynamics_mean_pairwise_cosine_squared": float(
            pairwise_cosines.square().mean()
        ),
    }
    return result
