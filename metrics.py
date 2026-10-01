"""Quantitative fidelity, semantic-alignment, and rollout metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from src.fuzzy_dynamics.config import REASONING_MODES
from src.fuzzy_dynamics.semantic_scores import prediction_refinement_score, route_score


STATE_COMPONENTS = ("z", "concept", "prediction", "uncertainty")


def _macro_defined(values: list[float | None]) -> float | None:
    return sum(values) / len(values) if values and all(value is not None for value in values) else None


def _validate_pair(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must have the same shape, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if predicted.ndim < 2:
        raise ValueError("Metric tensors must include samples and a feature dimension.")
    if predicted.shape[-1] == 0:
        raise ValueError("The feature dimension cannot be empty.")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("Regression metrics require finite predicted and target values.")


def _r2_score(predicted: torch.Tensor, target: torch.Tensor) -> float | None:
    """Variance-weighted multi-output R², or ``None`` for a constant target."""
    _validate_pair(predicted, target)
    predicted64 = predicted.to(torch.float64)
    target64 = target.to(torch.float64)
    reduce_dims = tuple(range(target64.ndim - 1))
    target_mean = target64.mean(dim=reduce_dims, keepdim=True)
    total_sum_squares = (target64 - target_mean).square().sum()
    if float(total_sum_squares) == 0.0:
        return None
    residual_sum_squares = (predicted64 - target64).square().sum()
    return float(1.0 - residual_sum_squares / total_sum_squares)


def _regression_summary(
    predicted: torch.Tensor, target: torch.Tensor
) -> dict[str, Any]:
    _validate_pair(predicted, target)
    residual = predicted - target
    r2 = _r2_score(predicted, target)
    reduce_dims = tuple(range(target.ndim - 1))
    centered_target = target.to(torch.float64) - target.to(torch.float64).mean(
        dim=reduce_dims, keepdim=True
    )
    mean_squared_error = residual.to(torch.float64).square().mean()
    target_variance = centered_target.square().mean()
    target_scale = target_variance.sqrt()
    nrmse = (
        float(mean_squared_error.sqrt() / target_scale)
        if float(target_scale) > 0.0
        else None
    )
    return {
        "mse": float(residual.square().mean()),
        "rmse": float(mean_squared_error.sqrt()),
        "nrmse": nrmse,
        "nrmse_defined": nrmse is not None,
        "nrmse_reason": None if nrmse is not None else "constant_target",
        "nrmse_normalization": "target_std_after_per_feature_centering",
        "mae": float(residual.abs().mean()),
        "r2": r2,
        "r2_defined": r2 is not None,
        "r2_reason": None if r2 is not None else "constant_target",
        "target_sum_squares": float(centered_target.square().sum()),
        "target_variance": float(target_variance),
        "cosine_similarity": float(F.cosine_similarity(predicted, target, dim=-1).mean()),
    }


def _component_slices(
    dimensions: Mapping[str, int], state_dim: int
) -> dict[str, slice]:
    missing = [name for name in STATE_COMPONENTS if name not in dimensions]
    if missing:
        raise ValueError(f"Missing state-component dimensions: {', '.join(missing)}")
    slices: dict[str, slice] = {}
    start = 0
    for name in STATE_COMPONENTS:
        size = int(dimensions[name])
        if size <= 0:
            raise ValueError(f"State-component dimension '{name}' must be positive.")
        slices[name] = slice(start, start + size)
        start += size
    if start != state_dim:
        raise ValueError(
            f"State-component dimensions sum to {start}, but the state dimension is {state_dim}."
        )
    return slices


def reconstruction_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    component_dimensions: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate one-step state-transition reconstruction.

    R² centers each state coordinate independently over queries and layers.  If
    component dimensions are supplied, the function additionally reports the
    four state-block scores and their unweighted macro average.
    """
    metrics: dict[str, Any] = _regression_summary(predicted, target)
    metrics["per_layer"] = []
    for layer in range(predicted.shape[1]):
        layer_summary = _regression_summary(predicted[:, layer], target[:, layer])
        metrics["per_layer"].append({"layer": layer, **layer_summary})
    layer_r2 = [entry["r2"] for entry in metrics["per_layer"]]
    metrics["layer_macro_r2"] = _macro_defined(layer_r2)
    requested_bands = {
        "early_layers_0_8_macro_r2": (0, 9),
        "middle_layers_9_17_macro_r2": (9, 18),
        "late_layers_18_29_macro_r2": (18, 30),
        "final_layers_30_31_macro_r2": (30, 32),
    }
    for name, (start, stop) in requested_bands.items():
        selected = layer_r2[start : min(stop, len(layer_r2))]
        metrics[name] = _macro_defined(selected) if start < len(layer_r2) else None

    if component_dimensions is not None:
        slices = _component_slices(component_dimensions, predicted.shape[-1])
        components = {
            name: _regression_summary(predicted[..., section], target[..., section])
            for name, section in slices.items()
        }
        metrics["components"] = components
        component_r2 = [value["r2"] for value in components.values() if value["r2"] is not None]
        metrics["macro_r2"] = (
            sum(component_r2) / len(component_r2)
            if len(component_r2) == len(components)
            else None
        )
        metrics["macro_r2_available_components"] = (
            sum(component_r2) / len(component_r2) if component_r2 else None
        )
        metrics["macro_r2_defined_components"] = len(component_r2)
    return metrics


def r2_at_horizon(
    predicted_endpoints: torch.Tensor, target_endpoints: torch.Tensor
) -> float | None:
    """Compute Appendix C.1 R2@k over all valid start-layer pairs."""
    _validate_pair(predicted_endpoints, target_endpoints)
    if predicted_endpoints.ndim != 3:
        raise ValueError("Endpoint tensors must have shape [queries, starts, hidden_dim].")
    target64 = target_endpoints.to(torch.float64)
    predicted64 = predicted_endpoints.to(torch.float64)
    target_mean = target64.mean(dim=0, keepdim=True)
    denominator = (target64 - target_mean).square().sum()
    if float(denominator) == 0.0:
        return None
    return float((1.0 - (predicted64 - target64).square().sum() / denominator).detach())


def transition_cosine_similarity(
    predicted_delta: torch.Tensor, target_delta: torch.Tensor, epsilon: float = 1e-8
) -> float:
    """Compute Appendix C.1.2 Delta CosSim for predicted transitions."""
    _validate_pair(predicted_delta, target_delta)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    numerator = (target_delta * predicted_delta).sum(dim=-1)
    denominator = target_delta.norm(dim=-1) * predicted_delta.norm(dim=-1) + epsilon
    return float((numerator / denominator).mean())


def rollout_metrics(
    predicted_states: torch.Tensor,
    target_states: torch.Tensor,
    component_dimensions: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate a multi-layer rollout, excluding the supplied initial state."""
    _validate_pair(predicted_states, target_states)
    if predicted_states.ndim != 3:
        raise ValueError("Rollout states must have shape [queries, layers + 1, state_dim].")
    if predicted_states.shape[1] < 2:
        raise ValueError("A rollout must contain an initial state and at least one prediction.")
    if not torch.allclose(predicted_states[:, 0], target_states[:, 0]):
        raise ValueError("Rollout evaluation requires predicted s_0 to equal the observed s_0.")

    predicted = predicted_states[:, 1:]
    target = target_states[:, 1:]
    squared_l2 = (predicted - target).square().sum(dim=-1)
    summary: dict[str, Any] = {
        "kind": "conditional_on_observed_attention_and_mlp",
        "initial_state_mse": 0.0,
        "rollout_error": float(squared_l2.mean()),
        "mse": float((predicted - target).square().mean()),
        "rmse": float((predicted - target).square().mean().sqrt()),
        "final_state_squared_l2": float(squared_l2[:, -1].mean()),
        "final_state_mse": float((predicted[:, -1] - target[:, -1]).square().mean()),
        "per_horizon": [],
    }
    for horizon in range(predicted.shape[1]):
        horizon_summary = _regression_summary(predicted[:, horizon], target[:, horizon])
        summary["per_horizon"].append(
            {
                "horizon": horizon + 1,
                "squared_l2": float(squared_l2[:, horizon].mean()),
                **horizon_summary,
            }
        )
    requested_horizons: dict[str, Any] = {}
    for horizon in (2, 4, 8):
        if horizon <= len(summary["per_horizon"]):
            requested_horizons[str(horizon)] = {
                "available": True,
                **summary["per_horizon"][horizon - 1],
            }
        else:
            requested_horizons[str(horizon)] = {
                "available": False,
                "horizon": horizon,
                "reason": "trajectory_too_short",
            }
    summary["requested_horizons"] = requested_horizons
    positive_horizon = 0
    for entry in summary["per_horizon"]:
        if entry["r2"] is None or entry["r2"] <= 0:
            break
        positive_horizon = int(entry["horizon"])
    summary["positive_r2_horizon"] = positive_horizon

    if component_dimensions is not None:
        slices = _component_slices(component_dimensions, predicted.shape[-1])
        components = {
            name: _regression_summary(predicted[..., section], target[..., section])
            for name, section in slices.items()
        }
        summary["components"] = components
        component_r2 = [value["r2"] for value in components.values() if value["r2"] is not None]
        summary["macro_r2"] = (
            sum(component_r2) / len(component_r2)
            if len(component_r2) == len(components)
            else None
        )
        summary["macro_r2_available_components"] = (
            sum(component_r2) / len(component_r2) if component_r2 else None
        )
        summary["macro_r2_defined_components"] = len(component_r2)
    return summary


def binary_ranking_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute tie-aware AUROC and average precision without sklearn."""
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape.")
    valid = torch.isfinite(scores)
    if labels.is_floating_point():
        valid &= torch.isfinite(labels)
    if valid_mask is not None:
        if valid_mask.shape != scores.shape:
            raise ValueError("valid_mask must have the same shape as scores.")
        valid &= valid_mask.bool()

    flat_scores = scores[valid].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    selected_labels = labels[valid].detach().to(device="cpu").reshape(-1)
    if not torch.all((selected_labels == 0) | (selected_labels == 1)):
        raise ValueError("labels must contain only binary 0/1 values on valid entries.")
    flat_labels = selected_labels.bool()
    total = int(flat_labels.numel())
    positives = int(flat_labels.sum())
    negatives = total - positives
    result: dict[str, Any] = {
        "available": positives > 0 and negatives > 0,
        "num_examples": total,
        "num_positive": positives,
        "num_negative": negatives,
        "prevalence": positives / total if total else None,
        "auroc": None,
        "average_precision": None,
    }
    if total == 0:
        result["reason"] = "no_valid_events"
        return result
    if positives == 0 or negatives == 0:
        result["reason"] = "both_positive_and_negative_events_are_required"
        return result

    # Mann-Whitney U with average ranks gives an AUROC that is correct under ties.
    ascending = torch.argsort(flat_scores, stable=True)
    sorted_scores = flat_scores[ascending]
    sorted_labels = flat_labels[ascending]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    starts = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]))
    average_ranks = starts.to(torch.float64) + (counts.to(torch.float64) + 1.0) / 2.0
    ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = ranks[sorted_labels].sum()
    auc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    # Threshold-grouped AP is deterministic when multiple examples share a score.
    descending = torch.argsort(flat_scores, descending=True, stable=True)
    desc_scores = flat_scores[descending]
    desc_labels = flat_labels[descending]
    _, desc_counts = torch.unique_consecutive(desc_scores, return_counts=True)
    group_starts = torch.cat((torch.zeros(1, dtype=torch.long), desc_counts.cumsum(0)[:-1]))
    group_positives = torch.stack(
        [
            desc_labels[start : start + count].sum()
            for start, count in zip(group_starts.tolist(), desc_counts.tolist())
        ]
    ).to(torch.float64)
    cumulative_positives = group_positives.cumsum(0)
    cumulative_total = desc_counts.cumsum(0).to(torch.float64)
    precision = cumulative_positives / cumulative_total
    average_precision = ((group_positives / positives) * precision).sum()

    result["auroc"] = float(auc)
    result["average_precision"] = float(average_precision)
    return result


def _scalar_layers(values: torch.Tensor, expected_layers: int, name: str) -> torch.Tensor:
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 2 or values.shape[1] != expected_layers:
        raise ValueError(
            f"{name} must have shape [queries, {expected_layers}] or "
            f"[queries, {expected_layers}, 1]."
        )
    return values


def automatic_semantic_event_scores(
    attention: torch.Tensor,
    mlp: torch.Tensor,
    prediction: torch.Tensor,
    bridge_logprob: torch.Tensor | None = None,
    answer_logprob: torch.Tensor | None = None,
    include_operation_proxies: bool = False,
) -> dict[str, dict[str, Any]]:
    """Construct the event proxies proposed in ``ideas/evaluation metrics.md``.

    F3/F4/F5 are built by default because those are the modes for which the
    diagnostic definitions provide events. Optional F1/F2 operation-norm
    proxies use raw component outputs. All automatic signals are computed only
    after training and measure proxy agreement, not independent semantic validity. Independent
    annotated or intervention-derived labels can instead be passed directly to
    :func:`semantic_alignment_metrics`.
    """
    if attention.ndim != 3 or mlp.shape != attention.shape:
        raise ValueError("attention and mlp must have matching [queries, layers, dim] shapes.")
    num_layers = attention.shape[1]
    result: dict[str, dict[str, Any]] = {}
    if include_operation_proxies:
        result.update(
            {
                "knowledge_enrichment": {
                    "scores": mlp.norm(dim=-1),
                    "valid_mask": torch.isfinite(mlp).all(dim=-1),
                    "source": "high_raw_mlp_output_norm_posthoc_proxy",
                    "positive_only": False,
                    "independent": False,
                },
                "information_routing": {
                    "scores": route_score(attention),
                    "valid_mask": torch.isfinite(attention).all(dim=-1),
                    "source": "high_raw_attention_output_norm_posthoc_proxy",
                    "positive_only": False,
                    "independent": False,
                },
            }
        )

    bridge_change = None
    if bridge_logprob is not None:
        bridge = _scalar_layers(bridge_logprob, num_layers + 1, "bridge_logprob")
        bridge_change = bridge[:, 1:] - bridge[:, :-1]
        result["concept_composition"] = {
            "scores": bridge_change,
            "valid_mask": torch.isfinite(bridge_change),
            "source": "positive_first_token_bridge_log_probability_change",
            "positive_only": True,
            "independent": False,
        }

    if prediction is None:
        raise ValueError("prediction probabilities are required for Eq. (26).")
    refinement_score = prediction_refinement_score(prediction)
    result["prediction_refinement"] = {
        "scores": refinement_score,
        "valid_mask": torch.isfinite(refinement_score),
        "source": "joint_top1_top2_margin_increase_and_uncertainty_decrease",
        "positive_only": True,
        "independent": False,
    }

    answer_change = None
    if answer_logprob is not None:
        answer = _scalar_layers(answer_logprob, num_layers + 1, "answer_logprob")
        answer_change = answer[:, 1:] - answer[:, :-1]

    if bridge_change is not None and answer_change is not None:
        transition_score = (
            (-bridge_change).clamp_min(0.0) * answer_change.clamp_min(0.0)
        ).sqrt()
        result["hop_transition"] = {
            "scores": transition_score,
            "valid_mask": torch.isfinite(bridge_change) & torch.isfinite(answer_change),
            "source": "joint_first_token_bridge_logprob_decrease_and_answer_logprob_increase",
            "positive_only": True,
            "independent": False,
        }
    return result


def _quantile_event_labels(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    quantile: float,
    positive_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("event_quantile must be strictly between 0 and 1.")
    if scores.ndim != 2 or valid_mask.shape != scores.shape:
        raise ValueError("Event scores and masks must have shape [queries, layers].")
    usable = valid_mask.bool() & torch.isfinite(scores)
    if positive_only:
        usable &= scores >= 0
        scores = scores.clamp_min(0.0)
    labels = torch.zeros_like(scores, dtype=torch.bool)
    if usable.any():
        threshold = torch.quantile(scores[usable].float(), quantile)
        labels = usable & (scores >= threshold)
    return labels, valid_mask.bool() & torch.isfinite(scores)


def mode_contribution_scores(
    memberships: torch.Tensor,
    local_deltas: torch.Tensor,
) -> torch.Tensor:
    """Return q[i,l,k] = mu[i,l,k] * ||F_k(chi[i,l])||_2."""
    if memberships.ndim != 3 or memberships.shape[-1] != len(REASONING_MODES):
        raise ValueError(
            f"memberships must have shape [queries, layers, {len(REASONING_MODES)}]."
        )
    if local_deltas.ndim != 4 or local_deltas.shape[:-1] != memberships.shape:
        raise ValueError(
            "local_deltas must have shape [queries, layers, modes, state_dim] "
            "aligned with memberships."
        )
    if local_deltas.shape[-1] == 0:
        raise ValueError("local_deltas must have a non-empty state dimension.")
    if not torch.isfinite(memberships).all() or not torch.isfinite(local_deltas).all():
        raise ValueError("Semantic contribution scores require finite tensors.")
    return memberships * local_deltas.norm(dim=-1)


def _resolved_event_labels(
    definition: Mapping[str, Any],
    template: torch.Tensor,
    event_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    valid_mask = definition.get(
        "valid_mask", torch.ones_like(template, dtype=torch.bool)
    )
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != template.shape:
        raise ValueError("Event valid_mask must align with semantic scores.")
    if "labels" in definition:
        labels = definition["labels"]
        if not isinstance(labels, torch.Tensor) or labels.shape != template.shape:
            raise ValueError("Event labels must align with semantic scores.")
        selected_labels = labels[valid_mask.bool()]
        if not torch.all((selected_labels == 0) | (selected_labels == 1)):
            raise ValueError("Event labels must contain only binary 0/1 values.")
        label_rule = "provided_binary_labels"
    elif "scores" in definition:
        event_scores = definition["scores"]
        if not isinstance(event_scores, torch.Tensor) or event_scores.shape != template.shape:
            raise ValueError("Event scores must align with semantic scores.")
        labels, valid_mask = _quantile_event_labels(
            event_scores,
            valid_mask,
            event_quantile,
            bool(definition.get("positive_only", False)),
        )
        label_rule = f"global_top_{1.0 - event_quantile:.3f}_quantile"
    else:
        raise ValueError("Each event definition needs 'labels' or 'scores'.")
    return labels.bool(), valid_mask.bool(), label_rule


def semantic_alignment_matrix(
    mode_scores: torch.Tensor,
    event_definitions: Mapping[str, Mapping[str, Any]],
    event_quantile: float = 0.75,
    score_definition: str = "unspecified",
) -> dict[str, Any]:
    """Evaluate every mode score against every semantic event (a 5x5 matrix)."""
    if mode_scores.ndim != 3 or mode_scores.shape[-1] != len(REASONING_MODES):
        raise ValueError(
            f"mode_scores must have shape [queries, layers, {len(REASONING_MODES)}]."
        )
    if not torch.isfinite(mode_scores).all():
        raise ValueError("Semantic alignment scores must be finite.")
    if not 0.0 < event_quantile < 1.0:
        raise ValueError("event_quantile must be strictly between 0 and 1.")

    resolved_events: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
    event_metadata: dict[str, Any] = {}
    for event_mode in REASONING_MODES:
        definition = event_definitions.get(event_mode)
        if definition is None:
            event_metadata[event_mode] = {
                "available": False,
                "reason": "event_definition_unavailable",
                "source": None,
                "independently_labeled_event": False,
            }
            continue
        labels, valid_mask, label_rule = _resolved_event_labels(
            definition,
            mode_scores[..., 0],
            event_quantile,
        )
        positives = labels & valid_mask
        resolved_events[event_mode] = (labels, valid_mask, label_rule)
        event_metadata[event_mode] = {
            "available": bool(valid_mask.any()),
            "source": definition.get("source", "unspecified"),
            "independently_labeled_event": bool(definition.get("independent", False)),
            "label_rule": label_rule,
            "num_queries": int(mode_scores.shape[0]),
            "num_queries_with_valid_layers": int(valid_mask.any(dim=1).sum()),
            "num_queries_with_positive_events": int(positives.any(dim=1).sum()),
            "coverage": float(valid_mask.float().mean()),
            "num_positive": int(positives.sum()),
            "num_valid": int(valid_mask.sum()),
        }

    cells: dict[str, dict[str, Any]] = {}
    available_cells = 0
    for score_index, score_mode in enumerate(REASONING_MODES):
        row: dict[str, Any] = {}
        for event_mode in REASONING_MODES:
            resolved = resolved_events.get(event_mode)
            if resolved is None:
                row[event_mode] = {
                    "available": False,
                    "reason": "event_definition_unavailable",
                    "auroc": None,
                    "average_precision": None,
                }
                continue
            labels, valid_mask, _label_rule = resolved
            ranking = binary_ranking_metrics(
                mode_scores[..., score_index], labels, valid_mask=valid_mask
            )
            row[event_mode] = ranking
            available_cells += int(ranking["available"])
        cells[score_mode] = row

    row_summaries: dict[str, Any] = {}
    diagonal_aurocs: list[float] = []
    diagonal_aps: list[float] = []
    auroc_gaps: list[float] = []
    ap_gaps: list[float] = []
    for score_mode in REASONING_MODES:
        diagonal = cells[score_mode][score_mode]
        off_diagonal = [
            cells[score_mode][event_mode]
            for event_mode in REASONING_MODES
            if event_mode != score_mode
        ]
        off_aurocs = [
            float(cell["auroc"])
            for cell in off_diagonal
            if cell.get("auroc") is not None
        ]
        off_aps = [
            float(cell["average_precision"])
            for cell in off_diagonal
            if cell.get("average_precision") is not None
        ]
        diagonal_auroc = diagonal.get("auroc")
        diagonal_ap = diagonal.get("average_precision")
        auroc_gap = (
            float(diagonal_auroc) - sum(off_aurocs) / len(off_aurocs)
            if diagonal_auroc is not None and len(off_aurocs) == len(REASONING_MODES) - 1
            else None
        )
        ap_gap = (
            float(diagonal_ap) - sum(off_aps) / len(off_aps)
            if diagonal_ap is not None and len(off_aps) == len(REASONING_MODES) - 1
            else None
        )
        row_summaries[score_mode] = {
            "diagonal_auroc": diagonal_auroc,
            "mean_off_diagonal_auroc": (
                sum(off_aurocs) / len(off_aurocs) if off_aurocs else None
            ),
            "auroc_selectivity_gap": auroc_gap,
            "diagonal_average_precision": diagonal_ap,
            "mean_off_diagonal_average_precision": (
                sum(off_aps) / len(off_aps) if off_aps else None
            ),
            "average_precision_selectivity_gap": ap_gap,
        }
        if diagonal_auroc is not None:
            diagonal_aurocs.append(float(diagonal_auroc))
        if diagonal_ap is not None:
            diagonal_aps.append(float(diagonal_ap))
        if auroc_gap is not None:
            auroc_gaps.append(auroc_gap)
        if ap_gap is not None:
            ap_gaps.append(ap_gap)

    complete = available_cells == len(REASONING_MODES) ** 2
    all_independent = all(
        bool(event_metadata[mode].get("independently_labeled_event", False))
        for mode in REASONING_MODES
    )
    return {
        "schema_version": 1,
        "score_definition": score_definition,
        "score_order": list(REASONING_MODES),
        "event_order": list(REASONING_MODES),
        "event_quantile": event_quantile,
        "event_metadata": event_metadata,
        "cells": cells,
        "row_summaries": row_summaries,
        "available_cells": available_cells,
        "total_cells": len(REASONING_MODES) ** 2,
        "complete_five_by_five": complete,
        "all_events_independently_labeled": all_independent,
        "diagnostic_macro_diagonal_auroc": (
            sum(diagonal_aurocs) / len(diagonal_aurocs) if diagonal_aurocs else None
        ),
        "diagnostic_macro_diagonal_average_precision": (
            sum(diagonal_aps) / len(diagonal_aps) if diagonal_aps else None
        ),
        "diagnostic_mean_auroc_selectivity_gap": (
            sum(auroc_gaps) / len(auroc_gaps) if auroc_gaps else None
        ),
        "diagnostic_mean_average_precision_selectivity_gap": (
            sum(ap_gaps) / len(ap_gaps) if ap_gaps else None
        ),
        "strict_macro_diagonal_auroc": (
            sum(diagonal_aurocs) / len(diagonal_aurocs)
            if complete and all_independent
            else None
        ),
        "strict_macro_diagonal_average_precision": (
            sum(diagonal_aps) / len(diagonal_aps)
            if complete and all_independent
            else None
        ),
        "strict_metric_reason": (
            None
            if complete and all_independent
            else "requires_complete_independent_events_for_all_five_modes"
        ),
    }


def semantic_alignment_metrics(
    memberships: torch.Tensor,
    event_definitions: Mapping[str, Mapping[str, Any]],
    event_quantile: float = 0.75,
) -> dict[str, Any]:
    """Compute per-mode AUROC/AP and strict five-mode macro scores.

    Each event definition may contain binary ``labels`` directly, or continuous
    ``scores`` that are converted to global top-quantile events. Missing or
    single-class modes remain explicit and are not silently folded into the
    strict five-mode macro score.
    """
    if memberships.ndim != 3 or memberships.shape[-1] != len(REASONING_MODES):
        raise ValueError(
            f"memberships must have shape [queries, layers, {len(REASONING_MODES)}]."
        )
    mode_results: dict[str, Any] = {}
    available_aurocs: list[float] = []
    available_aps: list[float] = []

    for mode_index, mode in enumerate(REASONING_MODES):
        definition = event_definitions.get(mode)
        if definition is None:
            mode_results[mode] = {
                "available": False,
                "reason": "event_definition_unavailable",
                "auroc": None,
                "average_precision": None,
            }
            continue
        valid_mask = definition.get(
            "valid_mask", torch.ones_like(memberships[..., mode_index], dtype=torch.bool)
        )
        if "labels" in definition:
            labels = definition["labels"]
            label_rule = "provided_binary_labels"
        elif "scores" in definition:
            labels, valid_mask = _quantile_event_labels(
                definition["scores"],
                valid_mask,
                event_quantile,
                bool(definition.get("positive_only", False)),
            )
            label_rule = f"global_top_{1.0 - event_quantile:.3f}_quantile"
        else:
            raise ValueError(f"Event definition for {mode} needs 'labels' or 'scores'.")

        ranking = binary_ranking_metrics(
            memberships[..., mode_index], labels, valid_mask=valid_mask
        )
        ranking.update(
            {
                "source": definition.get("source", "unspecified"),
                "independently_labeled_event": bool(definition.get("independent", False)),
                "label_rule": label_rule,
                "num_queries": int(memberships.shape[0]),
                "num_queries_with_valid_layers": int(valid_mask.any(dim=1).sum()),
                "num_queries_with_positive_events": int(
                    (labels.bool() & valid_mask.bool()).any(dim=1).sum()
                ),
                "coverage": float(valid_mask.float().mean()),
            }
        )
        mode_results[mode] = ranking
        if ranking["available"]:
            available_aurocs.append(ranking["auroc"])
            available_aps.append(ranking["average_precision"])

    all_available = len(available_aurocs) == len(REASONING_MODES)
    all_independent = all(
        bool(event_definitions.get(mode, {}).get("independent", False))
        for mode in REASONING_MODES
    )
    strict_available = all_available and all_independent
    available_modes = [
        mode for mode in REASONING_MODES if mode_results[mode].get("available", False)
    ]
    return {
        "modes": mode_results,
        "event_quantile": event_quantile,
        "num_available_modes": len(available_aurocs),
        "available_modes": available_modes,
        "available_mode_fraction": len(available_aurocs) / len(REASONING_MODES),
        "macro_auroc": (
            sum(available_aurocs) / len(available_aurocs) if strict_available else None
        ),
        "macro_average_precision": (
            sum(available_aps) / len(available_aps) if strict_available else None
        ),
        "diagnostic_macro_auroc_available_modes": (
            sum(available_aurocs) / len(available_aurocs) if available_aurocs else None
        ),
        "diagnostic_macro_average_precision_available_modes": (
            sum(available_aps) / len(available_aps) if available_aps else None
        ),
        "strict_macro_requires_all_five_modes": True,
        "strict_macro_requires_independent_events": True,
        "all_events_independently_labeled": all_independent,
        "strict_macro_reason": (
            None
            if strict_available
            else "requires_defined_independent_events_for_all_five_modes"
        ),
    }
