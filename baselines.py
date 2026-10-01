"""Leakage-safe baselines for the main dynamics-prediction experiment."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch


ProjectedBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ProjectedBatchFactory = Callable[[], Iterable[ProjectedBatch]]
PassCleanup = Callable[[], None]


@dataclass(frozen=True)
class LinearDynamicsBaseline:
    """Ridge regression from ``[state, attention, MLP]`` to state change."""

    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    weight: torch.Tensor

    def predict(
        self,
        states: torch.Tensor,
        attention: torch.Tensor,
        mlp: torch.Tensor,
    ) -> torch.Tensor:
        if states.shape[:-1] != attention.shape[:-1] or attention.shape != mlp.shape:
            raise ValueError("Baseline states and operation signals must align.")
        output_device = states.device
        output_dtype = states.dtype
        calculation_device = self.weight.device
        features = torch.cat((states, attention, mlp), dim=-1).to(
            device=calculation_device, dtype=torch.float64
        )
        standardized = (features - self.feature_mean) / self.feature_scale
        design = torch.cat(
            (standardized, torch.ones_like(standardized[..., :1])), dim=-1
        )
        return torch.matmul(design, self.weight).to(
            device=output_device, dtype=output_dtype
        )


@dataclass(frozen=True)
class GuardedRollout:
    """Finite prefix and diagnostics from a recursively applied baseline."""

    states: torch.Tensor
    complete: bool
    divergence_horizon: int | None
    divergence_reason: str | None


def fit_linear_dynamics_baseline(
    states: torch.Tensor,
    attention: torch.Tensor,
    mlp: torch.Tensor,
    ridge: float = 1e-3,
) -> LinearDynamicsBaseline:
    """Fit a global linear dynamics model using training trajectories only."""
    if states.ndim != 3 or attention.ndim != 3 or mlp.ndim != 3:
        raise ValueError("Baseline tensors must have shape [queries, layers, features].")
    if states.shape[1] != attention.shape[1] + 1 or attention.shape != mlp.shape:
        raise ValueError("Baseline state and operation trajectories do not align.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    current = states[:, :-1].reshape(-1, states.shape[-1]).to(torch.float64)
    target = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1]).to(
        torch.float64
    )
    features = torch.cat(
        (
            current,
            attention.reshape(-1, attention.shape[-1]).to(torch.float64),
            mlp.reshape(-1, mlp.shape[-1]).to(torch.float64),
        ),
        dim=-1,
    )
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-8)
    standardized = (features - feature_mean) / feature_scale
    design = torch.cat(
        (standardized, torch.ones(standardized.shape[0], 1, dtype=torch.float64)),
        dim=-1,
    )
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * ridge
    penalty[-1, -1] = 0.0
    weight = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
    return LinearDynamicsBaseline(feature_mean, feature_scale, weight)


def fit_linear_dynamics_baseline_streaming(
    batches: ProjectedBatchFactory,
    ridge: float = 1e-3,
    calculation_device: torch.device | str = "cpu",
    pass_cleanup: PassCleanup | None = None,
) -> LinearDynamicsBaseline:
    """Fit the exact standardized ridge baseline from bounded-memory batches.

    Two deterministic passes compute the same population feature statistics and
    normal equations as :func:`fit_linear_dynamics_baseline` without retaining
    the ``[queries * layers, hidden_size]`` design and target matrices.
    """
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    device = torch.device(calculation_device)
    feature_sum: torch.Tensor | None = None
    feature_square_sum: torch.Tensor | None = None
    row_count = 0
    output_dim: int | None = None

    for states, attention, mlp in batches():
        if states.ndim != 3 or attention.ndim != 3 or mlp.ndim != 3:
            raise ValueError("Baseline tensors must have shape [queries, layers, features].")
        if states.shape[1] != attention.shape[1] + 1 or attention.shape != mlp.shape:
            raise ValueError("Baseline state and operation trajectories do not align.")
        current = states[:, :-1].reshape(-1, states.shape[-1]).to(
            device=device, dtype=torch.float64
        )
        features = torch.cat(
            (
                current,
                attention.reshape(-1, attention.shape[-1]).to(
                    device=device, dtype=torch.float64
                ),
                mlp.reshape(-1, mlp.shape[-1]).to(
                    device=device, dtype=torch.float64
                ),
            ),
            dim=-1,
        )
        if feature_sum is None:
            feature_sum = torch.zeros(
                features.shape[-1], dtype=torch.float64, device=device
            )
            feature_square_sum = torch.zeros_like(feature_sum)
            output_dim = int(states.shape[-1])
        elif features.shape[-1] != feature_sum.shape[0] or states.shape[-1] != output_dim:
            raise ValueError("Streaming baseline batch dimensions changed between passes.")
        feature_sum.add_(features.sum(dim=0))
        assert feature_square_sum is not None
        feature_square_sum.add_(features.square().sum(dim=0))
        row_count += int(features.shape[0])

    if feature_sum is None or feature_square_sum is None or output_dim is None or row_count == 0:
        raise ValueError("Streaming baseline received no training rows.")
    feature_mean = feature_sum / row_count
    feature_variance = (
        feature_square_sum / row_count - feature_mean.square()
    ).clamp_min(0.0)
    feature_scale = feature_variance.sqrt().clamp_min(1e-8)
    if pass_cleanup is not None:
        pass_cleanup()
    feature_dim = int(feature_mean.shape[0])
    normal = torch.zeros(
        feature_dim + 1, feature_dim + 1, dtype=torch.float64, device=device
    )
    right = torch.zeros(
        feature_dim + 1, output_dim, dtype=torch.float64, device=device
    )
    second_pass_rows = 0
    for states, attention, mlp in batches():
        current = states[:, :-1].reshape(-1, states.shape[-1]).to(
            device=device, dtype=torch.float64
        )
        target = (states[:, 1:] - states[:, :-1]).reshape(
            -1, states.shape[-1]
        ).to(device=device, dtype=torch.float64)
        features = torch.cat(
            (
                current,
                attention.reshape(-1, attention.shape[-1]).to(
                    device=device, dtype=torch.float64
                ),
                mlp.reshape(-1, mlp.shape[-1]).to(
                    device=device, dtype=torch.float64
                ),
            ),
            dim=-1,
        )
        standardized = (features - feature_mean) / feature_scale
        design = torch.cat(
            (
                standardized,
                torch.ones(
                    standardized.shape[0], 1, dtype=torch.float64, device=device
                ),
            ),
            dim=1,
        )
        normal.addmm_(design.T, design)
        right.addmm_(design.T, target)
        second_pass_rows += int(design.shape[0])
    if second_pass_rows != row_count:
        raise ValueError("Streaming baseline batch factory changed between passes.")
    penalty = torch.eye(feature_dim + 1, dtype=torch.float64, device=device) * ridge
    penalty[-1, -1] = 0.0
    weight = torch.linalg.solve(normal + penalty, right)
    return LinearDynamicsBaseline(feature_mean, feature_scale, weight)


def mean_transition_delta(states: torch.Tensor) -> torch.Tensor:
    """Return the training-set mean transition separately for every layer."""
    if states.ndim != 3 or states.shape[1] < 2:
        raise ValueError("states must contain [queries, at least two layers, state_dim].")
    return (states[:, 1:] - states[:, :-1]).mean(dim=0)


def recursive_rollout(
    initial_state: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    """Accumulate predicted changes beginning at an observed initial state."""
    if initial_state.ndim != 2 or deltas.ndim != 3:
        raise ValueError("Expected initial_state [N,D] and deltas [N,L,D].")
    if initial_state.shape[0] != deltas.shape[0] or initial_state.shape[1] != deltas.shape[2]:
        raise ValueError("Initial states and predicted deltas do not align.")
    return torch.cat(
        (initial_state.unsqueeze(1), initial_state.unsqueeze(1) + deltas.cumsum(dim=1)),
        dim=1,
    )


def linear_recursive_rollout(
    baseline: LinearDynamicsBaseline,
    initial_state: torch.Tensor,
    attention: torch.Tensor,
    mlp: torch.Tensor,
) -> torch.Tensor:
    """Recursively apply the global linear model with observed operation signals."""
    current = initial_state
    predicted = [current]
    for layer in range(attention.shape[1]):
        delta = baseline.predict(current, attention[:, layer], mlp[:, layer])
        current = current + delta
        predicted.append(current)
    return torch.stack(predicted, dim=1)


def guarded_linear_recursive_rollout(
    baseline: LinearDynamicsBaseline,
    initial_state: torch.Tensor,
    attention: torch.Tensor,
    mlp: torch.Tensor,
    max_abs_state: float = 1e6,
) -> GuardedRollout:
    """Roll out until completion or the first numerically divergent state.

    A divergent state is never appended, so every state returned in the finite
    prefix remains safe for metric computation and strict JSON serialization.
    """
    if max_abs_state <= 0:
        raise ValueError("max_abs_state must be positive.")
    current = initial_state
    predicted = [current]
    for layer in range(attention.shape[1]):
        delta = baseline.predict(current, attention[:, layer], mlp[:, layer])
        next_state = current + delta
        horizon = layer + 1
        if not torch.isfinite(next_state).all():
            return GuardedRollout(
                torch.stack(predicted, dim=1), False, horizon, "non_finite_state"
            )
        if float(next_state.abs().max()) > max_abs_state:
            return GuardedRollout(
                torch.stack(predicted, dim=1), False, horizon, "state_magnitude_limit"
            )
        current = next_state
        predicted.append(current)
    return GuardedRollout(torch.stack(predicted, dim=1), True, None, None)
