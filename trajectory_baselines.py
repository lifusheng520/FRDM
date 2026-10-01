"""State-only trajectory baselines for Experiment 1.

The implementations in this module are deliberately independent of the FRDM
training loop.  They are fitted only on projected states from the checkpoint
training split and expose a small layer-wise prediction interface used by the
existing evaluation metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from .baselines import GuardedRollout


StateBatchFactory = Callable[[], Iterable[torch.Tensor]]


def _validate_states(states: torch.Tensor) -> None:
    if states.ndim != 3 or states.shape[1] < 2 or states.shape[2] < 1:
        raise ValueError(
            "states must have shape [queries, at least two layers, state_dim]."
        )
    if not torch.isfinite(states).all():
        raise ValueError("Baseline fitting requires finite states.")


def _canonicalize_component_signs(components: torch.Tensor) -> torch.Tensor:
    """Remove the arbitrary sign of eigen/SVD vectors deterministically."""
    if components.ndim != 2:
        raise ValueError("components must be a matrix with vectors in columns.")
    largest = components.abs().argmax(dim=0)
    columns = torch.arange(components.shape[1], device=components.device)
    signs = torch.sign(components[largest, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return components * signs


def _left_singular_system(
    samples: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return leading left-singular vectors/values without materializing Vh."""
    gram = samples.T @ samples
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.arange(
        eigenvalues.shape[0] - 1,
        eigenvalues.shape[0] - rank - 1,
        -1,
    )
    basis = _canonicalize_component_signs(eigenvectors[:, order])
    singular_values = eigenvalues[order].clamp_min(0.0).sqrt()
    return basis, singular_values


@dataclass(frozen=True)
class LinesOfThoughtBaseline:
    """Layer-wise SVD singular-value-ratio extrapolation.

    This is the deterministic, state-only variant described in
    ``ideas/3. baseline details.md``.  No stochastic residual is added.
    """

    current_basis: torch.Tensor
    next_basis: torch.Tensor
    singular_value_ratio: torch.Tensor
    requested_rank: int
    effective_rank: int
    singular_value_epsilon: float

    @property
    def transitions(self) -> int:
        return int(self.current_basis.shape[0])

    def predict_layer(self, states: torch.Tensor, layer: int) -> torch.Tensor:
        if states.ndim != 2 or states.shape[1] != self.current_basis.shape[1]:
            raise ValueError("states must have shape [queries, fitted_state_dim].")
        if layer < 0 or layer >= self.transitions:
            raise IndexError(f"Layer {layer} outside 0-{self.transitions - 1}.")
        dtype = states.dtype
        output_device = states.device
        values = states.to(
            device=self.current_basis.device, dtype=torch.float64
        )
        coordinates = values @ self.current_basis[layer]
        coordinates = coordinates * self.singular_value_ratio[layer]
        return (coordinates @ self.next_basis[layer].T).to(
            device=output_device, dtype=dtype
        )

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        _validate_states(states)
        if states.shape[1] - 1 != self.transitions:
            raise ValueError("Evaluation and fitted trajectories have different lengths.")
        return torch.stack(
            [self.predict_layer(states[:, layer], layer) for layer in range(self.transitions)],
            dim=1,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "variant": "layerwise_svd_singular_value_ratio",
            "input": "state_only",
            "svd_centered": False,
            "stochastic_residual": False,
            "requested_rank": self.requested_rank,
            "effective_rank": self.effective_rank,
            "singular_value_epsilon": self.singular_value_epsilon,
        }


def fit_lines_of_thought_baseline(
    states: torch.Tensor,
    rank: int = 40,
    singular_value_epsilon: float = 1e-8,
) -> LinesOfThoughtBaseline:
    """Fit the layer-wise LoT mapping from training trajectories only."""
    _validate_states(states)
    if rank <= 0:
        raise ValueError("rank must be positive.")
    if singular_value_epsilon <= 0:
        raise ValueError("singular_value_epsilon must be positive.")
    values = states.detach().to(device="cpu", dtype=torch.float64)
    effective_rank = min(rank, values.shape[0], values.shape[2])
    current_basis: list[torch.Tensor] = []
    next_basis: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    for layer in range(values.shape[1] - 1):
        current_u, current_s = _left_singular_system(
            values[:, layer], effective_rank
        )
        next_u, next_s = _left_singular_system(
            values[:, layer + 1], effective_rank
        )
        ratio = torch.where(
            current_s > singular_value_epsilon,
            next_s / current_s,
            torch.zeros_like(current_s),
        )
        current_basis.append(current_u)
        next_basis.append(next_u)
        ratios.append(ratio)
    return LinesOfThoughtBaseline(
        current_basis=torch.stack(current_basis),
        next_basis=torch.stack(next_basis),
        singular_value_ratio=torch.stack(ratios),
        requested_rank=rank,
        effective_rank=effective_rank,
        singular_value_epsilon=singular_value_epsilon,
    )


def fit_lines_of_thought_baseline_streaming(
    batches: StateBatchFactory,
    rank: int = 40,
    singular_value_epsilon: float = 1e-8,
    calculation_device: torch.device | str = "cpu",
) -> LinesOfThoughtBaseline:
    """Fit the exact uncentered layer-wise LoT SVD from streamed states."""
    if rank <= 0:
        raise ValueError("rank must be positive.")
    if singular_value_epsilon <= 0:
        raise ValueError("singular_value_epsilon must be positive.")
    device = torch.device(calculation_device)
    grams: list[torch.Tensor] | None = None
    query_count = 0
    hidden_size = 0
    for states in batches():
        _validate_states(states)
        if grams is None:
            hidden_size = int(states.shape[-1])
            grams = [
                torch.zeros(
                    hidden_size, hidden_size, dtype=torch.float64, device=device
                )
                for _ in range(states.shape[1])
            ]
        elif len(grams) != states.shape[1] or hidden_size != states.shape[-1]:
            raise ValueError("Streaming LoT batch dimensions changed.")
        values = states.to(device=device, dtype=torch.float64)
        for layer, gram in enumerate(grams):
            layer_values = values[:, layer]
            gram.addmm_(layer_values.T, layer_values)
        query_count += int(states.shape[0])
    if grams is None or query_count == 0:
        raise ValueError("Streaming LoT received no training states.")
    effective_rank = min(rank, query_count, hidden_size)
    bases: list[torch.Tensor] = []
    singular_values: list[torch.Tensor] = []
    for gram in grams:
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.arange(
            eigenvalues.shape[0] - 1,
            eigenvalues.shape[0] - effective_rank - 1,
            -1,
            device=device,
        )
        basis = _canonicalize_component_signs(eigenvectors[:, order])
        bases.append(basis)
        singular_values.append(eigenvalues[order].clamp_min(0.0).sqrt())
    ratios = [
        torch.where(
            singular_values[layer] > singular_value_epsilon,
            singular_values[layer + 1] / singular_values[layer],
            torch.zeros_like(singular_values[layer]),
        )
        for layer in range(len(bases) - 1)
    ]
    return LinesOfThoughtBaseline(
        current_basis=torch.stack(bases[:-1]),
        next_basis=torch.stack(bases[1:]),
        singular_value_ratio=torch.stack(ratios),
        requested_rank=rank,
        effective_rank=effective_rank,
        singular_value_epsilon=singular_value_epsilon,
    )


def _principal_components(
    values: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    _, vectors = torch.linalg.eigh(covariance)
    components = vectors[:, -rank:].flip(dims=(1,))
    return mean, _canonicalize_component_signs(components)


def _squared_distances(values: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return (
        values.square().sum(dim=1, keepdim=True)
        + centroids.square().sum(dim=1).unsqueeze(0)
        - 2.0 * values @ centroids.T
    ).clamp_min_(0.0)


def _fit_kmeans(
    values: torch.Tensor,
    modes: int,
    max_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if modes > values.shape[0]:
        raise ValueError("modes cannot exceed the number of transition examples.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(values.shape[0], (1,), generator=generator).item())
    centroids = [values[first].clone()]
    closest = (values - centroids[0]).square().sum(dim=1)
    for _ in range(1, modes):
        total = closest.sum()
        if float(total) == 0.0:
            index = len(centroids) % values.shape[0]
        else:
            index = int(
                torch.multinomial(closest / total, 1, generator=generator).item()
            )
        centroids.append(values[index].clone())
        closest = torch.minimum(
            closest, (values - centroids[-1]).square().sum(dim=1)
        )
    centroid_tensor = torch.stack(centroids)
    labels = torch.zeros(values.shape[0], dtype=torch.long)
    completed_iterations = 0
    for iteration in range(max_iterations):
        distances = _squared_distances(values, centroid_tensor)
        new_labels = distances.argmin(dim=1)
        updated: list[torch.Tensor] = []
        min_distances = distances.min(dim=1).values
        for mode in range(modes):
            selected = values[new_labels == mode]
            if selected.shape[0] == 0:
                updated.append(values[min_distances.argmax()].clone())
            else:
                updated.append(selected.mean(dim=0))
        new_centroids = torch.stack(updated)
        completed_iterations = iteration + 1
        converged = torch.equal(new_labels, labels) and torch.allclose(
            new_centroids, centroid_tensor, rtol=0.0, atol=1e-10
        )
        labels = new_labels
        centroid_tensor = new_centroids
        if converged:
            break
    labels = _squared_distances(values, centroid_tensor).argmin(dim=1)
    return centroid_tensor, labels, completed_iterations


def _ridge_solution(
    design: torch.Tensor,
    target: torch.Tensor,
    penalty: torch.Tensor,
) -> torch.Tensor:
    return _ridge_solution_from_statistics(
        design.T @ design,
        design.T @ target,
        penalty,
    )


def _ridge_solution_from_statistics(
    normal_matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
    penalty: torch.Tensor,
) -> torch.Tensor:
    """Solve ridge regression from pre-accumulated sufficient statistics."""
    regularized = normal_matrix + penalty
    try:
        return torch.linalg.solve(regularized, right_hand_side)
    except RuntimeError:
        return torch.linalg.lstsq(regularized, right_hand_side).solution


@dataclass(frozen=True)
class HardSwitchingSLDSBaseline:
    """PCA/K-means hard-switching state-only linear dynamics baseline."""

    projection_mean: torch.Tensor
    projection_components: torch.Tensor
    centroids: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    weights: torch.Tensor
    transition_matrix: torch.Tensor
    initial_distribution: torch.Tensor
    cluster_counts: torch.Tensor
    requested_rank: int
    effective_rank: int
    ridge: float
    seed: int
    kmeans_iterations: int
    transition_smoothing: float

    @property
    def modes(self) -> int:
        return int(self.centroids.shape[0])

    def infer_mode(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or states.shape[1] != self.projection_mean.shape[0]:
            raise ValueError("states must have shape [queries, fitted_state_dim].")
        output_device = states.device
        projected = (
            states.to(device=self.projection_mean.device, dtype=torch.float64)
            - self.projection_mean
        ) @ self.projection_components
        return _squared_distances(projected, self.centroids).argmin(dim=1).to(
            output_device
        )

    def predict_delta(self, states: torch.Tensor) -> torch.Tensor:
        dtype = states.dtype
        output_device = states.device
        calculation_device = self.weights.device
        values = states.to(device=calculation_device, dtype=torch.float64)
        projected = (
            values - self.projection_mean
        ) @ self.projection_components
        modes = _squared_distances(projected, self.centroids).argmin(dim=1)
        standardized = (values - self.feature_mean) / self.feature_scale
        design = torch.cat(
            (
                standardized,
                torch.ones(
                    standardized.shape[0],
                    1,
                    dtype=torch.float64,
                    device=calculation_device,
                ),
            ),
            dim=1,
        )
        predicted = torch.empty_like(values)
        for mode in range(self.modes):
            mask = modes == mode
            if mask.any():
                predicted[mask] = design[mask] @ self.weights[mode]
        return predicted.to(device=output_device, dtype=dtype)

    def predict_layer(self, states: torch.Tensor, layer: int) -> torch.Tensor:
        del layer
        return states + self.predict_delta(states)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        _validate_states(states)
        current = states[:, :-1]
        flat = current.reshape(-1, current.shape[-1])
        return (flat + self.predict_delta(flat)).reshape_as(current)

    def manifest(self) -> dict[str, object]:
        return {
            "variant": "hard_switching_pca_kmeans_ridge",
            "input": "state_only",
            "modes": self.modes,
            "requested_rank": self.requested_rank,
            "effective_rank": self.effective_rank,
            "ridge": self.ridge,
            "seed": self.seed,
            "kmeans_initialization": "seeded_kmeans_plus_plus",
            "kmeans_iterations": self.kmeans_iterations,
            "pca_usage": "mode_assignment",
            "regression_space": "standardized_full_state",
            "mode_inference": "nearest_pca_centroid_from_current_state",
            "transition_usage": "diagnostic_only_for_hard_switching_variant",
            "transition_smoothing": self.transition_smoothing,
            "cluster_counts": self.cluster_counts.tolist(),
            "initial_distribution": self.initial_distribution.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
        }


def fit_hard_switching_slds_baseline(
    states: torch.Tensor,
    modes: int = 4,
    rank: int = 40,
    ridge: float = 1e-3,
    max_iterations: int = 25,
    seed: int = 0,
    transition_smoothing: float = 1.0,
) -> HardSwitchingSLDSBaseline:
    """Fit the reproducible hard-switching SLDS variant on training states."""
    _validate_states(states)
    if modes <= 0 or rank <= 0 or max_iterations <= 0:
        raise ValueError("modes, rank, and max_iterations must be positive.")
    if ridge < 0 or transition_smoothing < 0:
        raise ValueError("ridge and transition_smoothing must be non-negative.")
    values = states.detach().to(device="cpu", dtype=torch.float64)
    current = values[:, :-1]
    target_delta = values[:, 1:] - current
    flat_current = current.reshape(-1, current.shape[-1])
    flat_target = target_delta.reshape(-1, target_delta.shape[-1])
    effective_rank = min(rank, flat_current.shape[0], flat_current.shape[1])
    projection_mean, components = _principal_components(flat_current, effective_rank)
    projected = (flat_current - projection_mean) @ components
    centroids, labels, iterations = _fit_kmeans(
        projected, modes=modes, max_iterations=max_iterations, seed=seed
    )

    feature_mean = flat_current.mean(dim=0)
    feature_scale = flat_current.std(dim=0, unbiased=False).clamp_min(1e-8)
    standardized = (flat_current - feature_mean) / feature_scale
    design = torch.cat(
        (standardized, torch.ones(standardized.shape[0], 1, dtype=torch.float64)),
        dim=1,
    )
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * ridge
    penalty[-1, -1] = 0.0
    weights: list[torch.Tensor] = []
    counts = torch.bincount(labels, minlength=modes)
    fallback_weight = _ridge_solution(design, flat_target, penalty)
    for mode in range(modes):
        selected_design = design[labels == mode]
        selected_target = flat_target[labels == mode]
        weights.append(
            fallback_weight
            if selected_design.shape[0] == 0
            else _ridge_solution(selected_design, selected_target, penalty)
        )

    sequence_labels = labels.reshape(current.shape[0], current.shape[1])
    transition_counts = torch.full(
        (modes, modes), transition_smoothing, dtype=torch.float64
    )
    for previous, following in zip(
        sequence_labels[:, :-1].reshape(-1), sequence_labels[:, 1:].reshape(-1)
    ):
        transition_counts[previous, following] += 1.0
    outgoing = transition_counts.sum(dim=1, keepdim=True)
    transition_matrix = transition_counts / outgoing.clamp_min(1.0)
    no_outgoing = outgoing.squeeze(1) == 0
    if no_outgoing.any():
        transition_matrix[no_outgoing] = 1.0 / modes
    initial_counts = torch.bincount(
        sequence_labels[:, 0], minlength=modes
    ).to(torch.float64)
    initial_distribution = initial_counts / initial_counts.sum()
    return HardSwitchingSLDSBaseline(
        projection_mean=projection_mean,
        projection_components=components,
        centroids=centroids,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=torch.stack(weights),
        transition_matrix=transition_matrix,
        initial_distribution=initial_distribution,
        cluster_counts=counts,
        requested_rank=rank,
        effective_rank=effective_rank,
        ridge=ridge,
        seed=seed,
        kmeans_iterations=iterations,
        transition_smoothing=transition_smoothing,
    )


def fit_hard_switching_slds_baseline_streaming(
    batches: StateBatchFactory,
    modes: int = 4,
    rank: int = 40,
    ridge: float = 1e-3,
    max_iterations: int = 25,
    seed: int = 0,
    transition_smoothing: float = 1.0,
    calculation_device: torch.device | str = "cpu",
) -> HardSwitchingSLDSBaseline:
    """Fit the full-state hard-switching SLDS with bounded host memory."""
    if modes <= 0 or rank <= 0 or max_iterations <= 0:
        raise ValueError("modes, rank, and max_iterations must be positive.")
    if ridge < 0 or transition_smoothing < 0:
        raise ValueError("ridge and transition_smoothing must be non-negative.")
    device = torch.device(calculation_device)
    value_sum: torch.Tensor | None = None
    value_square_sum: torch.Tensor | None = None
    second_moment: torch.Tensor | None = None
    row_count = 0
    query_count = 0
    transitions: int | None = None
    hidden_size: int | None = None
    for states in batches():
        _validate_states(states)
        if transitions is None:
            transitions = int(states.shape[1] - 1)
            hidden_size = int(states.shape[-1])
            value_sum = torch.zeros(hidden_size, dtype=torch.float64, device=device)
            value_square_sum = torch.zeros_like(value_sum)
            second_moment = torch.zeros(
                hidden_size, hidden_size, dtype=torch.float64, device=device
            )
        elif transitions != states.shape[1] - 1 or hidden_size != states.shape[-1]:
            raise ValueError("Streaming SLDS batch dimensions changed.")
        current = states[:, :-1].reshape(-1, states.shape[-1]).to(
            device=device, dtype=torch.float64
        )
        assert value_sum is not None and value_square_sum is not None
        assert second_moment is not None
        value_sum.add_(current.sum(dim=0))
        value_square_sum.add_(current.square().sum(dim=0))
        second_moment.addmm_(current.T, current)
        row_count += int(current.shape[0])
        query_count += int(states.shape[0])
    if (
        value_sum is None
        or value_square_sum is None
        or second_moment is None
        or hidden_size is None
        or transitions is None
        or row_count == 0
    ):
        raise ValueError("Streaming SLDS received no training states.")
    projection_mean = value_sum / row_count
    covariance = (
        second_moment - row_count * torch.outer(projection_mean, projection_mean)
    ) / max(row_count - 1, 1)
    effective_rank = min(rank, row_count, hidden_size)
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = _canonicalize_component_signs(
        eigenvectors[:, -effective_rank:].flip(dims=(1,))
    )
    del covariance, second_moment, eigenvectors

    projected_chunks: list[torch.Tensor] = []
    projected_rows = 0
    for states in batches():
        current = states[:, :-1].reshape(-1, states.shape[-1]).to(
            device=device, dtype=torch.float64
        )
        projected = (current - projection_mean) @ components
        projected_chunks.append(projected.cpu())
        projected_rows += int(projected.shape[0])
    if projected_rows != row_count:
        raise ValueError("Streaming SLDS batch factory changed between passes.")
    projected_all = torch.cat(projected_chunks)
    centroids_cpu, labels, iterations = _fit_kmeans(
        projected_all, modes=modes, max_iterations=max_iterations, seed=seed
    )
    del projected_chunks, projected_all

    feature_mean = projection_mean
    feature_variance = (
        value_square_sum / row_count - feature_mean.square()
    ).clamp_min(0.0)
    feature_scale = feature_variance.sqrt().clamp_min(1e-8)
    normal = torch.zeros(
        modes,
        hidden_size + 1,
        hidden_size + 1,
        dtype=torch.float64,
        device=device,
    )
    right = torch.zeros(
        modes,
        hidden_size + 1,
        hidden_size,
        dtype=torch.float64,
        device=device,
    )
    offset = 0
    for states in batches():
        values = states.to(device=device, dtype=torch.float64)
        current = values[:, :-1].reshape(-1, hidden_size)
        target = (values[:, 1:] - values[:, :-1]).reshape(-1, hidden_size)
        batch_labels = labels[offset : offset + current.shape[0]].to(device)
        standardized = (current - feature_mean) / feature_scale
        design = torch.cat(
            (
                standardized,
                torch.ones(
                    standardized.shape[0], 1, dtype=torch.float64, device=device
                ),
            ),
            dim=1,
        )
        for mode in range(modes):
            selected = batch_labels == mode
            if selected.any():
                selected_design = design[selected]
                selected_target = target[selected]
                normal[mode].addmm_(selected_design.T, selected_design)
                right[mode].addmm_(selected_design.T, selected_target)
        offset += int(current.shape[0])
    if offset != row_count:
        raise ValueError("Streaming SLDS batch factory changed during regression.")
    penalty = torch.eye(hidden_size + 1, dtype=torch.float64, device=device) * ridge
    penalty[-1, -1] = 0.0
    fallback_weight = _ridge_solution_from_statistics(
        normal.sum(dim=0), right.sum(dim=0), penalty
    )
    counts = torch.bincount(labels, minlength=modes)
    weights = torch.stack(
        [
            fallback_weight
            if int(counts[mode]) == 0
            else _ridge_solution_from_statistics(normal[mode], right[mode], penalty)
            for mode in range(modes)
        ]
    )

    sequence_labels = labels.reshape(query_count, transitions)
    transition_counts = torch.full(
        (modes, modes), transition_smoothing, dtype=torch.float64
    )
    flat_pairs = sequence_labels[:, :-1] * modes + sequence_labels[:, 1:]
    transition_counts.add_(
        torch.bincount(flat_pairs.reshape(-1), minlength=modes * modes).reshape(
            modes, modes
        )
    )
    outgoing = transition_counts.sum(dim=1, keepdim=True)
    transition_matrix = transition_counts / outgoing.clamp_min(1.0)
    no_outgoing = outgoing.squeeze(1) == 0
    if no_outgoing.any():
        transition_matrix[no_outgoing] = 1.0 / modes
    initial_counts = torch.bincount(
        sequence_labels[:, 0], minlength=modes
    ).to(torch.float64)
    initial_distribution = initial_counts / initial_counts.sum()
    return HardSwitchingSLDSBaseline(
        projection_mean=projection_mean,
        projection_components=components,
        centroids=centroids_cpu.to(device),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
        transition_matrix=transition_matrix.to(device),
        initial_distribution=initial_distribution.to(device),
        cluster_counts=counts.to(device),
        requested_rank=rank,
        effective_rank=effective_rank,
        ridge=ridge,
        seed=seed,
        kmeans_iterations=iterations,
        transition_smoothing=transition_smoothing,
    )


def guarded_state_only_rollout(
    baseline: LinesOfThoughtBaseline | HardSwitchingSLDSBaseline,
    initial_state: torch.Tensor,
    transitions: int,
    max_abs_state: float = 1e6,
) -> GuardedRollout:
    """Recursively apply a state-only baseline with finite-state guards."""
    if initial_state.ndim != 2:
        raise ValueError("initial_state must have shape [queries, state_dim].")
    if transitions <= 0 or max_abs_state <= 0:
        raise ValueError("transitions and max_abs_state must be positive.")
    if isinstance(baseline, LinesOfThoughtBaseline) and transitions > baseline.transitions:
        raise ValueError("Requested rollout exceeds the fitted LoT transitions.")
    current = initial_state
    predicted = [current]
    for layer in range(transitions):
        next_state = baseline.predict_layer(current, layer)
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
