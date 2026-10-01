"""Layer-specific truncated-SVD coordinates from FRDM equations (14)--(15)."""

from __future__ import annotations

import torch
from torch import nn


class LayerwiseLowRankReference(nn.Module):
    """Frozen per-layer means, left singular vectors, and singular values."""

    def __init__(self, num_layers: int, hidden_size: int, dynamics_dim: int) -> None:
        super().__init__()
        if dynamics_dim > hidden_size:
            raise ValueError("dynamics_dim cannot exceed hidden_size.")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dynamics_dim = dynamics_dim
        self.register_buffer("means", torch.zeros(num_layers, hidden_size))
        self.register_buffer(
            "bases", torch.zeros(num_layers, hidden_size, dynamics_dim)
        )
        self.register_buffer("singular_values", torch.ones(num_layers, dynamics_dim))
        self.register_buffer(
            "effective_ranks", torch.zeros(num_layers, dtype=torch.long)
        )
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(
        self,
        hidden_trajectories: torch.Tensor,
        indices: torch.Tensor | None = None,
        max_samples: int = 8192,
    ) -> None:
        """Fit only on training trajectories shaped ``[N, L+1, d]``."""
        expected_tail = (self.num_layers + 1, self.hidden_size)
        if hidden_trajectories.ndim != 3 or hidden_trajectories.shape[1:] != expected_tail:
            raise ValueError(
                "hidden_trajectories must have shape "
                f"[N, {self.num_layers + 1}, {self.hidden_size}], got "
                f"{tuple(hidden_trajectories.shape)}."
            )
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        if indices is None:
            indices = torch.arange(
                min(hidden_trajectories.shape[0], max_samples), dtype=torch.long
            )
        elif indices.numel() > max_samples:
            indices = indices[:max_samples]
        sample_count = indices.numel()
        if sample_count < 2:
            raise ValueError("Layer-specific SVD requires at least two trajectories.")
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("indices must be a one-dimensional long tensor.")
        if int(indices.min()) < 0 or int(indices.max()) >= hidden_trajectories.shape[0]:
            raise IndexError("SVD fitting indices are outside the activation cache.")
        indices = indices.cpu()
        for layer in range(self.num_layers):
            layer_values = hidden_trajectories[:, layer]
            if indices is not None:
                layer_values = layer_values.index_select(0, indices)
            layer_values = layer_values.detach().to(
                self.means.device, torch.float32
            )
            mean = layer_values.mean(dim=0)
            centered = layer_values - mean
            rank = min(
                self.dynamics_dim,
                centered.shape[0] - 1,
                self.hidden_size,
            )
            # H_l in the paper is [d, N], so its left singular vectors are the
            # right singular vectors of the conventional [N, d] data matrix.
            _, singular_values, right_vectors = torch.pca_lowrank(
                centered, q=rank, center=False, niter=3
            )
            self.means[layer].copy_(mean)
            self.bases[layer].zero_()
            self.singular_values[layer].fill_(1.0)
            self.bases[layer, :, :rank].copy_(right_vectors[:, :rank])
            self.singular_values[layer, :rank].copy_(
                singular_values[:rank].clamp_min(1e-6)
            )
            self.effective_ranks[layer] = rank
        self.fitted.fill_(True)

    def _check(self, values: torch.Tensor, final_dim: int) -> None:
        if not bool(self.fitted):
            raise RuntimeError("LayerwiseLowRankReference must be fitted first.")
        if values.ndim != 3 or values.shape[1] != self.num_layers:
            raise ValueError(
                f"Expected [batch, {self.num_layers}, {final_dim}], got "
                f"{tuple(values.shape)}."
            )
        if values.shape[-1] != final_dim:
            raise ValueError(f"Expected final dimension {final_dim}.")

    def reference(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return r_{i,l}=Sigma_l^-1 U_l^T(h_{i,l}-mean_l)."""
        self._check(hidden, self.hidden_size)
        centered = hidden.float() - self.means
        coordinates = torch.einsum("bld,lds->bls", centered, self.bases)
        return (coordinates / self.singular_values).to(hidden.dtype)

    def decode(self, transitions: torch.Tensor) -> torch.Tensor:
        """Return U_l Sigma_l Delta_{i,l} in the original hidden space."""
        self._check(transitions, self.dynamics_dim)
        scaled = transitions.float() * self.singular_values
        return torch.einsum("bls,lds->bld", scaled, self.bases).to(
            transitions.dtype
        )

    def reference_window(self, hidden: torch.Tensor, start_layer: int) -> torch.Tensor:
        """Apply consecutive layer references to a rollout-start axis."""
        if not bool(self.fitted):
            raise RuntimeError("LayerwiseLowRankReference must be fitted first.")
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Expected hidden states [batch, starts, hidden_size].")
        stop = start_layer + hidden.shape[1]
        if start_layer < 0 or stop > self.num_layers:
            raise IndexError("Reference window exceeds the fitted layers.")
        centered = hidden.float() - self.means[start_layer:stop]
        coordinates = torch.einsum(
            "bld,lds->bls", centered, self.bases[start_layer:stop]
        )
        return (
            coordinates / self.singular_values[start_layer:stop]
        ).to(hidden.dtype)

    def decode_window(
        self, transitions: torch.Tensor, start_layer: int
    ) -> torch.Tensor:
        """Decode consecutive layer transitions on a rollout-start axis."""
        if not bool(self.fitted):
            raise RuntimeError("LayerwiseLowRankReference must be fitted first.")
        if transitions.ndim != 3 or transitions.shape[-1] != self.dynamics_dim:
            raise ValueError("Expected transitions [batch, starts, dynamics_dim].")
        stop = start_layer + transitions.shape[1]
        if start_layer < 0 or stop > self.num_layers:
            raise IndexError("Decode window exceeds the fitted layers.")
        scaled = transitions.float() * self.singular_values[start_layer:stop]
        return torch.einsum(
            "bls,lds->bld", scaled, self.bases[start_layer:stop]
        ).to(transitions.dtype)

    def reference_at(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("LayerwiseLowRankReference must be fitted first.")
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected [batch, {self.hidden_size}] hidden states.")
        if not 0 <= layer < self.num_layers:
            raise IndexError(layer)
        centered = hidden.float() - self.means[layer]
        coordinates = torch.matmul(centered, self.bases[layer])
        return (coordinates / self.singular_values[layer]).to(hidden.dtype)

    def decode_at(self, transitions: torch.Tensor, layer: int) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("LayerwiseLowRankReference must be fitted first.")
        if transitions.ndim != 2 or transitions.shape[-1] != self.dynamics_dim:
            raise ValueError(f"Expected [batch, {self.dynamics_dim}] transitions.")
        if not 0 <= layer < self.num_layers:
            raise IndexError(layer)
        scaled = transitions.float() * self.singular_values[layer]
        return torch.matmul(scaled, self.bases[layer].T).to(transitions.dtype)
