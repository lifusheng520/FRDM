"""End-to-end implementation of Fuzzy Reasoning Dynamics Modeling."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .config import FRDM_CHECKPOINT_SCHEMA_VERSION, FRDM_METHOD_ID, FuzzyDynamicsConfig
from .local_dynamics import GlobalReasoningDynamics, LocalReasoningDynamics
from .low_rank import LayerwiseLowRankReference
from .membership import GaussianMembership
from .projectors import ReasoningProjectors


SELECTIVE_INTERVENTIONS = (
    "remove_mlp",
    "remove_attention",
    "remove_concept",
    "remove_belief_uncertainty",
)


class FuzzyReasoningDynamics(nn.Module):
    """FRDM equations (4)--(18), with a frozen layer-specific SVD basis."""

    def __init__(self, config: FuzzyDynamicsConfig) -> None:
        super().__init__()
        self.config = config
        self.projectors = ReasoningProjectors(
            hidden_size=config.hidden_size,
            prediction_input_dim=config.prediction_input_dim,
            query_input_dim=config.query_input_dim,
            z_dim=config.z_dim,
            concept_dim=config.concept_dim,
            prediction_dim=config.prediction_dim,
            operation_dim=config.operation_dim,
            hidden_dim=config.projector_hidden_dim,
            dropout=config.projector_dropout,
        )
        self.local_dynamics = LocalReasoningDynamics(
            z_dim=config.z_dim,
            concept_dim=config.concept_dim,
            prediction_dim=config.prediction_dim,
            operation_dim=config.operation_dim,
            dynamics_dim=config.dynamics_dim,
            hidden_dim=config.dynamics_hidden_dim,
            dropout=config.dynamics_dropout,
            homogeneous=config.homogeneous_local_dynamics,
        )
        self.low_rank = LayerwiseLowRankReference(
            config.num_layers, config.hidden_size, config.dynamics_dim
        )
        self.membership = GaussianMembership(
            config.num_modes,
            sigma_min=config.membership_sigma_min,
            epsilon=config.membership_epsilon,
            initial_center=0.4 * math.sqrt(config.dynamics_dim),
            initial_spread=0.3 * math.sqrt(config.dynamics_dim),
        )
        self.global_dynamics = (
            GlobalReasoningDynamics(
                state_dim=config.state_dim,
                operation_dim=config.operation_dim,
                dynamics_dim=config.dynamics_dim,
                hidden_dim=config.dynamics_hidden_dim,
                dropout=config.dynamics_dropout,
            )
            if config.dynamics_mixing == "global"
            else None
        )

    @torch.no_grad()
    def fit_low_rank_reference(
        self,
        hidden_trajectories: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        self.low_rank.fit(hidden_trajectories, indices=indices)

    def _project_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state, z, concept, prediction, uncertainty = self.projectors.states(
            batch["hidden"],
            batch["prediction"],
            batch["query_embedding"],
        )
        attention, mlp = self.projectors.operations(
            batch["attention"], batch["mlp"]
        )
        return state, z, concept, prediction, uncertainty, attention, mlp

    def _compose(
        self,
        local_dynamics: torch.Tensor,
        memberships: torch.Tensor,
        mixing_weights: torch.Tensor,
        state: torch.Tensor,
        attention: torch.Tensor,
        mlp: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.dynamics_mixing == "global":
            assert self.global_dynamics is not None
            return self.global_dynamics(state, attention, mlp)
        if self.config.dynamics_mixing == "hard":
            indices = memberships.argmax(dim=-1, keepdim=True)
            indices = indices.unsqueeze(-1).expand(
                *indices.shape, local_dynamics.shape[-1]
            )
            return local_dynamics.gather(-2, indices).squeeze(-2)
        if self.config.dynamics_mixing == "uniform":
            return local_dynamics.mean(dim=-2)
        return (mixing_weights.unsqueeze(-1) * local_dynamics).sum(dim=-2)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        intervention: str | None = None,
    ) -> dict[str, Any]:
        if intervention is not None and intervention not in SELECTIVE_INTERVENTIONS:
            raise ValueError(
                f"Unknown intervention {intervention!r}; expected one of "
                f"{SELECTIVE_INTERVENTIONS}."
            )
        state, z, concept, prediction, uncertainty, attention, mlp = self._project_batch(
            batch
        )
        if batch["hidden"].shape[1] != self.config.num_layers + 1:
            raise ValueError("Activation cache layer count does not match model config.")

        if intervention == "remove_concept":
            concept = torch.zeros_like(concept)
        elif intervention == "remove_belief_uncertainty":
            prediction = torch.zeros_like(prediction)
        if intervention == "remove_attention":
            attention = torch.zeros_like(attention)
        elif intervention == "remove_mlp":
            mlp = torch.zeros_like(mlp)

        state = torch.cat((z, concept, prediction, uncertainty), dim=-1)
        current_state = state[:, :-1]
        local_dynamics = self.local_dynamics(
            z[:, :-1],
            concept[:, :-1],
            prediction[:, :-1],
            uncertainty[:, :-1],
            attention,
            mlp,
        )
        reference = self.low_rank.reference(batch["hidden"][:, :-1])
        memberships, mixing_weights, distances = self.membership(
            local_dynamics, reference
        )
        latent_transition = self._compose(
            local_dynamics,
            memberships,
            mixing_weights,
            current_state,
            attention,
            mlp,
        )
        predicted_delta = self.low_rank.decode(latent_transition)
        predicted_next = batch["hidden"][:, :-1] + predicted_delta
        target_next = batch["hidden"][:, 1:]
        concept_change = concept[:, 1:] - concept[:, :-1]

        outputs: dict[str, Any] = {
            "states": batch["hidden"],
            "reasoning_states": state,
            "z": z,
            "concept": concept,
            "prediction": prediction,
            "uncertainty": uncertainty,
            "attention": attention,
            "mlp": mlp,
            "concept_change": concept_change,
            "local_dynamics": local_dynamics,
            "local_deltas": local_dynamics,
            "low_rank_reference": reference,
            "membership_distances": distances,
            "memberships": memberships,
            "mixing_weights": mixing_weights,
            "latent_transition": latent_transition,
            "predicted_delta": predicted_delta,
            "target_delta": target_next - batch["hidden"][:, :-1],
            "predicted_next_hidden": predicted_next,
            "target_next_hidden": target_next,
            "membership_epsilon": self.config.membership_epsilon,
        }
        for name in ("bridge_logprob", "answer_logprob", "margin"):
            if name in batch:
                outputs[name] = batch[name]
        if intervention is not None:
            outputs["intervention"] = intervention
        return outputs

    def conditional_rollout(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Recursively predict h while conditioning on observed b, u, a, and m."""
        if batch["hidden"].shape[1] != self.config.num_layers + 1:
            raise ValueError("Activation cache layer count does not match model config.")
        attention, mlp = self.projectors.operations(batch["attention"], batch["mlp"])
        current = batch["hidden"][:, 0]
        predicted_states = [current]
        predicted_deltas = []
        rollout_memberships = []
        rollout_weights = []
        rollout_local_dynamics = []

        for layer in range(self.config.num_layers):
            state, z, concept, prediction, uncertainty = self.projectors.states(
                current.unsqueeze(1),
                batch["prediction"][:, layer : layer + 1],
                batch["query_embedding"],
            )
            local = self.local_dynamics(
                z,
                concept,
                prediction,
                uncertainty,
                attention[:, layer : layer + 1],
                mlp[:, layer : layer + 1],
            ).squeeze(1)
            reference = self.low_rank.reference_at(current, layer)
            memberships, weights, _ = self.membership(local, reference)
            latent_transition = self._compose(
                local,
                memberships,
                weights,
                state.squeeze(1),
                attention[:, layer],
                mlp[:, layer],
            )
            delta = self.low_rank.decode_at(latent_transition, layer)
            current = current + delta
            predicted_states.append(current)
            predicted_deltas.append(delta)
            rollout_memberships.append(memberships)
            rollout_weights.append(weights)
            rollout_local_dynamics.append(local)

        return {
            "true_states": batch["hidden"],
            "predicted_states": torch.stack(predicted_states, dim=1),
            "predicted_delta": torch.stack(predicted_deltas, dim=1),
            "memberships": torch.stack(rollout_memberships, dim=1),
            "mixing_weights": torch.stack(rollout_weights, dim=1),
            "local_dynamics": torch.stack(rollout_local_dynamics, dim=1),
            "local_deltas": torch.stack(rollout_local_dynamics, dim=1),
            "rollout_kind": "conditional_on_observed_belief_uncertainty_attention_and_mlp",
        }

    def conditional_horizon_rollout(
        self, batch: dict[str, torch.Tensor], horizon: int
    ) -> dict[str, torch.Tensor | int | str]:
        """Predict every observed start state forward by exactly ``horizon`` layers."""
        if horizon <= 0 or horizon > self.config.num_layers:
            raise ValueError(
                f"horizon must lie in [1, {self.config.num_layers}], got {horizon}."
            )
        starts = self.config.num_layers - horizon + 1
        endpoints = []
        attention, mlp = self.projectors.operations(batch["attention"], batch["mlp"])
        for start_layer in range(starts):
            current = batch["hidden"][:, start_layer]
            for step in range(horizon):
                layer = start_layer + step
                state, z, concept, prediction, uncertainty = self.projectors.states(
                    current.unsqueeze(1),
                    batch["prediction"][:, layer : layer + 1],
                    batch["query_embedding"],
                )
                local = self.local_dynamics(
                    z,
                    concept,
                    prediction,
                    uncertainty,
                    attention[:, layer : layer + 1],
                    mlp[:, layer : layer + 1],
                ).squeeze(1)
                reference = self.low_rank.reference_at(current, layer)
                memberships, weights, _ = self.membership(local, reference)
                latent_transition = self._compose(
                    local,
                    memberships,
                    weights,
                    state.squeeze(1),
                    attention[:, layer],
                    mlp[:, layer],
                )
                current = current + self.low_rank.decode_at(latent_transition, layer)
            endpoints.append(current)
        predicted_endpoints = torch.stack(endpoints, dim=1)
        return {
            "horizon": horizon,
            "predicted_endpoints": predicted_endpoints,
            "target_endpoints": batch["hidden"][:, horizon : horizon + starts],
            "rollout_kind": "sliding_conditional_on_observed_belief_uncertainty_attention_and_mlp",
        }

    def checkpoint(self, **metadata: Any) -> dict[str, Any]:
        return {
            "schema_version": FRDM_CHECKPOINT_SCHEMA_VERSION,
            "method": FRDM_METHOD_ID,
            "model_config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
            **metadata,
        }
