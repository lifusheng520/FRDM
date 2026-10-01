"""Configuration for Fuzzy Reasoning Dynamics Modeling (FRDM)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


REASONING_MODES = (
    "knowledge_enrichment",
    "information_routing",
    "concept_composition",
    "prediction_refinement",
    "hop_transition",
)

FRDM_INPUT_SCHEMA = "FRDM_equations_4_to_8_tuned_lens_probabilities_v4"
FRDM_METHOD_ID = "FRDM_equations_4_to_19"
FRDM_CHECKPOINT_SCHEMA_VERSION = 3
FRDM_MANIFEST_SCHEMA_VERSION = 4


@dataclass
class FuzzyDynamicsConfig:
    """Dimensions and architectural choices for equations (4)--(18)."""

    hidden_size: int
    prediction_input_dim: int
    query_input_dim: int
    num_layers: int
    z_dim: int = 64
    concept_dim: int = 32
    prediction_dim: int = 32
    operation_dim: int = 32
    dynamics_dim: int = 64
    projector_hidden_dim: int = 128
    dynamics_hidden_dim: int = 128
    projector_dropout: float = 0.0
    dynamics_dropout: float = 0.1
    membership_sigma_min: float = 1e-4
    membership_epsilon: float = 1e-8
    dynamics_mixing: str = "fuzzy"
    homogeneous_local_dynamics: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_size",
            "prediction_input_dim",
            "query_input_dim",
            "num_layers",
            "z_dim",
            "concept_dim",
            "prediction_dim",
            "operation_dim",
            "dynamics_dim",
            "projector_hidden_dim",
            "dynamics_hidden_dim",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.dynamics_dim > self.hidden_size:
            raise ValueError("dynamics_dim cannot exceed hidden_size for truncated SVD.")
        if not 0.0 <= self.projector_dropout < 1.0:
            raise ValueError("projector_dropout must lie in [0, 1).")
        if not 0.0 <= self.dynamics_dropout < 1.0:
            raise ValueError("dynamics_dropout must lie in [0, 1).")
        if self.membership_sigma_min <= 0 or self.membership_epsilon <= 0:
            raise ValueError("Membership numerical constants must be positive.")
        if self.dynamics_mixing not in {"fuzzy", "hard", "uniform", "global"}:
            raise ValueError(
                "dynamics_mixing must be 'fuzzy', 'hard', 'uniform', or 'global'."
            )

    @property
    def state_dim(self) -> int:
        """Dimension of the descriptor state s_l=[z_l;c_l;b_l;u_l]."""
        return self.z_dim + self.concept_dim + self.prediction_dim + 1

    @property
    def num_modes(self) -> int:
        return len(REASONING_MODES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FuzzyDynamicsConfig":
        return cls(**values)


@dataclass
class LossConfig:
    """The three terms in the paper's equation (19)."""

    dynamics_weight: float = 1.0
    diversity_weight: float = 0.6
    sparsity_weight: float = 0.4

    def __post_init__(self) -> None:
        for name in (
            "dynamics_weight",
            "diversity_weight",
            "sparsity_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if abs(self.diversity_weight + self.sparsity_weight - 1.0) > 1e-8:
            raise ValueError("diversity_weight + sparsity_weight must equal 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LossConfig":
        return cls(**values)


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings reported in Appendix C.4."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1).")
        if not self.random_seeds:
            raise ValueError("random_seeds must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
