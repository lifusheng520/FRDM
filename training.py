"""Training loop for the FRDM experiment in Appendix C.4."""

from __future__ import annotations

import copy
import random
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from .config import FuzzyDynamicsConfig, LossConfig, TrainingConfig
from .losses import fuzzy_dynamics_loss
from .system import FuzzyReasoningDynamics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device, non_blocking=True)
        for name, value in batch.items()
    }


def _evaluate(
    model: FuzzyReasoningDynamics,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_config: LossConfig,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            loss, _ = fuzzy_dynamics_loss(
                model(_move_batch(batch, device)), loss_config
            )
            total += float(loss)
            batches += 1
    if batches == 0:
        raise ValueError("validation loader is empty.")
    return total / batches


def train_frdm(
    model: FuzzyReasoningDynamics,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    training_config: TrainingConfig | None = None,
    loss_config: LossConfig | None = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Train one FRDM seed and restore the best validation checkpoint."""
    training_config = training_config or TrainingConfig()
    loss_config = loss_config or LossConfig()
    device = torch.device(device)
    if not bool(model.low_rank.fitted):
        raise RuntimeError(
            "Fit the layer-specific SVD reference on the training split before training."
        )
    _seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(training_config.epochs):
        model.train()
        training_total = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = fuzzy_dynamics_loss(
                model(_move_batch(batch, device)), loss_config
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm
            )
            optimizer.step()
            training_total += float(loss.detach())
            batches += 1
        if batches == 0:
            raise ValueError("training loader is empty.")
        validation_loss = _evaluate(model, validation_loader, loss_config, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": training_total / batches,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_validation_loss": best_validation,
        "seed": seed,
        "training_config": training_config.to_dict(),
        "loss_config": loss_config.to_dict(),
    }


def train_six_seeds(
    model_factory: Callable[[], FuzzyReasoningDynamics],
    train_loader_factory: Callable[[int], DataLoader[dict[str, torch.Tensor]]],
    validation_loader_factory: Callable[[int], DataLoader[dict[str, torch.Tensor]]],
    training_config: TrainingConfig | None = None,
    loss_config: LossConfig | None = None,
    device: torch.device | str = "cpu",
) -> list[dict[str, Any]]:
    """Run the six-seed protocol reported in the paper."""
    training_config = training_config or TrainingConfig()
    return [
        train_frdm(
            model_factory(),
            train_loader_factory(seed),
            validation_loader_factory(seed),
            training_config,
            loss_config,
            device,
            seed,
        )
        for seed in training_config.random_seeds
    ]
