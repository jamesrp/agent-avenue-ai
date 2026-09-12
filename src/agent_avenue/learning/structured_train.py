"""Deterministic CPU training for the structured residual model v2."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from .structured_model import (
    INPUT_WIDTH,
    StructuredModelError,
    StructuredResidualMLP,
    create_structured_model,
    validate_state_dict,
)
from .train import BinaryMetrics, CalibrationBin, EpochMetrics, PhaseMetrics


class StructuredTrainingError(ValueError):
    """Raised when v2 training inputs or deterministic settings are malformed."""


@dataclass(frozen=True, slots=True)
class StructuredTrainingConfig:
    """Fixed AdamW and CPU determinism contract for every Step-3 v2 fit."""

    seed: int = 0
    projection_seed: int | None = None
    shuffle_seed: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 50
    early_stopping_patience: int = 8
    min_delta: float = 0.0
    cpu_threads: int = 1
    calibration_bins: int = 10
    deterministic_algorithms: bool = True
    data_loader_workers: int = field(default=0, init=False)
    loss: str = field(default="binary-cross-entropy-with-logits", init=False)
    optimizer: str = field(default="adamw", init=False)
    gradient_clipping: str = field(default="disabled", init=False)

    def __post_init__(self) -> None:
        for name in ("seed", "projection_seed", "shuffle_seed"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 0 <= value < 2**63):
                raise StructuredTrainingError(f"{name} must be an integer in [0, 2**63)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise StructuredTrainingError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise StructuredTrainingError("weight_decay must be finite and non-negative")
        for name in ("batch_size", "max_epochs", "early_stopping_patience", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise StructuredTrainingError(f"{name} must be a positive integer")
        if type(self.calibration_bins) is not int or self.calibration_bins <= 0:
            raise StructuredTrainingError("calibration_bins must be a positive integer")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise StructuredTrainingError("min_delta must be finite and non-negative")
        if self.deterministic_algorithms is not True:
            raise StructuredTrainingError(
                "structured training requires deterministic_algorithms=True"
            )

    @property
    def effective_projection_seed(self) -> int:
        return self.seed if self.projection_seed is None else self.projection_seed

    @property
    def effective_shuffle_seed(self) -> int:
        return self.seed if self.shuffle_seed is None else self.shuffle_seed

    def normalized(self) -> dict[str, object]:
        result = asdict(self)
        result["projection_seed"] = self.effective_projection_seed
        result["shuffle_seed"] = self.effective_shuffle_seed
        return cast(dict[str, object], result)


@dataclass(frozen=True, slots=True)
class StructuredTrainingData:
    """Detached, validated v2 input arrays and complete-game grouping identifiers."""

    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    game_index: NDArray[np.int64]
    phase: NDArray[np.str_] | None = None

    def __post_init__(self) -> None:
        raw_features = np.asarray(self.features)
        raw_targets = np.asarray(self.targets)
        raw_games = np.asarray(self.game_index)
        if raw_games.dtype.kind not in "iu":
            raise StructuredTrainingError("game_index must use an integer dtype")
        features = np.asarray(raw_features, dtype=np.float32)
        targets = np.asarray(raw_targets, dtype=np.float32)
        games = np.asarray(raw_games, dtype=np.int64)
        phase = None if self.phase is None else np.asarray(self.phase, dtype=np.str_)
        count = features.shape[0] if features.ndim == 2 else 0
        if features.ndim != 2 or features.shape[1] != INPUT_WIDTH:
            raise StructuredTrainingError(f"features must have shape [N, {INPUT_WIDTH}]")
        if count == 0 or targets.shape != (count,) or games.shape != (count,):
            raise StructuredTrainingError(
                "features, targets, and game_index must be non-empty matching rows"
            )
        if phase is not None and phase.shape != (count,):
            raise StructuredTrainingError("phase must be one-dimensional and match features")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise StructuredTrainingError("features and targets must be finite")
        if not np.isin(targets, (0.0, 1.0)).all() or np.any(games < 0):
            raise StructuredTrainingError("targets must be binary and game indices non-negative")
        features = np.ascontiguousarray(features).copy()
        targets = np.ascontiguousarray(targets).copy()
        games = np.ascontiguousarray(games).copy()
        features.setflags(write=False)
        targets.setflags(write=False)
        games.setflags(write=False)
        if phase is not None:
            phase = np.ascontiguousarray(phase).copy()
            phase.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "game_index", games)
        object.__setattr__(self, "phase", phase)

    @property
    def sample_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def game_count(self) -> int:
        return int(np.unique(self.game_index).size)

    @classmethod
    def from_value(cls, value: object, *, name: str = "split") -> StructuredTrainingData:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                return cls(
                    value["features"], value["targets"], value["game_index"], value.get("phase")
                )
            except KeyError as exc:
                raise StructuredTrainingError(f"{name} mapping is missing {exc.args[0]!r}") from exc
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) in (3, 4)
        ):
            return cls(value[0], value[1], value[2], value[3] if len(value) == 4 else None)
        try:
            raw = cast(Any, value)
            return cls(
                raw.features,
                raw.targets,
                raw.game_index,
                getattr(raw, "phase", None),
            )
        except AttributeError as exc:
            raise StructuredTrainingError(
                f"{name} must expose features, targets, and game_index arrays"
            ) from exc


@dataclass(frozen=True, slots=True)
class StructuredTrainingResult:
    model: StructuredResidualMLP
    config: StructuredTrainingConfig
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    best_validation_loss: float
    train_metrics: BinaryMetrics
    validation_metrics: BinaryMetrics
    examples_per_second: float = field(compare=False)
    wall_clock_seconds: float = field(compare=False)

    def metrics_dict(self) -> dict[str, object]:
        return {
            "best_epoch": self.best_epoch,
            "epochs_completed": self.epochs_completed,
            "stopped_early": self.stopped_early,
            "best_validation_loss": self.best_validation_loss,
            "train": self.train_metrics.to_dict(),
            "validation": self.validation_metrics.to_dict(),
            "history": [asdict(epoch) for epoch in self.history],
            "runtime": {
                "examples_per_second": self.examples_per_second,
                "wall_clock_seconds": self.wall_clock_seconds,
            },
        }


def _split_pair(dataset: object) -> tuple[object, object]:
    if isinstance(dataset, Mapping) and "train" in dataset:
        for name in ("validation", "val"):
            if name in dataset:
                return dataset["train"], dataset[name]
    if hasattr(dataset, "train"):
        for name in ("validation", "val"):
            if hasattr(dataset, name):
                return dataset.train, getattr(dataset, name)
    raise StructuredTrainingError("validation data is required")


def _tensors(data: StructuredTrainingData) -> tuple[Tensor, Tensor]:
    return torch.tensor(data.features, dtype=torch.float32), torch.tensor(
        data.targets, dtype=torch.float32
    )


def _clone_state(model: StructuredResidualMLP) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@contextmanager
def _strict_cpu(config: StructuredTrainingConfig) -> Iterator[None]:
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
        torch.set_num_threads(prior_threads)


def _metrics(
    model: StructuredResidualMLP,
    data: StructuredTrainingData,
    features: Tensor,
    targets: Tensor,
    *,
    calibration_bins: int,
) -> BinaryMetrics:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
    loss_values = losses.cpu().numpy().astype(np.float64, copy=False)
    probabilities_values = probabilities.cpu().numpy().astype(np.float64, copy=False)
    target_values = data.targets.astype(np.float64, copy=False)
    game_losses = tuple(
        float(loss_values[data.game_index == game].mean()) for game in np.unique(data.game_index)
    )
    calibration: list[CalibrationBin] = []
    assignments = np.minimum(
        (probabilities_values * calibration_bins).astype(np.int64), calibration_bins - 1
    )
    for index in range(calibration_bins):
        selected = assignments == index
        calibration.append(
            CalibrationBin(
                index / calibration_bins,
                (index + 1) / calibration_bins,
                int(selected.sum()),
                float(probabilities_values[selected].mean()) if selected.any() else None,
                float(target_values[selected].mean()) if selected.any() else None,
            )
        )
    phase: list[PhaseMetrics] = []
    if data.phase is not None:
        for phase_name in np.unique(data.phase):
            selected = data.phase == phase_name
            phase_probs = probabilities_values[selected]
            phase_targets = target_values[selected]
            phase.append(
                PhaseMetrics(
                    str(phase_name),
                    int(selected.sum()),
                    float(loss_values[selected].mean()),
                    float(np.square(phase_probs - phase_targets).mean()),
                    float(((phase_probs >= 0.5) == phase_targets).mean()),
                    float(phase_targets.mean()),
                )
            )
    return BinaryMetrics(
        sample_count=data.sample_count,
        game_count=data.game_count,
        pooled_loss=float(loss_values.mean()),
        equal_game_loss=float(sum(game_losses) / len(game_losses)),
        brier_score=float(np.square(probabilities_values - target_values).mean()),
        accuracy=float(((probabilities_values >= 0.5) == target_values).mean()),
        outcome_balance=float(target_values.mean()),
        calibration=tuple(calibration),
        phase=tuple(phase),
    )


def evaluate_structured_model(
    model: StructuredResidualMLP, data: object, *, calibration_bins: int = 10
) -> BinaryMetrics:
    """Evaluate pooled/equal-game BCE, Brier, accuracy, calibration, and phase metrics."""
    split = StructuredTrainingData.from_value(data)
    features, targets = _tensors(split)
    return _metrics(model, split, features, targets, calibration_bins=calibration_bins)


def train_structured_model(
    train: object,
    validation: object | None = None,
    *,
    q0_state_dict: Mapping[str, object] | None = None,
    config: StructuredTrainingConfig | None = None,
    model: StructuredResidualMLP | None = None,
    initial_state_dict: Mapping[str, object] | None = None,
) -> StructuredTrainingResult:
    """Train a q0-embedded v2 model deterministically and restore best equal-game BCE."""
    config = config or StructuredTrainingConfig()
    if validation is None:
        train, validation = _split_pair(train)
    train_data = StructuredTrainingData.from_value(train, name="train")
    validation_data = StructuredTrainingData.from_value(validation, name="validation")
    if model is not None and (q0_state_dict is not None or initial_state_dict is not None):
        raise StructuredTrainingError("provide model or q0/initial state, not both")
    if model is None:
        if q0_state_dict is None:
            raise StructuredTrainingError("q0_state_dict is required to create a structured model")
        fitted = create_structured_model(
            q0_state_dict, projection_seed=config.effective_projection_seed
        )
    else:
        fitted = model
    if next(fitted.parameters()).device.type != "cpu":
        raise StructuredTrainingError("structured training requires a CPU model")
    if initial_state_dict is not None:
        try:
            validate_state_dict(initial_state_dict)
            fitted.load_state_dict(initial_state_dict, strict=True)
        except (StructuredModelError, RuntimeError) as exc:
            raise StructuredTrainingError("initial state is incompatible with v2") from exc
    train_features, train_targets = _tensors(train_data)
    validation_features, validation_targets = _tensors(validation_data)
    shuffle = torch.Generator(device="cpu")
    shuffle.manual_seed(config.effective_shuffle_seed)
    history: list[EpochMetrics] = []
    best_state: dict[str, Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    early_counter = 0
    examples = 0
    started = time.perf_counter()
    with _strict_cpu(config):
        optimizer = torch.optim.AdamW(
            fitted.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
            fused=False,
        )
        for epoch in range(1, config.max_epochs + 1):
            fitted.train()
            order = torch.randperm(train_data.sample_count, generator=shuffle)
            for start in range(0, train_data.sample_count, config.batch_size):
                indices = order[start : start + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    fitted(train_features[indices]), train_targets[indices]
                )
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                examples += int(indices.numel())
            train_metrics = _metrics(
                fitted,
                train_data,
                train_features,
                train_targets,
                calibration_bins=config.calibration_bins,
            )
            validation_metrics = _metrics(
                fitted,
                validation_data,
                validation_features,
                validation_targets,
                calibration_bins=config.calibration_bins,
            )
            improved = validation_metrics.equal_game_loss < best_loss - config.min_delta
            if improved:
                best_loss = validation_metrics.equal_game_loss
                best_epoch = epoch
                best_state = _clone_state(fitted)
                early_counter = 0
            else:
                early_counter += 1
            history.append(
                EpochMetrics(epoch, train_metrics, validation_metrics, improved, early_counter)
            )
            if early_counter >= config.early_stopping_patience:
                break
        if best_state is None:  # pragma: no cover - finite BCE always improves on epoch one
            raise RuntimeError("structured training did not produce a best model")
        fitted.load_state_dict(best_state, strict=True)
        fitted.eval()
        final_train = _metrics(
            fitted,
            train_data,
            train_features,
            train_targets,
            calibration_bins=config.calibration_bins,
        )
        final_validation = _metrics(
            fitted,
            validation_data,
            validation_features,
            validation_targets,
            calibration_bins=config.calibration_bins,
        )
    elapsed = time.perf_counter() - started
    return StructuredTrainingResult(
        fitted,
        config,
        tuple(history),
        best_epoch,
        len(history),
        len(history) < config.max_epochs,
        best_loss,
        final_train,
        final_validation,
        examples / elapsed if elapsed else math.inf,
        elapsed,
    )


train = train_structured_model

__all__ = [
    "StructuredTrainingConfig",
    "StructuredTrainingData",
    "StructuredTrainingError",
    "StructuredTrainingResult",
    "evaluate_structured_model",
    "train",
    "train_structured_model",
]
