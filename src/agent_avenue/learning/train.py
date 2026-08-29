"""Deterministic CPU training and metrics for ``candidate-mlp-v1``."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from .model import INPUT_WIDTH, CandidateMLP, ModelError, create_model, validate_state_dict


class TrainingError(ValueError):
    """Raised when training configuration or materialized data is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Versioned baseline optimizer and determinism settings."""

    seed: int = 0
    model_seed: int | None = None
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
        for name in ("seed", "model_seed", "shuffle_seed"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 0 <= value < 2**63):
                raise TrainingError(f"{name} must be an integer in [0, 2**63)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise TrainingError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise TrainingError("weight_decay must be finite and non-negative")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise TrainingError("batch_size must be a positive integer")
        if type(self.max_epochs) is not int or self.max_epochs <= 0:
            raise TrainingError("max_epochs must be a positive integer")
        if type(self.early_stopping_patience) is not int or self.early_stopping_patience <= 0:
            raise TrainingError("early_stopping_patience must be a positive integer")
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise TrainingError("min_delta must be finite and non-negative")
        if type(self.cpu_threads) is not int or self.cpu_threads <= 0:
            raise TrainingError("cpu_threads must be a positive integer")
        if self.deterministic_algorithms is not True:
            raise TrainingError("strict baseline training requires deterministic_algorithms=True")
        if type(self.calibration_bins) is not int or self.calibration_bins <= 0:
            raise TrainingError("calibration_bins must be a positive integer")

    @property
    def effective_model_seed(self) -> int:
        return self.seed if self.model_seed is None else self.model_seed

    @property
    def effective_shuffle_seed(self) -> int:
        return self.seed if self.shuffle_seed is None else self.shuffle_seed

    def normalized(self) -> dict[str, object]:
        value = asdict(self)
        value["model_seed"] = self.effective_model_seed
        value["shuffle_seed"] = self.effective_shuffle_seed
        return value


@dataclass(frozen=True, slots=True)
class TrainingData:
    """One materialized train or validation split."""

    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    game_index: NDArray[np.int64]
    phase: NDArray[np.str_] | None = None

    def __post_init__(self) -> None:
        raw_features = np.asarray(self.features)
        raw_targets = np.asarray(self.targets)
        raw_game_index = np.asarray(self.game_index)
        if raw_game_index.dtype.kind not in "iu":
            raise TrainingError("game_index must use an integer dtype")
        features = np.asarray(raw_features, dtype=np.float32)
        targets = np.asarray(raw_targets, dtype=np.float32)
        game_index = np.asarray(raw_game_index, dtype=np.int64)
        phase = None if self.phase is None else np.asarray(self.phase, dtype=np.str_)

        if features.ndim != 2 or features.shape[1] != INPUT_WIDTH:
            raise TrainingError(f"features must have shape [N, {INPUT_WIDTH}]")
        count = features.shape[0]
        if count == 0:
            raise TrainingError("training splits must contain at least one sample")
        if targets.shape != (count,) or game_index.shape != (count,):
            raise TrainingError("targets and game_index must be one-dimensional and match features")
        if phase is not None and phase.shape != (count,):
            raise TrainingError("phase must be one-dimensional and match features")
        if not np.isfinite(features).all():
            raise TrainingError("features contain non-finite values")
        if not np.isfinite(targets).all() or not np.isin(targets, (0.0, 1.0)).all():
            raise TrainingError("targets must contain only finite binary values")
        if np.any(game_index < 0):
            raise TrainingError("game_index values must be non-negative")

        # Detach from mutable dataset buffers.  This makes a run insensitive to caller mutation.
        detached_features = np.ascontiguousarray(features).copy()
        detached_targets = np.ascontiguousarray(targets).copy()
        detached_game_index = np.ascontiguousarray(game_index).copy()
        detached_phase = None if phase is None else np.ascontiguousarray(phase).copy()
        for array in (detached_features, detached_targets, detached_game_index, detached_phase):
            if array is not None:
                array.setflags(write=False)
        object.__setattr__(self, "features", detached_features)
        object.__setattr__(self, "targets", detached_targets)
        object.__setattr__(self, "game_index", detached_game_index)
        object.__setattr__(self, "phase", detached_phase)

    @property
    def sample_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def game_count(self) -> int:
        return int(np.unique(self.game_index).size)

    @classmethod
    def from_value(cls, value: object, *, name: str = "split") -> TrainingData:
        """Adapt a dataset split exposing arrays by attributes, mapping, or a 3/4-tuple."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                return cls(
                    value["features"],
                    value["targets"],
                    value["game_index"],
                    value.get("phase"),
                )
            except KeyError as exc:
                raise TrainingError(f"{name} mapping is missing {exc.args[0]!r}") from exc
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) in (3, 4)
        ):
            phase = value[3] if len(value) == 4 else None
            return cls(value[0], value[1], value[2], phase)
        try:
            features = np.asarray(_attribute(value, "features"), dtype=np.float32)
            targets = np.asarray(_attribute(value, "targets"), dtype=np.float32)
            game_index = np.asarray(_attribute(value, "game_index"), dtype=np.int64)
            phase_value = _attribute(value, "phase", None)
            phase = None if phase_value is None else np.asarray(phase_value, dtype=np.str_)
            return cls(features, targets, game_index, phase)
        except AttributeError as exc:
            raise TrainingError(
                f"{name} must expose features, targets, and game_index arrays"
            ) from exc


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    outcome_rate: float | None


@dataclass(frozen=True, slots=True)
class PhaseMetrics:
    phase: str
    count: int
    pooled_loss: float
    brier_score: float
    accuracy: float
    outcome_balance: float


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    sample_count: int
    game_count: int
    pooled_loss: float
    equal_game_loss: float
    brier_score: float
    accuracy: float
    outcome_balance: float
    calibration: tuple[CalibrationBin, ...]
    phase: tuple[PhaseMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train: BinaryMetrics
    validation: BinaryMetrics
    improved: bool
    early_stop_counter: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: CandidateMLP
    config: TrainingConfig
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


_MISSING = object()


def _attribute(value: object, name: str, default: object = _MISSING) -> object:
    try:
        return getattr(value, name)
    except AttributeError:
        if default is _MISSING:
            raise
        return default


def _split_pair(dataset: object) -> tuple[object, object]:
    if isinstance(dataset, Mapping):
        for validation_name in ("validation", "val"):
            if "train" in dataset and validation_name in dataset:
                return dataset["train"], dataset[validation_name]
    for validation_name in ("validation", "val"):
        if hasattr(dataset, "train") and hasattr(dataset, validation_name):
            return _attribute(dataset, "train"), _attribute(dataset, validation_name)
    # Accommodate materializers that expose prefixed arrays on one object.
    if hasattr(dataset, "train_features"):
        train = (
            _attribute(dataset, "train_features"),
            _attribute(dataset, "train_targets"),
            _attribute(dataset, "train_game_index"),
            _attribute(dataset, "train_phase", None),
        )
        validation = (
            _attribute(dataset, "validation_features"),
            _attribute(dataset, "validation_targets"),
            _attribute(dataset, "validation_game_index"),
            _attribute(dataset, "validation_phase", None),
        )
        return train, validation
    raise TrainingError("validation data is required")


def _tensors(data: TrainingData) -> tuple[Tensor, Tensor]:
    # torch.from_numpy warns for read-only arrays, so copy into owned tensors explicitly.
    return torch.tensor(data.features, dtype=torch.float32), torch.tensor(
        data.targets, dtype=torch.float32
    )


def _clone_state(model: CandidateMLP) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@contextmanager
def _strict_cpu(config: TrainingConfig) -> Iterator[None]:
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(config.deterministic_algorithms, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


def _metrics(
    model: CandidateMLP,
    data: TrainingData,
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
    losses_array = losses.cpu().numpy().astype(np.float64, copy=False)
    probabilities_array = probabilities.cpu().numpy().astype(np.float64, copy=False)
    targets_array = data.targets.astype(np.float64, copy=False)

    game_losses = tuple(
        float(losses_array[data.game_index == game].mean()) for game in np.unique(data.game_index)
    )
    pooled_loss = float(losses_array.mean())
    brier = float(np.square(probabilities_array - targets_array).mean())
    accuracy = float(((probabilities_array >= 0.5) == targets_array).mean())
    balance = float(targets_array.mean())

    bins: list[CalibrationBin] = []
    assignments = np.minimum(
        (probabilities_array * calibration_bins).astype(np.int64), calibration_bins - 1
    )
    for index in range(calibration_bins):
        selected = assignments == index
        count = int(selected.sum())
        bins.append(
            CalibrationBin(
                lower=index / calibration_bins,
                upper=(index + 1) / calibration_bins,
                count=count,
                mean_probability=float(probabilities_array[selected].mean()) if count else None,
                outcome_rate=float(targets_array[selected].mean()) if count else None,
            )
        )

    phase_metrics: list[PhaseMetrics] = []
    if data.phase is not None:
        for phase in np.unique(data.phase):
            selected = data.phase == phase
            phase_probabilities = probabilities_array[selected]
            phase_targets = targets_array[selected]
            phase_metrics.append(
                PhaseMetrics(
                    phase=str(phase),
                    count=int(selected.sum()),
                    pooled_loss=float(losses_array[selected].mean()),
                    brier_score=float(np.square(phase_probabilities - phase_targets).mean()),
                    accuracy=float(((phase_probabilities >= 0.5) == phase_targets).mean()),
                    outcome_balance=float(phase_targets.mean()),
                )
            )

    return BinaryMetrics(
        sample_count=data.sample_count,
        game_count=data.game_count,
        pooled_loss=pooled_loss,
        equal_game_loss=float(sum(game_losses) / len(game_losses)),
        brier_score=brier,
        accuracy=accuracy,
        outcome_balance=balance,
        calibration=tuple(bins),
        phase=tuple(phase_metrics),
    )


def evaluate_model(
    model: CandidateMLP,
    data: object,
    *,
    calibration_bins: int = 10,
) -> BinaryMetrics:
    """Evaluate pooled, equal-game, calibration, outcome, and optional phase metrics."""
    split = TrainingData.from_value(data)
    features, targets = _tensors(split)
    return _metrics(model, split, features, targets, calibration_bins=calibration_bins)


def train_model(
    train: object,
    validation: object | None = None,
    *,
    config: TrainingConfig | None = None,
    model: CandidateMLP | None = None,
    initial_state_dict: Mapping[str, object] | None = None,
) -> TrainingResult:
    """Train deterministically on CPU and restore the best validation checkpoint.

    ``train`` and ``validation`` may be :class:`TrainingData`, mappings, objects exposing the three
    materialized arrays, or 3/4-tuples.  If ``validation`` is omitted, ``train`` may instead expose
    ``train`` and ``validation`` splits (or prefixed arrays), which keeps this API compatible with
    the adjacent dataset implementation without coupling to its concrete classes.
    """
    config = config or TrainingConfig()
    if validation is None:
        train, validation = _split_pair(train)
    train_data = TrainingData.from_value(train, name="train")
    validation_data = TrainingData.from_value(validation, name="validation")

    if model is not None and initial_state_dict is not None:
        raise TrainingError("provide either model or initial_state_dict, not both")
    fitted_model = model if model is not None else create_model(config.effective_model_seed)
    if next(fitted_model.parameters()).device.type != "cpu":
        raise TrainingError("strict baseline training requires a CPU model")
    if initial_state_dict is not None:
        try:
            validate_state_dict(initial_state_dict)
            fitted_model.load_state_dict(initial_state_dict, strict=True)
        except (ModelError, RuntimeError) as exc:
            raise TrainingError("initial state is incompatible with candidate-mlp-v1") from exc

    train_features, train_targets = _tensors(train_data)
    validation_features, validation_targets = _tensors(validation_data)
    shuffle = torch.Generator(device="cpu")
    shuffle.manual_seed(config.effective_shuffle_seed)

    history: list[EpochMetrics] = []
    best_state: dict[str, Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    early_stop_counter = 0
    examples_seen = 0
    started = time.perf_counter()

    with _strict_cpu(config):
        optimizer = torch.optim.AdamW(
            fitted_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
            fused=False,
        )
        for epoch in range(1, config.max_epochs + 1):
            fitted_model.train()
            order = torch.randperm(train_data.sample_count, generator=shuffle)
            for start in range(0, train_data.sample_count, config.batch_size):
                indices = order[start : start + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = fitted_model(train_features[indices])
                loss = F.binary_cross_entropy_with_logits(logits, train_targets[indices])
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                examples_seen += int(indices.numel())

            train_metrics = _metrics(
                fitted_model,
                train_data,
                train_features,
                train_targets,
                calibration_bins=config.calibration_bins,
            )
            validation_metrics = _metrics(
                fitted_model,
                validation_data,
                validation_features,
                validation_targets,
                calibration_bins=config.calibration_bins,
            )
            candidate_loss = validation_metrics.equal_game_loss
            improved = candidate_loss < best_loss - config.min_delta
            if improved:
                best_loss = candidate_loss
                best_epoch = epoch
                best_state = _clone_state(fitted_model)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            history.append(
                EpochMetrics(
                    epoch=epoch,
                    train=train_metrics,
                    validation=validation_metrics,
                    improved=improved,
                    early_stop_counter=early_stop_counter,
                )
            )
            if early_stop_counter >= config.early_stopping_patience:
                break

        if best_state is None:  # pragma: no cover - first finite validation loss always improves
            raise RuntimeError("training failed to produce a best model")
        fitted_model.load_state_dict(best_state, strict=True)
        fitted_model.eval()
        final_train_metrics = _metrics(
            fitted_model,
            train_data,
            train_features,
            train_targets,
            calibration_bins=config.calibration_bins,
        )
        final_validation_metrics = _metrics(
            fitted_model,
            validation_data,
            validation_features,
            validation_targets,
            calibration_bins=config.calibration_bins,
        )

    elapsed = time.perf_counter() - started
    return TrainingResult(
        model=fitted_model,
        config=config,
        history=tuple(history),
        best_epoch=best_epoch,
        epochs_completed=len(history),
        stopped_early=len(history) < config.max_epochs,
        best_validation_loss=best_loss,
        train_metrics=final_train_metrics,
        validation_metrics=final_validation_metrics,
        examples_per_second=examples_seen / elapsed if elapsed else math.inf,
        wall_clock_seconds=elapsed,
    )


# Conventional shorter spelling for script-level callers.
train = train_model
