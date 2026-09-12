"""Paired Step-4 MC-plus-rollout training primitives for the frozen structured-v2 model.

The treatment changes only the declared soft all-action auxiliary gradient.  It reuses the exact
CPU optimizer, MC shuffle generator, model, validation metric, and selected-action BCE from
``structured_train``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from agent_avenue.rollout.targets import RolloutTargetSet

from .structured_checkpoint import structured_tensor_digest
from .structured_model import (
    INPUT_WIDTH,
    StructuredResidualMLP,
    create_structured_model,
    validate_state_dict,
)
from .structured_train import (
    StructuredTrainingConfig,
    StructuredTrainingData,
    _clone_state,
    _metrics,
    _split_pair,
    _strict_cpu,
    _tensors,
)
from .train import BinaryMetrics, EpochMetrics

ROLLOUT_LOSS_VERSION = "position-balanced-soft-bce-v1"
ROLLOUT_SHARD_POSITIONS = 128
STEP4_TREATMENT_LAMBDA = 0.20
STEP4_REPLICATE_HORIZONS: dict[str, int] = {
    "replicate-1": 26,
    "replicate-2": 33,
    "replicate-3": 31,
}


class RolloutTrainingError(ValueError):
    """Raised when frozen rollout supervision/training inputs are malformed."""


@dataclass(frozen=True, slots=True)
class RolloutTrainingData:
    """Safe 519-vector rows grouped by immutable panel position, not action count."""

    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    position_index: NDArray[np.int64]
    position_count: int

    def __post_init__(self) -> None:
        raw_features = np.asarray(self.features)
        raw_targets = np.asarray(self.targets)
        raw_positions = np.asarray(self.position_index)
        if raw_positions.dtype.kind not in "iu":
            raise RolloutTrainingError("position_index must use an integer dtype")
        features = np.asarray(raw_features, dtype=np.float32)
        targets = np.asarray(raw_targets, dtype=np.float32)
        positions = np.asarray(raw_positions, dtype=np.int64)
        count = features.shape[0] if features.ndim == 2 else 0
        if (
            features.ndim != 2
            or features.shape[1] != INPUT_WIDTH
            or count == 0
            or targets.shape != (count,)
            or positions.shape != (count,)
        ):
            raise RolloutTrainingError("rollout rows must be aligned non-empty 519-value vectors")
        if type(self.position_count) is not int or self.position_count <= 0:
            raise RolloutTrainingError("position_count must be a positive integer")
        if (
            not np.isfinite(features).all()
            or not np.isfinite(targets).all()
            or np.any(targets < 0.0)
            or np.any(targets > 1.0)
            or np.any(positions < 0)
            or np.any(positions >= self.position_count)
            or set(positions.tolist()) != set(range(self.position_count))
        ):
            raise RolloutTrainingError(
                "rollout rows have invalid soft labels or incomplete positions"
            )
        features = np.ascontiguousarray(features).copy()
        targets = np.ascontiguousarray(targets).copy()
        positions = np.ascontiguousarray(positions).copy()
        features.setflags(write=False)
        targets.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "position_index", positions)

    @classmethod
    def from_target_set(cls, targets: RolloutTargetSet) -> RolloutTrainingData:
        position_by_id = {
            position.safe_identity: index for index, position in enumerate(targets.positions)
        }
        return cls(
            np.asarray([row.features for row in targets.rows], dtype=np.float32),
            np.asarray([row.target for row in targets.rows], dtype=np.float32),
            np.asarray([position_by_id[row.safe_identity] for row in targets.rows], dtype=np.int64),
            len(targets.positions),
        )

    def cyclic_position_shard(
        self, cursor: int, *, size: int = ROLLOUT_SHARD_POSITIONS
    ) -> tuple[int, ...]:
        """Return one complete cyclic position shard, never a candidate-row shard."""
        if type(cursor) is not int or cursor < 0:
            raise RolloutTrainingError("rollout cursor must be a non-negative integer")
        if type(size) is not int or size <= 0:
            raise RolloutTrainingError("rollout shard size must be a positive integer")
        return tuple((cursor + offset) % self.position_count for offset in range(size))


@dataclass(frozen=True, slots=True)
class RolloutTreatmentConfig:
    """The frozen auxiliary coefficient and matched fixed-horizon treatment schedule."""

    training: StructuredTrainingConfig
    fixed_epochs: int
    lambda_roll: float = STEP4_TREATMENT_LAMBDA
    rollout_shard_positions: int = ROLLOUT_SHARD_POSITIONS

    def __post_init__(self) -> None:
        if (
            type(self.fixed_epochs) is not int
            or not 1 <= self.fixed_epochs <= self.training.max_epochs
        ):
            raise RolloutTrainingError("fixed_epochs must be within the MC training maximum")
        if not math.isfinite(self.lambda_roll) or self.lambda_roll < 0.0:
            raise RolloutTrainingError("lambda_roll must be finite and non-negative")
        if self.rollout_shard_positions != ROLLOUT_SHARD_POSITIONS:
            raise RolloutTrainingError("Step-4 rollout shards must contain exactly 128 positions")


def step4_fixed_horizon(replicate_id: str) -> int:
    """Return the retained Step-3 M optimizer horizon for one matched replicate."""
    try:
        return STEP4_REPLICATE_HORIZONS[replicate_id]
    except KeyError as exc:
        raise RolloutTrainingError("replicate has no frozen Step-4 treatment horizon") from exc


@dataclass(frozen=True, slots=True)
class OptimizerTrace:
    """Canonical tensor digests captured after every optimizer step and epoch."""

    step_digests: tuple[str, ...]
    epoch_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RolloutTrainingResult:
    """Treatment result selected by unchanged MC equal-game validation BCE."""

    model: StructuredResidualMLP
    config: RolloutTreatmentConfig
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_validation_loss: float
    train_metrics: BinaryMetrics
    validation_metrics: BinaryMetrics
    trace: OptimizerTrace
    examples_per_second: float
    wall_clock_seconds: float

    @property
    def epochs_completed(self) -> int:
        return self.config.fixed_epochs


def position_balanced_rollout_bce(
    logits: Tensor,
    targets: Tensor,
    position_index: Tensor,
    *,
    position_order: tuple[int, ...] | None = None,
) -> Tensor:
    """Compute mean_position(mean_candidate(soft BCE)) exactly as frozen for Step-4."""
    if logits.ndim != 1 or targets.shape != logits.shape or position_index.shape != logits.shape:
        raise RolloutTrainingError("rollout logits, targets, and positions must be aligned vectors")
    if logits.numel() == 0 or position_index.dtype != torch.int64:
        raise RolloutTrainingError("rollout loss requires non-empty int64 position groups")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    groups = (
        torch.tensor(position_order, dtype=torch.int64)
        if position_order is not None
        else torch.unique(position_index, sorted=True)
    )
    if groups.numel() == 0:
        raise RolloutTrainingError("rollout loss requires at least one position group")
    return torch.stack([losses[position_index == group].mean() for group in groups]).mean()


def _rollout_tensors(data: RolloutTrainingData) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.tensor(data.features, dtype=torch.float32),
        torch.tensor(data.targets, dtype=torch.float32),
        torch.tensor(data.position_index, dtype=torch.int64),
    )


def _rollout_shard_indices(positions: Tensor, selected_positions: tuple[int, ...]) -> Tensor:
    return torch.cat(
        tuple(
            torch.nonzero(positions == position, as_tuple=False).squeeze(1)
            for position in selected_positions
        )
    )


def _initial_model(
    *,
    q0_state_dict: Mapping[str, object],
    config: StructuredTrainingConfig,
    initial_state_dict: Mapping[str, object] | None,
) -> StructuredResidualMLP:
    model = create_structured_model(q0_state_dict, projection_seed=config.effective_projection_seed)
    if initial_state_dict is not None:
        try:
            validate_state_dict(initial_state_dict)
            model.load_state_dict(initial_state_dict, strict=True)
        except (ValueError, RuntimeError) as exc:
            raise RolloutTrainingError("initial structured state is incompatible") from exc
    return model


def train_rollout_treatment(
    train: object,
    validation: object | None = None,
    *,
    rollout: RolloutTrainingData,
    q0_state_dict: Mapping[str, object],
    config: RolloutTreatmentConfig,
    initial_state_dict: Mapping[str, object] | None = None,
) -> RolloutTrainingResult:
    """Train the matched Step-4 treatment for a fixed MC horizon and select by MC validation."""
    if validation is None:
        train, validation = _split_pair(train)
    train_data = StructuredTrainingData.from_value(train, name="train")
    validation_data = StructuredTrainingData.from_value(validation, name="validation")
    model = _initial_model(
        q0_state_dict=q0_state_dict,
        config=config.training,
        initial_state_dict=initial_state_dict,
    )
    train_features, train_targets = _tensors(train_data)
    validation_features, validation_targets = _tensors(validation_data)
    rollout_features, rollout_targets, rollout_positions = _rollout_tensors(rollout)
    shuffle = torch.Generator(device="cpu")
    shuffle.manual_seed(config.training.effective_shuffle_seed)
    history: list[EpochMetrics] = []
    step_digests: list[str] = []
    epoch_digests: list[str] = []
    best_state: dict[str, Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    cursor = 0
    examples = 0
    started = time.perf_counter()
    with _strict_cpu(config.training):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            foreach=False,
            fused=False,
        )
        for epoch in range(1, config.fixed_epochs + 1):
            model.train()
            order = torch.randperm(train_data.sample_count, generator=shuffle)
            for start in range(0, train_data.sample_count, config.training.batch_size):
                indices = order[start : start + config.training.batch_size]
                optimizer.zero_grad(set_to_none=True)
                mc_loss = F.binary_cross_entropy_with_logits(
                    model(train_features[indices]), train_targets[indices]
                )
                if config.lambda_roll == 0.0:
                    loss = mc_loss
                else:
                    shard = rollout.cyclic_position_shard(
                        cursor, size=config.rollout_shard_positions
                    )
                    cursor = (cursor + config.rollout_shard_positions) % rollout.position_count
                    rollout_indices = _rollout_shard_indices(rollout_positions, shard)
                    roll_loss = position_balanced_rollout_bce(
                        model(rollout_features[rollout_indices]),
                        rollout_targets[rollout_indices],
                        rollout_positions[rollout_indices],
                        position_order=shard,
                    )
                    loss = mc_loss + config.lambda_roll * roll_loss
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                examples += int(indices.numel())
                step_digests.append(structured_tensor_digest(model.state_dict()))
            train_metrics = _metrics(
                model,
                train_data,
                train_features,
                train_targets,
                calibration_bins=config.training.calibration_bins,
            )
            validation_metrics = _metrics(
                model,
                validation_data,
                validation_features,
                validation_targets,
                calibration_bins=config.training.calibration_bins,
            )
            improved = validation_metrics.equal_game_loss < best_loss - config.training.min_delta
            if improved:
                best_loss = validation_metrics.equal_game_loss
                best_epoch = epoch
                best_state = _clone_state(model)
            history.append(EpochMetrics(epoch, train_metrics, validation_metrics, improved, 0))
            epoch_digests.append(structured_tensor_digest(model.state_dict()))
        if best_state is None:  # pragma: no cover - finite BCE improves on first epoch
            raise RuntimeError("rollout treatment did not produce a best model")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        final_train = _metrics(
            model,
            train_data,
            train_features,
            train_targets,
            calibration_bins=config.training.calibration_bins,
        )
        final_validation = _metrics(
            model,
            validation_data,
            validation_features,
            validation_targets,
            calibration_bins=config.training.calibration_bins,
        )
    elapsed = time.perf_counter() - started
    return RolloutTrainingResult(
        model,
        config,
        tuple(history),
        best_epoch,
        best_loss,
        final_train,
        final_validation,
        OptimizerTrace(tuple(step_digests), tuple(epoch_digests)),
        examples / elapsed if elapsed else math.inf,
        elapsed,
    )


def _reference_mc_trace(
    train: StructuredTrainingData,
    *,
    q0_state_dict: Mapping[str, object],
    training: StructuredTrainingConfig,
    epochs: int,
    initial_state_dict: Mapping[str, object] | None,
) -> OptimizerTrace:
    """Independent MC-only loop retained solely for epochwise lambda=0 regression checks."""
    model = _initial_model(
        q0_state_dict=q0_state_dict, config=training, initial_state_dict=initial_state_dict
    )
    features, targets = _tensors(train)
    shuffle = torch.Generator(device="cpu")
    shuffle.manual_seed(training.effective_shuffle_seed)
    steps: list[str] = []
    epoch_digests: list[str] = []
    with _strict_cpu(training):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
            foreach=False,
            fused=False,
        )
        for _ in range(epochs):
            model.train()
            order = torch.randperm(train.sample_count, generator=shuffle)
            for start in range(0, train.sample_count, training.batch_size):
                indices = order[start : start + training.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    model(features[indices]), targets[indices]
                )
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                steps.append(structured_tensor_digest(model.state_dict()))
            epoch_digests.append(structured_tensor_digest(model.state_dict()))
    return OptimizerTrace(tuple(steps), tuple(epoch_digests))


@dataclass(frozen=True, slots=True)
class LambdaZeroTraceRegression:
    """Epochwise/stepwise exact reproduction evidence for the no-auxiliary control path."""

    passed: bool
    control: OptimizerTrace
    treatment: OptimizerTrace
    first_step_mismatch: int | None
    first_epoch_mismatch: int | None


def lambda_zero_trace_regression(
    train: object,
    validation: object | None = None,
    *,
    rollout: RolloutTrainingData,
    q0_state_dict: Mapping[str, object],
    config: RolloutTreatmentConfig,
    initial_state_dict: Mapping[str, object] | None = None,
) -> LambdaZeroTraceRegression:
    """Compare independent MC-only and lambda=0 paths after every optimizer step and epoch."""
    if config.lambda_roll != 0.0:
        raise RolloutTrainingError("lambda-zero regression requires lambda_roll exactly zero")
    if validation is None:
        train, validation = _split_pair(train)
    train_data = StructuredTrainingData.from_value(train, name="train")
    control = _reference_mc_trace(
        train_data,
        q0_state_dict=q0_state_dict,
        training=config.training,
        epochs=config.fixed_epochs,
        initial_state_dict=initial_state_dict,
    )
    treatment = train_rollout_treatment(
        train_data,
        rollout=rollout,
        q0_state_dict=q0_state_dict,
        config=config,
        initial_state_dict=initial_state_dict,
        validation=StructuredTrainingData.from_value(validation, name="validation"),
    ).trace
    step_mismatch = next(
        (
            index
            for index, pair in enumerate(
                zip(control.step_digests, treatment.step_digests, strict=True)
            )
            if pair[0] != pair[1]
        ),
        None,
    )
    epoch_mismatch = next(
        (
            index
            for index, pair in enumerate(
                zip(control.epoch_digests, treatment.epoch_digests, strict=True)
            )
            if pair[0] != pair[1]
        ),
        None,
    )
    passed = (
        control.step_digests == treatment.step_digests
        and control.epoch_digests == treatment.epoch_digests
    )
    return LambdaZeroTraceRegression(passed, control, treatment, step_mismatch, epoch_mismatch)


__all__ = [
    "ROLLOUT_LOSS_VERSION",
    "ROLLOUT_SHARD_POSITIONS",
    "STEP4_REPLICATE_HORIZONS",
    "STEP4_TREATMENT_LAMBDA",
    "LambdaZeroTraceRegression",
    "OptimizerTrace",
    "RolloutTrainingData",
    "RolloutTrainingError",
    "RolloutTrainingResult",
    "RolloutTreatmentConfig",
    "lambda_zero_trace_regression",
    "position_balanced_rollout_bce",
    "step4_fixed_horizon",
    "train_rollout_treatment",
]
