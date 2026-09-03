"""Offline chosen-action outcome evaluation over retained game records."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from agent_avenue.storage import GameRecord, game_record_fingerprint

from .dataset import extract_game_samples
from .model import CandidateMLP
from .train import BinaryMetrics, TrainingData, evaluate_model


class RecordEvaluationError(ValueError):
    """Raised when retained records cannot form a valid offline evaluation set."""


@dataclass(frozen=True, slots=True)
class GamePredictionMetrics:
    record_fingerprint: str
    game_id: str
    pair_id: str | None
    decision_count: int
    log_loss: float
    brier_score: float
    accuracy: float
    outcome_balance: float

    def to_data(self) -> dict[str, object]:
        return {
            "record_fingerprint": self.record_fingerprint,
            "game_id": self.game_id,
            "pair_id": self.pair_id,
            "decision_count": self.decision_count,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "outcome_balance": self.outcome_balance,
        }


@dataclass(frozen=True, slots=True)
class RecordEvaluation:
    metrics: BinaryMetrics
    games: tuple[GamePredictionMetrics, ...]
    behavior_policy_decisions: dict[str, int]

    def to_data(self, *, include_games: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "chosen_action_outcome_metrics": self.metrics.to_dict(),
            "behavior_policy_decisions": dict(sorted(self.behavior_policy_decisions.items())),
        }
        if include_games:
            data["games"] = [game.to_data() for game in self.games]
        return data


def evaluate_records(
    model: CandidateMLP,
    records: Iterable[GameRecord],
    *,
    verify_code: bool = True,
    calibration_bins: int = 10,
) -> RecordEvaluation:
    """Score selected actions from verified records under actor-relative terminal labels."""
    buffered = tuple(records)
    if not buffered:
        raise RecordEvaluationError("record evaluation requires at least one game")

    features: list[tuple[float, ...]] = []
    targets: list[float] = []
    game_indices: list[int] = []
    phases: list[str] = []
    policy_counts = Counter[str]()
    boundaries: list[tuple[int, int]] = []
    offset = 0
    for game_index, record in enumerate(buffered):
        samples = extract_game_samples(record, verify_code=verify_code)
        if not samples:
            raise RecordEvaluationError("evaluation records must contain decisions")
        features.extend(sample.features for sample in samples)
        targets.extend(sample.target for sample in samples)
        game_indices.extend([game_index] * len(samples))
        phases.extend(sample.phase.value for sample in samples)
        policy_counts.update(sample.behavior_policy_id for sample in samples)
        boundaries.append((offset, offset + len(samples)))
        offset += len(samples)

    data = TrainingData(
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(game_indices, dtype=np.int64),
        np.asarray(phases, dtype=np.str_),
    )
    metrics = evaluate_model(model, data, calibration_bins=calibration_bins)
    feature_tensor = torch.tensor(data.features, dtype=torch.float32, device="cpu")
    target_tensor = torch.tensor(data.targets, dtype=torch.float32, device="cpu")
    model.eval()
    with torch.inference_mode():
        logits = model(feature_tensor)
        losses = F.binary_cross_entropy_with_logits(logits, target_tensor, reduction="none")
        probabilities = torch.sigmoid(logits)
    loss_values = losses.cpu().numpy().astype(np.float64, copy=False)
    probability_values = probabilities.cpu().numpy().astype(np.float64, copy=False)
    target_values = data.targets.astype(np.float64, copy=False)

    game_metrics: list[GamePredictionMetrics] = []
    for record, (start, end) in zip(buffered, boundaries, strict=True):
        game_losses = loss_values[start:end]
        game_probabilities = probability_values[start:end]
        game_targets = target_values[start:end]
        log_loss = float(game_losses.mean())
        brier = float(np.square(game_probabilities - game_targets).mean())
        accuracy = float(((game_probabilities >= 0.5) == game_targets).mean())
        balance = float(game_targets.mean())
        if any(not math.isfinite(value) for value in (log_loss, brier, accuracy, balance)):
            raise RecordEvaluationError("model evaluation produced non-finite metrics")
        game_metrics.append(
            GamePredictionMetrics(
                game_record_fingerprint(record),
                record.game_id,
                record.pair_id,
                record.decision_count,
                log_loss,
                brier,
                accuracy,
                balance,
            )
        )
    return RecordEvaluation(metrics, tuple(game_metrics), dict(policy_counts))
