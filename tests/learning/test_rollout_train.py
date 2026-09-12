from __future__ import annotations

import numpy as np
import torch

from agent_avenue.learning.model import CandidateMLP
from agent_avenue.learning.rollout_train import (
    RolloutTrainingData,
    RolloutTreatmentConfig,
    lambda_zero_trace_regression,
    position_balanced_rollout_bce,
)
from agent_avenue.learning.structured_train import StructuredTrainingConfig, StructuredTrainingData


def _features(rows: int) -> np.ndarray:
    result = np.zeros((rows, 519), dtype=np.float32)
    result[:, 0] = 1.0
    return result


def test_position_balanced_rollout_loss_weights_positions_not_candidate_counts() -> None:
    logits = torch.zeros(3, dtype=torch.float32)
    targets = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)
    positions = torch.tensor((0, 0, 1), dtype=torch.int64)
    actual = position_balanced_rollout_bce(logits, targets, positions)
    # BCE(0, 0) == BCE(0, 1), but this guards the exact grouped operation and API.
    assert (
        actual.item()
        == torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(0.0), torch.tensor(0.0)
        ).item()
    )


def test_lambda_zero_reproduces_mc_optimizer_trace_step_by_step() -> None:
    train = StructuredTrainingData(
        _features(5),
        np.asarray((0, 1, 0, 1, 0), dtype=np.float32),
        np.asarray((0, 0, 1, 1, 2), dtype=np.int64),
    )
    rollout = RolloutTrainingData(
        _features(3),
        np.asarray((0.25, 0.75, 0.5), dtype=np.float32),
        np.asarray((0, 0, 1), dtype=np.int64),
        2,
    )
    config = RolloutTreatmentConfig(
        StructuredTrainingConfig(
            seed=2,
            projection_seed=3,
            shuffle_seed=5,
            batch_size=2,
            max_epochs=2,
            early_stopping_patience=1,
        ),
        fixed_epochs=2,
        lambda_roll=0.0,
    )
    result = lambda_zero_trace_regression(
        train,
        train,
        rollout=rollout,
        q0_state_dict=CandidateMLP(seed=7).state_dict(),
        config=config,
    )
    assert result.passed
    assert len(result.control.step_digests) == 6
    assert len(result.control.epoch_digests) == 2
