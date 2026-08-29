import numpy as np
import torch

from agent_avenue.learning.checkpoint import tensor_digest
from agent_avenue.learning.train import TrainingConfig, TrainingData, train_model


def _fixture() -> tuple[TrainingData, TrainingData]:
    # A deliberately easy target determined by one safe feature. Each held-out game has a
    # different number of decisions so equal-game and pooled aggregation are both exercised.
    rng = np.random.default_rng(7)
    features = np.zeros((32, 87), dtype=np.float32)
    features[:, 0] = np.concatenate((np.full(16, -1.0), np.full(16, 1.0)))
    features[:, 1:] = rng.normal(0, 0.01, size=(32, 86)).astype(np.float32)
    targets = (features[:, 0] > 0).astype(np.float32)
    games = np.repeat(np.arange(8), 4)
    phases = np.where(np.arange(32) % 2, "play", "recruit")
    train = TrainingData(features[:24], targets[:24], games[:24], phases[:24])
    validation = TrainingData(features[24:], targets[24:], games[24:], phases[24:])
    return train, validation


def test_training_is_bitwise_deterministic_and_reports_metrics() -> None:
    train, validation = _fixture()
    config = TrainingConfig(
        seed=123,
        learning_rate=0.02,
        weight_decay=0.0,
        batch_size=8,
        max_epochs=15,
        early_stopping_patience=5,
    )
    first = train_model(train, validation, config=config)
    second = train_model(train, validation, config=config)

    assert tensor_digest(first.model.state_dict()) == tensor_digest(second.model.state_dict())
    assert first.history == second.history
    assert first.validation_metrics.accuracy == 1.0
    assert first.validation_metrics.game_count == 2
    assert len(first.validation_metrics.calibration) == 10
    assert {metric.phase for metric in first.validation_metrics.phase} == {"play", "recruit"}
    assert first.best_epoch <= first.epochs_completed <= config.max_epochs
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.model.state_dict().values(), second.model.state_dict().values(), strict=True
        )
    )


def test_early_stopping_uses_equal_game_validation_loss() -> None:
    train, validation = _fixture()
    result = train_model(
        train,
        validation,
        config=TrainingConfig(
            seed=9,
            learning_rate=1e-12,
            weight_decay=0.0,
            batch_size=32,
            max_epochs=20,
            early_stopping_patience=2,
            min_delta=1.0,
        ),
    )
    assert result.stopped_early
    assert result.epochs_completed == 3
    assert result.best_epoch == 1
