from pathlib import Path

import numpy as np
import pytest
import torch

from agent_avenue.agents import GreedyHeuristicAgent, GreedyHeuristicConfig
from agent_avenue.engine import GameConfig
from agent_avenue.learning.checkpoint import CheckpointCompatibilityError, load_checkpoint
from agent_avenue.learning.dataset import materialize_dataset, save_dataset
from agent_avenue.learning.model import CandidateMLP
from agent_avenue.learning.structured_checkpoint import (
    StructuredCheckpointIntegrityError,
    load_structured_checkpoint,
    save_structured_checkpoint,
)
from agent_avenue.learning.structured_dataset import (
    load_structured_dataset,
    materialize_structured_dataset,
    save_structured_dataset,
)
from agent_avenue.learning.structured_model import create_structured_model
from agent_avenue.learning.structured_train import StructuredTrainingConfig, train_structured_model
from agent_avenue.runners import AgentSpec, GameSpec, run_game


def _records(count: int = 6):  # type: ignore[no-untyped-def]
    config = GreedyHeuristicConfig()
    spec = AgentSpec("greedy-public-v1", config.to_data(), GreedyHeuristicAgent)
    return tuple(
        run_game(
            GameSpec(
                "structured-test",
                f"game-{index}",
                None,
                GameConfig(),
                200 + index,
                (spec, spec),
                (300 + index, 400 + index),
            )
        )
        for index in range(count)
    )


def test_structured_materializer_preserves_v1_rows_splits_labels_and_provenance(
    tmp_path: Path,
) -> None:
    records = _records()
    v1 = materialize_dataset(records, split_seed=17)
    v1_path, _ = save_dataset(v1, tmp_path / "v1.npz")
    structured = materialize_structured_dataset(records, v1_dataset=v1_path)

    assert np.array_equal(structured.train.targets, v1.train.targets)
    assert np.array_equal(structured.validation.targets, v1.validation.targets)
    assert np.array_equal(structured.train.game_index, v1.train.game_index)
    assert np.array_equal(structured.validation.phase, v1.validation.phase)
    assert structured.train.features.shape[1] == 519
    assert all("policy" not in name for name in structured.manifest["feature_names"])
    assert structured.manifest["provenance"]["behavior_policy_id"] == "audit-only-not-feature"

    saved, _ = save_structured_dataset(structured, tmp_path / "structured.npz")
    loaded = load_structured_dataset(saved)
    assert loaded.fingerprint == structured.fingerprint
    assert np.array_equal(loaded.train.record_fingerprint, structured.train.record_fingerprint)


def test_structured_training_is_deterministic_cpu_smoke(tmp_path: Path) -> None:
    records = _records()
    v1 = materialize_dataset(records, split_seed=19)
    v1_path, _ = save_dataset(v1, tmp_path / "v1.npz")
    dataset = materialize_structured_dataset(records, v1_dataset=v1_path)
    config = StructuredTrainingConfig(
        seed=41,
        projection_seed=43,
        shuffle_seed=47,
        max_epochs=3,
        early_stopping_patience=2,
        batch_size=32,
    )
    q0 = CandidateMLP(seed=11).state_dict()
    first = train_structured_model(dataset, q0_state_dict=q0, config=config)
    second = train_structured_model(dataset, q0_state_dict=q0, config=config)

    assert first.best_epoch == second.best_epoch
    assert first.best_validation_loss == second.best_validation_loss
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.model.state_dict().values(), second.model.state_dict().values(), strict=True
        )
    )


def test_structured_checkpoint_round_trip_tamper_and_cross_version_rejection(
    tmp_path: Path,
) -> None:
    model = create_structured_model(CandidateMLP(seed=5), projection_seed=7)
    path = tmp_path / "structured-checkpoint"
    saved = save_structured_checkpoint(
        path,
        model,
        metrics={"validation": {"equal_game_loss": 0.5}},
        q0_parent_checkpoint_fingerprint="a" * 64,
        q0_parent_tensor_digest="b" * 64,
        dataset_fingerprint="c" * 64,
        source_corpus_fingerprints=("d" * 64,),
        split_identities={"train_pair_ids": "e" * 64, "validation_pair_ids": "f" * 64},
        training_config={"batch_size": 1024},
        training_seeds={"structured_init": 7, "shuffle": 9},
        created_at="2026-09-12T00:00:00+00:00",
    )
    loaded = load_structured_checkpoint(path)
    assert loaded.checkpoint_fingerprint == saved.checkpoint_fingerprint
    assert not loaded.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(path)

    (path / "metrics.json").write_text('{"validation":{"equal_game_loss":0.7}}\n')
    with pytest.raises(StructuredCheckpointIntegrityError, match=r"metrics\.json"):
        load_structured_checkpoint(path)
