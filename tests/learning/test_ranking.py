from pathlib import Path

import numpy as np
import torch

from agent_avenue.agents import GreedyHeuristicAgent, GreedyHeuristicConfig
from agent_avenue.engine import GameConfig
from agent_avenue.learning import (
    RankingDataset,
    RankingSplit,
    RankingTrainingConfig,
    load_ranking_dataset,
    materialize_dataset,
    materialize_ranking_dataset,
    save_ranking_dataset,
    tensor_digest,
    train_ranking_model,
)
from agent_avenue.learning.model import create_model
from agent_avenue.runners import AgentSpec, GameSpec, run_game
from agent_avenue.storage import game_record_fingerprint


def _records(count: int = 4):  # type: ignore[no-untyped-def]
    config = GreedyHeuristicConfig()
    spec = AgentSpec("greedy-public-v1", config.to_data(), GreedyHeuristicAgent)
    return tuple(
        run_game(
            GameSpec(
                "ranking-test",
                f"game-{index}",
                None,
                GameConfig(),
                800 + index,
                (spec, spec),
                (900 + index, 1000 + index),
            )
        )
        for index in range(count)
    )


def test_ranking_materialization_is_deterministic_and_uses_mc_split(tmp_path: Path) -> None:
    records = _records()
    source = "a" * 64
    mc = materialize_dataset(records, split_seed=71, source_corpus_fingerprint=source)
    first = materialize_ranking_dataset(
        records,
        mc_manifest=mc.manifest,
        source_corpus_fingerprint=source,
    )
    second = materialize_ranking_dataset(
        tuple(reversed(records)),
        mc_manifest=mc.manifest,
        source_corpus_fingerprint=source,
    )

    assert first.fingerprint == second.fingerprint
    assert np.array_equal(first.train.candidate_features, second.train.candidate_features)
    assert (
        first.manifest["information_boundary"] == "PlayerObservation + legal semantic Action only"
    )
    assert first.manifest["teacher_semantics"].endswith("not-counterfactual-return")
    assert first.train.position_count == int(mc.train.targets.size)
    assert first.validation.position_count == int(mc.validation.targets.size)
    assert first.train.candidate_count >= first.train.position_count
    assert first.train.pair_count > 0
    assert set(first.manifest["split"]["train_game_fingerprints"]) == {
        game_record_fingerprint(record)
        for record in records
        if game_record_fingerprint(record) in set(mc.manifest["split"]["train_game_fingerprints"])
    }

    arrays, _ = save_ranking_dataset(first, tmp_path / "ranking.npz")
    loaded = load_ranking_dataset(arrays)
    assert loaded.fingerprint == first.fingerprint
    assert np.array_equal(
        loaded.validation.preferred_candidate_index, first.validation.preferred_candidate_index
    )


def _split(features: np.ndarray, games: np.ndarray) -> RankingSplit:
    position_count = features.shape[0] // 2
    return RankingSplit(
        features.astype(np.float32),
        np.arange(0, features.shape[0] + 1, 2, dtype=np.int64),
        np.arange(0, features.shape[0], 2, dtype=np.int64),
        np.arange(1, features.shape[0], 2, dtype=np.int64),
        np.arange(position_count + 1, dtype=np.int64),
        games.astype(np.int64),
        np.arange(position_count, dtype=np.int64),
        np.asarray(["play"] * position_count, dtype=np.str_),
    )


def test_ranking_training_is_bitwise_deterministic_and_preserves_global_rng() -> None:
    features = np.zeros((16, 87), dtype=np.float32)
    features[0::2, 0] = 1.0
    features[1::2, 0] = -1.0
    train = _split(features[:12], np.asarray([0, 0, 1, 1, 2, 2]))
    validation = _split(features[12:], np.asarray([3, 3]))
    dataset = RankingDataset(train, validation, {"dataset_fingerprint": "b" * 64})
    parent = create_model(seed=19)
    config = RankingTrainingConfig(
        seed=23,
        learning_rate=0.02,
        weight_decay=0.0,
        batch_positions=3,
        max_epochs=12,
        early_stopping_patience=4,
    )

    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = train_ranking_model(dataset, config=config, initial_state_dict=parent.state_dict())
    after = torch.random.get_rng_state().clone()
    second = train_ranking_model(dataset, config=config, initial_state_dict=parent.state_dict())

    assert torch.equal(before, after)
    assert tensor_digest(first.model.state_dict()) == tensor_digest(second.model.state_dict())
    assert first.history == second.history
    assert first.validation_metrics.pair_accuracy == 1.0
    assert first.validation_metrics.equal_game_loss < 0.7


def test_tie_only_positions_are_retained_but_not_supervised() -> None:
    features = np.zeros((6, 87), dtype=np.float32)
    split = RankingSplit(
        features,
        np.asarray([0, 2, 4, 6]),
        np.asarray([2]),
        np.asarray([3]),
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 0, 1]),
        np.asarray([0, 1, 0]),
        np.asarray(["play", "recruit", "play"]),
    )
    assert split.position_count == 3
    assert split.supervised_positions.tolist() == [1]
