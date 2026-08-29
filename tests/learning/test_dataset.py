from pathlib import Path

import numpy as np
import pytest

from agent_avenue.agents import GreedyHeuristicAgent, GreedyHeuristicConfig
from agent_avenue.engine import GameConfig
from agent_avenue.learning.dataset import (
    DatasetError,
    extract_game_samples,
    load_dataset,
    materialize_dataset,
    save_dataset,
)
from agent_avenue.runners import AgentSpec, GameSpec, run_game
from agent_avenue.storage import load_corpus, write_corpus


def _records(count: int = 4):  # type: ignore[no-untyped-def]
    config = GreedyHeuristicConfig()
    spec = AgentSpec("greedy-public-v1", config.to_data(), GreedyHeuristicAgent)
    return tuple(
        run_game(
            GameSpec(
                "dataset-test",
                f"game-{index}",
                None,
                GameConfig(),
                100 + index,
                (spec, spec),
                (200 + index, 300 + index),
            )
        )
        for index in range(count)
    )


def test_replay_extraction_is_ordered_safe_and_actor_relative() -> None:
    record = _records(1)[0]
    samples = extract_game_samples(record)
    assert len(samples) == record.decision_count
    assert [sample.decision_index for sample in samples] == list(range(record.decision_count))
    assert all(len(sample.features) == 87 for sample in samples)
    assert {sample.target for sample in samples} == {0.0, 1.0}
    assert all(sample.target == float(sample.actor is record.winner) for sample in samples)
    assert all(sample.winner is record.winner for sample in samples)
    assert all("seed" not in sample.__dataclass_fields__ for sample in samples)


def test_corpus_dataset_round_trip_and_group_split(tmp_path: Path) -> None:
    records = _records()
    corpus_path = tmp_path / "corpus"
    written = write_corpus(
        corpus_path,
        records,
        run_id="dataset-test",
        behavior_policy="greedy-public-v1",
        root_seed=9,
    )
    loaded_manifest, loaded_records = load_corpus(corpus_path)
    assert loaded_manifest.corpus_fingerprint == written.corpus_fingerprint
    assert loaded_records == records

    first = materialize_dataset(
        loaded_records,
        split_seed=17,
        source_corpus_fingerprint=written.corpus_fingerprint,
    )
    second = materialize_dataset(
        tuple(reversed(loaded_records)),
        split_seed=17,
        source_corpus_fingerprint=written.corpus_fingerprint,
    )
    assert first.fingerprint == second.fingerprint
    assert np.array_equal(first.train.features, second.train.features)
    train_games = set(first.manifest["split"]["train_game_fingerprints"])
    validation_games = set(first.manifest["split"]["validation_game_fingerprints"])
    assert train_games.isdisjoint(validation_games)

    arrays, _ = save_dataset(first, tmp_path / "dataset.npz")
    loaded = load_dataset(arrays)
    assert loaded.fingerprint == first.fingerprint
    assert np.array_equal(loaded.validation.targets, first.validation.targets)

    payload = bytearray(arrays.read_bytes())
    payload[-1] ^= 1
    arrays.write_bytes(payload)
    with pytest.raises(DatasetError, match="digest"):
        load_dataset(arrays)
