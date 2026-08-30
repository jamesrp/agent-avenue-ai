from __future__ import annotations

from pathlib import Path

import pytest

from agent_avenue.agents import RandomAgent, RandomAgentConfig
from agent_avenue.engine import GameConfig
from agent_avenue.runners import AgentSpec, GameSpec, run_game, run_resumable_corpus
from agent_avenue.runners.corpus import corpus_declaration
from agent_avenue.storage import (
    CorpusDeclaration,
    CorpusError,
    load_corpus,
    open_resumable_corpus,
)


def _specs(count: int = 4) -> tuple[GameSpec, ...]:
    config = RandomAgentConfig()
    first = AgentSpec("random-a", config.to_data(), RandomAgent)
    second = AgentSpec("random-b", config.to_data(), RandomAgent)
    return tuple(
        GameSpec(
            "resume-run",
            f"game-{index:06d}",
            None,
            GameConfig(),
            100 + index,
            (first, second),
            (200 + index, 300 + index),
        )
        for index in range(count)
    )


def _declaration(specs: tuple[GameSpec, ...]) -> CorpusDeclaration:
    declaration, _ = corpus_declaration(
        specs,
        behavior_policy="fixture-policy",
        root_seed=7,
        generation=1,
        configuration={"fixture": True},
    )
    return declaration


def test_interrupted_corpus_resumes_only_missing_games_and_matches_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _specs()
    resumed_path = tmp_path / "resumed"
    with open_resumable_corpus(resumed_path, _declaration(specs)) as corpus:
        corpus.append(run_game(specs[0]))
        corpus.append(run_game(specs[1]))
        assert corpus.missing_game_ids() == ("game-000002", "game-000003")

    import agent_avenue.runners.corpus as corpus_runner

    calls: list[str] = []
    original = corpus_runner.run_game

    def counting_run(spec: GameSpec):  # type: ignore[no-untyped-def]
        calls.append(spec.game_id)
        return original(spec)

    monkeypatch.setattr(corpus_runner, "run_game", counting_run)
    resumed = run_resumable_corpus(
        resumed_path,
        specs,
        behavior_policy="fixture-policy",
        root_seed=7,
        generation=1,
        configuration={"fixture": True},
    )
    assert calls == ["game-000002", "game-000003"]

    clean_path = tmp_path / "clean"
    clean = run_resumable_corpus(
        clean_path,
        specs,
        behavior_policy="fixture-policy",
        root_seed=7,
        generation=1,
        configuration={"fixture": True},
    )
    assert resumed.corpus_fingerprint == clean.corpus_fingerprint
    assert resumed.record_fingerprints == clean.record_fingerprints
    assert not (resumed_path / "staging").exists()
    assert not (resumed_path / "progress.json").exists()
    assert [record.game_id for record in load_corpus(resumed_path)[1]] == [
        spec.game_id for spec in specs
    ]


def test_resume_rejects_changed_declaration_and_corrupt_shard(tmp_path: Path) -> None:
    specs = _specs(2)
    path = tmp_path / "corpus"
    with open_resumable_corpus(path, _declaration(specs)) as corpus:
        corpus.append(run_game(specs[0]))

    stale = path / "staging" / ".game-000001.json.gz.tmp-crash"
    stale.write_bytes(b"partial")
    with open_resumable_corpus(path, _declaration(specs)) as corpus:
        assert corpus.missing_game_ids() == ("game-000001",)
    assert not stale.exists()

    changed = CorpusDeclaration(
        "resume-run",
        "different-policy",
        tuple(spec.game_id for spec in specs),
        root_seed=7,
        generation=1,
        configuration={"fixture": True},
    )
    with pytest.raises(CorpusError, match="declaration mismatch"):
        open_resumable_corpus(path, changed)

    changed_specs = list(specs)
    changed_specs[1] = GameSpec(
        "resume-run",
        specs[1].game_id,
        None,
        GameConfig(),
        999,
        specs[1].seats,
        specs[1].agent_seeds,
    )
    with pytest.raises(CorpusError, match="declaration mismatch"):
        run_resumable_corpus(
            path,
            changed_specs,
            behavior_policy="fixture-policy",
            root_seed=7,
            generation=1,
            configuration={"fixture": True},
        )

    (path / "staging" / "game-000000.json.gz").write_bytes(b"not gzip")
    with (
        open_resumable_corpus(path, _declaration(specs)) as corpus,
        pytest.raises(CorpusError, match="verify corpus shard"),
    ):
        corpus.missing_game_ids()
