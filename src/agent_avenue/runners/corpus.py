"""Restart-safe execution of declared game schedules into corpus storage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from agent_avenue.storage import (
    CorpusDeclaration,
    CorpusError,
    CorpusManifest,
    GameRecord,
    load_corpus,
    open_resumable_corpus,
)

from .game import GameSpec, run_game


def _spec_data(spec: GameSpec) -> dict[str, object]:
    from agent_avenue.engine.setup import normalize_config

    return {
        "run_id": spec.run_id,
        "game_id": spec.game_id,
        "pair_id": spec.pair_id,
        "game_config": normalize_config(spec.config),
        "setup_seed": spec.setup_seed,
        "seats": [
            {
                "agent_id": seat.agent_id,
                "config": json.loads(json.dumps(dict(seat.config), sort_keys=True)),
                "seed": spec.agent_seeds[index],
                "seed_derivation": spec.agent_seed_derivations[index],
            }
            for index, seat in enumerate(spec.seats)
        ],
    }


def _spec_fingerprint(spec: GameSpec) -> str:
    encoded = json.dumps(
        _spec_data(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _record_matches_spec(record: GameRecord, spec: GameSpec) -> bool:
    from agent_avenue.engine.setup import normalize_config

    return (
        record.run_id == spec.run_id
        and record.game_id == spec.game_id
        and record.pair_id == spec.pair_id
        and normalize_config(record.replay.config) == normalize_config(spec.config)
        and record.replay.seed == spec.setup_seed
        and all(
            record.seats[index].agent_id == spec.seats[index].agent_id
            and dict(record.seats[index].config) == dict(spec.seats[index].config)
            and record.seats[index].seed == spec.agent_seeds[index]
            and record.seats[index].seed_derivation == spec.agent_seed_derivations[index]
            for index in range(2)
        )
    )


def corpus_declaration(
    specs: Iterable[GameSpec],
    *,
    behavior_policy: str,
    root_seed: int | None,
    generation: int | None,
    configuration: Mapping[str, object],
) -> tuple[CorpusDeclaration, tuple[GameSpec, ...]]:
    """Freeze a complete game schedule into a resumable corpus declaration."""
    scheduled = tuple(specs)
    if not scheduled:
        raise CorpusError("resumable corpus schedule cannot be empty")
    run_ids = {spec.run_id for spec in scheduled}
    game_ids = tuple(spec.game_id for spec in scheduled)
    if len(run_ids) != 1 or len(set(game_ids)) != len(game_ids):
        raise CorpusError("corpus schedule requires one run id and unique game ids")
    schedule_fingerprints = [_spec_fingerprint(spec) for spec in scheduled]
    declared_configuration = {
        **json.loads(json.dumps(dict(configuration), sort_keys=True)),
        "schedule_fingerprints": schedule_fingerprints,
    }
    return (
        CorpusDeclaration(
            run_id=scheduled[0].run_id,
            behavior_policy=behavior_policy,
            expected_game_ids=game_ids,
            root_seed=root_seed,
            generation=generation,
            configuration=declared_configuration,
        ),
        scheduled,
    )


def run_resumable_corpus(
    directory: Path,
    specs: Iterable[GameSpec],
    *,
    behavior_policy: str,
    root_seed: int | None,
    generation: int | None,
    configuration: Mapping[str, object],
) -> CorpusManifest:
    """Run only missing scheduled games and atomically finalize their ordered corpus."""
    declaration, scheduled = corpus_declaration(
        specs,
        behavior_policy=behavior_policy,
        root_seed=root_seed,
        generation=generation,
        configuration=configuration,
    )
    game_ids = declaration.expected_game_ids
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest, records = load_corpus(directory)
        if manifest.declaration_fingerprint != declaration.fingerprint:
            raise CorpusError("completed corpus does not match the requested schedule")
        if tuple(record.game_id for record in records) != game_ids or any(
            not _record_matches_spec(record, spec)
            for record, spec in zip(records, scheduled, strict=True)
        ):
            raise CorpusError("completed corpus games do not match the requested schedule")
        return manifest

    by_id = {spec.game_id: spec for spec in scheduled}
    with open_resumable_corpus(directory, declaration) as corpus:
        for record in corpus.completed_records():
            if not _record_matches_spec(record, by_id[record.game_id]):
                raise CorpusError("staged corpus record does not match its scheduled game")
        for game_id in corpus.missing_game_ids():
            corpus.append(run_game(by_id[game_id]))
        return corpus.finalize()
