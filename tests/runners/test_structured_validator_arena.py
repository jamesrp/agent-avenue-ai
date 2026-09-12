"""Focused adversarial tests for the independent Step-3 arena validator."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_avenue.agents import (
    RNG_ALGORITHM,
    SEED_DERIVATION,
    RandomAgent,
    RandomAgentConfig,
    derive_seed,
)
from agent_avenue.engine import GameConfig, PlayerId
from agent_avenue.runners import (
    AgentSpec,
    ArenaConfig,
    arena_report_from_records,
    run_game,
    schedule_arena,
)
from agent_avenue.storage import AgentSeatRecord


def _validator_module() -> Any:
    path = Path("scripts/validate_structured_model_v2.py")
    spec = importlib.util.spec_from_file_location("structured_validator_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator() -> Any:
    return _validator_module()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _arena_fixture(validator: Any) -> dict[str, object]:
    cell = {
        "key": "C1-v2-vs-v1",
        "replicate_id": "replicate-1",
        "candidate_arm": "C",
        "candidate_kind": "v2",
        "opponent": "matched-v1",
        "paired_blocks": 1,
        "master_seed": 41,
        "run_id": "m7-structured-model-v2-replicate-1-C1-v2-vs-v1",
    }
    agent_a_id, agent_b_id, agent_a_rng, agent_b_rng = validator.expected_agent_id(cell)
    config = RandomAgentConfig().to_data()
    agent_a = AgentSpec(agent_a_id, config, RandomAgent, rng_identity=agent_a_rng)
    agent_b = AgentSpec(agent_b_id, config, RandomAgent, rng_identity=agent_b_rng)
    arena = ArenaConfig(
        cell["run_id"], agent_a, agent_b, cell["paired_blocks"], cell["master_seed"]
    )
    records = tuple(run_game(spec) for spec in schedule_arena(arena))
    report = arena_report_from_records(arena, records, elapsed_seconds=1.25).to_data()
    expected_configs = (dict(agent_a.config), dict(agent_b.config))
    return {
        "cell": cell,
        "records": records,
        "report": report,
        "expected_configs": expected_configs,
    }


def _reconstruct(
    validator: Any, fixture: dict[str, object], **overrides: object
) -> dict[str, object]:
    cell = overrides.get("cell", fixture["cell"])
    records = overrides.get("records", fixture["records"])
    retained = overrides.get("report", fixture["report"])
    expected_configs = overrides.get("expected_configs", fixture["expected_configs"])
    return validator.local_arena_report(cell, records, retained, expected_configs)


def test_local_arena_report_reconstructs_complete_report_byte_for_byte(
    validator: Any,
) -> None:
    fixture = _arena_fixture(validator)
    rebuilt = _reconstruct(validator, fixture)
    assert _canonical(rebuilt) == _canonical(fixture["report"])


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        (
            "paired outcome",
            lambda report: report["paired_seed_outcomes"][0].update(
                agent_a_wins=2 if report["paired_seed_outcomes"][0]["agent_a_wins"] != 2 else 1
            ),
        ),
        (
            "agent-a seat stats",
            lambda report: report["agent_a_by_seat"]["player_one"].update(wins=1),
        ),
        ("average score margin", lambda report: report.update(average_score_margin=99.0)),
        ("terminal reason counts", lambda report: report["terminal_reasons"].update(condition=99)),
        ("average turns", lambda report: report.update(average_turns=99.0)),
        ("average decisions", lambda report: report.update(average_decisions=99.0)),
    ],
)
def test_report_aggregate_mutations_are_rejected(
    validator: Any, field: str, mutate: Any
) -> None:
    fixture = _arena_fixture(validator)
    mutated = copy.deepcopy(fixture["report"])
    mutate(mutated)
    try:
        rebuilt = _reconstruct(validator, fixture, report=mutated)
    except validator.ValidationError:
        return
    assert _canonical(rebuilt) != _canonical(mutated), field


def test_record_winner_mutation_reconstructs_differently(validator: Any) -> None:
    fixture = _arena_fixture(validator)
    records = list(fixture["records"])
    records[0] = replace(records[0], winner=records[0].winner.other())
    rebuilt = _reconstruct(validator, fixture, records=tuple(records))
    assert _canonical(rebuilt) != _canonical(fixture["report"])


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("setup seed", lambda record: replace(record.replay, seed=record.replay.seed + 1)),
        ("pair ID", lambda record: replace(record, pair_id="pair-tampered")),
        ("game ID", lambda record: replace(record, game_id="game-tampered")),
        ("run ID", lambda record: replace(record, run_id="run-tampered")),
    ],
)
def test_record_schedule_identity_mutations_raise(
    validator: Any, field: str, mutate: Any
) -> None:
    fixture = _arena_fixture(validator)
    records = list(fixture["records"])
    mutated = mutate(records[0])
    if hasattr(mutated, "seed") and not hasattr(mutated, "replay"):
        mutated = replace(records[0], replay=mutated)
    records[0] = mutated
    with pytest.raises(validator.ValidationError):
        _reconstruct(validator, fixture, records=tuple(records))


@pytest.mark.parametrize(
    ("field", "mutate_seat"),
    [
        ("seat agent ID", lambda seat: replace(seat, agent_id="tampered-agent")),
        ("seat config", lambda seat: replace(seat, config={"type": "tampered"})),
        ("seat seed", lambda seat: replace(seat, seed=seat.seed + 1)),
        ("seed derivation", lambda seat: replace(seat, seed_derivation="tampered-derivation")),
        ("RNG domain", lambda seat: replace(seat, rng_domain="agent:tampered-domain")),
        ("RNG algorithm", lambda seat: replace(seat, rng_algorithm="tampered-rng")),
    ],
)
def test_record_seat_identity_mutations_raise(
    validator: Any, field: str, mutate_seat: Any
) -> None:
    fixture = _arena_fixture(validator)
    records = list(fixture["records"])
    record = records[0]
    records[0] = replace(record, seats=(mutate_seat(record.seats[0]), record.seats[1]))
    with pytest.raises(validator.ValidationError):
        _reconstruct(validator, fixture, records=tuple(records))


def test_record_and_report_policy_configs_are_not_taken_from_retained_report(
    validator: Any,
) -> None:
    fixture = _arena_fixture(validator)
    mutated = copy.deepcopy(fixture["report"])
    mutated["agents"]["a"]["config"] = {"type": "report-only-tamper"}
    rebuilt = _reconstruct(validator, fixture, report=mutated)
    assert rebuilt["agents"]["a"]["config"] == fixture["expected_configs"][0]
    assert _canonical(rebuilt) != _canonical(mutated)


def test_noncanonical_game_config_is_not_accepted_even_when_report_and_record_agree(
    validator: Any,
) -> None:
    fixture = _arena_fixture(validator)
    alternate = GameConfig(starting_player=PlayerId.PLAYER_TWO)
    mutated_records = tuple(
        replace(record, replay=replace(record.replay, config=alternate))
        for record in fixture["records"]
    )
    mutated_report = copy.deepcopy(fixture["report"])
    mutated_report["game_config"] = validator.normalize_config(alternate)
    with pytest.raises(validator.ValidationError, match="canonical"):
        _reconstruct(validator, fixture, records=mutated_records, report=mutated_report)


def _alignment_cell(validator: Any, key: str, arm: str | None, opponent: str) -> dict[str, object]:
    if key.endswith("v2-vs-v1"):
        shared = "architecture"
    else:
        shared = "heuristic" if opponent == "heuristic" else opponent
    return {
        "key": key,
        "replicate_id": "replicate-1",
        "candidate_arm": arm,
        "candidate_kind": "parent" if arm is None else "v2",
        "opponent": opponent,
        "paired_blocks": 1,
        "master_seed": 41,
        "run_id": f"m7-structured-model-v2-replicate-1-{key}",
        "shared_group": shared,
    }


def _aligned_records(validator: Any) -> dict[str, tuple[object, ...]]:
    config = RandomAgentConfig().to_data()
    base_cell = _alignment_cell(validator, "C1-v2-vs-v1", "C", "matched-v1")
    a_id, b_id, a_rng, b_rng = validator.expected_agent_id(base_cell)
    arena = ArenaConfig(
        "base",
        AgentSpec(a_id, config, RandomAgent, rng_identity=a_rng),
        AgentSpec(b_id, config, RandomAgent, rng_identity=b_rng),
        1,
        41,
    )
    base_records = tuple(run_game(spec) for spec in schedule_arena(arena))
    keys = {
        "C1-v2-vs-v1": ("C", "matched-v1"),
        "M1-v2-vs-v1": ("M", "matched-v1"),
        "C1-v2-vs-q0-parent": ("C", "q0-parent"),
        "M1-v2-vs-q0-parent": ("M", "q0-parent"),
        "C1-v2-vs-random": ("C", "random"),
        "M1-v2-vs-random": ("M", "random"),
        "C1-v2-vs-historical-q0": ("C", "historical-q0"),
        "M1-v2-vs-historical-q0": ("M", "historical-q0"),
        "C1-v2-vs-q1": ("C", "q1"),
        "M1-v2-vs-q1": ("M", "q1"),
        "C1-v2-vs-q2": ("C", "q2"),
        "M1-v2-vs-q2": ("M", "q2"),
        "C1-v2-vs-q3": ("C", "q3"),
        "M1-v2-vs-q3": ("M", "q3"),
        "C1-v2-vs-q4": ("C", "q4"),
        "M1-v2-vs-q4": ("M", "q4"),
        "C1-v2-vs-heuristic": ("C", "heuristic"),
        "M1-v2-vs-heuristic": ("M", "heuristic"),
        "q0-parent-vs-heuristic-reference-1": (None, "heuristic"),
    }
    records_by_key: dict[str, tuple[object, ...]] = {}
    for key, (arm, opponent) in keys.items():
        cell = _alignment_cell(validator, key, arm, opponent)
        a_id, b_id, a_rng, b_rng = validator.expected_agent_id(cell)
        setup = derive_seed(41, "arena:pair:0:setup") & ((1 << 64) - 1)
        seed_a = derive_seed(41, f"arena:pair:0:agent:{a_rng}")
        seed_b = derive_seed(41, f"arena:pair:0:agent:{b_rng}")
        records: list[object] = []
        for index, base in enumerate(base_records):
            a_first = index == 0
            ids = (a_id, b_id) if a_first else (b_id, a_id)
            seeds = (seed_a, seed_b) if a_first else (seed_b, seed_a)
            domains = (
                (f"agent:{a_rng}", f"agent:{b_rng}")
                if a_first
                else (f"agent:{b_rng}", f"agent:{a_rng}")
            )
            seats = tuple(
                AgentSeatRecord(
                    player,
                    agent_id,
                    config,
                    seed,
                    RNG_ALGORITHM,
                    SEED_DERIVATION,
                    domain,
                )
                for player, agent_id, seed, domain in zip(
                    PlayerId, ids, seeds, domains, strict=True
                )
            )
            records.append(
                replace(
                    base,
                    run_id=cell["run_id"],
                    game_id=f"pair-000000-{'a' if a_first else 'b'}-first",
                    pair_id="pair-000000",
                    seats=(seats[0], seats[1]),
                    replay=replace(base.replay, seed=setup),
                )
            )
        records_by_key[key] = tuple(records)
    return records_by_key


def test_cross_cell_architecture_and_heuristic_alignment_mutations_raise(
    validator: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validator, "REPLICATES", ("replicate-1",))
    records_by_key = _aligned_records(validator)
    validator.check_cross_cell_alignment(records_by_key)

    mutated_architecture = dict(records_by_key)
    record = mutated_architecture["M1-v2-vs-v1"][0]
    mutated_architecture["M1-v2-vs-v1"] = (
        replace(record, replay=replace(record.replay, seed=record.replay.seed + 1)),
        mutated_architecture["M1-v2-vs-v1"][1],
    )
    with pytest.raises(validator.ValidationError, match="setup"):
        validator.check_cross_cell_alignment(mutated_architecture)

    mutated_seat_rng = dict(records_by_key)
    record = mutated_seat_rng["M1-v2-vs-v1"][0]
    seat = replace(record.seats[0], seed=record.seats[0].seed + 1)
    mutated_seat_rng["M1-v2-vs-v1"] = (
        replace(record, seats=(seat, record.seats[1])),
        mutated_seat_rng["M1-v2-vs-v1"][1],
    )
    with pytest.raises(validator.ValidationError, match="physical seat/RNG"):
        validator.check_cross_cell_alignment(mutated_seat_rng)

    mutated_opponent = dict(records_by_key)
    record = mutated_opponent["M1-v2-vs-heuristic"][0]
    opponent_index = next(
        index for index, seat in enumerate(record.seats) if seat.agent_id == "heuristic"
    )
    seats = list(record.seats)
    seats[opponent_index] = replace(seats[opponent_index], config={"type": "tampered"})
    mutated_opponent["M1-v2-vs-heuristic"] = (
        replace(record, seats=(seats[0], seats[1])),
        mutated_opponent["M1-v2-vs-heuristic"][1],
    )
    with pytest.raises(validator.ValidationError, match="opponent"):
        validator.check_cross_cell_alignment(mutated_opponent)

    mutated_opponent_rng = dict(records_by_key)
    record = mutated_opponent_rng["q0-parent-vs-heuristic-reference-1"][0]
    opponent_index = next(
        index for index, seat in enumerate(record.seats) if seat.agent_id == "heuristic"
    )
    seats = list(record.seats)
    seats[opponent_index] = replace(seats[opponent_index], rng_domain="agent:tampered")
    mutated_opponent_rng["q0-parent-vs-heuristic-reference-1"] = (
        replace(record, seats=(seats[0], seats[1])),
        mutated_opponent_rng["q0-parent-vs-heuristic-reference-1"][1],
    )
    with pytest.raises(validator.ValidationError, match="opponent"):
        validator.check_cross_cell_alignment(mutated_opponent_rng)


def test_validator_source_guard_excludes_all_production_reconstruction_entry_points(
    validator: Any,
) -> None:
    validator.source_guard()
    tree = ast.parse(Path("scripts/validate_structured_model_v2.py").read_text())
    forbidden_modules = {
        "agent_avenue.encoding.candidate_v1",
        "agent_avenue.encoding.candidate_structured_v2",
        "agent_avenue.runners.structured_experiment",
        "agent_avenue.runners.arena",
    }
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
    assert imported_modules.isdisjoint(forbidden_modules)
    forbidden_calls = {
        "encode_v1",
        "encode_v2",
        "arena_report_from_records",
        "structured_statistics",
        "structured_selection",
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint(forbidden_calls)
