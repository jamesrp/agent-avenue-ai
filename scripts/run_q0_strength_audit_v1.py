#!/usr/bin/env python3
"""Run the approved fresh q0 strength and tactical-leak audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from agent_avenue.agents import (
    Agent,
    GreedyHeuristicAgent,
    RandomAgent,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
    derive_seed,
)
from agent_avenue.agents.learned import LearnedValueAgent
from agent_avenue.encoding import encode_candidate
from agent_avenue.engine import Action
from agent_avenue.learning import LoadedCheckpoint, load_checkpoint
from agent_avenue.observation.model import PlayerObservation
from agent_avenue.runners import (
    AgentSpec,
    ArenaConfig,
    audit_public_forced_wins,
    audit_recruit_patterns,
    paired_policy_difference_interval,
    run_resumable_arena,
)
from agent_avenue.storage import code_fingerprint, load_corpus, rules_fingerprint

CYCLE_ID = "m7-q0-strength-audit-v1"
ROOT_SEED = 2026090901
PAIR_COUNT = 1000
Q0_ID = "q0-terminal-safety-v1"
OFFENSE_ID = "q0-terminal-offense-v1"
HISTORICAL_ID = "historical-q0"
HEURISTIC_ID = "greedy-public-v1"
RANDOM_ID = "random"
Q_IDS = tuple(f"q{generation}-terminal-safety-v1" for generation in range(1, 5))
POLICY_ORDER = (Q0_ID, OFFENSE_ID, *Q_IDS, HISTORICAL_ID, HEURISTIC_ID, RANDOM_ID)
ANCHORS = (HISTORICAL_ID, HEURISTIC_ID, RANDOM_ID)
FIELD_OPPONENTS = (*Q_IDS, *ANCHORS)
CHECKPOINT_PATHS = {
    Q0_ID: Path("runs/terminal-safety-v1/q0-a1/checkpoint"),
    Q_IDS[0]: Path("runs/terminal-safety-v1/q1-a1/candidate"),
    Q_IDS[1]: Path("runs/terminal-safety-v1/q2-a1/candidate"),
    Q_IDS[2]: Path("runs/terminal-safety-v1/q3-a1/candidate"),
    Q_IDS[3]: Path("runs/terminal-safety-v1/q4-a1/candidate"),
    HISTORICAL_ID: Path("runs/terminal-safety-v1/inputs/historical-q0"),
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), check=True, capture_output=True, text=True).stdout.strip()


def _agent_config(agent: Agent) -> dict[str, object]:
    config = getattr(agent, "config", None)
    to_data = getattr(config, "to_data", None)
    if callable(to_data):
        value = to_data()
    else:
        config_to_data = getattr(agent, "config_to_data", None)
        if not callable(config_to_data):
            raise RuntimeError("agent has no normalized configuration")
        value = config_to_data()
    if not isinstance(value, dict):
        raise RuntimeError("agent configuration is not an object")
    return value


def _learned_factory(checkpoint: LoadedCheckpoint, *, shielded: bool, offense: bool) -> Agent:
    agent: Agent = LearnedValueAgent.from_checkpoint(checkpoint)
    if shielded:
        agent = TerminalSafetyAgent(agent)
    if offense:
        agent = TerminalOffenseAgent(agent)
    return agent


def _load_policies() -> tuple[dict[str, AgentSpec], dict[str, LoadedCheckpoint]]:
    checkpoints = {agent_id: load_checkpoint(path) for agent_id, path in CHECKPOINT_PATHS.items()}
    specs: dict[str, AgentSpec] = {}
    for agent_id in (Q0_ID, *Q_IDS, HISTORICAL_ID):
        checkpoint = checkpoints[agent_id]
        shielded = agent_id != HISTORICAL_ID

        def factory(checkpoint: LoadedCheckpoint = checkpoint, shielded: bool = shielded) -> Agent:
            return _learned_factory(checkpoint, shielded=shielded, offense=False)

        specs[agent_id] = AgentSpec(agent_id, _agent_config(factory()), factory)

    q0_checkpoint = checkpoints[Q0_ID]

    def offense_factory(checkpoint: LoadedCheckpoint = q0_checkpoint) -> Agent:
        return _learned_factory(checkpoint, shielded=True, offense=True)

    specs[OFFENSE_ID] = AgentSpec(OFFENSE_ID, _agent_config(offense_factory()), offense_factory)
    heuristic = GreedyHeuristicAgent()
    specs[HEURISTIC_ID] = AgentSpec(HEURISTIC_ID, _agent_config(heuristic), GreedyHeuristicAgent)
    random = RandomAgent()
    specs[RANDOM_ID] = AgentSpec(RANDOM_ID, _agent_config(random), RandomAgent)
    return specs, checkpoints


def _plan(
    output: Path,
    specs: dict[str, AgentSpec],
    checkpoints: dict[str, LoadedCheckpoint],
    pair_count: int,
) -> dict[str, object]:
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise RuntimeError("fresh strength audit requires a tracked-clean source tree")
    common_arena_seed = derive_seed(ROOT_SEED, f"{CYCLE_ID}:common-arena") & ((1 << 63) - 1)
    payload: dict[str, object] = {
        "version": "q0-strength-audit-plan-v1",
        "cycle_id": CYCLE_ID,
        "source": {
            "git_revision": _git("rev-parse", "HEAD"),
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "rules_fingerprint": rules_fingerprint(),
            "code_fingerprint": code_fingerprint(),
        },
        "root_seed": ROOT_SEED,
        "common_arena_seed": common_arena_seed,
        "pair_count_per_matchup": pair_count,
        "games_per_matchup": pair_count * 2,
        "policy_order": list(POLICY_ORDER),
        "matchup_count": len(tuple(combinations(POLICY_ORDER, 2))),
        "total_games": len(tuple(combinations(POLICY_ORDER, 2))) * pair_count * 2,
        "policies": {
            agent_id: {
                "config": dict(specs[agent_id].config),
                "checkpoint_path": (
                    str(CHECKPOINT_PATHS[Q0_ID])
                    if agent_id == OFFENSE_ID
                    else str(CHECKPOINT_PATHS[agent_id])
                    if agent_id in CHECKPOINT_PATHS
                    else None
                ),
                "checkpoint_fingerprint": (
                    checkpoints[Q0_ID].checkpoint_fingerprint
                    if agent_id == OFFENSE_ID
                    else checkpoints[agent_id].checkpoint_fingerprint
                    if agent_id in checkpoints
                    else None
                ),
            }
            for agent_id in POLICY_ORDER
        },
        "output": str(output),
        "tactical_definition": "public-guaranteed-current-turn-win-v1",
    }
    payload["plan_fingerprint"] = _fingerprint(payload)
    return payload


def _report_semantics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"elapsed_seconds", "games_per_second"}
    }


def _matchup_key(first: str, second: str) -> str:
    return f"{first}--{second}"


def _run_tournament(
    output: Path,
    *,
    plan: dict[str, Any],
    specs: dict[str, AgentSpec],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    matchups = tuple(combinations(POLICY_ORDER, 2))
    for index, (first, second) in enumerate(matchups, start=1):
        key = _matchup_key(first, second)
        print(f"[{index:02d}/{len(matchups)}] {first} vs {second}", flush=True)
        config = ArenaConfig(
            run_id=f"{CYCLE_ID}-{index:02d}",
            agent_a=specs[first],
            agent_b=specs[second],
            pair_count=int(plan["pair_count_per_matchup"]),
            master_seed=int(plan["common_arena_seed"]),
        )
        retained = run_resumable_arena(
            output / "arena-records" / key,
            config,
            generation=None,
            corpus_configuration={
                "version": "q0-strength-audit-arena-v1",
                "cycle_id": CYCLE_ID,
                "plan_fingerprint": plan["plan_fingerprint"],
                "matchup": [first, second],
                "common_setup_block": True,
            },
        )
        report_data = retained.report.to_data()
        artifact_path = output / "arenas" / f"{key}.json"
        if artifact_path.exists():
            existing = _read(artifact_path)
            if (
                existing.get("plan_fingerprint") != plan["plan_fingerprint"]
                or existing.get("records_corpus_fingerprint")
                != retained.records_manifest.corpus_fingerprint
                or _report_semantics(existing["report"]) != _report_semantics(report_data)
            ):
                raise RuntimeError(f"existing arena artifact changed: {key}")
            report_data = existing["report"]
        else:
            artifact = {
                "version": "q0-strength-audit-arena-artifact-v1",
                "plan_fingerprint": plan["plan_fingerprint"],
                "matchup": [first, second],
                "records_corpus_fingerprint": retained.records_manifest.corpus_fingerprint,
                "report": report_data,
            }
            artifact["artifact_fingerprint"] = _fingerprint(artifact)
            _write(artifact_path, artifact)
        reports[key] = report_data
    return reports


def _record_directories(output: Path) -> tuple[Path, ...]:
    return tuple(
        output / "arena-records" / _matchup_key(first, second)
        for first, second in combinations(POLICY_ORDER, 2)
    )


def _validate_common_block(output: Path, expected_games: int) -> dict[str, object]:
    common: tuple[int, ...] | None = None
    corpora: dict[str, str] = {}
    total_records = 0
    total_decisions = 0
    for directory in _record_directories(output):
        manifest, records = load_corpus(directory)
        pair_seeds = tuple(record.replay.seed for record in records[::2])
        if len(records) != expected_games:
            raise RuntimeError(f"unexpected game count in {directory}")
        if any(
            records[index].replay.seed != records[index + 1].replay.seed
            for index in range(0, len(records), 2)
        ):
            raise RuntimeError(f"seat-swapped pair seed mismatch in {directory}")
        if common is None:
            common = pair_seeds
        elif pair_seeds != common:
            raise RuntimeError("all-pairs tournament did not reuse one common setup block")
        corpora[directory.name] = manifest.corpus_fingerprint
        total_records += len(records)
        total_decisions += sum(record.decision_count for record in records)
    if common is None:
        raise RuntimeError("tournament produced no corpora")
    data: dict[str, object] = {
        "version": "q0-strength-audit-common-block-validation-v1",
        "matchup_count": len(corpora),
        "pair_count": len(common),
        "total_records": total_records,
        "total_decisions": total_decisions,
        "setup_seed_fingerprint": _fingerprint(list(common)),
        "corpus_fingerprints": dict(sorted(corpora.items())),
        "status": "passed",
    }
    data["artifact_fingerprint"] = _fingerprint(data)
    return data


def _iter_records(output: Path):  # type: ignore[no-untyped-def]
    for directory in _record_directories(output):
        _, records = load_corpus(directory)
        yield from records


def _scorer(checkpoint: LoadedCheckpoint):  # type: ignore[no-untyped-def]
    def score(observation: PlayerObservation, actions: tuple[Action, ...]) -> tuple[float, ...]:
        vectors = [encode_candidate(observation, action).vector for action in actions]
        features = torch.tensor(vectors, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            logits = checkpoint.model(features)
        return tuple(float(value) for value in logits.tolist())

    return score


def _find_report(reports: dict[str, dict[str, Any]], first: str, second: str) -> dict[str, Any]:
    order = {agent_id: index for index, agent_id in enumerate(POLICY_ORDER)}
    left, right = (first, second) if order[first] < order[second] else (second, first)
    return reports[_matchup_key(left, right)]


def _target_pair_wins(report: dict[str, Any], target: str) -> tuple[int, ...]:
    agent_a = report["agents"]["a"]["id"]
    values = tuple(int(row["agent_a_wins"]) for row in report["paired_seed_outcomes"])
    return values if target == agent_a else tuple(2 - value for value in values)


def _target_summary(report: dict[str, Any], target: str) -> dict[str, object]:
    agent_a = report["agents"]["a"]["id"]
    agent_b = report["agents"]["b"]["id"]
    if target not in {agent_a, agent_b}:
        raise RuntimeError("target agent is absent from report")
    target_is_a = target == agent_a
    paired = report["paired_bootstrap_confidence_interval_95"]
    low, high = (float(paired["interval"][0]), float(paired["interval"][1]))
    a_rate = float(report["agent_a_win_rate"])
    a_seats = report["agent_a_by_seat"]
    if target_is_a:
        rate = a_rate
        interval = [low, high]
        seats = {
            "player_one": float(a_seats["player_one"]["win_rate"]),
            "player_two": float(a_seats["player_two"]["win_rate"]),
        }
        margin = float(report["average_score_margin"])
        opponent = agent_b
    else:
        rate = 1 - a_rate
        interval = [1 - high, 1 - low]
        seats = {
            "player_one": 1 - float(a_seats["player_two"]["win_rate"]),
            "player_two": 1 - float(a_seats["player_one"]["win_rate"]),
        }
        margin = -float(report["average_score_margin"])
        opponent = agent_a
    return {
        "opponent": opponent,
        "games": int(report["total_games"]),
        "win_rate": rate,
        "paired_bootstrap_interval_95": interval,
        "by_seat": seats,
        "average_score_margin": margin,
        "average_turns": float(report["average_turns"]),
        "average_decisions": float(report["average_decisions"]),
        "terminal_reasons": report["terminal_reasons"],
    }


def _policy_matchups(
    reports: dict[str, dict[str, Any]], target: str, opponents: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    return {
        opponent: _target_summary(_find_report(reports, target, opponent), target)
        for opponent in opponents
    }


def _mean_rate(rows: dict[str, dict[str, object]]) -> float:
    return sum(float(row["win_rate"]) for row in rows.values()) / len(rows)


def _matrix(reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for row in POLICY_ORDER:
        result[row] = {}
        for column in POLICY_ORDER:
            result[row][column] = (
                None
                if row == column
                else float(_target_summary(_find_report(reports, row, column), row)["win_rate"])
            )
    return result


def _offense_effects(
    reports: dict[str, dict[str, Any]], common_arena_seed: int
) -> dict[str, object]:
    opponents: dict[str, object] = {}
    for opponent in FIELD_OPPONENTS:
        treatment_report = _find_report(reports, OFFENSE_ID, opponent)
        control_report = _find_report(reports, Q0_ID, opponent)
        opponents[opponent] = {
            "offense": _target_summary(treatment_report, OFFENSE_ID),
            "control": _target_summary(control_report, Q0_ID),
            "paired_difference": paired_policy_difference_interval(
                _target_pair_wins(treatment_report, OFFENSE_ID),
                _target_pair_wins(control_report, Q0_ID),
                master_seed=common_arena_seed,
                domain=f"{CYCLE_ID}:offense-minus-control:{opponent}",
            ),
        }
    differences = [float(row["paired_difference"]["point_estimate"]) for row in opponents.values()]
    return {
        "direct_offense_vs_control": _target_summary(
            _find_report(reports, OFFENSE_ID, Q0_ID), OFFENSE_ID
        ),
        "by_shared_opponent": opponents,
        "equal_opponent_mean_point_difference": sum(differences) / len(differences),
    }


def _selected_counts(tactical: dict[str, Any], agent_id: str) -> dict[str, Any]:
    return tactical["by_agent"][agent_id]


def _summarize(
    *,
    plan: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    tactical: dict[str, Any],
    recruit: dict[str, Any],
) -> dict[str, object]:
    q0_matchups = _policy_matchups(reports, Q0_ID, FIELD_OPPONENTS)
    q0_anchor = {opponent: q0_matchups[opponent] for opponent in ANCHORS}
    q0_specialists = {opponent: q0_matchups[opponent] for opponent in Q_IDS}
    q0_counts = _selected_counts(tactical, Q0_ID)
    offense_counts = _selected_counts(tactical, OFFENSE_ID)
    if offense_counts["counts"]["missed_forced_wins"] != 0:
        raise RuntimeError("terminal-offense policy missed a publicly guaranteed win")
    if q0_counts["counts"]["executed_avoidable_provable_losses"] != 0:
        raise RuntimeError("selected q0 violated the terminal-safety contract")
    offer_rows = [
        {"ordered_offer": key, **value}
        for key, value in recruit["by_ordered_offer"].items()
        if int(value["decisions"]) >= 100
    ]
    offer_rows.sort(key=lambda row: (-int(row["decisions"]), str(row["ordered_offer"])))
    leak_rows = [
        row
        for row in offer_rows
        if (float(row["face_up_choice_rate"]) >= 0.9 or float(row["face_up_choice_rate"]) <= 0.1)
        and float(row["target_game_win_rate"]) <= 0.45
    ]
    q0_rates = {opponent: float(row["win_rate"]) for opponent, row in q0_matchups.items()}
    result: dict[str, object] = {
        "version": "q0-strength-audit-result-v1",
        "status": "completed",
        "evidence_class": "fresh-common-block-descriptive-strength-and-tactical-audit",
        "cycle_id": CYCLE_ID,
        "plan_fingerprint": plan["plan_fingerprint"],
        "source": plan["source"],
        "games": {
            "total": validation["total_records"],
            "decisions": validation["total_decisions"],
            "matchups": validation["matchup_count"],
            "pairs_per_matchup": validation["pair_count"],
            "common_setup_seed_fingerprint": validation["setup_seed_fingerprint"],
        },
        "tournament": {
            "policy_order": list(POLICY_ORDER),
            "win_rate_matrix": _matrix(reports),
            "q0_matchups": q0_matchups,
            "q0_equal_anchor_mean_win_rate": _mean_rate(q0_anchor),
            "q0_equal_specialist_mean_win_rate": _mean_rate(q0_specialists),
            "q0_equal_field_mean_win_rate": _mean_rate(q0_matchups),
            "q0_worst_matchup": min(q0_rates, key=q0_rates.get),
            "q0_worst_win_rate": min(q0_rates.values()),
        },
        "terminal_offense_effects": _offense_effects(reports, int(plan["common_arena_seed"])),
        "tactical_audit": {
            "artifact_fingerprint": tactical["artifact_fingerprint"],
            "q0": q0_counts,
            "q0_terminal_offense": offense_counts,
        },
        "recruit_patterns": {
            "artifact_fingerprint": recruit["artifact_fingerprint"],
            "overall": recruit["overall"],
            "by_face_up": recruit["by_face_up"],
            "by_face_up_copy_count": recruit["by_face_up_copy_count"],
            "by_opponent": recruit["by_opponent"],
            "ordered_offers_minimum_100": offer_rows,
            "fixed_candidate_leak_rows": leak_rows,
        },
        "validation": {
            "common_block": validation,
            "offense_missed_forced_wins": offense_counts["counts"]["missed_forced_wins"],
            "q0_executed_avoidable_provable_losses": q0_counts["counts"][
                "executed_avoidable_provable_losses"
            ],
            "status": "passed",
        },
        "limitations": [
            "No opponent in the field is a solved-game oracle or grandmaster reference.",
            "q1-q4 are known q0-specialized descendants, not globally validated champions.",
            (
                "Ordered hidden-card offer slices are offline opponent-known diagnostics; "
                "outcome rates are descriptive, not causal."
            ),
            (
                "The offense wrapper tests only guaranteed current-turn wins and says nothing "
                "about deeper tactics."
            ),
        ],
        "result_fingerprint": "",
    }
    result["result_fingerprint"] = _fingerprint(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/m7-q0-strength-audit-v1"))
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    result_path = args.output / "result.json"
    if result_path.exists():
        result = _read(result_path)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0

    specs, checkpoints = _load_policies()
    plan = _plan(args.output, specs, checkpoints, args.pairs)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and _read(plan_path) != plan:
        raise RuntimeError("existing strength-audit plan differs from the frozen request")
    _write(plan_path, plan)

    reports = _run_tournament(args.output, plan=plan, specs=specs)
    validation = _validate_common_block(args.output, args.pairs * 2)
    _write(args.output / "analysis" / "common-block-validation.json", validation)

    tactical = audit_public_forced_wins(
        _iter_records(args.output),
        source_label=str(plan["plan_fingerprint"]),
        candidate_scorers={Q0_ID: _scorer(checkpoints[Q0_ID])},
        example_agent_ids=frozenset({Q0_ID}),
        max_examples_per_agent=24,
        verify_records=False,
    )
    _write(args.output / "analysis" / "tactical-audit.json", tactical)
    recruit = audit_recruit_patterns(
        _iter_records(args.output),
        target_agent_id=Q0_ID,
        source_label=str(plan["plan_fingerprint"]),
        verify_records=False,
    )
    _write(args.output / "analysis" / "q0-recruit-patterns.json", recruit)

    result = _summarize(
        plan=plan,
        reports=reports,
        validation=validation,
        tactical=tactical,
        recruit=recruit,
    )
    _write(result_path, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command boundary
        print(f"q0 strength audit failed: {exc}", file=sys.stderr)
        raise
