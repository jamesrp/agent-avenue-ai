#!/usr/bin/env python3
"""Independently recompute the retained q0 strength-audit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from agent_avenue.agents import RandomAgent
from agent_avenue.encoding import encode_candidate
from agent_avenue.engine import Action
from agent_avenue.learning import load_checkpoint
from agent_avenue.observation.model import PlayerObservation
from agent_avenue.runners import (
    AgentSpec,
    ArenaConfig,
    arena_report_from_records,
    audit_public_forced_wins,
    audit_recruit_patterns,
)
from agent_avenue.storage import load_corpus

Q0_ID = "q0-terminal-safety-v1"
OFFENSE_ID = "q0-terminal-offense-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _matchup_key(first: str, second: str) -> str:
    return f"{first}--{second}"


def _record_directories(root: Path, policy_order: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(
        root / "arena-records" / _matchup_key(first, second)
        for first, second in combinations(policy_order, 2)
    )


def _iter_records(root: Path, policy_order: tuple[str, ...]):  # type: ignore[no-untyped-def]
    for directory in _record_directories(root, policy_order):
        _, records = load_corpus(directory)
        yield from records


def _scorer(checkpoint_path: Path):  # type: ignore[no-untyped-def]
    checkpoint = load_checkpoint(checkpoint_path)

    def score(observation: PlayerObservation, actions: tuple[Action, ...]) -> tuple[float, ...]:
        vectors = [encode_candidate(observation, action).vector for action in actions]
        features = torch.tensor(vectors, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            logits = checkpoint.model(features)
        return tuple(float(value) for value in logits.tolist())

    return score


def _dummy_spec(agent_id: str, config: dict[str, object]) -> AgentSpec:
    return AgentSpec(agent_id, config, RandomAgent)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "validation.json"
    plan = _read(root / "plan.json")
    result = _read(root / "result.json")
    policy_order = tuple(str(value) for value in plan["policy_order"])
    policies = plan["policies"]
    if not isinstance(policies, dict):
        raise RuntimeError("plan policies are malformed")

    plan_payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    if _fingerprint(plan_payload) != plan["plan_fingerprint"]:
        raise RuntimeError("plan fingerprint mismatch")
    result_payload = dict(result)
    result_payload["result_fingerprint"] = ""
    if _fingerprint(result_payload) != result["result_fingerprint"]:
        raise RuntimeError("result fingerprint mismatch")

    arena_fingerprints: dict[str, str] = {}
    common_seeds: tuple[int, ...] | None = None
    total_records = 0
    total_decisions = 0
    for first, second in combinations(policy_order, 2):
        key = _matchup_key(first, second)
        artifact = _read(root / "arenas" / f"{key}.json")
        manifest, records = load_corpus(root / "arena-records" / key)
        if artifact["records_corpus_fingerprint"] != manifest.corpus_fingerprint:
            raise RuntimeError(f"arena corpus fingerprint mismatch: {key}")
        report = artifact["report"]
        first_config = policies[first]["config"]
        second_config = policies[second]["config"]
        if not isinstance(first_config, dict) or not isinstance(second_config, dict):
            raise RuntimeError("plan policy config is malformed")
        recomputed = arena_report_from_records(
            ArenaConfig(
                run_id=str(report["run_id"]),
                agent_a=_dummy_spec(first, first_config),
                agent_b=_dummy_spec(second, second_config),
                pair_count=int(plan["pair_count_per_matchup"]),
                master_seed=int(plan["common_arena_seed"]),
            ),
            records,
            elapsed_seconds=float(report["elapsed_seconds"]),
        ).to_data()
        if recomputed != report:
            raise RuntimeError(f"arena aggregate mismatch: {key}")
        artifact_payload = {
            key: value for key, value in artifact.items() if key != "artifact_fingerprint"
        }
        if _fingerprint(artifact_payload) != artifact["artifact_fingerprint"]:
            raise RuntimeError(f"arena artifact fingerprint mismatch: {key}")
        pair_seeds = tuple(record.replay.seed for record in records[::2])
        if common_seeds is None:
            common_seeds = pair_seeds
        elif pair_seeds != common_seeds:
            raise RuntimeError(f"common setup block mismatch: {key}")
        arena_fingerprints[key] = str(artifact["artifact_fingerprint"])
        total_records += len(records)
        total_decisions += sum(record.decision_count for record in records)

    tactical = audit_public_forced_wins(
        _iter_records(root, policy_order),
        source_label=str(plan["plan_fingerprint"]),
        candidate_scorers={Q0_ID: _scorer(Path(str(policies[Q0_ID]["checkpoint_path"])))},
        example_agent_ids=frozenset({Q0_ID}),
        max_examples_per_agent=24,
        verify_records=False,
    )
    retained_tactical = _read(root / "analysis" / "tactical-audit.json")
    if tactical != retained_tactical:
        raise RuntimeError("tactical audit mismatch")

    recruit = audit_recruit_patterns(
        _iter_records(root, policy_order),
        target_agent_id=Q0_ID,
        source_label=str(plan["plan_fingerprint"]),
        verify_records=False,
    )
    retained_recruit = _read(root / "analysis" / "q0-recruit-patterns.json")
    if recruit != retained_recruit:
        raise RuntimeError("recruit-pattern audit mismatch")

    if common_seeds is None:
        raise RuntimeError("no arena records found")
    q0_counts = tactical["by_agent"][Q0_ID]["counts"]
    offense_counts = tactical["by_agent"][OFFENSE_ID]["counts"]
    if q0_counts["executed_avoidable_provable_losses"] != 0:
        raise RuntimeError("q0 terminal-safety invariant failed")
    if offense_counts["missed_forced_wins"] != 0:
        raise RuntimeError("terminal-offense invariant failed")
    if total_records != int(result["games"]["total"]):
        raise RuntimeError("result game count mismatch")
    if total_decisions != int(result["games"]["decisions"]):
        raise RuntimeError("result decision count mismatch")

    validation: dict[str, object] = {
        "version": "q0-strength-audit-independent-validation-v1",
        "status": "passed",
        "validation_source_revision": subprocess.run(
            ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
        ).stdout.strip(),
        "plan_fingerprint": plan["plan_fingerprint"],
        "result_fingerprint": result["result_fingerprint"],
        "matchup_count": len(arena_fingerprints),
        "arena_artifact_fingerprints": dict(sorted(arena_fingerprints.items())),
        "record_count": total_records,
        "decision_count": total_decisions,
        "common_setup_seed_fingerprint": _fingerprint(list(common_seeds)),
        "tactical_audit_fingerprint": tactical["artifact_fingerprint"],
        "recruit_pattern_audit_fingerprint": recruit["artifact_fingerprint"],
        "q0_executed_avoidable_provable_losses": q0_counts["executed_avoidable_provable_losses"],
        "offense_missed_forced_wins": offense_counts["missed_forced_wins"],
        "artifact_fingerprint": "",
    }
    validation["artifact_fingerprint"] = _fingerprint(validation)
    _write(output, validation)
    print(json.dumps(validation, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
