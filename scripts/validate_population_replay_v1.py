#!/usr/bin/env python3
"""Locally replay and independently validate retained M7 population-replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, cast

from agent_avenue.agents import (
    Agent,
    DeterministicRandom,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
    derive_seed,
)
from agent_avenue.runners.arena import ArenaConfig, arena_report_from_records
from agent_avenue.runners.corpus import validate_record_matches_spec
from agent_avenue.runners.game import AgentSpec
from agent_avenue.runners.population import (
    POPULATION_PAIR_COUNT,
    POPULATION_POLICY_IDS,
    POPULATION_REPLAY_CYCLE_ID,
    POPULATION_REPLAY_REPLICATE_IDS,
    POPULATION_REPLAY_ROOT_SEED,
    PopulationArm,
    PopulationCorpusConfig,
    audit_population_alignment,
    audit_population_arm,
    build_population_corpus_plan,
)
from agent_avenue.runners.population_experiment import (
    CANDIDATE_ARMS,
    HEURISTIC,
    HISTORICAL_Q0,
    NESTED_BOOTSTRAP_LOWER_INDEX,
    NESTED_BOOTSTRAP_RESAMPLES,
    NESTED_BOOTSTRAP_UPPER_INDEX,
    RANDOM,
    build_population_policy_bundle,
)
from agent_avenue.runners.safety_audit import audit_terminal_safety
from agent_avenue.runners.strength_audit import audit_public_forced_wins
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    game_record_fingerprint,
    load_corpus,
    repository_root,
    rules_fingerprint,
)

PLAN_VERSION: Final = "m7-population-replay-plan-v1"
RESULT_VERSION: Final = "m7-population-replay-result-v1"
VALIDATION_VERSION: Final = "m7-population-replay-independent-validation-v2"
GLOBAL_BOOTSTRAP_DOMAIN: Final = f"{POPULATION_REPLAY_CYCLE_ID}:global-nested-bootstrap:v1"


class LocalValidationError(RuntimeError):
    """Raised when retained evidence cannot be reconstructed locally."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalValidationError(f"unable to read artifact: {path}") from exc
    if not isinstance(value, dict):
        raise LocalValidationError(f"artifact must be an object: {path}")
    return cast(dict[str, object], value)


def _write_immutable(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        if _read(path) != value:
            raise LocalValidationError("existing independent validation differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(_canonical(value) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _source_identity() -> dict[str, object]:
    root = repository_root()
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        tracked_status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=no"),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        tracked_diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD", "--", "."),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        full_status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LocalValidationError("unable to inspect frozen source") from exc
    lock = root / "uv.lock"
    return {
        "git_revision": revision,
        "uv_lock_sha256": _sha256(lock),
        "tracked_tree_clean": not bool(tracked_status.strip()),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "runner_clean_check": {
            "version": "population-replay-runner-clean-source-v1",
            "tracked_and_nonignored_untracked_clean": not bool(full_status.strip()),
            "status_sha256": hashlib.sha256(full_status).hexdigest(),
            "checks": {
                "tracked_changes": "git-status-porcelain-v1",
                "untracked": "all-nonignored-files",
                "ignored_artifacts": "excluded-by-git-status-ignored-no",
            },
        },
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }


def _slice_specs(specs: tuple[object, ...], pairs: int) -> tuple[object, ...]:
    return specs[: pairs * 2]


def _ordered(ids: Iterable[str]) -> list[str]:
    found = set(ids)
    return [policy for policy in POPULATION_POLICY_IDS if policy in found]


def _coverage(
    records: tuple[GameRecord, ...], manifest: Mapping[str, object], corpus_plan: object
) -> dict[str, object]:
    split = manifest.get("split")
    members = corpus_plan.config.member_by_agent_id
    if not isinstance(split, Mapping):
        raise LocalValidationError("dataset split is malformed")
    by_fp = {game_record_fingerprint(record): record for record in records}
    result: dict[str, object] = {}
    for name in ("train", "validation"):
        values = split.get(f"{name}_game_fingerprints")
        if not isinstance(values, list):
            raise LocalValidationError("dataset split fingerprints are malformed")
        pairs: set[str] = set()
        policies: set[str] = set()
        for value in values:
            if not isinstance(value, str) or value not in by_fp:
                raise LocalValidationError("dataset references an unknown record")
            record = by_fp[value]
            if record.pair_id is None:
                raise LocalValidationError("dataset record has no paired block")
            pairs.add(record.pair_id)
            for seat in record.seats:
                member = members.get(seat.agent_id)
                if member is not None:
                    policies.add(member.policy_id)
        result[name] = {
            "pair_count": len(pairs),
            "pair_ids": sorted(pairs),
            "policy_ids": _ordered(policies),
        }
    return result


def _matchup_summary(corpus_plan: object) -> dict[str, list[dict[str, object]]]:
    ordering = {policy: index for index, policy in enumerate(POPULATION_POLICY_IDS)}
    result: dict[str, list[dict[str, object]]] = {}
    for replicate in corpus_plan.replicates:
        counts: Counter[tuple[str, str]] = Counter()
        for matchup in replicate.treatment_assignment.matchups:
            left, right = sorted(
                (matchup.lane_a_policy_id, matchup.lane_b_policy_id),
                key=lambda value: ordering[value],
            )
            counts[(left, right)] += 1
        result[replicate.replicate_id] = [
            {"policy_a": left, "policy_b": right, "paired_blocks": count}
            for (left, right), count in sorted(
                counts.items(), key=lambda item: (ordering[item[0][0]], ordering[item[0][1]])
            )
        ]
    return result


def _cells(corpus_plan: object, pairs: int) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:
        replicate = corpus_plan.replicate(replicate_id)

        def add(
            key: str,
            arm: PopulationArm | None,
            opponent: str,
            declared_pairs: int,
            shared: str | None,
            *,
            direct: bool = False,
            current_replicate_id: str = replicate_id,
            current_replicate: object = replicate,
        ) -> None:
            count = min(pairs, declared_pairs)
            seed_key = shared or key
            domain = (
                f"{POPULATION_REPLAY_CYCLE_ID}:replicate:{current_replicate_id}:arena:{seed_key}:v1"
            )
            values.append(
                {
                    "replicate_id": current_replicate_id,
                    "key": key,
                    "run_id": f"{POPULATION_REPLAY_CYCLE_ID}-{current_replicate_id}-{key}",
                    "candidate_arm": arm,
                    "opponent": opponent,
                    "paired_blocks": count,
                    "games": count * 2,
                    "master_seed": derive_seed(current_replicate.seed_plan.arena_seed, domain)
                    & ((1 << 63) - 1),
                    "shared_group": shared,
                    "direct": direct,
                }
            )

        add("treatment-vs-control", None, "control", 500, None, direct=True)
        for arm in ("control", "treatment"):
            add(f"{arm}-vs-parent", arm, "q0", 500, "parent")
        for arm in ("control", "treatment"):
            add(f"{arm}-vs-heuristic", arm, HEURISTIC, 300, "heuristic")
        add("parent-vs-heuristic-reference", None, HEURISTIC, 300, "heuristic")
        for arm in ("control", "treatment"):
            add(f"{arm}-vs-random", arm, RANDOM, 200, "random")
            add(f"{arm}-vs-historical-q0", arm, HISTORICAL_Q0, 200, "historical-q0")
        for opponent in ("q1", "q2", "q3", "q4"):
            for arm in ("control", "treatment"):
                add(f"{arm}-vs-{opponent}", arm, opponent, 100, opponent)
    return values


def _config_data(agent: Agent) -> dict[str, object]:
    method = getattr(agent, "config_to_data", None)
    if callable(method):
        value = method()
    else:
        config = getattr(agent, "config", None)
        to_data = getattr(config, "to_data", None)
        if not callable(to_data):
            raise LocalValidationError("agent has no normalized config")
        value = to_data()
    if not isinstance(value, dict):
        raise LocalValidationError("agent config is malformed")
    return cast(dict[str, object], json.loads(_canonical(value)))


def _enveloped(path: Path, agent_id: str, rng_identity: str) -> AgentSpec:
    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import load_checkpoint

    checkpoint = load_checkpoint(path)

    def factory() -> Agent:
        return TerminalOffenseAgent(
            TerminalSafetyAgent(LearnedValueAgent.from_checkpoint(checkpoint))
        )

    return AgentSpec(agent_id, _config_data(factory()), factory, rng_identity=rng_identity)


def _arena_agents(
    cell: Mapping[str, object], output: Path, bundle: object
) -> tuple[AgentSpec, AgentSpec]:
    replicate = cast(str, cell["replicate_id"])
    key = cast(str, cell["key"])
    candidate_lane = f"{POPULATION_REPLAY_CYCLE_ID}:{replicate}:candidate-lane"
    if cell["direct"] is True:
        return (
            _enveloped(
                output / "checkpoints" / replicate / "treatment",
                "treatment-candidate",
                f"{POPULATION_REPLAY_CYCLE_ID}:{replicate}:treatment-direct-lane",
            ),
            _enveloped(
                output / "checkpoints" / replicate / "control",
                "control-candidate",
                f"{POPULATION_REPLAY_CYCLE_ID}:{replicate}:control-direct-lane",
            ),
        )
    opponent = cast(str, cell["opponent"])
    if key == "parent-vs-heuristic-reference":
        return _opponent("q0", replicate, bundle), _opponent(HEURISTIC, replicate, bundle)
    arm = cell["candidate_arm"]
    if arm not in {"control", "treatment"}:
        raise LocalValidationError("candidate arena arm is malformed")
    return _enveloped(
        output / "checkpoints" / replicate / str(arm), f"{arm}-candidate", candidate_lane
    ), _opponent(opponent, replicate, bundle)


def _opponent(opponent: str, replicate: str, bundle: object) -> AgentSpec:
    if opponent == HEURISTIC:
        agent = GreedyHeuristicAgent(GreedyHeuristicConfig())
        return AgentSpec(
            HEURISTIC,
            _config_data(agent),
            lambda: GreedyHeuristicAgent(GreedyHeuristicConfig()),
            rng_identity=f"{replicate}:heuristic",
        )
    if opponent == RANDOM:
        agent = RandomAgent(RandomAgentConfig())
        return AgentSpec(
            RANDOM,
            _config_data(agent),
            lambda: RandomAgent(RandomAgentConfig()),
            rng_identity=f"{replicate}:random",
        )
    if opponent == HISTORICAL_Q0:
        spec = bundle.historical_q0
        return AgentSpec(
            spec.agent_id,
            dict(spec.config),
            spec.factory,
            rng_identity=f"{replicate}:historical-q0",
        )
    if bundle.mode == "toy-random-smoke":

        def factory() -> Agent:
            return TerminalOffenseAgent(TerminalSafetyAgent(RandomAgent(RandomAgentConfig())))

        return AgentSpec(
            f"enveloped-{opponent}",
            _config_data(factory()),
            factory,
            rng_identity=f"{replicate}:{opponent}",
        )
    paths = bundle.checkpoint_identities
    row = paths.get(opponent)
    if not isinstance(row, Mapping) or not isinstance(row.get("locator"), str):
        raise LocalValidationError("opponent checkpoint identity is missing")
    return _enveloped(
        Path(cast(str, row["locator"])), f"enveloped-{opponent}", f"{replicate}:{opponent}"
    )


def _pair_scores(artifact: Mapping[str, object]) -> tuple[float, ...]:
    report = artifact.get("report")
    values = report.get("paired_seed_outcomes") if isinstance(report, Mapping) else None
    if not isinstance(values, list):
        raise LocalValidationError("arena report lacks paired outcomes")
    return tuple(float(cast(Mapping[str, object], value)["agent_a_wins"]) / 2 for value in values)


def _nested(rows: tuple[tuple[float, ...], ...], seed: int, domain: str) -> dict[str, object]:
    if len(rows) != 3 or any(not row for row in rows) or len({len(row) for row in rows}) != 1:
        raise LocalValidationError("nested rows are malformed")
    count = len(rows[0])
    stream_domain = f"{POPULATION_REPLAY_CYCLE_ID}:nested-bootstrap:{domain}:v1"
    stream_seed = derive_seed(seed, stream_domain)
    rng = DeterministicRandom(stream_seed, stream_domain)
    values: list[float] = []
    for _ in range(NESTED_BOOTSTRAP_RESAMPLES):
        means = []
        for replicate in (rng.randbelow(3) for _ in range(3)):
            row = rows[replicate]
            means.append(sum(row[rng.randbelow(count)] for _ in range(count)) / count)
        values.append(sum(means) / 3)
    values.sort()
    point = sum(sum(row) / count for row in rows) / 3
    return {
        "version": "m7-population-replay-nested-bootstrap-v1",
        "unit": "paired training replicate, then paired setup block",
        "replicate_count": 3,
        "blocks_per_replicate": count,
        "resamples": NESTED_BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "order_statistic_indices": {
            "lower": NESTED_BOOTSTRAP_LOWER_INDEX,
            "upper": NESTED_BOOTSTRAP_UPPER_INDEX,
        },
        "point_estimate": point,
        "interval": [values[NESTED_BOOTSTRAP_LOWER_INDEX], values[NESTED_BOOTSTRAP_UPPER_INDEX]],
        "seed": stream_seed,
        "domain": domain,
    }


def _difference(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise LocalValidationError("unmatched arena block lengths")
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _local_metrics(
    artifacts: Iterable[Mapping[str, object]], global_seed: int
) -> tuple[dict[str, object], dict[str, object]]:
    by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    for artifact in artifacts:
        cell = artifact.get("cell")
        if not isinstance(cell, Mapping):
            raise LocalValidationError("arena cell artifact is malformed")
        by_key[(cast(str, cell["replicate_id"]), cast(str, cell["key"]))] = artifact
    direct: list[tuple[float, ...]] = []
    treatment_parent: list[tuple[float, ...]] = []
    control_parent: list[tuple[float, ...]] = []
    treatment_random: list[tuple[float, ...]] = []
    treatment_parent_heuristic: list[tuple[float, ...]] = []
    diffs: dict[str, list[tuple[float, ...]]] = {
        name: [] for name in (HEURISTIC, RANDOM, HISTORICAL_Q0, "q1", "q2", "q3", "q4")
    }
    per_replicate: list[dict[str, object]] = []
    minimum_seat = 1.0
    tactical = True
    treatment_macros: list[float] = []
    control_macros: list[float] = []
    for replicate in POPULATION_REPLAY_REPLICATE_IDS:

        def scores(key: str, current_replicate: str = replicate) -> tuple[float, ...]:
            return _pair_scores(by_key[(current_replicate, key)])

        direct_row = scores("treatment-vs-control")
        tr_parent = scores("treatment-vs-parent")
        co_parent = scores("control-vs-parent")
        direct.append(direct_row)
        treatment_parent.append(tr_parent)
        control_parent.append(co_parent)
        diffs[HEURISTIC].append(
            _difference(scores("treatment-vs-heuristic"), scores("control-vs-heuristic"))
        )
        diffs[RANDOM].append(
            _difference(scores("treatment-vs-random"), scores("control-vs-random"))
        )
        diffs[HISTORICAL_Q0].append(
            _difference(scores("treatment-vs-historical-q0"), scores("control-vs-historical-q0"))
        )
        for opponent in ("q1", "q2", "q3", "q4"):
            diffs[opponent].append(
                _difference(scores(f"treatment-vs-{opponent}"), scores(f"control-vs-{opponent}"))
            )
        tr_random = scores("treatment-vs-random")
        treatment_random.append(tr_random)
        tr_parent_h = _difference(
            scores("treatment-vs-heuristic"), scores("parent-vs-heuristic-reference")
        )
        treatment_parent_heuristic.append(tr_parent_h)
        opponents = ("parent", "heuristic", "random", "historical-q0", "q1", "q2", "q3", "q4")
        treatment_macro = sum(
            sum(scores(f"treatment-vs-{name}")) / len(scores(f"treatment-vs-{name}"))
            for name in opponents
        ) / len(opponents)
        control_macro = sum(
            sum(scores(f"control-vs-{name}")) / len(scores(f"control-vs-{name}"))
            for name in opponents
        ) / len(opponents)
        treatment_macros.append(treatment_macro)
        control_macros.append(control_macro)
        seats: list[float] = []
        for (item_replicate, key), artifact in by_key.items():
            if item_replicate != replicate or not (
                key.startswith("control-vs-")
                or key.startswith("treatment-vs-")
                or key == "treatment-vs-control"
            ):
                continue
            report = cast(Mapping[str, object], artifact["report"])
            rows = cast(Mapping[str, object], report["agent_a_by_seat"])
            seats.extend(
                float(cast(Mapping[str, object], row)["win_rate"]) for row in rows.values()
            )
            if (
                not isinstance(artifact.get("tactical_invariants"), Mapping)
                or artifact["tactical_invariants"].get("passed") is not True
            ):
                tactical = False
        direct_rows = cast(
            Mapping[str, object],
            cast(Mapping[str, object], by_key[(replicate, "treatment-vs-control")])["report"],
        )["agent_a_by_seat"]
        seats.extend(
            1.0 - float(cast(Mapping[str, object], row)["win_rate"])
            for row in cast(Mapping[str, object], direct_rows).values()
        )
        replicate_min = min(seats)
        minimum_seat = min(minimum_seat, replicate_min)
        per_replicate.append(
            {
                "replicate_id": replicate,
                "direct_treatment_vs_control": sum(direct_row) / len(direct_row),
                "treatment_vs_parent": sum(tr_parent) / len(tr_parent),
                "control_vs_parent": sum(co_parent) / len(co_parent),
                "treatment_vs_random": sum(tr_random) / len(tr_random),
                "treatment_minus_control_heuristic": sum(diffs[HEURISTIC][-1])
                / len(diffs[HEURISTIC][-1]),
                "treatment_minus_parent_heuristic": sum(tr_parent_h) / len(tr_parent_h),
                "equal_opponent_treatment_macro": treatment_macro,
                "equal_opponent_control_macro": control_macro,
                "minimum_candidate_seat_rate": replicate_min,
            }
        )
    nested: dict[str, object] = {
        "treatment_vs_control": _nested(tuple(direct), global_seed, "treatment-vs-control"),
        "treatment_vs_parent": _nested(tuple(treatment_parent), global_seed, "treatment-vs-parent"),
        "control_vs_parent": _nested(tuple(control_parent), global_seed, "control-vs-parent"),
        "treatment_vs_random": _nested(tuple(treatment_random), global_seed, "treatment-vs-random"),
        "treatment_minus_parent_heuristic": _nested(
            tuple(treatment_parent_heuristic), global_seed, "treatment-minus-parent-heuristic"
        ),
    }
    for opponent, rows in diffs.items():
        nested[f"treatment_minus_control_{opponent}"] = _nested(
            tuple(rows), global_seed, f"treatment-minus-control-{opponent}"
        )
    points = [float(row["direct_treatment_vs_control"]) for row in per_replicate]
    variation = (sum((point - sum(points) / 3) ** 2 for point in points) / 2) ** 0.5
    summary = {
        "version": "m7-population-replay-statistics-v1",
        "replicates": per_replicate,
        "nested": nested,
        "global_bootstrap": {
            "domain": GLOBAL_BOOTSTRAP_DOMAIN,
            "seed": global_seed,
            "root_seed": POPULATION_REPLAY_ROOT_SEED,
            "rng_algorithm": "hmac-sha256-counter-v1",
        },
        "descriptive_equal_opponent_macros": {
            "treatment_mean": sum(treatment_macros) / 3,
            "control_mean": sum(control_macros) / 3,
        },
        "between_replicate": {"direct_treatment_vs_control_sample_standard_deviation": variation},
        "minimum_candidate_seat_rate": minimum_seat,
        "tactical_invariants_passed": tactical,
    }
    return summary, nested


def _local_decision(summary: Mapping[str, object], integrity: bool) -> dict[str, object]:
    nested = cast(Mapping[str, object], summary["nested"])
    reps = cast(list[object], summary["replicates"])

    def lower(name: str) -> float:
        return float(cast(list[object], cast(Mapping[str, object], nested[name])["interval"])[0])

    favored = sum(
        float(cast(Mapping[str, object], row)["direct_treatment_vs_control"]) > 0.5 for row in reps
    )
    conditions = {
        "direct_nested_lower_strictly_above_50_percent": lower("treatment_vs_control") > 0.5,
        "at_least_two_direct_replicates_favor_treatment": favored >= 2,
        "treatment_parent_nested_lower_above_50_percent": lower("treatment_vs_parent") > 0.5,
        "treatment_random_nested_lower_above_50_percent": lower("treatment_vs_random") > 0.5,
        "treatment_minus_parent_heuristic_lower_above_minus_5pp": lower(
            "treatment_minus_parent_heuristic"
        )
        > -0.05,
        "treatment_minus_control_heuristic_lower_above_minus_5pp": lower(
            f"treatment_minus_control_{HEURISTIC}"
        )
        > -0.05,
        "no_candidate_seat_point_below_45_percent": float(summary["minimum_candidate_seat_rate"])
        >= 0.45,
        "zero_enveloped_avoidable_losses_and_missed_guaranteed_wins": summary[
            "tactical_invariants_passed"
        ]
        is True,
        "replay_schedule_seed_split_checkpoint_compatibility": integrity,
    }
    crosses = any(
        float(cast(list[object], cast(Mapping[str, object], nested[name])["interval"])[0])
        <= 0.5
        <= float(cast(list[object], cast(Mapping[str, object], nested[name])["interval"])[1])
        for name in ("treatment_vs_control", "treatment_vs_parent", "treatment_vs_random")
    )
    return {
        "decision": "advancing_model_v1_collection_recipe"
        if all(conditions.values())
        else ("inconclusive_does_not_advance" if crosses else "does_not_advance"),
        "conditions": conditions,
        "direct_replicates_favoring_treatment": favored,
    }


def _check_alignment(records: Mapping[tuple[str, str], tuple[GameRecord, ...]]) -> None:
    for replicate in POPULATION_REPLAY_REPLICATE_IDS:
        for opponent in ("parent", "heuristic", "random", "historical-q0", "q1", "q2", "q3", "q4"):
            treatment = records[(replicate, f"treatment-vs-{opponent}")]
            control = records[(replicate, f"control-vs-{opponent}")]
            for left, right in zip(treatment, control, strict=True):
                if (
                    left.game_id != right.game_id
                    or left.pair_id != right.pair_id
                    or left.replay.seed != right.replay.seed
                    or tuple((seat.rng_identity, seat.seed) for seat in left.seats)
                    != tuple((seat.rng_identity, seat.seed) for seat in right.seats)
                ):
                    raise LocalValidationError("pairwise shared-opponent RNG alignment failed")
        triple = tuple(
            records[(replicate, key)]
            for key in (
                "treatment-vs-heuristic",
                "control-vs-heuristic",
                "parent-vs-heuristic-reference",
            )
        )
        for treatment, control, parent in zip(*triple, strict=True):
            trio = (treatment, control, parent)
            if (
                len({record.replay.seed for record in trio}) != 1
                or len({record.game_id for record in trio}) != 1
            ):
                raise LocalValidationError("three-way heuristic setup alignment failed")
            heuristic = []
            other = []
            for record in trio:
                index = next(
                    (i for i, seat in enumerate(record.seats) if seat.agent_id == HEURISTIC), None
                )
                if index is None:
                    raise LocalValidationError("heuristic seat missing")
                heuristic.append(
                    (index, record.seats[index].rng_identity, record.seats[index].seed)
                )
                other.append(record.seats[1 - index])
            if len(set(heuristic)) != 1:
                raise LocalValidationError("three-way heuristic seat/RNG alignment failed")
            if (
                other[0].rng_identity != other[1].rng_identity
                or other[0].seed != other[1].seed
                or other[2].rng_identity == other[0].rng_identity
                or other[2].seed == other[0].seed
            ):
                raise LocalValidationError("three-way parent/candidate lane alignment failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/m7-population-replay-v1"))
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    output = args.output
    plan = _read(output / "plan.json")
    if plan.get("version") != PLAN_VERSION or plan.get("plan_fingerprint") != _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    ):
        raise LocalValidationError("plan identity is malformed")
    source = _source_identity()
    if plan.get("source") != source:
        raise LocalValidationError(
            "validator requires exact clean frozen source, lock, code, and rules"
        )
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise LocalValidationError("plan execution is malformed")
    if execution.get("claim_eligible") is not True and not args.allow_smoke:
        raise LocalValidationError("bounded smoke requires --allow-smoke")
    pairs = execution.get("pairs_per_corpus_arm")
    if type(pairs) is not int:
        raise LocalValidationError("pair count is malformed")
    input_paths = plan.get("input_paths")
    if not isinstance(input_paths, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in input_paths.items()
    ):
        raise LocalValidationError("input paths are malformed")
    paths = {key: Path(value) for key, value in input_paths.items()}
    bundle = build_population_policy_bundle(paths, toy=execution.get("mode") == "toy-random-smoke")
    corpus_plan = build_population_corpus_plan(PopulationCorpusConfig(members=bundle.members))
    if corpus_plan.to_data() != plan.get("corpus_plan") or corpus_plan.fingerprint != plan.get(
        "corpus_plan_fingerprint"
    ):
        raise LocalValidationError("corpus plan does not reconstruct")
    if _matchup_summary(corpus_plan) != plan.get("treatment_matchup_counts"):
        raise LocalValidationError("treatment matchup summaries differ")
    bootstrap = plan.get("global_bootstrap")
    expected_seed = derive_seed(POPULATION_REPLAY_ROOT_SEED, GLOBAL_BOOTSTRAP_DOMAIN)
    if (
        not isinstance(bootstrap, Mapping)
        or bootstrap.get("domain") != GLOBAL_BOOTSTRAP_DOMAIN
        or bootstrap.get("seed") != expected_seed
    ):
        raise LocalValidationError("global bootstrap declaration differs")
    cells = _cells(corpus_plan, pairs)
    plan_cells = cast(Mapping[str, object], plan["arenas"])["cells"]
    if plan_cells != cells:
        raise LocalValidationError("arena cells do not reconstruct")
    if pairs == POPULATION_PAIR_COUNT:
        arenas = cast(Mapping[str, object], plan["arenas"])
        expected_counts = {
            "candidate_comparison_games_per_replicate": 7400,
            "parent_heuristic_reference_games_per_replicate": 600,
            "physical_games_per_replicate": 8000,
            "total_physical_games": 24000,
        }
        if any(arenas.get(key) != value for key, value in expected_counts.items()):
            raise LocalValidationError("corrected 8,000/24,000 arena count contract failed")
    retained_corpora: dict[tuple[str, PopulationArm], tuple[GameRecord, ...]] = {}
    corpus_artifacts: dict[tuple[str, PopulationArm], Mapping[str, object]] = {}
    for replicate in POPULATION_REPLAY_REPLICATE_IDS:
        for arm in CANDIDATE_ARMS:
            directory = output / "corpora" / replicate / arm
            manifest, records = load_corpus(directory)
            specs = _slice_specs(corpus_plan.game_specs(replicate, arm), pairs)
            if len(records) != len(specs):
                raise LocalValidationError("corpus record count differs")
            for record, spec in zip(records, specs, strict=True):
                validate_record_matches_spec(record, spec)
            if pairs == POPULATION_PAIR_COUNT:
                audit_population_arm(corpus_plan, replicate, arm, records)
            artifact = _read(output / "corpus-artifacts" / replicate / f"{arm}.json")
            if (
                artifact.get("plan_fingerprint") != plan["plan_fingerprint"]
                or cast(Mapping[str, object], artifact["records"]).get("corpus_fingerprint")
                != manifest.corpus_fingerprint
            ):
                raise LocalValidationError("corpus artifact differs")
            retained_corpora[(replicate, arm)] = records
            corpus_artifacts[(replicate, arm)] = artifact
        if pairs == POPULATION_PAIR_COUNT:
            audit_population_alignment(
                retained_corpora[(replicate, "control")], retained_corpora[(replicate, "treatment")]
            )
    from agent_avenue.learning import (
        inspect_checkpoint,
        load_checkpoint,
        load_dataset,
        materialize_dataset,
        tensor_digest,
    )

    dataset_artifacts: dict[tuple[str, PopulationArm], Mapping[str, object]] = {}
    split_rows: list[dict[str, object]] = []
    for replicate in POPULATION_REPLAY_REPLICATE_IDS:
        coverage_by_arm: dict[str, dict[str, object]] = {}
        for arm in CANDIDATE_ARMS:
            manifest = cast(Mapping[str, object], corpus_artifacts[(replicate, arm)]["records"])
            expected = materialize_dataset(
                retained_corpora[(replicate, arm)],
                split_seed=corpus_plan.replicate(replicate).seed_plan.split_seed,
                validation_fraction=0.1,
                source_corpus_fingerprint=cast(str, manifest["corpus_fingerprint"]),
            )
            path = output / "datasets" / replicate / f"{arm}.npz"
            dataset = load_dataset(path)
            artifact = _read(output / "dataset-audits" / replicate / f"{arm}.json")
            data = cast(Mapping[str, object], artifact["dataset"])
            coverage = _coverage(retained_corpora[(replicate, arm)], dataset.manifest, corpus_plan)
            if (
                dataset.fingerprint != expected.fingerprint
                or data.get("fingerprint") != dataset.fingerprint
                or data.get("array_sha256") != _sha256(path)
                or data.get("manifest_sha256") != _sha256(path.with_suffix(".json"))
                or artifact.get("provenance_coverage") != coverage
            ):
                raise LocalValidationError("dataset artifact does not reconstruct")
            if (
                arm == "treatment"
                and pairs == POPULATION_PAIR_COUNT
                and any(
                    set(cast(Mapping[str, object], coverage[name])["policy_ids"])
                    != set(POPULATION_POLICY_IDS)
                    for name in ("train", "validation")
                )
            ):
                raise LocalValidationError("treatment split coverage is incomplete")
            dataset_artifacts[(replicate, arm)] = artifact
            coverage_by_arm[arm] = coverage
        row: dict[str, object] = {"replicate_id": replicate}
        for name in ("train", "validation"):
            left = cast(Mapping[str, object], coverage_by_arm["control"][name])["pair_ids"]
            right = cast(Mapping[str, object], coverage_by_arm["treatment"][name])["pair_ids"]
            if not isinstance(left, list) or not isinstance(right, list) or set(left) != set(right):
                raise LocalValidationError("matched train/validation pair sets differ")
            row[name] = {
                "pair_count": len(left),
                "pair_ids_fingerprint": _fingerprint(sorted(left)),
            }
        row["natural_hash_split_counts"] = {
            "train": cast(Mapping[str, object], row["train"])["pair_count"],
            "validation": cast(Mapping[str, object], row["validation"])["pair_count"],
        }
        row["status"] = "passed"
        split_rows.append(row)
    split_artifact = _read(output / "dataset-split-alignment.json")
    if split_artifact.get("replicates") != split_rows:
        raise LocalValidationError("split alignment artifact differs")
    parent = load_checkpoint(paths["q0"])
    parent_tensor = tensor_digest(parent.model.state_dict())
    summary = _read(output / "training-summary.json")
    if not isinstance(summary.get("replicates"), list) or len(summary["replicates"]) != 3:
        raise LocalValidationError("training summary cardinality differs")
    for row in cast(list[object], summary["replicates"]):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("replicate_id"), str)
            or not isinstance(row.get("arms"), Mapping)
        ):
            raise LocalValidationError("training summary row malformed")
        replicate = cast(str, row["replicate_id"])
        expected_config = (
            __import__("agent_avenue.learning", fromlist=["TrainingConfig"])
            .TrainingConfig(
                seed=corpus_plan.replicate(replicate).seed_plan.initialization_seed
                & ((1 << 63) - 1),
                model_seed=corpus_plan.replicate(replicate).seed_plan.initialization_seed
                & ((1 << 63) - 1),
                shuffle_seed=corpus_plan.replicate(replicate).seed_plan.shuffle_seed
                & ((1 << 63) - 1),
                learning_rate=1e-3,
                weight_decay=1e-4,
                batch_size=1024,
                max_epochs=execution["training_max_epochs"],
                early_stopping_patience=8,
                cpu_threads=1,
            )
            .normalized()
        )
        for arm in CANDIDATE_ARMS:
            checkpoint_path = output / "checkpoints" / replicate / arm
            inspected = inspect_checkpoint(checkpoint_path)
            checkpoint = load_checkpoint(checkpoint_path)
            stored = cast(Mapping[str, object], cast(Mapping[str, object], row["arms"])[arm])
            metadata = cast(Mapping[str, object], checkpoint.manifest["metadata"])
            lineage = cast(Mapping[str, object], checkpoint.manifest["lineage"])
            training = cast(Mapping[str, object], checkpoint.manifest["training"])
            sources = cast(Mapping[str, object], checkpoint.manifest["sources"])
            corpus_fp = cast(Mapping[str, object], corpus_artifacts[(replicate, arm)]["records"])[
                "corpus_fingerprint"
            ]
            dataset_fp = cast(Mapping[str, object], dataset_artifacts[(replicate, arm)]["dataset"])[
                "fingerprint"
            ]
            if (
                inspected.checkpoint_fingerprint != stored.get("checkpoint_fingerprint")
                or inspected.tensor_digest != stored.get("tensor_digest")
                or metadata.get("plan_fingerprint") != plan["plan_fingerprint"]
                or metadata.get("initial_parent_tensor_digest") != parent_tensor
                or lineage.get("parent_checkpoint") != parent.checkpoint_fingerprint
                or training.get("config") != expected_config
                or sources.get("dataset_fingerprint") != dataset_fp
                or tuple(cast(tuple[object, ...], sources.get("corpus_fingerprints", ())))
                != (corpus_fp,)
            ):
                raise LocalValidationError("checkpoint lineage/config differs")
    arena_records: dict[tuple[str, str], tuple[GameRecord, ...]] = {}
    artifacts: list[dict[str, object]] = []
    for cell in cells:
        replicate = cast(str, cell["replicate_id"])
        key = cast(str, cell["key"])
        left, right = _arena_agents(cell, output, bundle)
        config = ArenaConfig(
            cast(str, cell["run_id"]),
            left,
            right,
            cast(int, cell["paired_blocks"]),
            cast(int, cell["master_seed"]),
        )
        manifest, records = load_corpus(output / "arena-records" / replicate / key)
        artifact = _read(output / "arenas" / replicate / f"{key}.json")
        report = cast(Mapping[str, object], artifact["report"])
        elapsed = report.get("elapsed_seconds")
        tactical = {"passed": None}
        tactical = {
            "passed": None,
            "safety": audit_terminal_safety(
                records, source_corpus_fingerprint=manifest.corpus_fingerprint
            ),
            "offense": audit_public_forced_wins(records, source_label=manifest.corpus_fingerprint),
        }
        checked = {}
        safety_agents = cast(
            Mapping[str, object], cast(Mapping[str, object], tactical["safety"])["by_agent"]
        )
        offense_agents = cast(
            Mapping[str, object], cast(Mapping[str, object], tactical["offense"])["by_agent"]
        )
        passed = True
        for agent_id, item in safety_agents.items():
            agent = cast(Mapping[str, object], item)
            agent_config = agent.get("config")
            if isinstance(agent_config, Mapping) and agent_config.get("type") == "terminal_offense":
                safety_counts = cast(Mapping[str, object], agent["counts"])
                offense_counts = cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], offense_agents[agent_id])["counts"],
                )
                ok = (
                    safety_counts.get("executed_avoidable_provable_losses") == 0
                    and offense_counts.get("missed_forced_wins") == 0
                )
                checked[agent_id] = {
                    "avoidable_immediate_losses": safety_counts.get(
                        "executed_avoidable_provable_losses"
                    ),
                    "missed_guaranteed_wins": offense_counts.get("missed_forced_wins"),
                    "passed": ok,
                }
                passed &= ok
        tactical["passed"] = passed and bool(checked)
        tactical["checked_agents"] = checked
        if (
            not isinstance(elapsed, int | float)
            or arena_report_from_records(config, records, elapsed_seconds=float(elapsed)).to_data()
            != report
            or artifact.get("tactical_invariants") != tactical
        ):
            raise LocalValidationError("arena aggregate or tactical audit differs")
        artifacts.append(artifact)
        arena_records[(replicate, key)] = records
    if len(artifacts) != 54:
        raise LocalValidationError("arena artifact cardinality differs")
    _check_alignment(arena_records)
    metrics, _ = _local_metrics(artifacts, expected_seed)
    stored_metrics = _read(output / "statistics.json")
    expected_metrics = {"plan_fingerprint": plan["plan_fingerprint"], **metrics}
    expected_metrics["artifact_fingerprint"] = _fingerprint(expected_metrics)
    if stored_metrics != expected_metrics:
        raise LocalValidationError("nested statistics differ")
    decision = _local_decision(metrics, integrity=True)
    result = _read(output / "result.json")
    expected_corpus_keys = {
        f"{replicate}:{arm}"
        for replicate in POPULATION_REPLAY_REPLICATE_IDS
        for arm in CANDIDATE_ARMS
    }
    expected_arena_keys = {f"{cell['replicate_id']}:{cell['key']}" for cell in cells}
    result_corpora = result.get("corpora")
    result_datasets = result.get("datasets")
    result_arenas = result.get("arena_artifacts")
    if (
        result.get("version") != RESULT_VERSION
        or result.get("decision") != decision
        or not isinstance(result_corpora, Mapping)
        or set(result_corpora) != expected_corpus_keys
        or not isinstance(result_datasets, Mapping)
        or set(result_datasets) != expected_corpus_keys
        or not isinstance(result_arenas, Mapping)
        or set(result_arenas) != expected_arena_keys
        or result.get("dataset_split_alignment_fingerprint")
        != split_artifact.get("artifact_fingerprint")
        or result.get("training_summary_fingerprint") != summary.get("artifact_fingerprint")
        or result.get("result_fingerprint")
        != _fingerprint(
            {key: value for key, value in result.items() if key != "result_fingerprint"}
        )
    ):
        raise LocalValidationError("result differs from local reconstruction")
    validation: dict[str, object] = {
        "version": VALIDATION_VERSION,
        "status": "passed",
        "plan_fingerprint": plan["plan_fingerprint"],
        "source": source,
        "checks": {
            "source_lock_code_rules": True,
            "six_corpus_schedules_assignments_and_alignment": True,
            "matched_splits_and_treatment_coverage": True,
            "dataset_files_and_fingerprints": True,
            "checkpoint_lineage_and_initialization": True,
            "arena_aggregates_pairwise_and_three_way_alignment": True,
            "tactical_invariants": True,
            "global_nested_bootstrap_and_decision": True,
            "artifact_cardinalities_and_references": True,
        },
        "artifact_fingerprint": "",
    }
    validation["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in validation.items() if key != "artifact_fingerprint"}
    )
    _write_immutable(output / "validation.json", validation)
    print(json.dumps(validation, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"population replay validation failed: {exc}", file=sys.stderr)
        raise
