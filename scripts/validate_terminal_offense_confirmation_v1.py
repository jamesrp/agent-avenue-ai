#!/usr/bin/env python3
"""Independently validate retained step-1 terminal-offense confirmation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_avenue.agents import (
    RNG_ALGORITHM,
    SEED_DERIVATION,
    DeterministicRandom,
    LearnedCandidateScores,
    RandomAgent,
    derive_seed,
    filter_immediate_win_actions,
    filter_terminal_actions,
)
from agent_avenue.agents.ordering import semantic_action_key
from agent_avenue.engine import (
    Action,
    GameConfig,
    GameState,
    Phase,
    PlayOfferAction,
    RecruitAction,
    apply_action,
    new_game,
)
from agent_avenue.engine.model import player_index
from agent_avenue.engine.setup import normalize_config
from agent_avenue.observation import observe
from agent_avenue.observation.model import PlayerObservation
from agent_avenue.runners import (
    ANCHOR_OPPONENTS,
    BOOTSTRAP_DOMAIN,
    BOOTSTRAP_LOWER_INDEX,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_UPPER_INDEX,
    CONTROL_ID,
    CYCLE_ID,
    DEFAULT_EXCLUDED_SETUP_ROOTS,
    DEFAULT_PAIRS_PER_FAMILY,
    FAMILY_IDS,
    FIELD_OPPONENTS,
    PRACTICAL_LIFT_THRESHOLD,
    Q0_RNG_IDENTITY,
    ROOT_SEED,
    SETUP_HOLDOUT_SCOPE_VERSION,
    TREATMENT_ID,
    AgentSpec,
    ArenaConfig,
    ArenaReport,
    ConfirmationCell,
    arena_report_from_records,
    confirmation_cells,
    independent_public_forced_win_oracle,
)
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    inspect_source_identity,
    load_corpus,
    rules_fingerprint,
)

_VALIDATOR_BOOTSTRAP_DOMAIN = BOOTSTRAP_DOMAIN
CandidateScorer = Callable[[PlayerObservation, tuple[Action, ...]], LearnedCandidateScores]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _toy_scorer(
    observation: PlayerObservation, actions: tuple[Action, ...]
) -> LearnedCandidateScores:
    semantic = tuple(sorted(actions, key=semantic_action_key))
    return LearnedCandidateScores.from_logits(semantic, tuple(0.0 for _ in semantic))


def _scorers(plan: dict[str, Any]) -> dict[str, CandidateScorer]:
    mode = plan["execution"]["mode"]
    if mode == "toy-random-smoke":
        return {CONTROL_ID: _toy_scorer, TREATMENT_ID: _toy_scorer}
    if mode != "learned-checkpoints":
        raise RuntimeError("unsupported confirmation policy mode")
    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import load_checkpoint

    checkpoint_data = plan["policies"][CONTROL_ID]["checkpoint"]
    checkpoint = load_checkpoint(Path(str(checkpoint_data["path"])))
    if checkpoint.checkpoint_fingerprint != checkpoint_data["checkpoint_fingerprint"]:
        raise RuntimeError("q0 checkpoint fingerprint differs from frozen plan")
    agent = LearnedValueAgent.from_checkpoint(checkpoint)

    def score(
        observation: PlayerObservation, actions: tuple[Action, ...]
    ) -> LearnedCandidateScores:
        return agent.score_candidates(observation, observation.decision, actions)

    return {CONTROL_ID: score, TREATMENT_ID: score}


def _require_frozen_source(plan: dict[str, Any]) -> None:
    current = inspect_source_identity()
    frozen = plan["source"]
    if not current.tracked_tree_clean:
        raise RuntimeError("validator requires a tracked-clean source tree")
    expected = {
        "git_revision": current.git_revision,
        "uv_lock_sha256": current.uv_lock_sha256,
        "tracked_tree_clean": True,
        "tracked_diff_sha256": current.tracked_diff_sha256,
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }
    actual = {key: frozen.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError("current source provenance does not match the frozen plan")


def _family_master(family: str) -> int:
    if family not in FAMILY_IDS:
        raise RuntimeError("unknown seed family")
    return derive_seed(ROOT_SEED, f"{CYCLE_ID}:seed-family:{family}") & ((1 << 63) - 1)


def _family_setups(family: str, pair_count: int) -> tuple[int, ...]:
    if pair_count < 1:
        raise RuntimeError("pair count must be positive")
    master_seed = _family_master(family)
    return tuple(
        derive_seed(master_seed, f"arena:pair:{index}:setup") & ((1 << 64) - 1)
        for index in range(pair_count)
    )


def _seed_validation(pair_count: int) -> dict[str, object]:
    blocks = {family: _family_setups(family, pair_count) for family in FAMILY_IDS}
    unique = all(len(set(block)) == len(block) for block in blocks.values())
    disjoint = set(blocks[FAMILY_IDS[0]]).isdisjoint(blocks[FAMILY_IDS[1]])
    return {
        "root_seed": ROOT_SEED,
        "family_seed_domains": {
            family: f"{CYCLE_ID}:seed-family:{family}" for family in FAMILY_IDS
        },
        "family_master_seeds": {family: _family_master(family) for family in FAMILY_IDS},
        "pair_count_per_family": pair_count,
        "setup_seed_fingerprints": {
            family: hashlib.sha256(
                json.dumps(list(block), separators=(",", ":")).encode()
            ).hexdigest()
            for family, block in blocks.items()
        },
        "unique_within_families": unique,
        "families_disjoint": disjoint,
        "status": "passed" if unique and disjoint else "failed",
    }


def _setup_identity(config: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(
        _canonical_json({"game_config": dict(config), "setup_seed": seed})
    ).hexdigest()


def _identity_set_fingerprint(values: set[str]) -> str:
    return hashlib.sha256(_canonical_json(sorted(values))).hexdigest()


def _independent_holdout_scan(
    *, pair_count: int, roots: tuple[Path, ...], current_output: Path
) -> dict[str, object]:
    if not roots:
        raise RuntimeError("setup holdout has no declared roots")
    config = normalize_config(GameConfig())
    current: dict[str, dict[str, object]] = {}
    for family in FAMILY_IDS:
        for pair_index, seed in enumerate(_family_setups(family, pair_count)):
            identity = _setup_identity(config, seed)
            if identity in current:
                raise RuntimeError("candidate seed families overlap")
            current[identity] = {
                "family": family,
                "pair_index": pair_index,
                "setup_seed": seed,
                "game_config": config,
            }
    output = current_output.resolve()
    seen: set[Path] = set()
    sources: dict[str, list[str]] = {}
    material: dict[str, dict[str, object]] = {}
    root_rows: list[dict[str, object]] = []
    corpora: list[dict[str, object]] = []
    scanned_records = 0
    for declared in roots:
        root = declared.resolve()
        if root == output:
            root_rows.append(
                {
                    "declared_root": str(declared),
                    "resolved_root": str(root),
                    "status": "excluded-current-output",
                    "corpus_count": 0,
                    "record_count": 0,
                    "ignored_manifest_count": 0,
                    "current_output_excluded": True,
                }
            )
            continue
        if not root.exists():
            root_rows.append(
                {
                    "declared_root": str(declared),
                    "resolved_root": str(root),
                    "status": "missing",
                    "corpus_count": 0,
                    "record_count": 0,
                    "ignored_manifest_count": 0,
                    "current_output_excluded": output.is_relative_to(root),
                }
            )
            continue
        if not root.is_dir():
            raise RuntimeError("declared setup holdout root is not a directory")
        corpus_count = 0
        record_count = 0
        ignored = 0
        for manifest_path in sorted(root.rglob("manifest.json")):
            resolved = manifest_path.resolve()
            if resolved.is_relative_to(output) or resolved in seen:
                continue
            seen.add(resolved)
            records_path = manifest_path.parent / "games.jsonl.gz"
            try:
                manifest_data = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                if records_path.exists():
                    raise RuntimeError("malformed completed corpus manifest") from exc
                ignored += 1
                continue
            is_corpus = records_path.exists() or (
                isinstance(manifest_data, dict) and "records_file" in manifest_data
            )
            if not is_corpus:
                ignored += 1
                continue
            try:
                manifest, records = load_corpus(
                    manifest_path.parent,
                    verify_code=False,
                    verify_replays=False,
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError("unable to load completed excluded corpus") from exc
            source = str(manifest_path.parent)
            corpus_count += 1
            record_count += len(records)
            scanned_records += len(records)
            corpora.append(
                {
                    "path": source,
                    "corpus_fingerprint": manifest.corpus_fingerprint,
                    "record_count": len(records),
                }
            )
            for record in records:
                record_config = normalize_config(record.replay.config)
                identity = _setup_identity(record_config, record.replay.seed)
                sources.setdefault(identity, []).append(source)
                material.setdefault(
                    identity,
                    {"setup_seed": record.replay.seed, "game_config": record_config},
                )
        root_rows.append(
            {
                "declared_root": str(declared),
                "resolved_root": str(root),
                "status": "present",
                "corpus_count": corpus_count,
                "record_count": record_count,
                "ignored_manifest_count": ignored,
                "current_output_excluded": output.is_relative_to(root),
            }
        )
    overlaps = sorted(set(current) & set(sources))
    data: dict[str, object] = {
        "version": "terminal-offense-confirmation-setup-holdout-v1",
        "scope": {
            "version": SETUP_HOLDOUT_SCOPE_VERSION,
            "declared_roots": [str(path) for path in roots],
            "recursive": True,
            "completed_corpus_detection": "manifest-and-games-jsonl-gzip-v1",
            "excludes_current_output_subtree": True,
        },
        "status": "passed" if not overlaps else "failed",
        "excluded_roots": root_rows,
        "current_output": str(output),
        "current_setup_count": len(current),
        "current_setup_fingerprint": _identity_set_fingerprint(set(current)),
        "prior_unique_setup_count": len(sources),
        "prior_setup_fingerprint": _identity_set_fingerprint(set(sources)),
        "scanned_corpus_count": len(corpora),
        "scanned_record_count": scanned_records,
        "scanned_corpora": sorted(corpora, key=lambda row: str(row["path"])),
        "overlap_count": len(overlaps),
        "overlap_fingerprint": _identity_set_fingerprint(set(overlaps)),
        "overlap_examples": [
            {
                "setup_identity": identity,
                "current": current[identity],
                "prior": material[identity],
                "prior_corpora": sorted(set(sources[identity])),
            }
            for identity in overlaps[:24]
        ],
        "artifact_fingerprint": "",
    }
    data["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in data.items() if key != "artifact_fingerprint"}
    )
    return data


def _target_pair_wins(report: ArenaReport, target_id: str) -> tuple[int, ...]:
    if target_id not in {report.agent_a_id, report.agent_b_id}:
        raise RuntimeError("target policy is absent from arena report")
    values = tuple(row.agent_a_wins for row in report.paired_seed_outcomes)
    return values if report.agent_a_id == target_id else tuple(2 - value for value in values)


def _cell(family: str, kind: str, opponent: str | None = None) -> ConfirmationCell:
    matches = tuple(
        value
        for value in confirmation_cells()
        if value.family == family and value.kind == kind and value.opponent_id == opponent
    )
    if len(matches) != 1:
        raise RuntimeError("confirmation cell lookup is not unique")
    return matches[0]


def _check_cell_schedule(
    cell: ConfirmationCell,
    records: tuple[GameRecord, ...],
    *,
    pair_count: int,
    rng_identities: Mapping[str, str],
) -> tuple[int, ...]:
    if len(records) != pair_count * 2:
        raise RuntimeError(f"record count mismatch: {cell.key}")
    master_seed = _family_master(cell.family)
    expected_setups = _family_setups(cell.family, pair_count)
    setup_block: list[int] = []
    for pair_index in range(pair_count):
        first = records[pair_index * 2]
        second = records[pair_index * 2 + 1]
        pair_id = f"pair-{pair_index:06d}"
        if (
            first.run_id != cell.run_id
            or second.run_id != cell.run_id
            or first.game_id != f"{pair_id}-a-first"
            or second.game_id != f"{pair_id}-b-first"
            or first.pair_id != pair_id
            or second.pair_id != pair_id
            or first.replay.seed != expected_setups[pair_index]
            or second.replay.seed != expected_setups[pair_index]
        ):
            raise RuntimeError(f"schedule identity mismatch: {cell.key}")
        expected_orders = (
            (cell.agent_a_id, cell.agent_b_id),
            (cell.agent_b_id, cell.agent_a_id),
        )
        for record, order in ((first, expected_orders[0]), (second, expected_orders[1])):
            if tuple(seat.agent_id for seat in record.seats) != order:
                raise RuntimeError(f"seat order mismatch: {cell.key}")
            for seat in record.seats:
                identity = rng_identities[seat.agent_id]
                expected_seed = derive_seed(
                    master_seed, f"arena:pair:{pair_index}:agent:{identity}"
                )
                if (
                    seat.seed != expected_seed
                    or seat.rng_identity != identity
                    or seat.rng_domain != f"agent:{identity}"
                ):
                    raise RuntimeError(f"agent RNG metadata mismatch: {cell.key}")
        setup_block.append(first.replay.seed)
    return tuple(setup_block)


def _independent_integrity(
    reports: Mapping[str, ArenaReport],
    manifests: Mapping[str, object],
    blocks: Mapping[str, tuple[int, ...]],
    *,
    pair_count: int,
    rng_identities: Mapping[str, str],
    record_count: int,
    decision_count: int,
) -> dict[str, object]:
    if set(reports) != {cell.key for cell in confirmation_cells()}:
        raise RuntimeError("incomplete confirmation report set")
    if blocks[FAMILY_IDS[0]] == blocks[FAMILY_IDS[1]]:
        raise RuntimeError("seed family setup blocks are identical")
    if not set(blocks[FAMILY_IDS[0]]).isdisjoint(blocks[FAMILY_IDS[1]]):
        raise RuntimeError("seed family setup blocks overlap")
    fingerprints: dict[str, str] = {}
    for key, manifest in manifests.items():
        fingerprint = getattr(manifest, "corpus_fingerprint", None)
        if not isinstance(fingerprint, str):
            raise RuntimeError("corpus manifest is malformed")
        fingerprints[key] = fingerprint
    return {
        "cell_count": len(reports),
        "record_count": record_count,
        "decision_count": decision_count,
        "pair_count_per_family": pair_count,
        "family_setup_seed_fingerprints": {
            family: hashlib.sha256(
                json.dumps(list(blocks[family]), separators=(",", ":")).encode()
            ).hexdigest()
            for family in FAMILY_IDS
        },
        "family_blocks_disjoint": True,
        "rng_identities": dict(sorted(rng_identities.items())),
        "corpus_fingerprints": dict(sorted(fingerprints.items())),
    }


def _metric_rows(reports: Mapping[str, ArenaReport]) -> dict[str, dict[str, tuple[float, ...]]]:
    rows: dict[str, dict[str, tuple[float, ...]]] = {}
    for family in FAMILY_IDS:
        family_metrics: dict[str, tuple[float, ...]] = {}
        differences: dict[str, tuple[float, ...]] = {}
        for opponent in FIELD_OPPONENTS:
            control = reports[_cell(family, "shared-opponent-control", opponent).key]
            treatment = reports[_cell(family, "shared-opponent-treatment", opponent).key]
            control_wins = _target_pair_wins(control, CONTROL_ID)
            treatment_wins = _target_pair_wins(treatment, TREATMENT_ID)
            if len(control_wins) != len(treatment_wins):
                raise RuntimeError("paired shared-opponent blocks differ")
            difference = tuple(
                (treatment_value - control_value) / 2
                for treatment_value, control_value in zip(treatment_wins, control_wins, strict=True)
            )
            differences[opponent] = difference
            family_metrics[f"opponent:{opponent}"] = difference
        block_count = len(next(iter(differences.values())))
        if any(len(values) != block_count for values in differences.values()):
            raise RuntimeError("common block alignment failed")
        family_metrics["anchor_macro"] = tuple(
            sum(differences[opponent][index] for opponent in ANCHOR_OPPONENTS)
            / len(ANCHOR_OPPONENTS)
            for index in range(block_count)
        )
        family_metrics["field_macro"] = tuple(
            sum(differences[opponent][index] for opponent in FIELD_OPPONENTS) / len(FIELD_OPPONENTS)
            for index in range(block_count)
        )
        direct = reports[_cell(family, "direct").key]
        family_metrics["direct_treatment_win_rate"] = tuple(
            wins / 2 for wins in _target_pair_wins(direct, TREATMENT_ID)
        )
        rows[family] = family_metrics
    return rows


def _independent_bootstrap(
    metrics: Mapping[str, Mapping[str, tuple[float, ...]]],
) -> dict[str, object]:
    if tuple(metrics) != FAMILY_IDS:
        raise RuntimeError("bootstrap strata are malformed")
    names = tuple(metrics[FAMILY_IDS[0]])
    if not names or any(tuple(metrics[family]) != names for family in FAMILY_IDS):
        raise RuntimeError("bootstrap metrics are not aligned")
    rows: dict[str, tuple[tuple[float, ...], ...]] = {}
    total_blocks = 0
    for family in FAMILY_IDS:
        lengths = {len(metrics[family][name]) for name in names}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise RuntimeError("bootstrap block lengths are malformed")
        count = next(iter(lengths))
        rows[family] = tuple(
            tuple(metrics[family][name][index] for name in names) for index in range(count)
        )
        total_blocks += count
    points = [0.0] * len(names)
    for family in FAMILY_IDS:
        for row in rows[family]:
            for index, value in enumerate(row):
                points[index] += value
    points = [value / total_blocks for value in points]

    seed = derive_seed(ROOT_SEED, _VALIDATOR_BOOTSTRAP_DOMAIN)
    rng = DeterministicRandom(seed, _VALIDATOR_BOOTSTRAP_DOMAIN)
    samples: list[list[float]] = [[] for _ in names]
    for _ in range(BOOTSTRAP_RESAMPLES):
        totals = [0.0] * len(names)
        for family in FAMILY_IDS:
            family_rows = rows[family]
            for _ in range(len(family_rows)):
                sampled = family_rows[rng.randbelow(len(family_rows))]
                for index, value in enumerate(sampled):
                    totals[index] += value
        for index, total in enumerate(totals):
            samples[index].append(total / total_blocks)
    metrics_data: dict[str, object] = {}
    for index, name in enumerate(names):
        ordered = sorted(samples[index])
        metrics_data[name] = {
            "point_estimate": points[index],
            "interval": [ordered[BOOTSTRAP_LOWER_INDEX], ordered[BOOTSTRAP_UPPER_INDEX]],
        }
    return {
        "metrics": metrics_data,
        "bootstrap_metadata": {
            "version": "stratified-joint-common-block-bootstrap-v1",
            "unit": "paired two-game setup block",
            "strata": list(FAMILY_IDS),
            "blocks_per_stratum": {family: len(rows[family]) for family in FAMILY_IDS},
            "resample_count": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "order_statistic_indices": {
                "lower": BOOTSTRAP_LOWER_INDEX,
                "upper": BOOTSTRAP_UPPER_INDEX,
            },
            "bootstrap_seed": seed,
            "bootstrap_rng_domain": _VALIDATOR_BOOTSTRAP_DOMAIN,
            "rng_algorithm": RNG_ALGORITHM,
            "seed_derivation": SEED_DERIVATION,
        },
    }


def _independent_statistics(reports: Mapping[str, ArenaReport]) -> dict[str, object]:
    bootstrap = _independent_bootstrap(_metric_rows(reports))
    metrics = bootstrap["metrics"]
    if not isinstance(metrics, dict):
        raise RuntimeError("bootstrap metric data is malformed")
    anchor = metrics["anchor_macro"]
    if not isinstance(anchor, dict) or not isinstance(anchor["interval"], list):
        raise RuntimeError("anchor metric is malformed")
    practical = float(anchor["interval"][0]) > PRACTICAL_LIFT_THRESHOLD
    return {
        "version": "terminal-offense-confirmation-statistics-v1",
        "treatment_minus_control_by_opponent": {
            opponent: metrics[f"opponent:{opponent}"] for opponent in FIELD_OPPONENTS
        },
        "anchor_macro": anchor,
        "field_macro": metrics["field_macro"],
        "direct_treatment_win_rate": metrics["direct_treatment_win_rate"],
        "direct_treatment_vs_control_interpretation": {
            "classification": "descriptive-only-shared-within-game-rng-identity",
            "reason": (
                "control and treatment intentionally share q0-terminal-core-v1 within each "
                "direct game, so this result is not causal matched evidence"
            ),
            "used_for_structural_adoption": False,
            "used_for_practical_lift_claim": False,
            "causal_matched_evidence": False,
            "causal_matched_evidence_scope": "shared-opponent control/treatment cells only",
        },
        "practical_lift_claim": {
            "criterion": "anchor-macro 95% lower bound > +0.25 percentage points",
            "threshold": PRACTICAL_LIFT_THRESHOLD,
            "passed": practical,
            "structural_adoption_blocking": False,
        },
        "bootstrap": bootstrap["bootstrap_metadata"],
    }


def _tactical_template() -> Counter[str]:
    return Counter(
        {
            "decisions": 0,
            "guaranteed_win_opportunities": 0,
            "guaranteed_win_conversions": 0,
            "guaranteed_win_misses": 0,
            "false_guaranteed_wins": 0,
            "q0_avoidable_immediate_loss_violations": 0,
        }
    )


def _tie_template() -> Counter[str]:
    return Counter(
        {
            "decisions": 0,
            "exact_max_logit_ties": 0,
            "exact_maximum_actions": 0,
            "ties_with_forced_maximum": 0,
            "ties_with_nonforced_maximum": 0,
            "ties_with_forced_and_nonforced_maxima": 0,
            "selected_in_exact_maxima": 0,
            "selected_outside_exact_maxima": 0,
        }
    )


def _counts(template: Counter[str], values: Mapping[str, int]) -> dict[str, int]:
    return {key: values[key] for key in template}


def _update_rows(
    overall: Counter[str],
    phases: dict[str, Counter[str]],
    opponents: dict[str, Counter[str]],
    *,
    phase: str,
    opponent: str,
    values: Mapping[str, int],
    template: Callable[[], Counter[str]],
) -> None:
    for row in (
        overall,
        phases.setdefault(phase, template()),
        opponents.setdefault(opponent, template()),
    ):
        row.update(values)


def _arm_rows(
    overall: Mapping[str, Counter[str]],
    phases: Mapping[str, Mapping[str, Counter[str]]],
    opponents: Mapping[str, Mapping[str, Counter[str]]],
    *,
    template: Callable[[], Counter[str]],
) -> dict[str, object]:
    empty = template()
    return {
        arm: {
            "counts": _counts(empty, overall[arm]),
            "by_phase": {
                phase: _counts(empty, counts)
                for phase, counts in sorted(phases.get(arm, {}).items())
            },
            "by_opponent": {
                opponent: _counts(empty, counts)
                for opponent, counts in sorted(opponents.get(arm, {}).items())
            },
        }
        for arm in ("control", "treatment")
    }


def _resolved_state(state: GameState, action: Action, following: Action | None) -> GameState:
    resolved = apply_action(state, action)
    if isinstance(action, PlayOfferAction):
        if not isinstance(following, RecruitAction):
            raise RuntimeError("play action has no following recruit resolution")
        resolved = apply_action(resolved, following)
    return resolved


def _learned_candidates(observation: PlayerObservation, treatment: bool) -> tuple[Action, ...]:
    if treatment:
        offense = filter_immediate_win_actions(observation, observation.legal_actions)
        reduced = replace(observation, legal_actions=offense.allowed_actions)
        return filter_terminal_actions(reduced, offense.allowed_actions).allowed_actions
    return filter_terminal_actions(observation, observation.legal_actions).allowed_actions


def _arm_index(record: GameRecord, arm_id: str) -> int:
    matches = tuple(index for index, seat in enumerate(record.seats) if seat.agent_id == arm_id)
    if len(matches) != 1:
        raise RuntimeError("matched record arm is malformed")
    return matches[0]


def _prefix_pair(control: GameRecord, treatment: GameRecord) -> dict[str, object]:
    control_index = _arm_index(control, CONTROL_ID)
    treatment_index = _arm_index(treatment, TREATMENT_ID)
    if control_index != treatment_index:
        raise RuntimeError("matched arms do not occupy the same seat")
    opponent_index = 1 - control_index
    control_arm = control.seats[control_index]
    treatment_arm = treatment.seats[treatment_index]
    control_opponent = control.seats[opponent_index]
    treatment_opponent = treatment.seats[opponent_index]
    if not (
        control.replay.seed == treatment.replay.seed
        and control_arm.seed == treatment_arm.seed
        and control_arm.rng_identity == treatment_arm.rng_identity == Q0_RNG_IDENTITY
        and control_opponent.agent_id == treatment_opponent.agent_id
        and control_opponent.seed == treatment_opponent.seed
        and control_opponent.rng_domain == treatment_opponent.rng_domain
    ):
        return {
            "metadata_aligned": False,
            "prefix_aligned": False,
            "endpoint_found": False,
            "classification": None,
            "aligned_actions": 0,
            "treatment_converted": False,
        }
    left = new_game(control.replay.config, control.replay.seed)
    right = new_game(treatment.replay.config, treatment.replay.seed)
    for index, (left_action, right_action) in enumerate(
        zip(control.replay.actions, treatment.replay.actions, strict=False)
    ):
        if left != right:
            return {
                "metadata_aligned": True,
                "prefix_aligned": False,
                "endpoint_found": False,
                "classification": None,
                "aligned_actions": index,
                "treatment_converted": False,
            }
        actor = left.active_player if left.phase is Phase.PLAY else left.active_player.other()
        if control.seats[player_index(actor)].agent_id == CONTROL_ID:
            observation = observe(left, actor)
            oracle = independent_public_forced_win_oracle(
                observation,
                observation.legal_actions,
                authoritative_state=left,
                exact_play_actions=(left_action,)
                if isinstance(left_action, PlayOfferAction)
                else (),
            )
            if oracle.forced_win_actions:
                control_converted = left_action in oracle.forced_win_actions
                treatment_converted = right_action in oracle.forced_win_actions
                return {
                    "metadata_aligned": True,
                    "prefix_aligned": True,
                    "endpoint_found": True,
                    "classification": (
                        "both_converted"
                        if control_converted and treatment_converted
                        else "control_missed_treatment_converted"
                        if treatment_converted
                        else "treatment_failed_guaranteed_win"
                    ),
                    "aligned_actions": index,
                    "treatment_converted": treatment_converted,
                }
        if left_action != right_action:
            return {
                "metadata_aligned": True,
                "prefix_aligned": False,
                "endpoint_found": False,
                "classification": None,
                "aligned_actions": index,
                "treatment_converted": False,
            }
        left = apply_action(left, left_action)
        right = apply_action(right, right_action)
    complete = len(control.replay.actions) == len(treatment.replay.actions)
    return {
        "metadata_aligned": True,
        "prefix_aligned": complete,
        "endpoint_found": False,
        "classification": None,
        "aligned_actions": min(len(control.replay.actions), len(treatment.replay.actions)),
        "treatment_converted": False,
    }


def _prefix_summary(
    family: str,
    opponent: str,
    control_records: tuple[GameRecord, ...],
    treatment_records: tuple[GameRecord, ...],
) -> Counter[str]:
    controls = {
        (record.pair_id, _arm_index(record, CONTROL_ID)): record for record in control_records
    }
    treatments = {
        (record.pair_id, _arm_index(record, TREATMENT_ID)): record for record in treatment_records
    }
    if None in {key[0] for key in controls} or set(controls) != set(treatments):
        raise RuntimeError("matched prefix records are incomplete")
    values = Counter(
        {
            "matched_games": 0,
            "metadata_mismatches": 0,
            "pre_endpoint_prefix_mismatches": 0,
            "guaranteed_win_endpoints": 0,
            "control_missed_treatment_converted": 0,
            "both_converted": 0,
            "treatment_endpoint_failures": 0,
            "games_without_guaranteed_win_endpoint": 0,
            "aligned_actions": 0,
        }
    )
    for key in sorted(controls, key=lambda item: (str(item[0]), item[1])):
        comparison = _prefix_pair(controls[key], treatments[key])
        endpoint = bool(comparison["endpoint_found"])
        classification = comparison["classification"]
        values.update(
            {
                "matched_games": 1,
                "metadata_mismatches": int(not bool(comparison["metadata_aligned"])),
                "pre_endpoint_prefix_mismatches": int(not bool(comparison["prefix_aligned"])),
                "guaranteed_win_endpoints": int(endpoint),
                "control_missed_treatment_converted": int(
                    classification == "control_missed_treatment_converted"
                ),
                "both_converted": int(classification == "both_converted"),
                "treatment_endpoint_failures": int(
                    endpoint and not bool(comparison["treatment_converted"])
                ),
                "games_without_guaranteed_win_endpoint": int(not endpoint),
                "aligned_actions": int(comparison["aligned_actions"]),
            }
        )
    return values


def _independent_audit(
    directories: Mapping[str, Path], scorers: Mapping[str, CandidateScorer]
) -> dict[str, object]:
    classifier = Counter(
        {
            "decisions": 0,
            "agreement_decisions": 0,
            "disagreement_decisions": 0,
            "candidate_action_classifications": 0,
            "exact_play_transition_cross_checks": 0,
        }
    )
    tactical = {"control": _tactical_template(), "treatment": _tactical_template()}
    tactical_phases: dict[str, dict[str, Counter[str]]] = {"control": {}, "treatment": {}}
    tactical_opponents: dict[str, dict[str, Counter[str]]] = {"control": {}, "treatment": {}}
    ties = {"control": _tie_template(), "treatment": _tie_template()}
    tie_phases: dict[str, dict[str, Counter[str]]] = {"control": {}, "treatment": {}}
    tie_opponents: dict[str, dict[str, Counter[str]]] = {"control": {}, "treatment": {}}
    prefix_rows: dict[str, Counter[str]] = {}
    pending: dict[tuple[str, str], tuple[GameRecord, ...]] = {}
    record_count = 0
    decision_count = 0

    for cell in confirmation_cells():
        _, records = load_corpus(directories[cell.key], verify_replays=False)
        for record in records:
            record_count += 1
            state = new_game(record.replay.config, record.replay.seed)
            actions = record.replay.actions
            for action_index, action in enumerate(actions):
                actor = (
                    state.active_player
                    if state.phase is Phase.PLAY
                    else state.active_player.other()
                )
                actor_index = player_index(actor)
                agent_id = record.seats[actor_index].agent_id
                opponent_id = record.seats[1 - actor_index].agent_id
                observation = observe(state, actor)
                independent = independent_public_forced_win_oracle(
                    observation,
                    observation.legal_actions,
                    authoritative_state=state,
                    exact_play_actions=(action,) if isinstance(action, PlayOfferAction) else (),
                )
                production = filter_immediate_win_actions(observation, observation.legal_actions)
                classifier.update(
                    {
                        "decisions": 1,
                        "agreement_decisions": int(
                            independent.forced_win_actions == production.forced_win_actions
                        ),
                        "disagreement_decisions": int(
                            independent.forced_win_actions != production.forced_win_actions
                        ),
                        "candidate_action_classifications": len(observation.legal_actions),
                        "exact_play_transition_cross_checks": (
                            independent.exact_transition_cross_checks
                        ),
                    }
                )
                following = actions[action_index + 1] if action_index + 1 < len(actions) else None
                resolved = _resolved_state(state, action, following)
                immediate_win = (
                    resolved.phase is Phase.TERMINAL
                    and resolved.outcome is not None
                    and resolved.outcome.winner is actor
                )
                arm = (
                    "control"
                    if agent_id == CONTROL_ID
                    else "treatment"
                    if agent_id == TREATMENT_ID
                    else None
                )
                if arm is not None:
                    safety = filter_terminal_actions(observation, observation.legal_actions)
                    opportunity = bool(independent.forced_win_actions)
                    converted = action in independent.forced_win_actions
                    _update_rows(
                        tactical[arm],
                        tactical_phases[arm],
                        tactical_opponents[arm],
                        phase=state.phase.value,
                        opponent=opponent_id,
                        values={
                            "decisions": 1,
                            "guaranteed_win_opportunities": int(opportunity),
                            "guaranteed_win_conversions": int(opportunity and converted),
                            "guaranteed_win_misses": int(opportunity and not converted),
                            "false_guaranteed_wins": int(converted and not immediate_win),
                            "q0_avoidable_immediate_loss_violations": int(
                                action in safety.provable_loss_actions
                                and not safety.forced_loss_fallback
                            ),
                        },
                        template=_tactical_template,
                    )
                    candidates = _learned_candidates(observation, arm == "treatment")
                    scored = scorers[agent_id](
                        replace(observation, legal_actions=candidates), candidates
                    )
                    if scored.actions != tuple(sorted(candidates, key=semantic_action_key)):
                        raise RuntimeError("candidate scorer action order is malformed")
                    maxima = scored.maximum_actions
                    forced_maxima = tuple(
                        item for item in maxima if item in independent.forced_win_actions
                    )
                    nonforced_maxima = tuple(
                        item for item in maxima if item not in independent.forced_win_actions
                    )
                    tied = scored.tie_count > 1
                    _update_rows(
                        ties[arm],
                        tie_phases[arm],
                        tie_opponents[arm],
                        phase=state.phase.value,
                        opponent=opponent_id,
                        values={
                            "decisions": 1,
                            "exact_max_logit_ties": int(tied),
                            "exact_maximum_actions": scored.tie_count,
                            "ties_with_forced_maximum": int(bool(tied and forced_maxima)),
                            "ties_with_nonforced_maximum": int(bool(tied and nonforced_maxima)),
                            "ties_with_forced_and_nonforced_maxima": int(
                                bool(tied and forced_maxima and nonforced_maxima)
                            ),
                            "selected_in_exact_maxima": int(action in maxima),
                            "selected_outside_exact_maxima": int(action not in maxima),
                        },
                        template=_tie_template,
                    )
                decision_count += 1
                state = apply_action(state, action)
        if cell.kind == "shared-opponent-control":
            assert cell.opponent_id is not None
            pending[(cell.family, cell.opponent_id)] = records
        elif cell.kind == "shared-opponent-treatment":
            assert cell.opponent_id is not None
            key = (cell.family, cell.opponent_id)
            if key not in pending:
                raise RuntimeError("treatment corpus preceded its control corpus")
            prefix_rows[f"{cell.family}:{cell.opponent_id}"] = _prefix_summary(
                cell.family, cell.opponent_id, pending.pop(key), records
            )
    if pending:
        raise RuntimeError("unmatched control corpora remain")
    prefix_counts: Counter[str] = Counter()
    for row in prefix_rows.values():
        prefix_counts.update(row)
    return {
        "record_count": record_count,
        "decision_count": decision_count,
        "oracle_counts": dict(classifier),
        "tactical": _arm_rows(
            tactical, tactical_phases, tactical_opponents, template=_tactical_template
        ),
        "ties": _arm_rows(ties, tie_phases, tie_opponents, template=_tie_template),
        "prefix": {
            "counts": dict(prefix_counts),
            "by_family_opponent": {key: dict(value) for key, value in sorted(prefix_rows.items())},
        },
    }


def _compare_audit(retained: dict[str, Any], independent: dict[str, object]) -> None:
    if (
        retained["record_count"] != independent["record_count"]
        or retained["decision_count"] != independent["decision_count"]
        or retained["independent_production_oracle_agreement"]["counts"]
        != independent["oracle_counts"]
        or retained["guaranteed_win_and_safety"] != independent["tactical"]
        or retained["exact_max_logit_ties"]["by_arm"] != independent["ties"]
        or retained["matched_action_prefixes"]["counts"] != independent["prefix"]["counts"]
        or retained["matched_action_prefixes"]["by_family_opponent"]
        != independent["prefix"]["by_family_opponent"]
    ):
        raise RuntimeError("independent replay-derived audit summary mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "validation.json"
    plan = _read(root / "plan.json")
    result = _read(root / "result.json")
    plan_payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    if _fingerprint(plan_payload) != plan["plan_fingerprint"]:
        raise RuntimeError("plan fingerprint mismatch")
    result_payload = {key: value for key, value in result.items() if key != "result_fingerprint"}
    if _fingerprint(result_payload) != result["result_fingerprint"]:
        raise RuntimeError("result fingerprint mismatch")
    if plan["cycle_id"] != CYCLE_ID or plan["root_seed"] != ROOT_SEED:
        raise RuntimeError("plan cycle or root seed differs from frozen design")
    if plan["frozen_default_design"]["pairs_per_family"] != DEFAULT_PAIRS_PER_FAMILY:
        raise RuntimeError("plan default pair count differs from frozen design")
    if plan["frozen_default_design"]["total_games"] != 60_000:
        raise RuntimeError("plan default game count differs from frozen design")
    holdout_scope = plan["setup_holdout_scope"]
    if (
        not isinstance(holdout_scope, dict)
        or holdout_scope.get("version") != SETUP_HOLDOUT_SCOPE_VERSION
        or holdout_scope.get("recursive") is not True
        or holdout_scope.get("excludes_current_output_subtree") is not True
        or not isinstance(holdout_scope.get("declared_roots"), list)
        or not holdout_scope["declared_roots"]
    ):
        raise RuntimeError("plan setup holdout scope is malformed")
    default_design = bool(plan["execution"]["uses_frozen_default_design"])
    if default_design and holdout_scope["declared_roots"] != [
        str(path) for path in DEFAULT_EXCLUDED_SETUP_ROOTS
    ]:
        raise RuntimeError("claim plan setup holdout scope differs from frozen design")
    if plan["execution_budget"] != {
        "claim_run_cutoff_seconds": 27_900,
        "claim_run_cutoff": "7h45m",
        "whole_step_compute_budget_seconds": 28_800,
        "whole_step_compute_budget": "8h",
        "incomplete_execution": "no result is emitted before every cell and audit completes",
        "allowed_resume_retries": 1,
    }:
        raise RuntimeError("plan execution budget differs from frozen design")
    if plan["direct_treatment_vs_control"]["causal_matched_evidence"] is not False:
        raise RuntimeError("plan direct comparison interpretation is malformed")
    if default_design:
        _require_frozen_source(plan)

    retained_holdout = _read(root / "analysis" / "setup-block-holdout.json")
    holdout_payload = {
        key: value for key, value in retained_holdout.items() if key != "artifact_fingerprint"
    }
    if _fingerprint(holdout_payload) != retained_holdout["artifact_fingerprint"]:
        raise RuntimeError("setup-block holdout artifact fingerprint mismatch")
    recomputed_holdout = _independent_holdout_scan(
        pair_count=int(plan["execution"]["pairs_per_family"]),
        roots=tuple(Path(value) for value in plan["setup_holdout_scope"]["declared_roots"]),
        current_output=root,
    )
    if recomputed_holdout != retained_holdout:
        raise RuntimeError("setup-block holdout scan differs from recomputation")
    if retained_holdout["overlap_count"] != 0:
        raise RuntimeError("confirmation setup block overlaps prior retained corpora")
    if plan["setup_holdout"] != {
        "artifact_fingerprint": retained_holdout["artifact_fingerprint"],
        "status": retained_holdout["status"],
        "overlap_count": retained_holdout["overlap_count"],
        "current_setup_fingerprint": retained_holdout["current_setup_fingerprint"],
        "prior_setup_fingerprint": retained_holdout["prior_setup_fingerprint"],
    }:
        raise RuntimeError("plan setup holdout summary mismatch")

    policy_data = plan["policies"]
    specs: dict[str, AgentSpec] = {}
    rng_identities: dict[str, str] = {}
    for agent_id, raw in policy_data.items():
        specs[agent_id] = AgentSpec(
            agent_id,
            raw["config"],
            RandomAgent,
            rng_identity=str(raw["rng_identity"]),
        )
        rng_identities[agent_id] = str(raw["rng_identity"])
    if (
        rng_identities[CONTROL_ID] != Q0_RNG_IDENTITY
        or rng_identities[TREATMENT_ID] != Q0_RNG_IDENTITY
    ):
        raise RuntimeError("control/treatment RNG identity is not frozen")

    pair_count = int(plan["execution"]["pairs_per_family"])
    reports: dict[str, ArenaReport] = {}
    manifests: dict[str, object] = {}
    directories: dict[str, Path] = {}
    blocks: dict[str, tuple[int, ...]] = {}
    cell_artifacts: dict[str, str] = {}
    record_count = 0
    decision_count = 0
    for cell in confirmation_cells():
        directory = root / "arena-records" / cell.family / cell.cell_id
        artifact = _read(root / "arenas" / cell.family / f"{cell.cell_id}.json")
        artifact_payload = {
            key: value for key, value in artifact.items() if key != "artifact_fingerprint"
        }
        if _fingerprint(artifact_payload) != artifact["artifact_fingerprint"]:
            raise RuntimeError(f"cell artifact fingerprint mismatch: {cell.key}")
        manifest, records = load_corpus(directory)
        if manifest.corpus_fingerprint != artifact["records_corpus_fingerprint"]:
            raise RuntimeError(f"cell corpus fingerprint mismatch: {cell.key}")
        block = _check_cell_schedule(
            cell, records, pair_count=pair_count, rng_identities=rng_identities
        )
        if cell.family in blocks and blocks[cell.family] != block:
            raise RuntimeError("family common setup block mismatch")
        blocks[cell.family] = block
        retained_report = artifact["report"]
        report = arena_report_from_records(
            ArenaConfig(
                run_id=cell.run_id,
                agent_a=specs[cell.agent_a_id],
                agent_b=specs[cell.agent_b_id],
                pair_count=pair_count,
                master_seed=_family_master(cell.family),
            ),
            records,
            elapsed_seconds=float(retained_report["elapsed_seconds"]),
        )
        if report.to_data() != retained_report:
            raise RuntimeError(f"cell arena report mismatch: {cell.key}")
        reports[cell.key] = report
        manifests[cell.key] = manifest
        directories[cell.key] = directory
        cell_artifacts[cell.key] = str(artifact["artifact_fingerprint"])
        record_count += len(records)
        decision_count += sum(record.decision_count for record in records)

    integrity = _independent_integrity(
        reports,
        manifests,
        blocks,
        pair_count=pair_count,
        rng_identities=rng_identities,
        record_count=record_count,
        decision_count=decision_count,
    )
    retained_integrity = _read(root / "analysis" / "integrity.json")
    integrity_keys = (
        "cell_count",
        "record_count",
        "decision_count",
        "pair_count_per_family",
        "family_setup_seed_fingerprints",
        "family_blocks_disjoint",
        "rng_identities",
        "corpus_fingerprints",
    )
    if any(integrity[key] != retained_integrity[key] for key in integrity_keys):
        raise RuntimeError("independent schedule/integrity summary mismatch")

    statistics = _independent_statistics(reports)
    retained_statistics = _read(root / "analysis" / "statistics.json")
    if statistics != retained_statistics:
        raise RuntimeError("independent stratified statistics mismatch")
    if statistics["direct_treatment_vs_control_interpretation"]["used_for_practical_lift_claim"]:
        raise RuntimeError("direct comparison incorrectly enters the practical-lift claim")

    audit = _independent_audit(directories, _scorers(plan))
    retained_audit = _read(root / "analysis" / "replay-audit.json")
    _compare_audit(retained_audit, audit)
    if result["statistics"] != statistics or result["replay_audit"] != retained_audit:
        raise RuntimeError("result does not embed retained independently checked analysis")
    if result["integrity"] != retained_integrity:
        raise RuntimeError("result does not embed retained integrity")
    if result["setup_block_holdout"] != retained_holdout:
        raise RuntimeError("result does not embed setup holdout evidence")
    if (
        result["direct_treatment_vs_control"]
        != statistics["direct_treatment_vs_control_interpretation"]
    ):
        raise RuntimeError("result direct comparison interpretation mismatch")
    if result["practical_lift_claim"] != statistics["practical_lift_claim"]:
        raise RuntimeError("result practical-lift claim differs from independent statistics")

    tactical = audit["tactical"]
    ties = audit["ties"]
    oracle_counts = audit["oracle_counts"]
    prefixes = audit["prefix"]["counts"]
    expected_structural = [
        bool(plan["execution"]["uses_frozen_default_design"])
        and plan["seed_validation"]["status"] == "passed"
        and bool(integrity["family_blocks_disjoint"]),
        retained_holdout["status"] == "passed" and retained_holdout["overlap_count"] == 0,
        True,
        oracle_counts["disagreement_decisions"] == 0,
        tactical["treatment"]["counts"]["guaranteed_win_misses"] == 0
        and tactical["treatment"]["counts"]["false_guaranteed_wins"] == 0,
        tactical["control"]["counts"]["q0_avoidable_immediate_loss_violations"] == 0
        and tactical["treatment"]["counts"]["q0_avoidable_immediate_loss_violations"] == 0,
        ties["control"]["counts"]["selected_outside_exact_maxima"] == 0
        and ties["treatment"]["counts"]["selected_outside_exact_maxima"] == 0,
        prefixes["metadata_mismatches"] == 0
        and prefixes["pre_endpoint_prefix_mismatches"] == 0
        and prefixes["treatment_endpoint_failures"] == 0,
    ]
    structural = result["structural_adoption"]
    retained_passes = [bool(row["passed"]) for row in structural["criteria"]]
    if (
        retained_passes != expected_structural
        or bool(structural["passed"]) != all(expected_structural)
        or bool(structural["eligible_default_design"])
        != bool(plan["execution"]["uses_frozen_default_design"])
    ):
        raise RuntimeError("structural adoption decision differs from independent recomputation")
    if result["cell_artifact_fingerprints"] != dict(sorted(cell_artifacts.items())):
        raise RuntimeError("result cell artifact fingerprints mismatch")
    if plan["seed_validation"] != _seed_validation(pair_count):
        raise RuntimeError("plan seed validation mismatch")

    validation: dict[str, object] = {
        "version": "terminal-offense-confirmation-independent-validation-v2",
        "status": "passed",
        "validation_source_revision": inspect_source_identity().git_revision,
        "plan_fingerprint": plan["plan_fingerprint"],
        "result_fingerprint": result["result_fingerprint"],
        "cell_count": integrity["cell_count"],
        "record_count": integrity["record_count"],
        "decision_count": integrity["decision_count"],
        "independent_path": {
            "schedule": "validator-local-record-schedule-check-v1",
            "statistics": "validator-local-stratified-joint-bootstrap-v1",
            "audit": "validator-local-replay-and-prefix-audit-v1",
        },
        "setup_holdout_fingerprint": retained_holdout["artifact_fingerprint"],
        "prior_setup_overlap_count": retained_holdout["overlap_count"],
        "oracle_disagreement_decisions": oracle_counts["disagreement_decisions"],
        "artifact_fingerprint": "",
    }
    validation["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in validation.items() if key != "artifact_fingerprint"}
    )
    _write(output, validation)
    print(json.dumps(validation, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
