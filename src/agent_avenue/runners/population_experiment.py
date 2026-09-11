"""Resumable implementation of the frozen M7 Step-2 population replay.

The scheduling primitives live in :mod:`agent_avenue.runners.population`; this module owns only
execution evidence.  It deliberately keeps NumPy/PyTorch and learned-agent imports inside explicit
execution helpers so importing the trusted runner package remains usable without the ``rl`` extra.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

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
from agent_avenue.engine.setup import normalize_config
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    inspect_source_identity,
    load_corpus,
    repository_root,
    rules_fingerprint,
)

from .arena import ArenaConfig, arena_report_from_records, run_resumable_arena, schedule_arena
from .corpus import run_resumable_corpus, validate_record_matches_spec
from .game import AgentSpec, GameSpec
from .population import (
    POPULATION_PAIR_COUNT,
    POPULATION_POLICY_IDS,
    POPULATION_REPLAY_CYCLE_ID,
    POPULATION_REPLAY_REPLICATE_IDS,
    PopulationArm,
    PopulationCorpusConfig,
    PopulationCorpusPlan,
    PopulationMember,
    audit_population_alignment,
    audit_population_arm,
    build_population_corpus_plan,
    enumerate_population_training_setup_blocks,
    population_setup_identity,
)
from .safety_audit import audit_terminal_safety
from .strength_audit import audit_public_forced_wins

PLAN_VERSION: Final = "m7-population-replay-plan-v1"
INPUT_AUDIT_VERSION: Final = "m7-population-replay-input-audit-v1"
CORPUS_ARTIFACT_VERSION: Final = "m7-population-replay-corpus-artifact-v1"
DATASET_AUDIT_VERSION: Final = "m7-population-replay-dataset-audit-v1"
TRAINING_SUMMARY_VERSION: Final = "m7-population-replay-training-summary-v1"
ARENA_ARTIFACT_VERSION: Final = "m7-population-replay-arena-artifact-v1"
RESULT_VERSION: Final = "m7-population-replay-result-v1"
VALIDATION_VERSION: Final = "m7-population-replay-independent-validation-v1"
NESTED_BOOTSTRAP_VERSION: Final = "m7-population-replay-nested-bootstrap-v1"
NESTED_BOOTSTRAP_RESAMPLES: Final = 20_000
NESTED_BOOTSTRAP_LOWER_INDEX: Final = 499
NESTED_BOOTSTRAP_UPPER_INDEX: Final = 19_499
CLAIM_RUN_CUTOFF_SECONDS: Final = 7 * 60 * 60 + 45 * 60
WHOLE_STEP_COMPUTE_BUDGET_SECONDS: Final = 8 * 60 * 60
ALLOWED_RESUME_RETRIES: Final = 1
FIXED_CREATED_AT: Final = "2026-09-11T00:00:00+00:00"

Q0_PARENT: Final = "q0"
HISTORICAL_Q0: Final = "historical-q0"
HEURISTIC: Final = "greedy-public-v1"
RANDOM: Final = "random"
CANDIDATE_ARMS: Final[tuple[PopulationArm, PopulationArm]] = ("control", "treatment")
DEFAULT_CHECKPOINT_PATHS: Final[dict[str, Path]] = {
    "q0": Path("runs/terminal-safety-v1/q0-a1/checkpoint"),
    "q1": Path("runs/terminal-safety-v1/q1-a1/candidate"),
    "q2": Path("runs/terminal-safety-v1/q2-a1/candidate"),
    "q3": Path("runs/terminal-safety-v1/q3-a1/candidate"),
    "q4": Path("runs/terminal-safety-v1/q4-a1/candidate"),
    HISTORICAL_Q0: Path("runs/terminal-safety-v1/inputs/historical-q0"),
}


class PopulationExperimentError(ValueError):
    """Raised when retained Step-2 evidence is incompatible with the frozen agreement."""


@dataclass(frozen=True, slots=True)
class PopulationExperimentConfig:
    """Paths and explicit bounded-smoke controls for the Step-2 runner.

    Defaults are the frozen claim design.  A non-``None`` ``smoke_pair_count`` is explicitly
    labelled non-claim evidence and only truncates already-frozen schedules; it never alters their
    seed domains, policy assignment, or default values.
    """

    output: Path
    checkpoint_paths: Mapping[str, Path] = field(default_factory=lambda: DEFAULT_CHECKPOINT_PATHS)
    holdout_roots: tuple[Path, ...] = (Path("runs"),)
    smoke_pair_count: int | None = None
    smoke_max_epochs: int | None = None

    def __post_init__(self) -> None:
        expected = {*POPULATION_POLICY_IDS[:5], HISTORICAL_Q0}
        if set(self.checkpoint_paths) != expected:
            raise PopulationExperimentError("checkpoint paths must contain q0-q4 and historical-q0")
        if not self.holdout_roots:
            raise PopulationExperimentError("population replay requires at least one holdout root")
        if (
            self.smoke_pair_count is not None
            and not 2 <= self.smoke_pair_count <= POPULATION_PAIR_COUNT
        ):
            raise PopulationExperimentError("smoke_pair_count must be in [2, 2000]")
        if self.smoke_max_epochs is not None and self.smoke_max_epochs < 1:
            raise PopulationExperimentError("smoke_max_epochs must be positive")

    @property
    def claim_default(self) -> bool:
        return self.smoke_pair_count is None and self.smoke_max_epochs is None

    @property
    def pairs_per_cell(self) -> int:
        return POPULATION_PAIR_COUNT if self.smoke_pair_count is None else self.smoke_pair_count

    @property
    def max_epochs(self) -> int:
        return 50 if self.smoke_max_epochs is None else self.smoke_max_epochs


@dataclass(frozen=True, slots=True)
class PopulationPolicyBundle:
    """Raw policy specs plus immutable checkpoint identities used to build a plan."""

    members: tuple[PopulationMember, ...]
    historical_q0: AgentSpec
    checkpoint_identities: Mapping[str, Mapping[str, object]]
    mode: str

    def member(self, policy_id: str) -> PopulationMember:
        matches = tuple(member for member in self.members if member.policy_id == policy_id)
        if len(matches) != 1:
            raise PopulationExperimentError(f"policy bundle lacks unique member {policy_id!r}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class PopulationArenaCell:
    """One retained fresh-development comparison.

    The parent-vs-heuristic reference is retained because it is required for the declared shared
    heuristic difference.  It adds 600 physical games per replicate beyond the agreement's stated
    7,400 arithmetic; the plan records this discrepancy rather than hiding it.
    """

    replicate_id: str
    key: str
    candidate_arm: PopulationArm | None
    opponent: str
    pair_count: int
    master_seed: int
    shared_group: str | None
    direct: bool = False

    @property
    def run_id(self) -> str:
        return f"{POPULATION_REPLAY_CYCLE_ID}-{self.replicate_id}-{self.key}"

    def to_data(self) -> dict[str, object]:
        return {
            "replicate_id": self.replicate_id,
            "key": self.key,
            "run_id": self.run_id,
            "candidate_arm": self.candidate_arm,
            "opponent": self.opponent,
            "paired_blocks": self.pair_count,
            "games": self.pair_count * 2,
            "master_seed": self.master_seed,
            "shared_group": self.shared_group,
            "direct": self.direct,
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _json_copy(value: object) -> object:
    return json.loads(_canonical_json(_jsonable(value)))


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PopulationExperimentError(f"unable to read JSON artifact: {path}") from exc
    if not isinstance(data, dict):
        raise PopulationExperimentError(f"artifact must be a JSON object: {path}")
    return cast(dict[str, object], data)


def _write_immutable(path: Path, value: Mapping[str, object], *, label: str) -> None:
    normalized = cast(dict[str, object], _json_copy(dict(value)))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != normalized:
            raise PopulationExperimentError(f"existing immutable {label} differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(_canonical_json(normalized) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        return str(resolved)


def _agent_config(agent: Agent) -> dict[str, object]:
    method = getattr(agent, "config_to_data", None)
    if callable(method):
        value = method()
    else:
        config = getattr(agent, "config", None)
        to_data = getattr(config, "to_data", None)
        if not callable(to_data):
            raise PopulationExperimentError("agent has no normalized configuration")
        value = to_data()
    if not isinstance(value, dict):
        raise PopulationExperimentError("agent configuration must be a JSON object")
    return cast(dict[str, object], _json_copy(value))


def build_population_policy_bundle(
    checkpoint_paths: Mapping[str, Path], *, toy: bool = False
) -> PopulationPolicyBundle:
    """Load raw q0-q4/history identity only on explicit optional-RL execution paths."""
    if toy:
        toy_members = tuple(
            PopulationMember(
                policy_id,
                AgentSpec(
                    f"raw-{policy_id}",
                    RandomAgentConfig().to_data(),
                    RandomAgent,
                ),
            )
            for policy_id in POPULATION_POLICY_IDS
        )
        historical = AgentSpec("historical-q0", RandomAgentConfig().to_data(), RandomAgent)
        return PopulationPolicyBundle(toy_members, historical, {}, "toy-random-smoke")

    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import load_checkpoint

    loaded = {policy_id: load_checkpoint(path) for policy_id, path in checkpoint_paths.items()}
    members: list[PopulationMember] = []
    identities: dict[str, Mapping[str, object]] = {}
    for policy_id in POPULATION_POLICY_IDS[:5]:
        checkpoint = loaded[policy_id]

        def factory(checkpoint: object = checkpoint) -> Agent:
            from agent_avenue.learning import LoadedCheckpoint

            if not isinstance(checkpoint, LoadedCheckpoint):  # pragma: no cover - closure guard
                raise PopulationExperimentError("learned checkpoint closure is malformed")
            return LearnedValueAgent.from_checkpoint(checkpoint)

        raw = factory()
        members.append(
            PopulationMember(policy_id, AgentSpec(f"raw-{policy_id}", _agent_config(raw), factory))
        )
        identities[policy_id] = {
            "locator": _portable(checkpoint_paths[policy_id]),
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "tensor_digest": checkpoint.manifest["tensor_digest"],
        }
    heuristic = GreedyHeuristicAgent(GreedyHeuristicConfig())
    random = RandomAgent(RandomAgentConfig())
    members.extend(
        (
            PopulationMember(
                HEURISTIC,
                AgentSpec(
                    "raw-greedy-public-v1",
                    _agent_config(heuristic),
                    lambda: GreedyHeuristicAgent(GreedyHeuristicConfig()),
                ),
            ),
            PopulationMember(
                RANDOM,
                AgentSpec(
                    "raw-random", _agent_config(random), lambda: RandomAgent(RandomAgentConfig())
                ),
            ),
        )
    )
    historical_checkpoint = loaded[HISTORICAL_Q0]

    def historical_factory(checkpoint: object = historical_checkpoint) -> Agent:
        from agent_avenue.learning import LoadedCheckpoint

        if not isinstance(checkpoint, LoadedCheckpoint):  # pragma: no cover
            raise PopulationExperimentError("historical checkpoint closure is malformed")
        return LearnedValueAgent.from_checkpoint(checkpoint)

    historical = AgentSpec("historical-q0", _agent_config(historical_factory()), historical_factory)
    identities[HISTORICAL_Q0] = {
        "locator": _portable(checkpoint_paths[HISTORICAL_Q0]),
        "checkpoint_fingerprint": historical_checkpoint.checkpoint_fingerprint,
        "tensor_digest": historical_checkpoint.manifest["tensor_digest"],
    }
    return PopulationPolicyBundle(tuple(members), historical, identities, "learned-checkpoints")


def _arena_master_seed(plan: PopulationCorpusPlan, replicate_id: str, key: str) -> int:
    replicate = plan.replicate(replicate_id)
    domain = f"{POPULATION_REPLAY_CYCLE_ID}:replicate:{replicate_id}:arena:{key}:v1"
    return derive_seed(replicate.seed_plan.arena_seed, domain) & ((1 << 63) - 1)


def population_arena_cells(
    corpus_plan: PopulationCorpusPlan, *, pair_count: int
) -> tuple[PopulationArenaCell, ...]:
    """Return all fresh named development cells in stable order.

    Shared opponent groups intentionally reuse only their group seed.  Candidate logical identities
    are supplied later and remain arm-independent for these paired comparisons.
    """
    if pair_count < 1 or pair_count > POPULATION_PAIR_COUNT:
        raise PopulationExperimentError("arena pair count must be in [1, 2000]")
    values: list[PopulationArenaCell] = []
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:

        def cell(
            key: str,
            arm: PopulationArm | None,
            opponent: str,
            pairs: int,
            shared: str | None,
            *,
            direct: bool = False,
            current_replicate_id: str = replicate_id,
        ) -> None:
            actual_pairs = min(pairs, pair_count)
            seed_key = shared or key
            values.append(
                PopulationArenaCell(
                    current_replicate_id,
                    key,
                    arm,
                    opponent,
                    actual_pairs,
                    _arena_master_seed(corpus_plan, current_replicate_id, seed_key),
                    shared,
                    direct,
                )
            )

        cell("treatment-vs-control", None, "control", 500, None, direct=True)
        for arm in CANDIDATE_ARMS:
            cell(f"{arm}-vs-parent", arm, Q0_PARENT, 500, "parent")
        for arm in CANDIDATE_ARMS:
            cell(f"{arm}-vs-heuristic", arm, HEURISTIC, 300, "heuristic")
        cell("parent-vs-heuristic-reference", None, HEURISTIC, 300, "heuristic-reference")
        for arm in CANDIDATE_ARMS:
            cell(f"{arm}-vs-random", arm, RANDOM, 200, "random")
            cell(f"{arm}-vs-historical-q0", arm, HISTORICAL_Q0, 200, "historical-q0")
        for opponent in ("q1", "q2", "q3", "q4"):
            for arm in CANDIDATE_ARMS:
                cell(f"{arm}-vs-{opponent}", arm, opponent, 100, opponent)
    return tuple(values)


def _declared_arena_game_counts(cells: Iterable[PopulationArenaCell]) -> dict[str, int]:
    all_cells = tuple(cells)
    candidate_games = sum(
        cell.pair_count * 2 for cell in all_cells if cell.key != "parent-vs-heuristic-reference"
    )
    reference_games = sum(
        cell.pair_count * 2 for cell in all_cells if cell.key == "parent-vs-heuristic-reference"
    )
    return {
        "candidate_comparison_games": candidate_games,
        "reference_games": reference_games,
        "physical_games": candidate_games + reference_games,
    }


def _holdout_scan(
    corpus_plan: PopulationCorpusPlan,
    cells: Iterable[PopulationArenaCell],
    *,
    roots: tuple[Path, ...],
    current_output: Path,
    smoke_pair_count: int | None,
) -> dict[str, object]:
    """Recursively inventory completed corpora and reject every non-intentional setup overlap."""
    current: dict[str, dict[str, object]] = {}
    corpus_pairs = POPULATION_PAIR_COUNT if smoke_pair_count is None else smoke_pair_count
    for candidate in enumerate_population_training_setup_blocks(corpus_plan):
        if candidate.pair_index >= corpus_pairs:
            continue
        current[candidate.setup_identity] = {
            "domain": "training",
            "replicate_id": candidate.replicate_id,
            "pair_id": candidate.pair_id,
            "setup_seed": candidate.setup_seed,
            "game_config": dict(candidate.game_config),
            "intentional_duplicate_arms": ["control", "treatment"],
        }
    for cell in cells:
        for spec in schedule_arena(
            ArenaConfig(
                cell.run_id,
                AgentSpec("schedule-a", {"type": "schedule"}, RandomAgent),
                AgentSpec("schedule-b", {"type": "schedule-b"}, RandomAgent),
                cell.pair_count,
                cell.master_seed,
            )
        ):
            identity = population_setup_identity(normalize_config(spec.config), spec.setup_seed)
            existing = current.get(identity)
            material: dict[str, object] = {
                "domain": "development",
                "replicate_id": cell.replicate_id,
                "arena": cell.key,
                "pair_id": spec.pair_id,
                "setup_seed": spec.setup_seed,
                "shared_group": cell.shared_group,
            }
            if existing is not None:
                same_arena_pair = (
                    existing.get("domain") == "development"
                    and existing.get("replicate_id") == cell.replicate_id
                    and existing.get("arena") == cell.key
                )
                same_declared_shared_group = (
                    existing.get("domain") == "development"
                    and cell.shared_group is not None
                    and existing.get("replicate_id") == cell.replicate_id
                    and existing.get("shared_group") == cell.shared_group
                )
                if not (same_arena_pair or same_declared_shared_group):
                    raise PopulationExperimentError(
                        "proposed training/development setup overlap: "
                        f"{existing['domain']} and development"
                    )
            else:
                current[identity] = material

    output = current_output.resolve()
    seen: set[Path] = set()
    prior: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    root_rows: list[dict[str, object]] = []
    for declared_root in roots:
        root = declared_root.resolve()
        if not root.exists():
            root_rows.append({"declared_root": str(declared_root), "status": "missing"})
            continue
        if not root.is_dir():
            raise PopulationExperimentError(f"holdout root is not a directory: {declared_root}")
        corpora = 0
        records_count = 0
        for manifest_path in sorted(root.rglob("manifest.json")):
            if manifest_path.resolve().is_relative_to(output):
                continue
            if manifest_path.resolve() in seen:
                continue
            seen.add(manifest_path.resolve())
            if not (manifest_path.parent / "games.jsonl.gz").exists():
                continue
            try:
                manifest, records = load_corpus(
                    manifest_path.parent, verify_code=False, verify_replays=False
                )
            except (OSError, ValueError) as exc:
                raise PopulationExperimentError(
                    f"unable to load completed corpus in holdout scan: {manifest_path.parent}"
                ) from exc
            corpora += 1
            records_count += len(records)
            rows.append(
                {
                    "path": str(manifest_path.parent.resolve()),
                    "corpus_fingerprint": manifest.corpus_fingerprint,
                    "record_count": len(records),
                }
            )
            for record in records:
                identity = population_setup_identity(
                    normalize_config(record.replay.config), record.replay.seed
                )
                prior.setdefault(identity, []).append(str(manifest_path.parent.resolve()))
        root_rows.append(
            {
                "declared_root": str(declared_root),
                "resolved_root": str(root),
                "status": "present",
                "corpus_count": corpora,
                "record_count": records_count,
                "current_output_excluded": output.is_relative_to(root),
            }
        )
    overlap = sorted(set(current) & set(prior))
    data: dict[str, object] = {
        "version": "m7-population-replay-setup-holdout-v1",
        "scope": {
            "recursive": True,
            "completed_corpus_detection": "manifest-and-games-jsonl-gzip-v1",
            "excludes_current_output_subtree": True,
            "intentional_duplicate": "control-treatment-within-one-training-replicate",
        },
        "status": "passed" if not overlap else "failed",
        "current_setup_count": len(current),
        "current_setup_fingerprint": _fingerprint(sorted(current)),
        "prior_setup_count": len(prior),
        "prior_setup_fingerprint": _fingerprint(sorted(prior)),
        "overlap_count": len(overlap),
        "overlap_fingerprint": _fingerprint(overlap),
        "overlap_examples": [
            {
                "setup_identity": identity,
                "current": current[identity],
                "prior_corpora": sorted(set(prior[identity])),
            }
            for identity in overlap[:24]
        ],
        "roots": root_rows,
        "scanned_corpora": rows,
        "artifact_fingerprint": "",
    }
    data["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in data.items() if key != "artifact_fingerprint"}
    )
    return data


def build_population_experiment_plan(
    config: PopulationExperimentConfig, bundle: PopulationPolicyBundle
) -> dict[str, object]:
    """Freeze source, checkpoint, assignment, corpus, and arena declarations before dispatch."""
    source = inspect_source_identity()
    corpus_config = PopulationCorpusConfig(members=bundle.members)
    corpus_plan = build_population_corpus_plan(corpus_config)
    cells = population_arena_cells(corpus_plan, pair_count=config.pairs_per_cell)
    count_data = _declared_arena_game_counts(cells)
    claim_reasons: list[str] = []
    if not config.claim_default:
        claim_reasons.append("bounded_smoke_execution")
    if bundle.mode != "learned-checkpoints":
        claim_reasons.append("toy_policy_bundle")
    if not source.tracked_tree_clean:
        claim_reasons.append("tracked_source_tree_is_dirty")
    claim_reasons.append("frozen_arena_total_conflict_7400_comparisons_plus_600_reference_games")
    # The agreement simultaneously requires a 300-pair parent reference and says 7,400 games.
    # Retain both facts faithfully rather than silently dropping evidence or games.
    if config.claim_default and (
        count_data["physical_games"] != 8_000 * len(POPULATION_REPLAY_REPLICATE_IDS)
    ):
        raise PopulationExperimentError("population arena physical-game schedule is malformed")
    source_data = {
        **source.to_data(),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }
    payload: dict[str, object] = {
        "version": PLAN_VERSION,
        "cycle_id": POPULATION_REPLAY_CYCLE_ID,
        "source": source_data,
        "input_paths": {key: _portable(path) for key, path in config.checkpoint_paths.items()},
        "checkpoint_identities": _json_copy(bundle.checkpoint_identities),
        "corpus_plan": corpus_plan.to_data(),
        "corpus_plan_fingerprint": corpus_plan.fingerprint,
        "execution": {
            "mode": bundle.mode,
            "evidence_class": "claim-eligible-default"
            if not claim_reasons
            else "smoke-only-nondefault",
            "claim_eligible": not claim_reasons,
            "claim_ineligibility_reasons": claim_reasons,
            "pairs_per_corpus_arm": config.pairs_per_cell,
            "corpus_games_per_arm": config.pairs_per_cell * 2,
            "training_max_epochs": config.max_epochs,
            "output": str(config.output.resolve()),
        },
        "frozen_default_design": {
            "root_seed": 2026091102,
            "replicates": list(POPULATION_REPLAY_REPLICATE_IDS),
            "pairs_per_corpus_arm": POPULATION_PAIR_COUNT,
            "games_per_corpus_arm": 4_000,
            "total_training_games": 24_000,
            "training": {
                "encoder": "candidate-public-v1",
                "model": "candidate-mlp-v1",
                "parameters": 11_393,
                "target": "acting-player-eventual-terminal-outcome-for-selected-action",
                "loss": "binary-cross-entropy-with-logits",
                "optimizer": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 1024,
                "max_epochs": 50,
                "patience": 8,
                "cpu_threads": 1,
                "deterministic_algorithms": True,
            },
            "claim_cutoff_seconds": CLAIM_RUN_CUTOFF_SECONDS,
            "whole_step_budget_seconds": WHOLE_STEP_COMPUTE_BUDGET_SECONDS,
            "allowed_resume_retries": ALLOWED_RESUME_RETRIES,
        },
        "holdout_scope": {
            "roots": [str(path.resolve()) for path in config.holdout_roots],
            "recursive": True,
            "excludes_current_output_subtree": True,
        },
        "arenas": {
            "cells": [cell.to_data() for cell in cells],
            "agreement_stated_games_per_replicate": 7_400,
            "comparison_games_per_replicate": count_data["candidate_comparison_games"] // 3,
            "required_parent_heuristic_reference_games_per_replicate": count_data["reference_games"]
            // 3,
            "physical_games_per_replicate": count_data["physical_games"] // 3,
            "agreement_arithmetic_note": (
                "The listed candidate comparisons total 7,400 games; adding the separately "
                "required 300-pair fresh parent-vs-heuristic reference totals 8,000 physical "
                "games."
            ),
        },
        "plan_fingerprint": "",
    }
    payload["plan_fingerprint"] = _fingerprint(
        {key: value for key, value in payload.items() if key != "plan_fingerprint"}
    )
    return payload


def _corpus_plan_from_bundle(bundle: PopulationPolicyBundle) -> PopulationCorpusPlan:
    return build_population_corpus_plan(PopulationCorpusConfig(members=bundle.members))


def _slice_specs(specs: tuple[GameSpec, ...], pairs: int) -> tuple[GameSpec, ...]:
    return specs[: pairs * 2]


def _validate_sliced_schedule(records: Iterable[GameRecord], specs: tuple[GameSpec, ...]) -> None:
    buffered = tuple(records)
    if len(buffered) != len(specs):
        raise PopulationExperimentError("bounded corpus has an incorrect record count")
    for record, spec in zip(buffered, specs, strict=True):
        validate_record_matches_spec(record, spec)


def _corpus_artifact(
    output: Path,
    plan: Mapping[str, object],
    corpus_plan: PopulationCorpusPlan,
    replicate_id: str,
    arm: PopulationArm,
    *,
    pairs: int,
) -> tuple[Path, tuple[GameRecord, ...], dict[str, object]]:
    specs = _slice_specs(corpus_plan.game_specs(replicate_id, arm), pairs)
    directory = output / "corpora" / replicate_id / arm
    manifest = run_resumable_corpus(
        directory,
        specs,
        behavior_policy="m7-population-terminal-offense-safety-epsilon-v1",
        root_seed=2026091102,
        generation=None,
        configuration={
            "version": CORPUS_ARTIFACT_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "replicate_id": replicate_id,
            "arm": arm,
            "corpus_plan_fingerprint": corpus_plan.fingerprint,
            "bounded_smoke": pairs != POPULATION_PAIR_COUNT,
        },
    )
    loaded_manifest, records = load_corpus(directory)
    if manifest.corpus_fingerprint != loaded_manifest.corpus_fingerprint:
        raise PopulationExperimentError("loaded corpus manifest changed after finalization")
    if pairs == POPULATION_PAIR_COUNT:
        audit = audit_population_arm(corpus_plan, replicate_id, arm, records).to_data()
    else:
        _validate_sliced_schedule(records, specs)
        audit = {"status": "passed", "bounded_schedule_records": len(records)}
    artifact: dict[str, object] = {
        "version": CORPUS_ARTIFACT_VERSION,
        "plan_fingerprint": plan["plan_fingerprint"],
        "replicate_id": replicate_id,
        "arm": arm,
        "records": {
            "locator": _portable(directory),
            "corpus_fingerprint": manifest.corpus_fingerprint,
        },
        "schedule_audit": audit,
        "artifact_fingerprint": "",
    }
    artifact["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    _write_immutable(
        output / "corpus-artifacts" / replicate_id / f"{arm}.json",
        artifact,
        label="corpus artifact",
    )
    return directory, records, artifact


def _coverage_for_records(
    records: tuple[GameRecord, ...],
    dataset_manifest: Mapping[str, object],
    corpus_plan: PopulationCorpusPlan,
) -> dict[str, object]:
    from agent_avenue.storage import game_record_fingerprint

    split = dataset_manifest.get("split")
    if not isinstance(split, Mapping):
        raise PopulationExperimentError("dataset has no split metadata")
    by_fingerprint = {game_record_fingerprint(record): record for record in records}
    members = corpus_plan.config.member_by_agent_id
    output: dict[str, object] = {}
    for name in ("train", "validation"):
        values = split.get(f"{name}_game_fingerprints")
        if not isinstance(values, list):
            raise PopulationExperimentError("dataset split game fingerprints are malformed")
        policy_ids: set[str] = set()
        pair_ids: set[str] = set()
        for value in values:
            if not isinstance(value, str) or value not in by_fingerprint:
                raise PopulationExperimentError("dataset refers to unknown source record")
            record = by_fingerprint[value]
            if record.pair_id is None:
                raise PopulationExperimentError(
                    "population replay dataset lost paired-block provenance"
                )
            pair_ids.add(record.pair_id)
            for seat in record.seats:
                member = members.get(seat.agent_id)
                if member is not None:
                    policy_ids.add(member.policy_id)
        output[name] = {"pair_count": len(pair_ids), "policy_ids": sorted(policy_ids)}
    return output


def _dataset_artifact(
    output: Path,
    plan: Mapping[str, object],
    corpus_plan: PopulationCorpusPlan,
    replicate_id: str,
    arm: PopulationArm,
    records: tuple[GameRecord, ...],
    corpus_fingerprint: str,
    *,
    require_coverage: bool,
) -> tuple[Path, dict[str, object]]:
    from agent_avenue.learning import load_dataset, materialize_dataset, save_dataset

    split_seed = corpus_plan.replicate(replicate_id).seed_plan.split_seed
    expected = materialize_dataset(
        records,
        split_seed=split_seed,
        validation_fraction=0.1,
        source_corpus_fingerprint=corpus_fingerprint,
    )
    path = output / "datasets" / replicate_id / f"{arm}.npz"
    if path.exists():
        dataset = load_dataset(path)
        if dataset.fingerprint != expected.fingerprint:
            raise PopulationExperimentError(
                "resumed dataset does not match deterministic extraction"
            )
    else:
        save_dataset(expected, path)
        dataset = load_dataset(path)
    coverage = _coverage_for_records(records, dataset.manifest, corpus_plan)
    if arm == "treatment" and require_coverage:
        required = list(POPULATION_POLICY_IDS)
        for split_name in ("train", "validation"):
            observed = cast(dict[str, object], coverage[split_name])["policy_ids"]
            if observed != required:
                raise PopulationExperimentError(
                    f"treatment {split_name} split does not cover every frozen population member"
                )
    artifact: dict[str, object] = {
        "version": DATASET_AUDIT_VERSION,
        "plan_fingerprint": plan["plan_fingerprint"],
        "replicate_id": replicate_id,
        "arm": arm,
        "dataset": {
            "locator": _portable(path),
            "fingerprint": dataset.fingerprint,
            "array_sha256": _sha256_file(path),
            "manifest_sha256": _sha256_file(path.with_suffix(".json")),
            "counts": dataset.manifest["counts"],
            "split": dataset.manifest["split"],
        },
        "provenance_coverage": coverage,
        "coverage_required": require_coverage,
        "artifact_fingerprint": "",
    }
    artifact["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    )
    _write_immutable(
        output / "dataset-audits" / replicate_id / f"{arm}.json", artifact, label="dataset audit"
    )
    return path, artifact


def _training_config(replicate: Any, *, max_epochs: int) -> Any:
    from agent_avenue.learning import TrainingConfig

    seed_plan = replicate.seed_plan
    return TrainingConfig(
        seed=seed_plan.initialization_seed & ((1 << 63) - 1),
        model_seed=seed_plan.initialization_seed & ((1 << 63) - 1),
        shuffle_seed=seed_plan.shuffle_seed & ((1 << 63) - 1),
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=1024,
        max_epochs=max_epochs,
        early_stopping_patience=8,
        cpu_threads=1,
    )


def _train_checkpoint(
    output: Path,
    plan: Mapping[str, object],
    corpus_plan: PopulationCorpusPlan,
    replicate_id: str,
    arm: PopulationArm,
    dataset_path: Path,
    dataset_fingerprint: str,
    corpus_fingerprint: str,
    parent_path: Path,
    *,
    max_epochs: int,
) -> dict[str, object]:
    from agent_avenue.learning import (
        inspect_checkpoint,
        load_checkpoint,
        load_dataset,
        save_checkpoint,
        tensor_digest,
        train_model,
    )

    replicate = corpus_plan.replicate(replicate_id)
    config = _training_config(replicate, max_epochs=max_epochs)
    parent = load_checkpoint(parent_path)
    parent_tensor = tensor_digest(parent.model.state_dict())
    destination = output / "checkpoints" / replicate_id / arm
    if not destination.exists():
        dataset = load_dataset(dataset_path)
        trained = train_model(dataset, config=config, initial_state_dict=parent.model.state_dict())
        save_checkpoint(
            destination,
            trained.model,
            metrics=trained.metrics_dict(),
            training_config=config.normalized(),
            training_seeds={
                "initialization": replicate.seed_plan.initialization_seed,
                "shuffle": replicate.seed_plan.shuffle_seed,
                "fresh_optimizer": True,
            },
            dataset_fingerprint=dataset_fingerprint,
            source_corpus_fingerprints=(corpus_fingerprint,),
            parent_checkpoint=parent.checkpoint_fingerprint,
            generation=1,
            metadata={
                "experiment": PLAN_VERSION,
                "plan_fingerprint": plan["plan_fingerprint"],
                "replicate_id": replicate_id,
                "arm": arm,
                "initial_parent_tensor_digest": parent_tensor,
                "paired_shuffle_seed": replicate.seed_plan.shuffle_seed,
                "initialization_mode": "identical-q0-parent-tensor-weights-fresh-adamw",
            },
            created_at=FIXED_CREATED_AT,
        )
    inspection = inspect_checkpoint(destination)
    loaded = load_checkpoint(destination)
    manifest = loaded.manifest
    lineage = manifest.get("lineage")
    metadata = manifest.get("metadata")
    training = manifest.get("training")
    sources = manifest.get("sources")
    if not all(isinstance(value, Mapping) for value in (lineage, metadata, training, sources)):
        raise PopulationExperimentError("trained checkpoint lacks required lineage metadata")
    lineage_data = cast(Mapping[str, object], lineage)
    metadata_data = cast(Mapping[str, object], metadata)
    training_data = cast(Mapping[str, object], training)
    sources_data = cast(Mapping[str, object], sources)
    if (
        lineage_data.get("parent_checkpoint") != parent.checkpoint_fingerprint
        or lineage_data.get("generation") != 1
        or metadata_data.get("plan_fingerprint") != plan["plan_fingerprint"]
        or metadata_data.get("arm") != arm
        or metadata_data.get("replicate_id") != replicate_id
        or metadata_data.get("initial_parent_tensor_digest") != parent_tensor
        or training_data.get("config") != config.normalized()
        or sources_data.get("dataset_fingerprint") != dataset_fingerprint
        or tuple(cast(tuple[object, ...], sources_data.get("corpus_fingerprints", ())))
        != (corpus_fingerprint,)
    ):
        raise PopulationExperimentError("trained checkpoint lineage/configuration mismatch")
    return {
        "locator": _portable(destination),
        "checkpoint_fingerprint": inspection.checkpoint_fingerprint,
        "tensor_digest": inspection.tensor_digest,
        "parent_checkpoint_fingerprint": parent.checkpoint_fingerprint,
        "parent_tensor_digest": parent_tensor,
        "training_config": config.normalized(),
        "metrics": _json_copy(loaded.metrics),
    }


def _enveloped_checkpoint_agent(path: Path, agent_id: str, rng_identity: str) -> AgentSpec:
    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import load_checkpoint

    checkpoint = load_checkpoint(path)

    def factory() -> Agent:
        return TerminalOffenseAgent(
            TerminalSafetyAgent(LearnedValueAgent.from_checkpoint(checkpoint))
        )

    return AgentSpec(agent_id, _agent_config(factory()), factory, rng_identity=rng_identity)


def _canonical_opponent(
    bundle: PopulationPolicyBundle, opponent: str, replicate_id: str
) -> AgentSpec:
    if opponent == HEURISTIC:
        agent = GreedyHeuristicAgent(GreedyHeuristicConfig())
        return AgentSpec(
            HEURISTIC,
            _agent_config(agent),
            lambda: GreedyHeuristicAgent(GreedyHeuristicConfig()),
            rng_identity=f"{replicate_id}:heuristic",
        )
    if opponent == RANDOM:
        random_agent = RandomAgent(RandomAgentConfig())
        return AgentSpec(
            RANDOM,
            _agent_config(random_agent),
            lambda: RandomAgent(RandomAgentConfig()),
            rng_identity=f"{replicate_id}:random",
        )
    if opponent == HISTORICAL_Q0:
        spec = bundle.historical_q0
        return AgentSpec(
            spec.agent_id,
            dict(spec.config),
            spec.factory,
            rng_identity=f"{replicate_id}:historical-q0",
        )
    if bundle.mode == "toy-random-smoke" and opponent in {Q0_PARENT, "q1", "q2", "q3", "q4"}:

        def toy_factory() -> Agent:
            return TerminalOffenseAgent(TerminalSafetyAgent(RandomAgent(RandomAgentConfig())))

        return AgentSpec(
            f"enveloped-{opponent}",
            _agent_config(toy_factory()),
            toy_factory,
            rng_identity=f"{replicate_id}:{opponent}",
        )
    if opponent == Q0_PARENT or opponent in {"q1", "q2", "q3", "q4"}:
        # q0-q4 parent/opponents use the fixed offense+safety evaluation envelope, without epsilon.
        return _enveloped_checkpoint_agent(
            Path(cast(str, bundle.checkpoint_identities[opponent]["locator"])),
            f"enveloped-{opponent}",
            f"{replicate_id}:{opponent}",
        )
    raise PopulationExperimentError(f"unknown arena opponent: {opponent}")


def _candidate_checkpoint_path(output: Path, replicate_id: str, arm: PopulationArm) -> Path:
    return output / "checkpoints" / replicate_id / arm


def _arena_agents(
    cell: PopulationArenaCell, output: Path, bundle: PopulationPolicyBundle
) -> tuple[AgentSpec, AgentSpec]:
    replicate_id = cell.replicate_id
    candidate_identity = f"{POPULATION_REPLAY_CYCLE_ID}:{replicate_id}:candidate-lane"
    if cell.direct:
        treatment = _enveloped_checkpoint_agent(
            _candidate_checkpoint_path(output, replicate_id, "treatment"),
            "treatment-candidate",
            f"{POPULATION_REPLAY_CYCLE_ID}:{replicate_id}:treatment-direct-lane",
        )
        control = _enveloped_checkpoint_agent(
            _candidate_checkpoint_path(output, replicate_id, "control"),
            "control-candidate",
            f"{POPULATION_REPLAY_CYCLE_ID}:{replicate_id}:control-direct-lane",
        )
        return treatment, control
    if cell.key == "parent-vs-heuristic-reference":
        return _canonical_opponent(bundle, Q0_PARENT, replicate_id), _canonical_opponent(
            bundle, HEURISTIC, replicate_id
        )
    if cell.candidate_arm is None:
        raise PopulationExperimentError("non-direct arena has no candidate arm")
    candidate = _enveloped_checkpoint_agent(
        _candidate_checkpoint_path(output, replicate_id, cell.candidate_arm),
        f"{cell.candidate_arm}-candidate",
        candidate_identity,
    )
    return candidate, _canonical_opponent(bundle, cell.opponent, replicate_id)


def _enveloped_invariants(
    records: tuple[GameRecord, ...], source_fingerprint: str
) -> dict[str, object]:
    safety = audit_terminal_safety(records, source_corpus_fingerprint=source_fingerprint)
    offense = audit_public_forced_wins(records, source_label=source_fingerprint)
    safety_agents = cast(dict[str, object], safety["by_agent"])
    offense_agents = cast(dict[str, object], offense["by_agent"])
    checked: dict[str, object] = {}
    passed = True
    for agent_id, value in safety_agents.items():
        if not isinstance(value, Mapping):
            raise PopulationExperimentError("safety audit agent row is malformed")
        config = value.get("config")
        if not isinstance(config, Mapping) or config.get("type") != "terminal_offense":
            continue
        safety_counts = value.get("counts")
        offense_row = offense_agents.get(agent_id)
        if not isinstance(safety_counts, Mapping) or not isinstance(offense_row, Mapping):
            raise PopulationExperimentError("tactical audit rows are malformed")
        offense_counts = offense_row.get("counts")
        if not isinstance(offense_counts, Mapping):
            raise PopulationExperimentError("offense audit counts are malformed")
        avoidable = safety_counts.get("executed_avoidable_provable_losses")
        missed = offense_counts.get("missed_forced_wins")
        if type(avoidable) is not int or type(missed) is not int:
            raise PopulationExperimentError("tactical invariant counts are malformed")
        checked[agent_id] = {
            "avoidable_immediate_losses": avoidable,
            "missed_guaranteed_wins": missed,
            "passed": avoidable == 0 and missed == 0,
        }
        passed &= avoidable == 0 and missed == 0
    return {
        "passed": passed and bool(checked),
        "checked_agents": checked,
        "safety": safety,
        "offense": offense,
    }


def _arena_artifact(
    output: Path,
    plan: Mapping[str, object],
    cell: PopulationArenaCell,
    bundle: PopulationPolicyBundle,
) -> dict[str, object]:
    agent_a, agent_b = _arena_agents(cell, output, bundle)
    config = ArenaConfig(cell.run_id, agent_a, agent_b, cell.pair_count, cell.master_seed)
    artifact_path = output / "arenas" / cell.replicate_id / f"{cell.key}.json"
    records_directory = output / "arena-records" / cell.replicate_id / cell.key
    if artifact_path.exists():
        artifact = _read_json(artifact_path)
        expected = {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
        if artifact.get("artifact_fingerprint") != _fingerprint(expected):
            raise PopulationExperimentError("existing arena artifact fingerprint mismatch")
        manifest, records = load_corpus(records_directory)
        report = artifact.get("report")
        runtime = report.get("elapsed_seconds") if isinstance(report, Mapping) else None
        if not isinstance(runtime, int | float):
            raise PopulationExperimentError("existing arena report runtime is malformed")
        if (
            arena_report_from_records(config, records, elapsed_seconds=float(runtime)).to_data()
            != report
        ):
            raise PopulationExperimentError("existing arena aggregate no longer matches records")
        records_data = artifact.get("records")
        if not isinstance(records_data, Mapping) or (
            records_data.get("corpus_fingerprint") != manifest.corpus_fingerprint
        ):
            raise PopulationExperimentError("existing arena corpus fingerprint mismatch")
        if artifact.get("tactical_invariants") != _enveloped_invariants(
            records, manifest.corpus_fingerprint
        ):
            raise PopulationExperimentError("existing arena tactical diagnostics changed")
        return artifact
    retained = run_resumable_arena(
        records_directory,
        config,
        generation=1,
        corpus_configuration={
            "version": ARENA_ARTIFACT_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "cell": cell.to_data(),
            "fresh_named_arena_domain": True,
        },
    )
    _, records = load_corpus(records_directory)
    saved_artifact: dict[str, object] = {
        "version": ARENA_ARTIFACT_VERSION,
        "plan_fingerprint": plan["plan_fingerprint"],
        "cell": cell.to_data(),
        "records": {
            "locator": _portable(records_directory),
            "corpus_fingerprint": retained.records_manifest.corpus_fingerprint,
        },
        "report": retained.report.to_data(),
        "tactical_invariants": _enveloped_invariants(
            records, retained.records_manifest.corpus_fingerprint
        ),
        "artifact_fingerprint": "",
    }
    saved_artifact["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in saved_artifact.items() if key != "artifact_fingerprint"}
    )
    _write_immutable(artifact_path, saved_artifact, label="arena artifact")
    return saved_artifact


def _pair_scores(artifact: Mapping[str, object]) -> tuple[float, ...]:
    report = artifact.get("report")
    if not isinstance(report, Mapping):
        raise PopulationExperimentError("arena artifact has no report")
    values = report.get("paired_seed_outcomes")
    if not isinstance(values, list):
        raise PopulationExperimentError("arena report has no paired outcomes")
    output: list[float] = []
    for value in values:
        if not isinstance(value, Mapping) or type(value.get("agent_a_wins")) is not int:
            raise PopulationExperimentError("paired outcome is malformed")
        output.append(cast(int, value["agent_a_wins"]) / 2)
    return tuple(output)


def _seat_rates(artifact: Mapping[str, object]) -> dict[str, float]:
    report = artifact.get("report")
    rows = report.get("agent_a_by_seat") if isinstance(report, Mapping) else None
    if not isinstance(rows, Mapping):
        raise PopulationExperimentError("arena report lacks candidate seat data")
    output: dict[str, float] = {}
    for name, row in rows.items():
        if (
            not isinstance(name, str)
            or not isinstance(row, Mapping)
            or not isinstance(row.get("win_rate"), int | float)
        ):
            raise PopulationExperimentError("candidate seat data is malformed")
        output[name] = float(cast(float, row["win_rate"]))
    return output


def nested_population_bootstrap(
    replicate_blocks: tuple[tuple[float, ...], ...], *, seed: int, domain: str
) -> dict[str, object]:
    """Deterministic two-level bootstrap: training replicates then paired setup blocks."""
    if len(replicate_blocks) != 3 or any(not values for values in replicate_blocks):
        raise PopulationExperimentError(
            "nested bootstrap requires exactly three non-empty replicates"
        )
    block_count = len(replicate_blocks[0])
    if any(len(values) != block_count for values in replicate_blocks):
        raise PopulationExperimentError(
            "nested bootstrap requires matched block counts per replicate"
        )
    point = sum(sum(values) / block_count for values in replicate_blocks) / len(replicate_blocks)
    stream_seed = derive_seed(seed, f"{POPULATION_REPLAY_CYCLE_ID}:nested-bootstrap:{domain}:v1")
    rng = DeterministicRandom(
        stream_seed, f"{POPULATION_REPLAY_CYCLE_ID}:nested-bootstrap:{domain}:v1"
    )
    samples: list[float] = []
    for _ in range(NESTED_BOOTSTRAP_RESAMPLES):
        selected_replicates = [rng.randbelow(3) for _ in range(3)]
        replicate_means: list[float] = []
        for index in selected_replicates:
            rows = replicate_blocks[index]
            replicate_means.append(
                sum(rows[rng.randbelow(block_count)] for _ in range(block_count)) / block_count
            )
        samples.append(sum(replicate_means) / 3)
    samples.sort()
    return {
        "version": NESTED_BOOTSTRAP_VERSION,
        "unit": "paired training replicate, then paired setup block",
        "replicate_count": 3,
        "blocks_per_replicate": block_count,
        "resamples": NESTED_BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "order_statistic_indices": {
            "lower": NESTED_BOOTSTRAP_LOWER_INDEX,
            "upper": NESTED_BOOTSTRAP_UPPER_INDEX,
        },
        "point_estimate": point,
        "interval": [samples[NESTED_BOOTSTRAP_LOWER_INDEX], samples[NESTED_BOOTSTRAP_UPPER_INDEX]],
        "seed": stream_seed,
        "domain": domain,
    }


def _artifact_by_key(
    artifacts: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str], Mapping[str, object]]:
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for artifact in artifacts:
        cell = artifact.get("cell")
        if not isinstance(cell, Mapping):
            raise PopulationExperimentError("arena artifact cell is malformed")
        replicate = cell.get("replicate_id")
        key = cell.get("key")
        if not isinstance(replicate, str) or not isinstance(key, str):
            raise PopulationExperimentError("arena artifact cell identity is malformed")
        result[(replicate, key)] = artifact
    return result


def _paired_difference(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise PopulationExperimentError("shared-opponent paired blocks are not aligned")
    return tuple(a - b for a, b in zip(left, right, strict=True))


def population_statistics(
    artifacts: Iterable[Mapping[str, object]], corpus_plan: PopulationCorpusPlan
) -> dict[str, object]:
    """Aggregate only replicate/block-aware development statistics and decision inputs."""
    by_key = _artifact_by_key(artifacts)
    direct: list[tuple[float, ...]] = []
    treatment_parent: list[tuple[float, ...]] = []
    control_parent: list[tuple[float, ...]] = []
    treatment_random: list[tuple[float, ...]] = []
    treatment_parent_heuristic: list[tuple[float, ...]] = []
    differences: dict[str, list[tuple[float, ...]]] = {
        name: [] for name in (HEURISTIC, RANDOM, HISTORICAL_Q0, "q1", "q2", "q3", "q4")
    }
    per_replicate: list[dict[str, object]] = []
    minimum_seat = 1.0
    all_tactical_passed = True
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:

        def scores(key: str, current_replicate_id: str = replicate_id) -> tuple[float, ...]:
            return _pair_scores(by_key[(current_replicate_id, key)])

        direct_scores = scores("treatment-vs-control")
        treatment_parent_scores = scores("treatment-vs-parent")
        control_parent_scores = scores("control-vs-parent")
        direct.append(direct_scores)
        treatment_parent.append(treatment_parent_scores)
        control_parent.append(control_parent_scores)
        differences[HEURISTIC].append(
            _paired_difference(scores("treatment-vs-heuristic"), scores("control-vs-heuristic"))
        )
        differences[RANDOM].append(
            _paired_difference(scores("treatment-vs-random"), scores("control-vs-random"))
        )
        differences[HISTORICAL_Q0].append(
            _paired_difference(
                scores("treatment-vs-historical-q0"), scores("control-vs-historical-q0")
            )
        )
        for opponent in ("q1", "q2", "q3", "q4"):
            differences[opponent].append(
                _paired_difference(
                    scores(f"treatment-vs-{opponent}"), scores(f"control-vs-{opponent}")
                )
            )
        treatment_random.append(scores("treatment-vs-random"))
        treatment_parent_heuristic.append(
            _paired_difference(
                scores("treatment-vs-heuristic"), scores("parent-vs-heuristic-reference")
            )
        )
        candidate_artifacts = [
            artifact
            for (replicate, key), artifact in by_key.items()
            if replicate == replicate_id
            and (
                key.startswith("control-vs-")
                or key.startswith("treatment-vs-")
                or key == "treatment-vs-control"
            )
        ]
        seat_values: list[float] = []
        for artifact in candidate_artifacts:
            seat_values.extend(_seat_rates(artifact).values())
            tactical = artifact.get("tactical_invariants")
            if not isinstance(tactical, Mapping) or tactical.get("passed") is not True:
                all_tactical_passed = False
        direct_artifact = by_key[(replicate_id, "treatment-vs-control")]
        direct_seats = _seat_rates(direct_artifact)
        # direct agent-a is treatment; invert it for the control candidate's same physical seats.
        seat_values.extend(1.0 - value for value in direct_seats.values())
        replicate_minimum = min(seat_values)
        minimum_seat = min(minimum_seat, replicate_minimum)
        per_replicate.append(
            {
                "replicate_id": replicate_id,
                "direct_treatment_vs_control": sum(direct_scores) / len(direct_scores),
                "treatment_vs_parent": sum(treatment_parent_scores) / len(treatment_parent_scores),
                "control_vs_parent": sum(control_parent_scores) / len(control_parent_scores),
                "treatment_vs_random": sum(treatment_random[-1]) / len(treatment_random[-1]),
                "treatment_minus_control_heuristic": sum(differences[HEURISTIC][-1])
                / len(differences[HEURISTIC][-1]),
                "treatment_minus_parent_heuristic": sum(treatment_parent_heuristic[-1])
                / len(treatment_parent_heuristic[-1]),
                "minimum_candidate_seat_rate": replicate_minimum,
            }
        )
    seed = corpus_plan.replicate(POPULATION_REPLAY_REPLICATE_IDS[0]).seed_plan.bootstrap_seed
    intervals: dict[str, object] = {
        "treatment_vs_control": nested_population_bootstrap(
            tuple(direct), seed=seed, domain="treatment-vs-control"
        ),
        "treatment_vs_parent": nested_population_bootstrap(
            tuple(treatment_parent), seed=seed, domain="treatment-vs-parent"
        ),
        "control_vs_parent": nested_population_bootstrap(
            tuple(control_parent), seed=seed, domain="control-vs-parent"
        ),
        "treatment_vs_random": nested_population_bootstrap(
            tuple(treatment_random), seed=seed, domain="treatment-vs-random"
        ),
        "treatment_minus_parent_heuristic": nested_population_bootstrap(
            tuple(treatment_parent_heuristic), seed=seed, domain="treatment-minus-parent-heuristic"
        ),
    }
    for opponent, rows in differences.items():
        intervals[f"treatment_minus_control_{opponent}"] = nested_population_bootstrap(
            tuple(rows), seed=seed, domain=f"treatment-minus-control-{opponent}"
        )
    direct_points = [
        float(cast(float, row["direct_treatment_vs_control"])) for row in per_replicate
    ]
    variation = math.sqrt(sum((value - sum(direct_points) / 3) ** 2 for value in direct_points) / 2)
    return {
        "version": "m7-population-replay-statistics-v1",
        "replicates": per_replicate,
        "nested": intervals,
        "between_replicate": {"direct_treatment_vs_control_sample_standard_deviation": variation},
        "minimum_candidate_seat_rate": minimum_seat,
        "tactical_invariants_passed": all_tactical_passed,
    }


def population_decision(
    statistics: Mapping[str, object], *, integrity_passed: bool
) -> dict[str, object]:
    """Apply the agreement's conjunctive advancement criteria without selecting a champion."""
    nested = statistics.get("nested")
    replicates = statistics.get("replicates")
    if not isinstance(nested, Mapping) or not isinstance(replicates, list):
        raise PopulationExperimentError("statistics are malformed")

    def lower(name: str) -> float:
        row = nested.get(name)
        interval = row.get("interval") if isinstance(row, Mapping) else None
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not isinstance(interval[0], int | float)
        ):
            raise PopulationExperimentError(f"nested interval missing: {name}")
        return float(interval[0])

    direct_favors = sum(
        1
        for row in replicates
        if isinstance(row, Mapping)
        and isinstance(row.get("direct_treatment_vs_control"), int | float)
        and float(row["direct_treatment_vs_control"]) > 0.5
    )
    minimum_seat = statistics.get("minimum_candidate_seat_rate")
    if not isinstance(minimum_seat, int | float):
        raise PopulationExperimentError("minimum candidate seat rate is malformed")
    conditions = {
        "direct_nested_lower_strictly_above_50_percent": lower("treatment_vs_control") > 0.5,
        "at_least_two_direct_replicates_favor_treatment": direct_favors >= 2,
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
        "no_candidate_seat_point_below_45_percent": float(minimum_seat) >= 0.45,
        "zero_enveloped_avoidable_losses_and_missed_guaranteed_wins": statistics.get(
            "tactical_invariants_passed"
        )
        is True,
        "replay_schedule_seed_split_checkpoint_compatibility": integrity_passed,
    }
    key_intervals = ("treatment_vs_control", "treatment_vs_parent", "treatment_vs_random")

    def interval_endpoint(name: str, endpoint: int) -> float:
        row = nested.get(name)
        values = row.get("interval") if isinstance(row, Mapping) else None
        if (
            not isinstance(values, list)
            or len(values) != 2
            or not isinstance(values[endpoint], int | float)
        ):
            raise PopulationExperimentError("nested interval is malformed")
        return float(values[endpoint])

    crosses_half = any(
        interval_endpoint(name, 0) <= 0.5 <= interval_endpoint(name, 1) for name in key_intervals
    )
    decision = (
        "advancing_model_v1_collection_recipe"
        if all(conditions.values())
        else ("inconclusive_does_not_advance" if crosses_half else "does_not_advance")
    )
    return {
        "decision": decision,
        "conditions": conditions,
        "direct_replicates_favoring_treatment": direct_favors,
    }


def _shared_arena_alignment(
    artifacts: Iterable[Mapping[str, object]], output: Path
) -> dict[str, object]:
    """Audit shared-opponent candidate setup, seats, opponent RNG, and candidate-lane RNG."""
    by_key = _artifact_by_key(artifacts)
    checked = 0
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:
        for opponent in (Q0_PARENT, HEURISTIC, RANDOM, HISTORICAL_Q0, "q1", "q2", "q3", "q4"):
            treatment_key = (
                f"treatment-vs-{'historical-q0' if opponent == HISTORICAL_Q0 else opponent}"
            )
            control_key = f"control-vs-{'historical-q0' if opponent == HISTORICAL_Q0 else opponent}"
            if (replicate_id, treatment_key) not in by_key or (
                replicate_id,
                control_key,
            ) not in by_key:
                continue
            treatment_dir = output / "arena-records" / replicate_id / treatment_key
            control_dir = output / "arena-records" / replicate_id / control_key
            _, treatment = load_corpus(treatment_dir)
            _, control = load_corpus(control_dir)
            if len(treatment) != len(control):
                raise PopulationExperimentError("shared-opponent arena game counts differ")
            for left, right in zip(treatment, control, strict=True):
                if (
                    left.game_id != right.game_id
                    or left.pair_id != right.pair_id
                    or left.replay.seed != right.replay.seed
                    or tuple(seat.rng_identity for seat in left.seats)
                    != tuple(seat.rng_identity for seat in right.seats)
                    or tuple(seat.seed for seat in left.seats)
                    != tuple(seat.seed for seat in right.seats)
                ):
                    raise PopulationExperimentError(
                        "shared-opponent setup/seat/RNG alignment failed"
                    )
            checked += 1
    return {"status": "passed", "shared_comparison_count": checked}


def _begin_execution(output: Path, plan_fingerprint: str) -> Path:
    path = output / "execution-state.json"
    if path.exists():
        data = _read_json(path)
        if data.get("plan_fingerprint") != plan_fingerprint:
            raise PopulationExperimentError("execution state belongs to a different plan")
        attempts = data.get("attempt_count")
        if type(attempts) is not int or attempts > ALLOWED_RESUME_RETRIES:
            raise PopulationExperimentError("the one allowed resume/retry is exhausted")
        count = attempts + 1
    else:
        count = 1
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(
        _canonical_json(
            {
                "version": "m7-population-replay-execution-state-v1",
                "plan_fingerprint": plan_fingerprint,
                "attempt_count": count,
                "status": "running",
                "allowed_resume_retries": ALLOWED_RESUME_RETRIES,
            }
        )
        + b"\n"
    )
    os.replace(temporary, path)
    return path


def _complete_execution(path: Path) -> None:
    data = _read_json(path)
    data["status"] = "completed"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(_canonical_json(data) + b"\n")
    os.replace(temporary, path)


def run_population_experiment(
    config: PopulationExperimentConfig, bundle: PopulationPolicyBundle | None = None
) -> dict[str, object]:
    """Run/resume all corpus, dataset, checkpoint, arena, audit, and result boundaries."""
    bundle = bundle or build_population_policy_bundle(config.checkpoint_paths)
    plan = build_population_experiment_plan(config, bundle)
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable(output / "plan.json", plan, label="plan")
    plan_execution = plan.get("execution")
    if config.claim_default and (
        not isinstance(plan_execution, Mapping) or plan_execution.get("claim_eligible") is not True
    ):
        raise PopulationExperimentError(
            "default claim dispatch is blocked by the frozen source or arena-count preflight"
        )
    corpus_plan = _corpus_plan_from_bundle(bundle)
    if corpus_plan.fingerprint != plan["corpus_plan_fingerprint"]:
        raise PopulationExperimentError("recomputed corpus plan differs from frozen plan")
    cells = population_arena_cells(corpus_plan, pair_count=config.pairs_per_cell)
    holdout = _holdout_scan(
        corpus_plan,
        cells,
        roots=config.holdout_roots,
        current_output=output,
        smoke_pair_count=config.smoke_pair_count,
    )
    _write_immutable(
        output / "input-audit.json",
        {
            "version": INPUT_AUDIT_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "setup_holdout": holdout,
            "artifact_fingerprint": _fingerprint(
                {
                    "version": INPUT_AUDIT_VERSION,
                    "plan_fingerprint": plan["plan_fingerprint"],
                    "setup_holdout": holdout,
                }
            ),
        },
        label="input audit",
    )
    if holdout["status"] != "passed":
        raise PopulationExperimentError(
            "proposed training/development setup overlaps a retained corpus"
        )
    existing_result = output / "result.json"
    if existing_result.exists():
        persisted_result = _read_json(existing_result)
        if persisted_result.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise PopulationExperimentError("existing result belongs to a different plan")
        if persisted_result.get("result_fingerprint") != _fingerprint(
            {key: value for key, value in persisted_result.items() if key != "result_fingerprint"}
        ):
            raise PopulationExperimentError("existing result fingerprint mismatch")
        return persisted_result
    execution = _begin_execution(output, cast(str, plan["plan_fingerprint"]))
    deadline = time.monotonic() + CLAIM_RUN_CUTOFF_SECONDS
    corpus_records: dict[tuple[str, PopulationArm], tuple[GameRecord, ...]] = {}
    corpus_artifacts: dict[tuple[str, PopulationArm], dict[str, object]] = {}
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:
        for arm in CANDIDATE_ARMS:
            if time.monotonic() >= deadline:
                raise PopulationExperimentError("claim cutoff reached before corpus completion")
            _, records, artifact = _corpus_artifact(
                output, plan, corpus_plan, replicate_id, arm, pairs=config.pairs_per_cell
            )
            corpus_records[(replicate_id, arm)] = records
            corpus_artifacts[(replicate_id, arm)] = artifact
        if config.pairs_per_cell == POPULATION_PAIR_COUNT:
            audit_population_alignment(
                corpus_records[(replicate_id, "control")],
                corpus_records[(replicate_id, "treatment")],
            )
        else:
            _validate_sliced_schedule(
                corpus_records[(replicate_id, "control")],
                _slice_specs(
                    corpus_plan.game_specs(replicate_id, "control"), config.pairs_per_cell
                ),
            )
            _validate_sliced_schedule(
                corpus_records[(replicate_id, "treatment")],
                _slice_specs(
                    corpus_plan.game_specs(replicate_id, "treatment"), config.pairs_per_cell
                ),
            )
    dataset_paths: dict[tuple[str, PopulationArm], Path] = {}
    dataset_audits: dict[tuple[str, PopulationArm], dict[str, object]] = {}
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:
        for arm in CANDIDATE_ARMS:
            artifact = corpus_artifacts[(replicate_id, arm)]
            records_data = cast(Mapping[str, object], artifact["records"])
            corpus_fingerprint = cast(str, records_data["corpus_fingerprint"])
            dataset_paths[(replicate_id, arm)], dataset_audits[(replicate_id, arm)] = (
                _dataset_artifact(
                    output,
                    plan,
                    corpus_plan,
                    replicate_id,
                    arm,
                    corpus_records[(replicate_id, arm)],
                    corpus_fingerprint,
                    require_coverage=config.claim_default,
                )
            )
    training_rows: list[dict[str, object]] = []
    input_paths = plan.get("input_paths")
    if not isinstance(input_paths, Mapping) or not isinstance(input_paths.get(Q0_PARENT), str):
        raise PopulationExperimentError("plan input paths are malformed")
    parent_path = Path(cast(str, input_paths[Q0_PARENT]))
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:
        row: dict[str, object] = {"replicate_id": replicate_id, "arms": {}}
        for arm in CANDIDATE_ARMS:
            source = cast(Mapping[str, object], corpus_artifacts[(replicate_id, arm)]["records"])
            dataset_data = dataset_audits[(replicate_id, arm)].get("dataset")
            if not isinstance(dataset_data, Mapping) or not isinstance(
                dataset_data.get("fingerprint"), str
            ):
                raise PopulationExperimentError("dataset audit fingerprint is malformed")
            trained = _train_checkpoint(
                output,
                plan,
                corpus_plan,
                replicate_id,
                arm,
                dataset_paths[(replicate_id, arm)],
                cast(str, dataset_data["fingerprint"]),
                cast(str, source["corpus_fingerprint"]),
                parent_path,
                max_epochs=config.max_epochs,
            )
            cast(dict[str, object], row["arms"])[arm] = trained
        control_config = cast(
            Mapping[str, object], cast(dict[str, object], row["arms"])["control"]
        )["training_config"]
        treatment_config = cast(
            Mapping[str, object], cast(dict[str, object], row["arms"])["treatment"]
        )["training_config"]
        if control_config != treatment_config:
            raise PopulationExperimentError(
                "paired control/treatment training configuration differs"
            )
        training_rows.append(row)
    training_summary: dict[str, object] = {
        "version": TRAINING_SUMMARY_VERSION,
        "plan_fingerprint": plan["plan_fingerprint"],
        "replicates": training_rows,
        "retention": "retain both arms/datasets and all six checkpoints for Step 3",
        "artifact_fingerprint": "",
    }
    training_summary["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in training_summary.items() if key != "artifact_fingerprint"}
    )
    _write_immutable(output / "training-summary.json", training_summary, label="training summary")
    arena_artifacts = [_arena_artifact(output, plan, cell, bundle) for cell in cells]
    alignment = _shared_arena_alignment(arena_artifacts, output)
    _write_immutable(
        output / "arena-alignment.json",
        {
            "plan_fingerprint": plan["plan_fingerprint"],
            **alignment,
            "artifact_fingerprint": _fingerprint(
                {"plan_fingerprint": plan["plan_fingerprint"], **alignment}
            ),
        },
        label="arena alignment",
    )
    statistics = population_statistics(arena_artifacts, corpus_plan)
    _write_immutable(
        output / "statistics.json",
        {
            "plan_fingerprint": plan["plan_fingerprint"],
            **statistics,
            "artifact_fingerprint": _fingerprint(
                {"plan_fingerprint": plan["plan_fingerprint"], **statistics}
            ),
        },
        label="statistics",
    )
    integrity_passed = holdout["status"] == "passed" and alignment["status"] == "passed"
    decision = population_decision(statistics, integrity_passed=integrity_passed)
    arena_fingerprints: dict[str, object] = {}
    for artifact in arena_artifacts:
        cell_data = cast(Mapping[str, object], artifact["cell"])
        arena_key = f"{cell_data['replicate_id']}:{cell_data['key']}"
        arena_fingerprints[arena_key] = artifact["artifact_fingerprint"]
    result: dict[str, object] = {
        "version": RESULT_VERSION,
        "cycle_id": POPULATION_REPLAY_CYCLE_ID,
        "status": "completed",
        "plan_fingerprint": plan["plan_fingerprint"],
        "source": plan["source"],
        "evidence_class": cast(Mapping[str, object], plan["execution"])["evidence_class"],
        "corpora": {
            f"{replicate}:{arm}": corpus_artifacts[(replicate, arm)]["artifact_fingerprint"]
            for replicate in POPULATION_REPLAY_REPLICATE_IDS
            for arm in CANDIDATE_ARMS
        },
        "datasets": {
            f"{replicate}:{arm}": dataset_audits[(replicate, arm)]["artifact_fingerprint"]
            for replicate in POPULATION_REPLAY_REPLICATE_IDS
            for arm in CANDIDATE_ARMS
        },
        "training_summary_fingerprint": training_summary["artifact_fingerprint"],
        "arena_artifacts": arena_fingerprints,
        "statistics": statistics,
        "decision": decision,
        "scientific_limit": (
            "No champion, web default, locked final, ranking, TD, weights, or feature change "
            "occurs in Step 2."
        ),
        "result_fingerprint": "",
    }
    result["result_fingerprint"] = _fingerprint(
        {key: value for key, value in result.items() if key != "result_fingerprint"}
    )
    _write_immutable(existing_result, result, label="result")
    _complete_execution(execution)
    return result


def validate_population_experiment(
    output: Path,
    *,
    allow_smoke: bool = False,
    bundle: PopulationPolicyBundle | None = None,
) -> dict[str, object]:
    """Independently reload and recompute every persisted boundary without dispatching a run."""
    plan = _read_json(output / "plan.json")
    if plan.get("version") != PLAN_VERSION:
        raise PopulationExperimentError("unsupported population plan version")
    if plan.get("plan_fingerprint") != _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    ):
        raise PopulationExperimentError("population plan fingerprint mismatch")
    source = inspect_source_identity().to_data()
    frozen_source = plan.get("source")
    if not isinstance(frozen_source, Mapping) or any(
        frozen_source.get(key) != value for key, value in source.items()
    ):
        raise PopulationExperimentError(
            "validator requires the exact frozen source and dependency lock"
        )
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise PopulationExperimentError("plan execution metadata is malformed")
    if execution.get("claim_eligible") is not True and not allow_smoke:
        raise PopulationExperimentError("non-default smoke validation requires --allow-smoke")
    input_paths = plan.get("input_paths")
    holdout_scope = plan.get("holdout_scope")
    roots_data = holdout_scope.get("roots") if isinstance(holdout_scope, Mapping) else None
    if not isinstance(roots_data, list) or not all(isinstance(value, str) for value in roots_data):
        raise PopulationExperimentError("plan holdout scope is malformed")
    if not isinstance(input_paths, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in input_paths.items()
    ):
        raise PopulationExperimentError("plan input paths are malformed")
    config = PopulationExperimentConfig(
        output=output,
        checkpoint_paths={key: Path(value) for key, value in input_paths.items()},
        holdout_roots=tuple(Path(value) for value in roots_data),
        smoke_pair_count=(
            None
            if execution.get("pairs_per_corpus_arm") == POPULATION_PAIR_COUNT
            else cast(int, execution.get("pairs_per_corpus_arm"))
        ),
        smoke_max_epochs=(
            None
            if execution.get("training_max_epochs") == 50
            else cast(int, execution.get("training_max_epochs"))
        ),
    )
    bundle = bundle or build_population_policy_bundle(
        config.checkpoint_paths, toy=execution.get("mode") == "toy-random-smoke"
    )
    recomputed_plan = build_population_experiment_plan(config, bundle)
    if recomputed_plan != plan:
        raise PopulationExperimentError("recomputed immutable plan differs")
    corpus_plan = _corpus_plan_from_bundle(bundle)
    pairs = config.pairs_per_cell
    corpus_artifacts: dict[tuple[str, PopulationArm], dict[str, object]] = {}
    records_by_arm: dict[tuple[str, PopulationArm], tuple[GameRecord, ...]] = {}
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:
        for arm in CANDIDATE_ARMS:
            artifact = _read_json(output / "corpus-artifacts" / replicate_id / f"{arm}.json")
            directory = output / "corpora" / replicate_id / arm
            manifest, records = load_corpus(directory)
            specs = _slice_specs(corpus_plan.game_specs(replicate_id, arm), pairs)
            _validate_sliced_schedule(records, specs)
            if pairs == POPULATION_PAIR_COUNT:
                audit_population_arm(corpus_plan, replicate_id, arm, records)
            if (
                cast(Mapping[str, object], artifact["records"])["corpus_fingerprint"]
                != manifest.corpus_fingerprint
            ):
                raise PopulationExperimentError("corpus artifact does not match retained corpus")
            corpus_artifacts[(replicate_id, arm)] = artifact
            records_by_arm[(replicate_id, arm)] = records
        if pairs == POPULATION_PAIR_COUNT:
            audit_population_alignment(
                records_by_arm[(replicate_id, "control")],
                records_by_arm[(replicate_id, "treatment")],
            )
    for replicate_id in POPULATION_REPLAY_REPLICATE_IDS:
        for arm in CANDIDATE_ARMS:
            artifact = _read_json(output / "dataset-audits" / replicate_id / f"{arm}.json")
            from agent_avenue.learning import load_dataset, materialize_dataset

            corpus_fp = cast(
                str,
                cast(Mapping[str, object], corpus_artifacts[(replicate_id, arm)]["records"])[
                    "corpus_fingerprint"
                ],
            )
            expected_dataset = materialize_dataset(
                records_by_arm[(replicate_id, arm)],
                split_seed=corpus_plan.replicate(replicate_id).seed_plan.split_seed,
                validation_fraction=0.1,
                source_corpus_fingerprint=corpus_fp,
            )
            dataset_path = output / "datasets" / replicate_id / f"{arm}.npz"
            dataset = load_dataset(dataset_path)
            if dataset.fingerprint != expected_dataset.fingerprint:
                raise PopulationExperimentError(
                    "dataset does not match replay/split reconstruction"
                )
            if artifact.get("provenance_coverage") != _coverage_for_records(
                records_by_arm[(replicate_id, arm)], dataset.manifest, corpus_plan
            ):
                raise PopulationExperimentError("dataset coverage audit changed")
    summary = _read_json(output / "training-summary.json")
    if summary.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise PopulationExperimentError("training summary plan mismatch")
    from agent_avenue.learning import inspect_checkpoint, load_checkpoint, tensor_digest

    parent_path = config.checkpoint_paths[Q0_PARENT]
    parent = load_checkpoint(parent_path)
    parent_tensor = tensor_digest(parent.model.state_dict())
    # Reloading every checkpoint verifies its sealed files; the following checks independently bind
    # it to the frozen parent, paired seeds/config, corpus, and deterministic dataset.
    for row in cast(list[object], summary["replicates"]):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("replicate_id"), str)
            or not isinstance(row.get("arms"), Mapping)
        ):
            raise PopulationExperimentError("training summary row is malformed")
        replicate_id = cast(str, row["replicate_id"])
        for arm in CANDIDATE_ARMS:
            stored = cast(Mapping[str, object], cast(Mapping[str, object], row["arms"])[arm])
            inspection = inspect_checkpoint(output / "checkpoints" / replicate_id / arm)
            checkpoint = load_checkpoint(output / "checkpoints" / replicate_id / arm)
            if (
                inspection.checkpoint_fingerprint != stored["checkpoint_fingerprint"]
                or inspection.tensor_digest != stored["tensor_digest"]
            ):
                raise PopulationExperimentError("checkpoint identity differs from training summary")
            checkpoint_manifest = checkpoint.manifest
            metadata = checkpoint_manifest.get("metadata")
            lineage = checkpoint_manifest.get("lineage")
            training = checkpoint_manifest.get("training")
            sources = checkpoint_manifest.get("sources")
            expected_config = _training_config(
                corpus_plan.replicate(replicate_id), max_epochs=config.max_epochs
            ).normalized()
            dataset_audit = _read_json(output / "dataset-audits" / replicate_id / f"{arm}.json")
            dataset_data = dataset_audit.get("dataset")
            corpus_data = cast(
                Mapping[str, object], corpus_artifacts[(replicate_id, arm)]["records"]
            )
            if not all(
                isinstance(value, Mapping) for value in (metadata, lineage, training, sources)
            ) or not isinstance(dataset_data, Mapping):
                raise PopulationExperimentError("checkpoint lineage/configuration is malformed")
            metadata_data = cast(Mapping[str, object], metadata)
            lineage_data = cast(Mapping[str, object], lineage)
            training_data = cast(Mapping[str, object], training)
            sources_data = cast(Mapping[str, object], sources)
            if (
                metadata_data.get("plan_fingerprint") != plan["plan_fingerprint"]
                or metadata_data.get("initial_parent_tensor_digest") != parent_tensor
                or lineage_data.get("parent_checkpoint") != parent.checkpoint_fingerprint
                or lineage_data.get("generation") != 1
                or training_data.get("config") != expected_config
                or sources_data.get("dataset_fingerprint") != dataset_data.get("fingerprint")
                or tuple(cast(tuple[object, ...], sources_data.get("corpus_fingerprints", ())))
                != (corpus_data["corpus_fingerprint"],)
            ):
                raise PopulationExperimentError("checkpoint lineage/configuration differs")
    cells = population_arena_cells(corpus_plan, pair_count=pairs)
    persisted_input_audit = _read_json(output / "input-audit.json")
    recomputed_holdout = _holdout_scan(
        corpus_plan,
        cells,
        roots=config.holdout_roots,
        current_output=output,
        smoke_pair_count=config.smoke_pair_count,
    )
    if persisted_input_audit.get("setup_holdout") != recomputed_holdout:
        raise PopulationExperimentError(
            "setup holdout audit differs from independent reconstruction"
        )
    artifacts: list[dict[str, object]] = []
    for cell in cells:
        artifact = _read_json(output / "arenas" / cell.replicate_id / f"{cell.key}.json")
        agent_a, agent_b = _arena_agents(cell, output, bundle)
        arena_config = ArenaConfig(cell.run_id, agent_a, agent_b, cell.pair_count, cell.master_seed)
        manifest, records = load_corpus(output / "arena-records" / cell.replicate_id / cell.key)
        report = cast(Mapping[str, object], artifact["report"])
        elapsed = report.get("elapsed_seconds")
        if (
            not isinstance(elapsed, int | float)
            or arena_report_from_records(
                arena_config, records, elapsed_seconds=float(elapsed)
            ).to_data()
            != report
        ):
            raise PopulationExperimentError("arena report no longer reconstructs from records")
        if artifact.get("tactical_invariants") != _enveloped_invariants(
            records, manifest.corpus_fingerprint
        ):
            raise PopulationExperimentError("arena tactical invariants differ")
        artifacts.append(artifact)
    alignment = _shared_arena_alignment(artifacts, output)
    stats = population_statistics(artifacts, corpus_plan)
    persisted_stats = _read_json(output / "statistics.json")
    expected_stats = {"plan_fingerprint": plan["plan_fingerprint"], **stats}
    expected_stats["artifact_fingerprint"] = _fingerprint(expected_stats)
    if persisted_stats != expected_stats:
        raise PopulationExperimentError("nested statistics artifact differs")
    decision = population_decision(stats, integrity_passed=alignment["status"] == "passed")
    result = _read_json(output / "result.json")
    if result.get("decision") != decision:
        raise PopulationExperimentError("result decision differs from independent recomputation")
    if result.get("result_fingerprint") != _fingerprint(
        {key: value for key, value in result.items() if key != "result_fingerprint"}
    ):
        raise PopulationExperimentError("result fingerprint mismatch")
    validation: dict[str, object] = {
        "version": VALIDATION_VERSION,
        "status": "passed",
        "plan_fingerprint": plan["plan_fingerprint"],
        "source": source,
        "checks": {
            "source_lock": True,
            "schedule_assignment_split": True,
            "checkpoint_lineage": True,
            "arena_aggregates_and_alignment": True,
            "safety_and_offense": True,
            "nested_statistics_and_decision": True,
        },
        "artifact_fingerprint": "",
    }
    validation["artifact_fingerprint"] = _fingerprint(
        {key: value for key, value in validation.items() if key != "artifact_fingerprint"}
    )
    _write_immutable(output / "validation.json", validation, label="validation")
    return validation
