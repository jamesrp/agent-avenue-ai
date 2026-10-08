"""Step-5 independent league: frozen schedule, audits, statistics, and promotion decision.

This module implements the production side of the approved
``m7-independent-league-promotion-v1`` agreement. It never trains, selects, or mutates deployment
settings. The separate validator script must reproduce every derived artifact without importing the
aggregation, bootstrap, tactical-prefix, holdout, or decision functions defined here.

All league statistics are carried as integer win numerators over fixed denominators so that two
independent implementations can agree exactly regardless of summation order.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

from agent_avenue.agents import (
    FALLBACK_VERSION,
    RESOLUTION_SCOPE,
    RNG_ALGORITHM,
    SEED_DERIVATION,
    UNCERTAINTY_VERSION,
    Agent,
    DeterministicRandom,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
    derive_seed,
    filter_immediate_win_actions,
    filter_terminal_actions,
)
from agent_avenue.engine import (
    Action,
    GameConfig,
    GameState,
    Phase,
    PlayerId,
    PlayOfferAction,
    RecruitAction,
    apply_action,
    new_game,
)
from agent_avenue.engine.model import player_index
from agent_avenue.engine.setup import normalize_config
from agent_avenue.engine.terminal import TERMINAL_EVALUATOR_VERSION
from agent_avenue.observation import observe
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    game_record_fingerprint,
    inspect_source_identity,
    load_corpus,
    rules_fingerprint,
)
from agent_avenue.storage.provenance import repository_root

from .arena import ArenaConfig, arena_report_from_records, run_resumable_arena, wilson_interval
from .game import AgentSpec

CYCLE_ID: Final = "m7-independent-league-promotion-v1"
REGISTRY_PATH: Final = Path("research/cycles/m7-independent-league-inputs.json")
REGISTRY_VERSION: Final = "m7-independent-league-inputs-v1"
REGISTRY_FINGERPRINT: Final = "df238b7d9c948563ee15e4e2cfc59fe1525a9721b10e24362a7d251c1dfb7d48"
ROOT_SEED: Final = 2026091705
PAIRS_PER_FAMILY: Final = 200
FAMILY_IDS: Final = ("family-a", "family-b")

INCUMBENT_ID: Final = "q0-terminal-safety-v1"
CHALLENGER_ID: Final = "q0-terminal-offense-v1"
Q0_CORE_RNG_IDENTITY: Final = "q0-terminal-core-v1"
Q_IDS: Final = tuple(f"q{generation}-terminal-safety-v1" for generation in range(1, 5))
M_IDS: Final = tuple(f"M{replicate}-structured-v2-offense-safety" for replicate in range(1, 4))
HISTORICAL_ID: Final = "historical-q0"
HEURISTIC_ID: Final = "greedy-public-v1"
RANDOM_ID: Final = "random-agent-v1"
ANCHOR_IDS: Final = (HISTORICAL_ID, HEURISTIC_ID, RANDOM_ID)
POLICY_ORDER: Final = (
    INCUMBENT_ID,
    CHALLENGER_ID,
    *Q_IDS,
    *M_IDS,
    HISTORICAL_ID,
    HEURISTIC_ID,
    RANDOM_ID,
)
COMMON_OPPONENT_IDS: Final = POLICY_ORDER[2:]
SAFETY_ENVELOPED_IDS: Final = (INCUMBENT_ID, CHALLENGER_ID, *Q_IDS, *M_IDS)
OFFENSE_ENVELOPED_IDS: Final = (CHALLENGER_ID, *M_IDS)
NON_M_IDS: Final = tuple(policy for policy in POLICY_ORDER if policy not in M_IDS)

BOOTSTRAP_RESAMPLES: Final = 20_000
BOOTSTRAP_LOWER_INDEX: Final = 499
BOOTSTRAP_UPPER_INDEX: Final = 19_499
RESAMPLE_CHUNK: Final = 64
PROMOTION_LIFT_THRESHOLD: Final = 0.0025
RANDOM_LOWER_THRESHOLD: Final = 0.5
SEAT_FLOOR: Final = 0.45
CLAIM_CUTOFF_SECONDS: Final = 465 * 60
HARD_LIMIT_SECONDS: Final = 480 * 60
PROJECTION_LIMIT_MINUTES: Final = 420.0
PROJECTION_SAFETY_FACTOR: Final = 1.20
ALLOWED_ATTEMPTS: Final = 2

DECISION_PROMOTE: Final = "promote_q0_terminal_offense_v1"
DECISION_RETAIN: Final = "retain_q0_terminal_safety_v1"
DECISION_BLOCKED: Final = "blocked_no_decision"

PLAN_VERSION: Final = "m7-independent-league-plan-v1"
CELL_VERSION: Final = "m7-independent-league-cell-v1"
TACTICAL_VERSION: Final = "m7-independent-league-tactical-v1"
PREFIX_VERSION: Final = "m7-independent-league-aligned-prefix-v1"
STATISTICS_VERSION: Final = "m7-independent-league-statistics-v1"
DECISION_VERSION: Final = "m7-independent-league-promotion-decision-v1"
HOLDOUT_VERSION: Final = "m7-independent-league-setup-holdout-v1"
EXECUTION_VERSION: Final = "m7-independent-league-execution-v1"
RESULT_VERSION: Final = "m7-independent-league-result-v1"
CHECKSUM_VERSION: Final = "m7-independent-league-checksums-v1"
PREFLIGHT_VERSION: Final = "m7-independent-league-runtime-preflight-v1"
SMOKE_NAMESPACE: Final = f"{CYCLE_ID}:nonclaim-smoke"
SMOKE_ROOT_DOMAIN: Final = f"{CYCLE_ID}:nonclaim-smoke-root:v1"

MATCHUP_BOOTSTRAP_DOMAIN: Final = "matchup-family-stratified-bootstrap:v1"
ANCHOR_BOOTSTRAP_DOMAIN: Final = "anchor-joint-family-stratified-bootstrap:v1"
M_FAMILY_BOOTSTRAP_DOMAIN: Final = "m-family-nested-bootstrap:v1"

# Terminal-safety configurations serialize four frozen version fields that the registry's compact
# declarations omit. They are pinned here so a changed shield cannot pass the declaration check.
TERMINAL_SAFETY_RUNTIME_EXTRAS: Final = {
    "fallback": FALLBACK_VERSION,
    "public_uncertainty": UNCERTAINTY_VERSION,
    "resolution_scope": RESOLUTION_SCOPE,
    "terminal_evaluator": TERMINAL_EVALUATOR_VERSION,
}
ARCHIVE_PATHS: Final = {
    "terminal_safety": Path("artifacts/archive/terminal-safety-v1-artifacts-2026-09-03.tar.gz"),
    "terminal_offense": Path("artifacts/archive/m7-terminal-offense-confirm-v1-2026-09-11.tar.gz"),
    "structured_v2": Path("artifacts/archive/m7-structured-model-v2-2026-09-12.tar.gz"),
    "rollout_step4": Path(
        "artifacts/archive/m7-counterfactual-rollout-supervision-v1-2026-09-17.tar.gz"
    ),
}
CHECKSUM_EXCLUDED_NAMES: Final = frozenset(
    {
        "checksums.json",
        "execution-state.json",
        "validation.json",
        "validator-runtime.json",
        "driver.stdout",
        "driver.stderr",
        "validation.stdout",
        "validation.stderr",
        "wrapper-exit.json",
        ".lock",
    }
)


class LeagueError(ValueError):
    """Raised when Step-5 inputs or evidence violate the frozen agreement."""


# ---------------------------------------------------------------------------------------------
# JSON and file helpers


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LeagueError(f"unable to read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise LeagueError(f"expected JSON object: {path}")
    return value


def sealed(version: str, **values: object) -> dict[str, object]:
    """Return a versioned artifact whose fingerprint covers every other field."""
    data: dict[str, object] = {"version": version, **values}
    data["artifact_fingerprint"] = fingerprint(data)
    return data


def seal_matches(value: Mapping[str, object]) -> bool:
    payload = {key: item for key, item in value.items() if key != "artifact_fingerprint"}
    return value.get("artifact_fingerprint") == fingerprint(payload)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_immutable(path: Path, value: Mapping[str, object]) -> None:
    """Write once; on resume require the exact same content."""
    if path.exists():
        if read_json(path) != json.loads(canonical_json(dict(value))):
            raise LeagueError(f"immutable artifact differs from recomputation: {path}")
        return
    _write_bytes_atomic(path, json.dumps(dict(value), sort_keys=True, indent=2).encode() + b"\n")


def write_mutable(path: Path, value: Mapping[str, object]) -> None:
    _write_bytes_atomic(path, json.dumps(dict(value), sort_keys=True, indent=2).encode() + b"\n")


# ---------------------------------------------------------------------------------------------
# Registry and frozen schedule


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load the committed input registry and authenticate its self-fingerprint and shape."""
    registry = read_json(path)
    if not seal_matches(registry):
        raise LeagueError("input registry self-fingerprint does not match its content")
    expected_keys = {
        "archive_sha256",
        "artifact_fingerprint",
        "checkpoints",
        "created_date",
        "cycle_id",
        "policies",
        "root_seed",
        "schedule",
        "source_evidence",
        "version",
    }
    if set(registry) != expected_keys:
        raise LeagueError("input registry fields do not match its version")
    if registry["version"] != REGISTRY_VERSION or registry["cycle_id"] != CYCLE_ID:
        raise LeagueError("input registry version or cycle is not Step-5 v1")
    if type(registry["root_seed"]) is not int:
        raise LeagueError("input registry root seed is malformed")
    policies = registry["policies"]
    if not isinstance(policies, list) or tuple(
        policy.get("id") for policy in policies if isinstance(policy, dict)
    ) != tuple(registry["schedule"]["policy_order"]):
        raise LeagueError("input registry policies do not follow the declared policy order")
    names = [entry.get("name") for entry in registry["checkpoints"] if isinstance(entry, dict)]
    if len(names) != len(registry["checkpoints"]) or len(set(names)) != len(names):
        raise LeagueError("input registry checkpoint names are malformed or duplicated")
    eligible = [policy["id"] for policy in policies if policy.get("promotion_eligible") is True]
    if eligible != [CHALLENGER_ID]:
        raise LeagueError("the challenger must be the only promotion-eligible policy")
    return registry


def is_claim_registry(registry: Mapping[str, object]) -> bool:
    return (
        registry.get("artifact_fingerprint") == REGISTRY_FINGERPRINT
        and registry.get("root_seed") == ROOT_SEED
        and tuple(cast(Mapping[str, Any], registry["schedule"])["policy_order"]) == POLICY_ORDER
    )


@dataclass(frozen=True, slots=True)
class LeagueCell:
    """One family-specific ordered arena cell; ``left`` is always the earlier policy."""

    family: str
    left: str
    right: str
    pair_count: int
    master_seed: int
    namespace: str

    @property
    def key(self) -> str:
        return f"{self.family}:{self.left}--vs--{self.right}"

    @property
    def run_id(self) -> str:
        return f"{self.namespace}:{self.family}:{self.left}:vs:{self.right}"

    @property
    def games(self) -> int:
        return self.pair_count * 2

    def to_data(self) -> dict[str, object]:
        return {
            "family": self.family,
            "games": self.games,
            "key": self.key,
            "left": self.left,
            "pair_count": self.pair_count,
            "right": self.right,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class LeagueDesign:
    """The seed namespace, families, and block count of one league execution."""

    namespace: str
    root_seed: int
    pair_count: int
    policy_order: tuple[str, ...] = POLICY_ORDER
    family_order: tuple[str, ...] = FAMILY_IDS

    def __post_init__(self) -> None:
        if self.pair_count < 1:
            raise LeagueError("league pair count must be positive")
        if len(set(self.policy_order)) != len(self.policy_order) or len(self.policy_order) < 2:
            raise LeagueError("league policy order must contain distinct policies")
        if len(set(self.family_order)) != len(self.family_order) or not self.family_order:
            raise LeagueError("league family order must contain distinct families")

    @property
    def is_claim_design(self) -> bool:
        return (
            self.namespace == CYCLE_ID
            and self.root_seed == ROOT_SEED
            and self.pair_count == PAIRS_PER_FAMILY
            and self.policy_order == POLICY_ORDER
            and self.family_order == FAMILY_IDS
        )

    def family_seed_domain(self, family: str) -> str:
        if family not in self.family_order:
            raise LeagueError(f"unknown seed family: {family}")
        return f"{self.namespace}:seed-family:{family}"

    def family_master_seed(self, family: str) -> int:
        return derive_seed(self.root_seed, self.family_seed_domain(family)) & ((1 << 63) - 1)

    def family_setup_seeds(self, family: str) -> tuple[int, ...]:
        master = self.family_master_seed(family)
        return tuple(
            derive_seed(master, f"arena:pair:{index}:setup") & ((1 << 64) - 1)
            for index in range(self.pair_count)
        )

    def cells(self) -> tuple[LeagueCell, ...]:
        cells: list[LeagueCell] = []
        for family in self.family_order:
            master = self.family_master_seed(family)
            for left_index, left in enumerate(self.policy_order):
                for right in self.policy_order[left_index + 1 :]:
                    cells.append(
                        LeagueCell(family, left, right, self.pair_count, master, self.namespace)
                    )
        return tuple(cells)

    def bootstrap_seed(self, domain: str) -> int:
        return derive_seed(self.root_seed, f"{self.namespace}:{domain}")

    def to_data(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "root_seed": self.root_seed,
            "pair_count": self.pair_count,
            "policy_order": list(self.policy_order),
            "family_order": list(self.family_order),
            "claim_design": self.is_claim_design,
        }


def claim_design() -> LeagueDesign:
    return LeagueDesign(CYCLE_ID, ROOT_SEED, PAIRS_PER_FAMILY)


def smoke_design(pair_count: int, *, root_seed: int = ROOT_SEED) -> LeagueDesign:
    """Separate nonclaim seed domains for the mandatory preclaim smoke."""
    smoke_root = derive_seed(root_seed, SMOKE_ROOT_DOMAIN) & ((1 << 63) - 1)
    return LeagueDesign(SMOKE_NAMESPACE, smoke_root, pair_count)


def setup_seed_fingerprint(seeds: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(seeds), separators=(",", ":")).encode()).hexdigest()


def schedule_data(design: LeagueDesign) -> dict[str, object]:
    families: dict[str, object] = {}
    all_seeds: list[set[int]] = []
    for family in design.family_order:
        seeds = design.family_setup_seeds(family)
        all_seeds.append(set(seeds))
        families[family] = {
            "seed_domain": design.family_seed_domain(family),
            "master_seed": design.family_master_seed(family),
            "pair_count": design.pair_count,
            "setup_seed_fingerprint": setup_seed_fingerprint(seeds),
            "unique_setups": len(set(seeds)) == len(seeds),
        }
    disjoint = all(
        first.isdisjoint(second)
        for index, first in enumerate(all_seeds)
        for second in all_seeds[index + 1 :]
    )
    cells = design.cells()
    return {
        "design": design.to_data(),
        "families": families,
        "families_disjoint": disjoint,
        "cells": [cell.to_data() for cell in cells],
        "total_cells": len(cells),
        "total_games": sum(cell.games for cell in cells),
        "seed_derivation": SEED_DERIVATION,
        "rng_algorithm": RNG_ALGORITHM,
    }


def verify_registry_schedule(registry: Mapping[str, Any]) -> dict[str, object]:
    """Reconstruct the frozen claim schedule and require exact equality with the registry."""
    design = claim_design()
    reconstructed = schedule_data(design)
    declared = registry["schedule"]
    failures: list[str] = []
    if not is_claim_registry(registry):
        failures.append("registry_is_not_the_committed_claim_registry")
    if declared.get("family_order") != list(design.family_order):
        failures.append("family_order")
    if declared.get("policy_order") != list(design.policy_order):
        failures.append("policy_order")
    if declared.get("seed_derivation") != SEED_DERIVATION:
        failures.append("seed_derivation")
    if declared.get("rng_algorithm") != RNG_ALGORITHM:
        failures.append("rng_algorithm")
    for family, data in cast(Mapping[str, Mapping[str, object]], reconstructed["families"]).items():
        expected = {key: data[key] for key in ("seed_domain", "master_seed", "pair_count")}
        expected["setup_seed_fingerprint"] = data["setup_seed_fingerprint"]
        if declared.get("families", {}).get(family) != expected:
            failures.append(f"family:{family}")
        if data["unique_setups"] is not True:
            failures.append(f"duplicate_setups:{family}")
    if reconstructed["families_disjoint"] is not True:
        failures.append("family_overlap")
    if declared.get("cells") != reconstructed["cells"]:
        failures.append("cells")
    if (
        declared.get("total_cells") != reconstructed["total_cells"]
        or declared.get("total_games") != reconstructed["total_games"]
    ):
        failures.append("totals")
    if reconstructed["total_games"] != 52_800 or reconstructed["total_cells"] != 132:
        failures.append("frozen_cardinality")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "total_cells": reconstructed["total_cells"],
        "total_games": reconstructed["total_games"],
        "family_setup_seed_fingerprints": {
            family: data["setup_seed_fingerprint"]
            for family, data in cast(
                Mapping[str, Mapping[str, object]], reconstructed["families"]
            ).items()
        },
    }


# ---------------------------------------------------------------------------------------------
# Input authentication


def _verify_file(path: Path, expected: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        return {"path": str(path), "expected_sha256": expected, "status": "missing"}
    actual = sha256_file(path)
    return {
        "path": str(path),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "status": "passed" if actual == expected else "mismatch",
    }


def verify_league_inputs(
    registry: Mapping[str, Any], root: Path, *, require_archives: bool
) -> dict[str, object]:
    """Authenticate every declared checkpoint file, source-evidence file, and archive."""
    checkpoint_rows: dict[str, object] = {}
    failures: list[str] = []
    for entry in registry["checkpoints"]:
        directory = root / entry["path"]
        declared_files = cast(Mapping[str, str], entry["files"])
        rows = {
            name: _verify_file(directory / name, digest) for name, digest in declared_files.items()
        }
        present = (
            sorted(item.name for item in directory.iterdir() if item.is_file())
            if directory.is_dir()
            else []
        )
        exact_set = present == sorted(declared_files)
        passed = exact_set and all(
            cast(Mapping[str, object], row)["status"] == "passed" for row in rows.values()
        )
        if not passed:
            failures.append(f"checkpoint:{entry['name']}")
        checkpoint_rows[entry["name"]] = {
            "path": entry["path"],
            "files": rows,
            "present_files": present,
            "exact_file_set": exact_set,
            "status": "passed" if passed else "failed",
        }
    evidence_rows: dict[str, object] = {}
    for name, entry in sorted(
        cast(Mapping[str, Mapping[str, Any]], registry["source_evidence"]).items()
    ):
        path = root / entry["path"]
        row = _verify_file(path, entry["sha256"])
        if row["status"] == "passed":
            content = read_json(path)
            for key in ("artifact_fingerprint", "plan_fingerprint"):
                if entry.get(key) is not None and content.get(key) != entry[key]:
                    row["status"] = f"{key}_mismatch"
        if row["status"] != "passed":
            failures.append(f"source_evidence:{name}")
        evidence_rows[name] = row
    archive_rows: dict[str, object] = {}
    for name, digest in sorted(cast(Mapping[str, str], registry["archive_sha256"]).items()):
        archive_path = ARCHIVE_PATHS.get(name)
        if archive_path is None:
            failures.append(f"archive_path_unknown:{name}")
            continue
        row = _verify_file(root / archive_path, digest)
        if row["status"] == "missing" and not require_archives:
            row["status"] = "missing-nonclaim-allowed"
        elif row["status"] != "passed":
            failures.append(f"archive:{name}")
        archive_rows[name] = row
    return sealed(
        "m7-independent-league-input-audit-v1",
        registry_fingerprint=registry["artifact_fingerprint"],
        checkpoints=checkpoint_rows,
        source_evidence=evidence_rows,
        archives=archive_rows,
        archives_required=require_archives,
        failures=failures,
        status="passed" if not failures else "failed",
    )


# ---------------------------------------------------------------------------------------------
# Policy construction and declaration checks


def agent_config(agent: object) -> dict[str, object]:
    source: Any = agent
    if callable(getattr(source, "config_to_data", None)):
        value = source.config_to_data()
    else:
        value = source.config.to_data()
    if not isinstance(value, dict):
        raise LeagueError("agent has no normalized configuration")
    return cast(dict[str, object], json.loads(canonical_json(value)))


RefResolver = Callable[[str, Mapping[str, object]], bool]


def declared_config_matches(
    declared: Mapping[str, object], runtime: Mapping[str, object], resolve_ref: RefResolver
) -> bool:
    """Return whether a runtime config realizes one nested registry declaration exactly.

    Every declared key must be present with an equal value. ``checkpoint_ref`` nodes resolve
    through ``resolve_ref``. The only runtime keys a declaration may omit are the pinned
    terminal-safety version fields.
    """
    if declared.get("type") == "checkpoint_ref":
        if set(declared) != {"name", "type"} or not isinstance(declared["name"], str):
            return False
        return resolve_ref(declared["name"], runtime)
    extras = set(runtime) - set(declared)
    if extras:
        if declared.get("type") != "terminal_safety":
            return False
        if extras != set(TERMINAL_SAFETY_RUNTIME_EXTRAS) or any(
            runtime[key] != value for key, value in TERMINAL_SAFETY_RUNTIME_EXTRAS.items()
        ):
            return False
    for key, value in declared.items():
        if key not in runtime:
            return False
        actual = runtime[key]
        if isinstance(value, Mapping):
            if not isinstance(actual, Mapping) or not declared_config_matches(
                value, actual, resolve_ref
            ):
                return False
        elif actual != value:
            return False
    return True


def _checkpoint_resolver(registry: Mapping[str, Any]) -> RefResolver:
    by_name = {entry["name"]: entry for entry in registry["checkpoints"]}

    def resolve(name: str, runtime: Mapping[str, object]) -> bool:
        entry = by_name.get(name)
        return entry is not None and (
            runtime.get("type") == entry["agent_kind"]
            and runtime.get("checkpoint_fingerprint") == entry["checkpoint_fingerprint"]
            and runtime.get("tensor_digest") == entry["tensor_digest"]
            and runtime.get("encoder_version") == entry["encoder_version"]
        )

    return resolve


def _toy_resolver(name: str, runtime: Mapping[str, object]) -> bool:
    return dict(runtime) == RandomAgentConfig().to_data()


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    """Executable league policies plus their frozen identities."""

    mode: str
    specs: Mapping[str, AgentSpec]
    checkpoint_identities: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


def _compose(declared: Mapping[str, Any], resolve_base: Callable[[str], Agent]) -> Agent:
    kind = declared.get("type")
    if kind == "checkpoint_ref":
        return resolve_base(cast(str, declared["name"]))
    if kind == "terminal_safety":
        return TerminalSafetyAgent(_compose(declared["base"], resolve_base))
    if kind == "terminal_offense":
        return TerminalOffenseAgent(_compose(declared["base"], resolve_base))
    if kind == "greedy_heuristic":
        return GreedyHeuristicAgent(GreedyHeuristicConfig())
    if kind == "random":
        return RandomAgent(RandomAgentConfig())
    raise LeagueError(f"unsupported declared policy component: {kind!r}")


def build_policy_bundle(registry: Mapping[str, Any], root: Path, *, mode: str) -> PolicyBundle:
    """Construct every registry policy and require its runtime config to realize the declaration.

    ``learned-checkpoints`` loads retained checkpoints (optional RL dependency). ``toy-random``
    substitutes a random base for every checkpoint reference; it is smoke-only and never claim
    eligible.
    """
    resolve_base: Callable[[str], Agent]
    resolver: RefResolver
    identities: dict[str, Mapping[str, object]] = {}
    if mode == "learned-checkpoints":
        # Optional RL imports stay behind the explicit learned-policy path.
        from agent_avenue.agents.learned import LearnedValueAgent
        from agent_avenue.agents.structured import StructuredValueAgent
        from agent_avenue.learning import load_checkpoint, load_structured_checkpoint

        loaded: dict[str, Any] = {}
        for entry in registry["checkpoints"]:
            path = root / entry["path"]
            if entry["agent_kind"] == "learned_value":
                checkpoint: Any = load_checkpoint(path)
            elif entry["agent_kind"] == "structured_value":
                checkpoint = load_structured_checkpoint(path)
            else:
                raise LeagueError(f"unsupported checkpoint kind: {entry['agent_kind']!r}")
            if (
                checkpoint.checkpoint_fingerprint != entry["checkpoint_fingerprint"]
                or checkpoint.manifest.get("tensor_digest") != entry["tensor_digest"]
            ):
                raise LeagueError(f"checkpoint identity differs from registry: {entry['name']}")
            loaded[entry["name"]] = (entry["agent_kind"], checkpoint)
            identities[entry["name"]] = {
                "path": entry["path"],
                "agent_kind": entry["agent_kind"],
                "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
                "tensor_digest": checkpoint.manifest.get("tensor_digest"),
                "encoder_version": entry["encoder_version"],
            }

        def learned_base(name: str) -> Agent:
            kind, checkpoint = loaded[name]
            if kind == "structured_value":
                return cast(Agent, StructuredValueAgent.from_checkpoint(checkpoint))
            return cast(Agent, LearnedValueAgent.from_checkpoint(checkpoint))

        resolve_base = learned_base
        resolver = _checkpoint_resolver(registry)
    elif mode == "toy-random":

        def toy_base(name: str) -> Agent:
            return RandomAgent(RandomAgentConfig())

        resolve_base = toy_base
        resolver = _toy_resolver
    else:
        raise LeagueError(f"unsupported policy mode: {mode!r}")

    specs: dict[str, AgentSpec] = {}
    for policy in registry["policies"]:
        declared = cast(Mapping[str, Any], policy["config"])

        def factory(declared: Mapping[str, Any] = declared) -> Agent:
            return _compose(declared, resolve_base)

        runtime = agent_config(factory())
        if not declared_config_matches(declared, runtime, resolver):
            raise LeagueError(
                f"runtime policy config does not realize its declaration: {policy['id']}"
            )
        specs[policy["id"]] = AgentSpec(
            policy["id"], runtime, factory, rng_identity=policy["rng_identity"]
        )
    if (
        specs[INCUMBENT_ID].rng_identity != Q0_CORE_RNG_IDENTITY
        or specs[CHALLENGER_ID].rng_identity != Q0_CORE_RNG_IDENTITY
    ):
        raise LeagueError("incumbent and challenger must share the q0 core RNG identity")
    other_identities = [
        spec.rng_identity for key, spec in specs.items() if key not in {INCUMBENT_ID, CHALLENGER_ID}
    ]
    if len(set(other_identities)) != len(other_identities) or Q0_CORE_RNG_IDENTITY in set(
        other_identities
    ):
        raise LeagueError("every other policy needs one distinct stable RNG identity")
    return PolicyBundle(mode, specs, identities)


# ---------------------------------------------------------------------------------------------
# Setup holdout


def _setup_identity(game_config: Mapping[str, object], setup_seed: int) -> str:
    return fingerprint({"game_config": dict(game_config), "setup_seed": setup_seed})


def scan_setup_holdout(
    design: LeagueDesign,
    *,
    roots: Sequence[Path],
    current_output: Path | None,
) -> dict[str, object]:
    """Compare both family blocks with every completed corpus under the declared roots."""
    if not roots:
        raise LeagueError("setup holdout requires at least one declared root")
    normalized = normalize_config(GameConfig())
    current: dict[str, dict[str, object]] = {}
    for family in design.family_order:
        for index, seed in enumerate(design.family_setup_seeds(family)):
            identity = _setup_identity(normalized, seed)
            if identity in current:
                raise LeagueError("league setup families overlap or repeat a setup")
            current[identity] = {"family": family, "pair_index": index, "setup_seed": seed}
    output = current_output.resolve() if current_output is not None else None
    seen: set[Path] = set()
    prior: dict[str, list[str]] = {}
    corpora: list[dict[str, object]] = []
    root_rows: list[dict[str, object]] = []
    for declared_root in roots:
        root = declared_root.resolve()
        if not root.exists():
            root_rows.append({"declared_root": str(declared_root), "status": "missing"})
            continue
        if not root.is_dir():
            raise LeagueError(f"holdout root is not a directory: {declared_root}")
        corpus_count = 0
        for manifest_path in sorted(root.rglob("manifest.json")):
            resolved = manifest_path.resolve()
            if (output is not None and resolved.is_relative_to(output)) or resolved in seen:
                continue
            seen.add(resolved)
            records_path = manifest_path.parent / "games.jsonl.gz"
            try:
                manifest_data = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                if records_path.exists():
                    raise LeagueError(f"malformed completed corpus: {manifest_path}") from exc
                continue
            if not records_path.exists() and not (
                isinstance(manifest_data, dict) and "records_file" in manifest_data
            ):
                continue
            try:
                manifest, records = load_corpus(
                    manifest_path.parent, verify_code=False, verify_replays=False
                )
            except (OSError, ValueError) as exc:
                raise LeagueError(
                    f"unable to load completed corpus: {manifest_path.parent}"
                ) from exc
            corpus_count += 1
            source = str(manifest_path.parent)
            corpora.append(
                {
                    "path": source,
                    "corpus_fingerprint": manifest.corpus_fingerprint,
                    "record_count": len(records),
                }
            )
            for record in records:
                identity = _setup_identity(
                    normalize_config(record.replay.config), record.replay.seed
                )
                prior.setdefault(identity, []).append(source)
        root_rows.append(
            {"declared_root": str(declared_root), "status": "present", "corpus_count": corpus_count}
        )
    overlaps = sorted(set(current) & set(prior))
    return sealed(
        HOLDOUT_VERSION,
        scope={
            "declared_roots": [str(path) for path in roots],
            "recursive": True,
            "completed_corpus_detection": "manifest-and-games-jsonl-gzip-v1",
            "excludes_current_output_subtree": current_output is not None,
        },
        design=design.to_data(),
        roots=root_rows,
        proposed_setups=[
            {"setup_identity": identity, **value} for identity, value in sorted(current.items())
        ],
        proposed_setup_count=len(current),
        proposed_setup_fingerprint=fingerprint(sorted(current)),
        prior_inventory=sorted(corpora, key=lambda row: str(row["path"])),
        prior_unique_setup_count=len(prior),
        prior_setup_fingerprint=fingerprint(sorted(prior)),
        overlap_count=len(overlaps),
        overlaps=[
            {
                "setup_identity": identity,
                "current": current[identity],
                "prior": sorted(set(prior[identity])),
            }
            for identity in overlaps
        ],
        status="passed" if not overlaps else "failed",
    )


# ---------------------------------------------------------------------------------------------
# Per-record tactical audit


TACTICAL_FIELDS: Final = (
    "decisions",
    "guaranteed_win_opportunities",
    "guaranteed_win_conversions",
    "missed_guaranteed_wins",
    "false_forced_wins",
    "executed_provable_losses",
    "executed_avoidable_provable_losses",
)


def empty_tactical() -> dict[str, int]:
    return dict.fromkeys(TACTICAL_FIELDS, 0)


def decision_actor(state: GameState) -> PlayerId:
    return state.active_player if state.phase is Phase.PLAY else state.active_player.other()


def _resolve_chosen(state: GameState, action: Action, following: Action | None) -> GameState:
    resolved = apply_action(state, action)
    if isinstance(action, PlayOfferAction) and resolved.phase is not Phase.TERMINAL:
        if not isinstance(following, RecruitAction):
            raise LeagueError("play action is not followed by its recruit resolution")
        resolved = apply_action(resolved, following)
    return resolved


def record_tactical_counts(records: Iterable[GameRecord]) -> dict[str, dict[str, int]]:
    """Replay records and count guaranteed-win and provable-loss behavior for every policy."""
    by_agent: dict[str, Counter[str]] = {}
    for record in records:
        state = new_game(record.replay.config, record.replay.seed)
        actions = record.replay.actions
        for index, action in enumerate(actions):
            actor = decision_actor(state)
            agent_id = record.seats[player_index(actor)].agent_id
            observation = observe(state, actor)
            forced = filter_immediate_win_actions(observation, observation.legal_actions)
            safety = filter_terminal_actions(observation, observation.legal_actions)
            following = actions[index + 1] if index + 1 < len(actions) else None
            resolved = _resolve_chosen(state, action, following)
            immediate_win = (
                resolved.phase is Phase.TERMINAL
                and resolved.outcome is not None
                and resolved.outcome.winner is actor
            )
            opportunity = bool(forced.forced_win_actions)
            converted = action in forced.forced_win_actions
            losing = action in safety.provable_loss_actions
            by_agent.setdefault(agent_id, Counter(empty_tactical())).update(
                {
                    "decisions": 1,
                    "guaranteed_win_opportunities": int(opportunity),
                    "guaranteed_win_conversions": int(converted),
                    "missed_guaranteed_wins": int(opportunity and not converted),
                    "false_forced_wins": int(converted and not immediate_win),
                    "executed_provable_losses": int(losing),
                    "executed_avoidable_provable_losses": int(
                        losing and not safety.forced_loss_fallback
                    ),
                }
            )
            state = apply_action(state, action)
    return {
        agent_id: {name: counts[name] for name in TACTICAL_FIELDS}
        for agent_id, counts in sorted(by_agent.items())
    }


# ---------------------------------------------------------------------------------------------
# Cell execution and summaries


def _seat_name(player: PlayerId) -> str:
    return player.value


def cell_summary(cell: LeagueCell, records: Sequence[GameRecord]) -> dict[str, object]:
    """Integer outcome summary of one completed family-specific cell, oriented to ``left``."""
    if len(records) != cell.games:
        raise LeagueError(f"cell record count differs from schedule: {cell.key}")
    pair_wins: list[int] = []
    seats: dict[str, dict[str, dict[str, int]]] = {
        policy: {_seat_name(player): {"games": 0, "wins": 0} for player in PlayerId}
        for policy in (cell.left, cell.right)
    }
    margin = 0
    turns = 0
    decisions = 0
    reasons: Counter[str] = Counter()
    for index in range(cell.pair_count):
        pair = records[index * 2 : index * 2 + 2]
        wins = 0
        for record in pair:
            winner_index = player_index(record.winner)
            winner_id = record.seats[winner_index].agent_id
            left_index = next(
                position for position, seat in enumerate(record.seats) if seat.agent_id == cell.left
            )
            for position, seat in enumerate(record.seats):
                row = seats[seat.agent_id][_seat_name(seat.player)]
                row["games"] += 1
                row["wins"] += int(position == winner_index)
            wins += int(winner_id == cell.left)
            margin += record.final_scores[left_index] - record.final_scores[1 - left_index]
            turns += record.turn_count
            decisions += record.decision_count
            reasons[record.terminal_reason] += 1
        pair_wins.append(wins)
    return {
        "key": cell.key,
        "family": cell.family,
        "left": cell.left,
        "right": cell.right,
        "pair_count": cell.pair_count,
        "games": cell.games,
        "left_pair_wins": pair_wins,
        "left_wins": sum(pair_wins),
        "seats": seats,
        "left_score_margin_total": margin,
        "turns_total": turns,
        "decisions_total": decisions,
        "terminal_reasons": dict(sorted(reasons.items())),
    }


def cell_arena_config(cell: LeagueCell, specs: Mapping[str, AgentSpec]) -> ArenaConfig:
    return ArenaConfig(
        cell.run_id, specs[cell.left], specs[cell.right], cell.pair_count, cell.master_seed
    )


def _records_directory(output: Path, cell: LeagueCell) -> Path:
    return output / "cells" / cell.family / f"{cell.left}--vs--{cell.right}" / "records"


def run_league_cell(
    output: Path,
    plan: Mapping[str, object],
    cell: LeagueCell,
    specs: Mapping[str, AgentSpec],
) -> dict[str, object]:
    """Run or exactly reconstruct one cell and retain its report, summary, and tactical counts."""
    directory = _records_directory(output, cell)
    artifact_path = directory.parent / "cell.json"
    arena = cell_arena_config(cell, specs)
    if artifact_path.exists():
        artifact = read_json(artifact_path)
        manifest, records = load_corpus(directory, verify_code=False)
        report = cast(Mapping[str, Any], artifact.get("report"))
        expected = {
            "report": arena_report_from_records(
                arena, records, elapsed_seconds=float(report["elapsed_seconds"])
            ).to_data(),
            "summary": cell_summary(cell, records),
            "tactical": record_tactical_counts(records),
            "corpus_fingerprint": manifest.corpus_fingerprint,
        }
        if not seal_matches(artifact) or any(
            artifact.get(key) != value for key, value in expected.items()
        ):
            raise LeagueError(f"resumed cell does not reconstruct exactly: {cell.key}")
        return artifact
    started = time.perf_counter()
    retained = run_resumable_arena(
        directory,
        arena,
        generation=None,
        corpus_configuration={
            "version": CELL_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "cell": cell.to_data(),
        },
    )
    manifest, records = load_corpus(directory, verify_replays=False)
    tactical_started = time.perf_counter()
    tactical = record_tactical_counts(records)
    artifact = sealed(
        CELL_VERSION,
        plan_fingerprint=plan["plan_fingerprint"],
        cell=cell.to_data(),
        master_seed=cell.master_seed,
        records_locator=directory.relative_to(output).as_posix(),
        corpus_fingerprint=manifest.corpus_fingerprint,
        report=retained.report.to_data(),
        summary=cell_summary(cell, records),
        tactical=tactical,
        timing_seconds={
            "arena_and_records": tactical_started - started,
            "tactical_replay": time.perf_counter() - tactical_started,
        },
    )
    write_immutable(artifact_path, artifact)
    return artifact


def load_cell_records(output: Path, cell: LeagueCell) -> tuple[GameRecord, ...]:
    _, records = load_corpus(
        _records_directory(output, cell), verify_code=False, verify_replays=False
    )
    return records


# ---------------------------------------------------------------------------------------------
# Aligned incumbent/challenger prefix audit


def _variant_index(record: GameRecord, policy: str) -> int:
    matches = [index for index, seat in enumerate(record.seats) if seat.agent_id == policy]
    if len(matches) != 1:
        raise LeagueError("aligned record must contain its q0 variant exactly once")
    return matches[0]


def compare_aligned_games(incumbent: GameRecord, challenger: GameRecord) -> dict[str, object]:
    """Compare one incumbent/challenger game pair against the same opponent stream.

    Histories must be identical until the first decision at which the q0 variant faces a public
    guaranteed current-turn win. There the challenger must choose a guaranteed action and the game
    must terminate immediately in its win.
    """
    seat = _variant_index(incumbent, INCUMBENT_ID)
    other = 1 - seat
    base: dict[str, object] = {
        "game_id": incumbent.game_id,
        "variant_seat_index": seat,
        "metadata_aligned": False,
        "prefix_aligned": False,
        "endpoint_index": None,
        "endpoint_classification": None,
        "challenger_converted": False,
        "terminated_in_guaranteed_win": False,
        "incumbent_won": player_index(incumbent.winner) == seat,
        "challenger_won": player_index(challenger.winner) == seat,
        "failure": None,
    }
    try:
        challenger_seat = _variant_index(challenger, CHALLENGER_ID)
    except LeagueError:
        challenger_seat = -1
    inc_variant, inc_opponent = incumbent.seats[seat], incumbent.seats[other]
    metadata = (
        challenger_seat == seat
        and incumbent.game_id == challenger.game_id
        and incumbent.pair_id == challenger.pair_id
        and incumbent.replay.seed == challenger.replay.seed
        and incumbent.replay.config == challenger.replay.config
        and inc_variant.seed == challenger.seats[seat].seed
        and inc_variant.rng_identity == challenger.seats[seat].rng_identity == Q0_CORE_RNG_IDENTITY
        and inc_opponent.agent_id == challenger.seats[other].agent_id
        and inc_opponent.seed == challenger.seats[other].seed
        and inc_opponent.rng_domain == challenger.seats[other].rng_domain
        and dict(inc_opponent.config) == dict(challenger.seats[other].config)
        and inc_opponent.player == challenger.seats[other].player
    )
    if not metadata:
        return {**base, "failure": "setup, seat, or RNG metadata differ"}
    base["metadata_aligned"] = True
    state = new_game(incumbent.replay.config, incumbent.replay.seed)
    inc_actions, ch_actions = incumbent.replay.actions, challenger.replay.actions
    index = 0
    while index < len(inc_actions) and index < len(ch_actions):
        actor = decision_actor(state)
        if player_index(actor) == seat:
            observation = observe(state, actor)
            forced = filter_immediate_win_actions(observation, observation.legal_actions)
            if forced.forced_win_actions:
                ch_action = ch_actions[index]
                inc_converted = inc_actions[index] in forced.forced_win_actions
                converted = ch_action in forced.forced_win_actions
                consumed = index + 1
                resolved = apply_action(state, ch_action)
                if (
                    isinstance(ch_action, PlayOfferAction)
                    and resolved.phase is not Phase.TERMINAL
                    and consumed < len(ch_actions)
                ):
                    resolved = apply_action(resolved, ch_actions[consumed])
                    consumed += 1
                terminated = (
                    converted
                    and resolved.phase is Phase.TERMINAL
                    and resolved.outcome is not None
                    and player_index(resolved.outcome.winner) == seat
                    and consumed == len(ch_actions)
                )
                classification = (
                    "both_converted"
                    if inc_converted and converted
                    else "incumbent_missed_challenger_converted"
                    if converted
                    else "challenger_failed_guaranteed_win"
                )
                return {
                    **base,
                    "prefix_aligned": True,
                    "endpoint_index": index,
                    "endpoint_classification": classification,
                    "challenger_converted": converted,
                    "terminated_in_guaranteed_win": terminated,
                    "failure": None
                    if terminated
                    else "challenger intervention did not end in its win",
                }
        if inc_actions[index] != ch_actions[index]:
            return {**base, "failure": f"actions diverged before endpoint at {index}"}
        state = apply_action(state, inc_actions[index])
        index += 1
    if len(inc_actions) != len(ch_actions) or incumbent.winner != challenger.winner:
        return {**base, "failure": "games without an endpoint diverged in length or outcome"}
    return {**base, "prefix_aligned": True}


PREFIX_FIELDS: Final = (
    "aligned_games",
    "metadata_failures",
    "prefix_failures",
    "endpoints",
    "both_converted",
    "incumbent_missed_challenger_converted",
    "challenger_failed_guaranteed_win",
    "post_intervention_termination_failures",
    "games_without_endpoint",
    "incumbent_win_challenger_loss_games",
    "incumbent_win_challenger_loss_blocks",
    "challenger_win_incumbent_loss_games",
)


def audit_aligned_prefixes(
    design: LeagueDesign, load_records: Callable[[LeagueCell], Sequence[GameRecord]]
) -> dict[str, object]:
    """Audit every aligned incumbent/challenger common-opponent game in both families."""
    cells = {cell.key: cell for cell in design.cells()}
    rows: dict[str, dict[str, int]] = {}
    failures: list[dict[str, object]] = []
    for family in design.family_order:
        for opponent in design.policy_order[2:]:
            incumbent_cell = cells[f"{family}:{INCUMBENT_ID}--vs--{opponent}"]
            challenger_cell = cells[f"{family}:{CHALLENGER_ID}--vs--{opponent}"]
            incumbent_records = load_records(incumbent_cell)
            challenger_records = load_records(challenger_cell)
            if len(incumbent_records) != len(challenger_records):
                raise LeagueError("aligned cells have different record counts")
            row = dict.fromkeys(PREFIX_FIELDS, 0)
            bad_blocks: set[str] = set()
            for inc, ch in zip(incumbent_records, challenger_records, strict=True):
                comparison = compare_aligned_games(inc, ch)
                classification = comparison["endpoint_classification"]
                endpoint = comparison["endpoint_index"] is not None
                regression = bool(comparison["incumbent_won"]) and not comparison["challenger_won"]
                if regression:
                    bad_blocks.add(str(inc.pair_id))
                values = {
                    "aligned_games": 1,
                    "metadata_failures": int(not comparison["metadata_aligned"]),
                    "prefix_failures": int(not comparison["prefix_aligned"]),
                    "endpoints": int(endpoint),
                    "both_converted": int(classification == "both_converted"),
                    "incumbent_missed_challenger_converted": int(
                        classification == "incumbent_missed_challenger_converted"
                    ),
                    "challenger_failed_guaranteed_win": int(
                        classification == "challenger_failed_guaranteed_win"
                    ),
                    "post_intervention_termination_failures": int(
                        endpoint and not comparison["terminated_in_guaranteed_win"]
                    ),
                    "games_without_endpoint": int(
                        comparison["metadata_aligned"] is True and not endpoint
                    ),
                    "incumbent_win_challenger_loss_games": int(regression),
                    "challenger_win_incumbent_loss_games": int(
                        bool(comparison["challenger_won"]) and not comparison["incumbent_won"]
                    ),
                }
                for name, value in values.items():
                    row[name] += value
                if comparison["failure"] is not None and len(failures) < 24:
                    failures.append(
                        {
                            "family": family,
                            "opponent": opponent,
                            "incumbent_record_fingerprint": game_record_fingerprint(inc),
                            "challenger_record_fingerprint": game_record_fingerprint(ch),
                            **comparison,
                        }
                    )
            row["incumbent_win_challenger_loss_blocks"] = len(bad_blocks)
            rows[f"{family}:{opponent}"] = row
    totals = dict.fromkeys(PREFIX_FIELDS, 0)
    for row in rows.values():
        for name in PREFIX_FIELDS:
            totals[name] += row[name]
    return sealed(
        PREFIX_VERSION,
        definition=(
            "identical semantic histories until the first q0-variant public guaranteed-win "
            "decision; the challenger must then choose a guaranteed action and win immediately"
        ),
        totals=totals,
        by_family_opponent=dict(sorted(rows.items())),
        failure_examples=failures,
    )


# ---------------------------------------------------------------------------------------------
# Statistics


def _sorted_interval(values: list[int], denominator: int) -> list[float]:
    values.sort()
    return [
        values[BOOTSTRAP_LOWER_INDEX] / denominator,
        values[BOOTSTRAP_UPPER_INDEX] / denominator,
    ]


def oriented_pair_wins(summary: Mapping[str, Any], policy: str) -> list[int]:
    """Pair wins of ``policy`` in one cell, regardless of its left/right position."""
    values = cast(list[int], summary["left_pair_wins"])
    if summary["left"] == policy:
        return list(values)
    if summary["right"] == policy:
        return [2 - value for value in values]
    raise LeagueError("policy is absent from the cell summary")


class _Summaries:
    def __init__(self, design: LeagueDesign, summaries: Mapping[str, Mapping[str, Any]]) -> None:
        expected = {cell.key for cell in design.cells()}
        if set(summaries) != expected:
            raise LeagueError("statistics require exactly every scheduled cell")
        self.design = design
        self.summaries = summaries
        self.index = {policy: position for position, policy in enumerate(design.policy_order)}

    def cell(self, family: str, first: str, second: str) -> Mapping[str, Any]:
        left, right = sorted((first, second), key=self.index.__getitem__)
        return self.summaries[f"{family}:{left}--vs--{right}"]

    def blocks(self, policy: str, opponent: str) -> dict[str, list[int]]:
        return {
            family: oriented_pair_wins(self.cell(family, policy, opponent), policy)
            for family in self.design.family_order
        }


def _stratified_resamples(
    rng: DeterministicRandom, design: LeagueDesign
) -> Iterator[tuple[list[int], ...]]:
    """Yield, per resample, one block-index list per family in family order.

    The stream equals sequential per-family ``randbelow(pair_count)`` draws. Every family holds
    ``pair_count`` blocks, so consecutive families and resamples share one bound and are fetched
    in chunks purely for throughput.
    """
    blocks, families = design.pair_count, len(design.family_order)
    per_resample = blocks * families
    remaining = BOOTSTRAP_RESAMPLES
    while remaining:
        chunk = min(RESAMPLE_CHUNK, remaining)
        draws = rng.randbelow_batch(blocks, per_resample * chunk)
        for start in range(0, per_resample * chunk, per_resample):
            yield tuple(
                draws[start + family * blocks : start + (family + 1) * blocks]
                for family in range(families)
            )
        remaining -= chunk


def _matchup_bootstrap(
    design: LeagueDesign, left: str, right: str, blocks: Mapping[str, list[int]]
) -> list[int]:
    if any(len(blocks[family]) != design.pair_count for family in design.family_order):
        raise LeagueError("matchup blocks do not match the design block count")
    domain = f"{MATCHUP_BOOTSTRAP_DOMAIN}:{left}--vs--{right}"
    rng = DeterministicRandom(design.bootstrap_seed(domain), f"{design.namespace}:{domain}")
    rows = [blocks[family] for family in design.family_order]
    return [
        sum(
            sum(values[index] for index in indexes)
            for values, indexes in zip(rows, draw, strict=True)
        )
        for draw in _stratified_resamples(rng, design)
    ]


def _seat_rows(summaries: _Summaries, policy: str, opponent: str) -> dict[str, dict[str, object]]:
    combined: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, dict[str, int]]] = {}
    for family in summaries.design.family_order:
        seats = cast(
            Mapping[str, Mapping[str, Mapping[str, int]]],
            summaries.cell(family, policy, opponent)["seats"],
        )[policy]
        by_family[family] = {seat: dict(values) for seat, values in seats.items()}
        for seat, values in seats.items():
            row = combined.setdefault(seat, {"games": 0, "wins": 0})
            row["games"] += values["games"]
            row["wins"] += values["wins"]

    def described(values: Mapping[str, int]) -> dict[str, object]:
        return {
            "games": values["games"],
            "wins": values["wins"],
            "win_rate": values["wins"] / values["games"],
            "wilson_95": list(wilson_interval(values["wins"], values["games"])),
        }

    return {
        seat: {
            "combined": described(combined[seat]),
            "by_family": {family: described(by_family[family][seat]) for family in by_family},
        }
        for seat in sorted(combined)
    }


def _matchup_row(
    summaries: _Summaries,
    reports: Mapping[str, Mapping[str, Any]],
    tactical: Mapping[str, Mapping[str, Mapping[str, int]]],
    left: str,
    right: str,
) -> dict[str, object]:
    design = summaries.design
    blocks = summaries.blocks(left, right)
    blocks_total = sum(len(values) for values in blocks.values())
    denominator = 2 * blocks_total
    wins = sum(sum(values) for values in blocks.values())
    totals = _matchup_bootstrap(design, left, right, blocks)
    family_rows: dict[str, object] = {}
    games = margin = turns = decisions = 0
    elapsed = 0.0
    reasons: Counter[str] = Counter()
    counts = {policy: empty_tactical() for policy in (left, right)}
    configs: set[str] = set()
    for family in design.family_order:
        summary = summaries.cell(family, left, right)
        report = reports[cast(str, summary["key"])]
        paired = cast(Mapping[str, Any], report["paired_bootstrap_confidence_interval_95"])
        family_wins = sum(blocks[family])
        family_rows[family] = {
            "left_wins": family_wins,
            "games": 2 * len(blocks[family]),
            "left_win_rate": family_wins / (2 * len(blocks[family])),
            "paired_bootstrap_95": list(paired["interval"]),
            "paired_bootstrap_seed": paired["bootstrap_seed"],
        }
        games += cast(int, summary["games"])
        sign = 1 if summary["left"] == left else -1
        margin += sign * cast(int, summary["left_score_margin_total"])
        turns += cast(int, summary["turns_total"])
        decisions += cast(int, summary["decisions_total"])
        reasons.update(cast(Mapping[str, int], summary["terminal_reasons"]))
        elapsed += float(report["elapsed_seconds"])
        for policy in (left, right):
            for name in TACTICAL_FIELDS:
                counts[policy][name] += tactical[cast(str, summary["key"])][policy][name]
        agents = cast(Mapping[str, Mapping[str, Any]], report["agents"])
        configs.add(
            fingerprint({agents[side]["id"]: agents[side]["config"] for side in ("a", "b")})
        )
    if len(configs) != 1:
        raise LeagueError(f"policy configurations differ across families: {left} vs {right}")
    agents = cast(
        Mapping[str, Mapping[str, Any]],
        reports[cast(str, summaries.cell(design.family_order[0], left, right)["key"])]["agents"],
    )
    return {
        "left": left,
        "right": right,
        "combined": {
            "left_wins": wins,
            "games": denominator,
            "left_win_rate": wins / denominator,
            "paired_block_bootstrap_95": _sorted_interval(totals, denominator),
            "bootstrap": {
                "method": "family-stratified-paired-block-percentile-v1",
                "resamples": BOOTSTRAP_RESAMPLES,
                "order_statistic_indices": [BOOTSTRAP_LOWER_INDEX, BOOTSTRAP_UPPER_INDEX],
                "seed": design.bootstrap_seed(f"{MATCHUP_BOOTSTRAP_DOMAIN}:{left}--vs--{right}"),
            },
        },
        "by_family": family_rows,
        "seats": {
            left: _seat_rows(summaries, left, right),
            right: _seat_rows(summaries, right, left),
        },
        "left_average_score_margin": margin / games,
        "average_turns": turns / games,
        "average_decisions": decisions / games,
        "terminal_reasons": dict(sorted(reasons.items())),
        "arena_elapsed_seconds": elapsed,
        "games_per_second": games / elapsed if elapsed > 0 else None,
        "tactical": counts,
        "config_fingerprints": {
            agents[side]["id"]: fingerprint(agents[side]["config"]) for side in ("a", "b")
        },
    }


def _anchor_contrasts(summaries: _Summaries) -> dict[str, object]:
    """Joint family-stratified common-block bootstrap of challenger-minus-incumbent scores."""
    design = summaries.design
    differences: dict[str, dict[str, list[int]]] = {}
    for anchor in ANCHOR_IDS:
        challenger = summaries.blocks(CHALLENGER_ID, anchor)
        incumbent = summaries.blocks(INCUMBENT_ID, anchor)
        differences[anchor] = {
            family: [
                after - before
                for after, before in zip(challenger[family], incumbent[family], strict=True)
            ]
            for family in design.family_order
        }
    blocks_total = sum(len(differences[ANCHOR_IDS[0]][family]) for family in design.family_order)
    denominator = 2 * blocks_total
    domain = ANCHOR_BOOTSTRAP_DOMAIN
    rng = DeterministicRandom(design.bootstrap_seed(domain), f"{design.namespace}:{domain}")
    samples: dict[str, list[int]] = {anchor: [] for anchor in ANCHOR_IDS}
    macro: list[int] = []
    for draw in _stratified_resamples(rng, design):
        sums = dict.fromkeys(ANCHOR_IDS, 0)
        for family, indexes in zip(design.family_order, draw, strict=True):
            # One index draw per block is shared by every anchor and by both arms.
            for index in indexes:
                for anchor in ANCHOR_IDS:
                    sums[anchor] += differences[anchor][family][index]
        for anchor in ANCHOR_IDS:
            samples[anchor].append(sums[anchor])
        macro.append(sum(sums.values()))
    points = {
        anchor: sum(sum(values) for values in differences[anchor].values()) for anchor in ANCHOR_IDS
    }
    return {
        "estimand": "score(challenger, O) - score(incumbent, O) on aligned blocks",
        "by_anchor": {
            anchor: {
                "numerator": points[anchor],
                "denominator": denominator,
                "point_estimate": points[anchor] / denominator,
                "interval_95": _sorted_interval(samples[anchor], denominator),
            }
            for anchor in ANCHOR_IDS
        },
        "macro": {
            "numerator": sum(points.values()),
            "denominator": denominator * len(ANCHOR_IDS),
            "point_estimate": sum(points.values()) / (denominator * len(ANCHOR_IDS)),
            "interval_95": _sorted_interval(macro, denominator * len(ANCHOR_IDS)),
        },
        "bootstrap": {
            "method": "joint-family-stratified-common-block-percentile-v1",
            "resamples": BOOTSTRAP_RESAMPLES,
            "order_statistic_indices": [BOOTSTRAP_LOWER_INDEX, BOOTSTRAP_UPPER_INDEX],
            "seed": design.bootstrap_seed(domain),
            "strata": list(design.family_order),
        },
    }


def _m_family(summaries: _Summaries) -> dict[str, object]:
    """Nested bootstrap: replicate identity outside, aligned family/block indexes inside."""
    design = summaries.design
    opponents = tuple(policy for policy in design.policy_order if policy not in M_IDS)
    blocks = {
        replicate: {opponent: summaries.blocks(replicate, opponent) for opponent in opponents}
        for replicate in M_IDS
    }
    blocks_total = sum(len(values) for values in blocks[M_IDS[0]][opponents[0]].values())
    denominator = 2 * blocks_total * len(M_IDS)
    domain = M_FAMILY_BOOTSTRAP_DOMAIN
    rng = DeterministicRandom(design.bootstrap_seed(domain), f"{design.namespace}:{domain}")
    per_opponent: dict[str, list[int]] = {opponent: [] for opponent in opponents}
    macro: list[int] = []
    worst: list[int] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        replicates = [M_IDS[index] for index in rng.randbelow_batch(len(M_IDS), len(M_IDS))]
        sums = dict.fromkeys(opponents, 0)
        count = design.pair_count
        draws = rng.randbelow_batch(count, count * len(design.family_order))
        for position, family in enumerate(design.family_order):
            indexes = draws[position * count : (position + 1) * count]
            for replicate in replicates:
                for opponent in opponents:
                    values = blocks[replicate][opponent][family]
                    sums[opponent] += sum(values[index] for index in indexes)
        for opponent in opponents:
            per_opponent[opponent].append(sums[opponent])
        macro.append(sum(sums.values()))
        worst.append(min(sums.values()))
    points = {
        opponent: sum(
            sum(sum(values) for values in blocks[replicate][opponent].values())
            for replicate in M_IDS
        )
        for opponent in opponents
    }
    worst_opponent = min(
        opponents, key=lambda opponent: (points[opponent], opponents.index(opponent))
    )
    return {
        "replicates": list(M_IDS),
        "opponents": list(opponents),
        "excludes_m_vs_m_cells": True,
        "selects_replicate": False,
        "by_opponent": {
            opponent: {
                "numerator": points[opponent],
                "denominator": denominator,
                "point_estimate": points[opponent] / denominator,
                "interval_95": _sorted_interval(per_opponent[opponent], denominator),
            }
            for opponent in opponents
        },
        "versus_incumbent": points[INCUMBENT_ID] / denominator,
        "versus_challenger": points[CHALLENGER_ID] / denominator,
        "equal_weight_macro": {
            "numerator": sum(points.values()),
            "denominator": denominator * len(opponents),
            "point_estimate": sum(points.values()) / (denominator * len(opponents)),
            "interval_95": _sorted_interval(macro, denominator * len(opponents)),
        },
        "worst_opponent": {
            "opponent": worst_opponent,
            "point_estimate": points[worst_opponent] / denominator,
            "resampled_minimum_interval_95": _sorted_interval(worst, denominator),
        },
        "bootstrap": {
            "method": "nested-replicate-outer-aligned-block-inner-percentile-v1",
            "resamples": BOOTSTRAP_RESAMPLES,
            "order_statistic_indices": [BOOTSTRAP_LOWER_INDEX, BOOTSTRAP_UPPER_INDEX],
            "seed": design.bootstrap_seed(domain),
        },
    }


def league_statistics(
    design: LeagueDesign,
    summaries: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    tactical: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, object]:
    """Compute the full matrix, per-policy summaries, M family, and anchor contrasts."""
    indexed = _Summaries(design, summaries)
    matchups: dict[str, object] = {}
    for left_index, left in enumerate(design.policy_order):
        for right in design.policy_order[left_index + 1 :]:
            matchups[f"{left}--vs--{right}"] = _matchup_row(indexed, reports, tactical, left, right)
    matrix: dict[str, dict[str, float]] = {policy: {} for policy in design.policy_order}
    numerators: dict[str, dict[str, int]] = {policy: {} for policy in design.policy_order}
    games_per_matchup = 2 * design.pair_count * len(design.family_order)
    for raw in matchups.values():
        row = cast(Mapping[str, Any], raw)
        left, right = row["left"], row["right"]
        wins = cast(int, row["combined"]["left_wins"])
        numerators[left][right] = wins
        numerators[right][left] = games_per_matchup - wins
        matrix[left][right] = wins / games_per_matchup
        matrix[right][left] = (games_per_matchup - wins) / games_per_matchup
    policies: dict[str, object] = {}
    for policy in design.policy_order:
        opponents = [other for other in design.policy_order if other != policy]
        total = sum(numerators[policy][other] for other in opponents)
        worst = min(
            opponents, key=lambda other: (numerators[policy][other], opponents.index(other))
        )
        policies[policy] = {
            "equal_opponent_macro": total / (games_per_matchup * len(opponents)),
            "equal_opponent_macro_numerator": total,
            "worst_opponent": worst,
            "worst_opponent_win_rate": matrix[policy][worst],
            "copeland_descriptive": sum(
                1 for other in opponents if 2 * numerators[policy][other] > games_per_matchup
            ),
        }
    data: dict[str, object] = {
        "design": design.to_data(),
        "matchups": matchups,
        "matrix": matrix,
        "policies": policies,
        "scalar_rankings_descriptive_only": True,
    }
    if set(M_IDS) <= set(design.policy_order):
        data["m_family"] = _m_family(indexed)
    if {INCUMBENT_ID, CHALLENGER_ID, *ANCHOR_IDS} <= set(design.policy_order):
        data["anchor_contrasts"] = _anchor_contrasts(indexed)
    return sealed(STATISTICS_VERSION, **data)


def aggregate_tactical(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    """Sum per-cell tactical counts by policy and evaluate the envelope integrity rules."""
    totals: dict[str, dict[str, int]] = {}
    for artifact in cells.values():
        for policy, counts in cast(Mapping[str, Mapping[str, int]], artifact["tactical"]).items():
            row = totals.setdefault(policy, empty_tactical())
            for name in TACTICAL_FIELDS:
                row[name] += counts[name]
    violations: list[str] = []
    for policy in SAFETY_ENVELOPED_IDS:
        if policy in totals and totals[policy]["executed_avoidable_provable_losses"]:
            violations.append(f"avoidable_loss:{policy}")
    for policy in M_IDS:
        if policy in totals and (
            totals[policy]["missed_guaranteed_wins"] or totals[policy]["false_forced_wins"]
        ):
            violations.append(f"offense_envelope:{policy}")
    return sealed(
        TACTICAL_VERSION,
        by_policy=dict(sorted(totals.items())),
        safety_enveloped=list(SAFETY_ENVELOPED_IDS),
        offense_enveloped=list(OFFENSE_ENVELOPED_IDS),
        q_policies_missed_wins_descriptive=list(Q_IDS),
        nonchallenger_envelope_violations=violations,
        status="passed" if not violations else "failed",
    )


# ---------------------------------------------------------------------------------------------
# Immutable decision


def promotion_decision(
    *,
    statistics: Mapping[str, Any],
    tactical: Mapping[str, Any],
    prefix: Mapping[str, Any],
    integrity: Mapping[str, bool],
) -> dict[str, object]:
    """Apply the frozen eight-condition promotion rule."""
    anchors = statistics["anchor_contrasts"]["macro"]
    random_row = statistics["matchups"][f"{CHALLENGER_ID}--vs--{RANDOM_ID}"]["combined"]
    totals = prefix["totals"]
    challenger = tactical["by_policy"][CHALLENGER_ID]
    incumbent = tactical["by_policy"][INCUMBENT_ID]
    seat_points: dict[str, float] = {}
    for opponent in (INCUMBENT_ID, HEURISTIC_ID, RANDOM_ID):
        left, right = (
            (INCUMBENT_ID, CHALLENGER_ID) if opponent == INCUMBENT_ID else (CHALLENGER_ID, opponent)
        )
        seats = statistics["matchups"][f"{left}--vs--{right}"]["seats"][CHALLENGER_ID]
        for seat, row in seats.items():
            seat_points[f"{opponent}:{seat}"] = row["combined"]["win_rate"]
    minimum_seat = min(seat_points.values())
    conditions = [
        {
            "id": "anchor_macro_lower_strictly_above_0.25pp",
            "value": anchors["interval_95"][0],
            "threshold": PROMOTION_LIFT_THRESHOLD,
            "passed": anchors["interval_95"][0] > PROMOTION_LIFT_THRESHOLD,
        },
        {
            "id": "zero_incumbent_win_challenger_loss_blocks",
            "value": totals["incumbent_win_challenger_loss_blocks"],
            "passed": totals["incumbent_win_challenger_loss_blocks"] == 0
            and totals["incumbent_win_challenger_loss_games"] == 0,
        },
        {
            "id": "zero_prefix_or_alignment_failures",
            "value": totals["metadata_failures"] + totals["prefix_failures"],
            "passed": totals["metadata_failures"] == 0
            and totals["prefix_failures"] == 0
            and totals["post_intervention_termination_failures"] == 0
            and totals["challenger_failed_guaranteed_win"] == 0,
        },
        {
            "id": "challenger_zero_missed_false_or_avoidable",
            "value": {
                "missed_guaranteed_wins": challenger["missed_guaranteed_wins"],
                "false_forced_wins": challenger["false_forced_wins"],
                "executed_avoidable_provable_losses": challenger[
                    "executed_avoidable_provable_losses"
                ],
            },
            "passed": challenger["missed_guaranteed_wins"] == 0
            and challenger["false_forced_wins"] == 0
            and challenger["executed_avoidable_provable_losses"] == 0,
        },
        {
            "id": "incumbent_zero_avoidable_losses",
            "value": incumbent["executed_avoidable_provable_losses"],
            "passed": incumbent["executed_avoidable_provable_losses"] == 0,
        },
        {
            "id": "challenger_vs_random_lower_strictly_above_50pct",
            "value": random_row["paired_block_bootstrap_95"][0],
            "threshold": RANDOM_LOWER_THRESHOLD,
            "passed": random_row["paired_block_bootstrap_95"][0] > RANDOM_LOWER_THRESHOLD,
        },
        {
            "id": "challenger_minimum_physical_seat_at_least_45pct",
            "value": minimum_seat,
            "by_seat": dict(sorted(seat_points.items())),
            "threshold": SEAT_FLOOR,
            "passed": minimum_seat >= SEAT_FLOOR,
        },
        {
            "id": "provenance_holdout_source_schedule_replay_checksum_deadline",
            "value": dict(sorted(integrity.items())),
            "passed": all(integrity.values()),
        },
    ]
    integrity_ok = all(integrity.values()) and tactical["status"] == "passed"
    if not integrity_ok:
        decision = DECISION_BLOCKED
    elif all(condition["passed"] for condition in conditions):
        decision = DECISION_PROMOTE
    else:
        decision = DECISION_RETAIN
    return sealed(
        DECISION_VERSION,
        statistics_fingerprint=statistics["artifact_fingerprint"],
        tactical_fingerprint=tactical["artifact_fingerprint"],
        prefix_fingerprint=prefix["artifact_fingerprint"],
        conditions=conditions,
        nonchallenger_envelope_status=tactical["status"],
        decision=decision,
        requires_independent_validation=True,
        deployment_change_authorized_by_runner=False,
    )


# ---------------------------------------------------------------------------------------------
# Execution state, checksums, and orchestration


def checksum_manifest(output: Path) -> dict[str, object]:
    files: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in CHECKSUM_EXCLUDED_NAMES:
            files[path.relative_to(output).as_posix()] = sha256_file(path)
    return sealed(CHECKSUM_VERSION, excluded_names=sorted(CHECKSUM_EXCLUDED_NAMES), files=files)


def _load_execution(path: Path) -> dict[str, Any]:
    state = read_json(path)
    if state.get("version") != EXECUTION_VERSION or not seal_matches(state):
        raise LeagueError("execution state is malformed")
    return state


def _store_execution(path: Path, state: Mapping[str, object]) -> None:
    payload = {key: value for key, value in state.items() if key != "artifact_fingerprint"}
    write_mutable(path, {**payload, "artifact_fingerprint": fingerprint(payload)})


def begin_execution(output: Path, plan: Mapping[str, object]) -> Path:
    """Record one initial attempt or the sole permitted resume of the same immutable plan."""
    path = output / "execution-state.json"
    if path.exists():
        state = _load_execution(path)
        if state["plan_fingerprint"] != plan["plan_fingerprint"]:
            raise LeagueError("execution state belongs to another plan")
        if state["completed"] is True:
            raise LeagueError("this league execution already completed; validate it instead")
        attempts = list(state["attempts"])
        if len(attempts) >= ALLOWED_ATTEMPTS:
            raise LeagueError("the single permitted Step-5 resume has already been used")
        attempts.append({"attempt": len(attempts) + 1, "started_at_epoch_seconds": time.time()})
        state["attempts"] = attempts
    else:
        state = {
            "version": EXECUTION_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "claim": cast(Mapping[str, object], plan["execution"])["claim"],
            "attempts": [{"attempt": 1, "started_at_epoch_seconds": time.time()}],
            "active_seconds": 0.0,
            "phase_seconds": {},
            "completed": False,
        }
    _store_execution(path, state)
    return path


@contextmanager
def timed_phase(path: Path, phase: str, *, cutoff_seconds: float | None) -> Iterator[None]:
    """Accumulate active phase time and enforce the claim cutoff on active execution time."""
    state = _load_execution(path)
    if cutoff_seconds is not None and float(state["active_seconds"]) >= cutoff_seconds:
        raise LeagueError(f"claim cutoff reached before {phase}")
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        state = _load_execution(path)
        phases = dict(state["phase_seconds"])
        phases[phase] = float(phases.get(phase, 0.0)) + elapsed
        state["phase_seconds"] = phases
        state["active_seconds"] = float(state["active_seconds"]) + elapsed
        _store_execution(path, state)
    if cutoff_seconds is not None and float(state["active_seconds"]) >= cutoff_seconds:
        raise LeagueError(f"claim cutoff reached after {phase}")


def source_identity() -> dict[str, object]:
    return {
        **inspect_source_identity().to_data(),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }


@dataclass(frozen=True, slots=True)
class LeagueRunConfig:
    """One league execution. Only the exact committed claim configuration is claim eligible."""

    output: Path
    registry_path: Path = REGISTRY_PATH
    inputs_root: Path = Path(".")
    policy_mode: str = "learned-checkpoints"
    smoke_pairs: int | None = None
    holdout_roots: tuple[Path, ...] = (Path("runs"),)
    preflight_path: Path | None = None


def validate_runtime_preflight(
    path: Path | None, *, code: object, registry_fingerprint: object
) -> tuple[dict[str, Any] | None, str | None]:
    """Require a passing measured smoke projection bound to this code and registry."""
    if path is None:
        return None, "missing_runtime_preflight"
    try:
        value = read_json(path)
    except LeagueError:
        return None, "unreadable_runtime_preflight"
    if (
        value.get("version") != PREFLIGHT_VERSION
        or not seal_matches(value)
        or value.get("claim_eligible") is not True
        or value.get("code_fingerprint") != code
        or value.get("registry_fingerprint") != registry_fingerprint
    ):
        return None, "invalid_runtime_preflight"
    return value, None


def build_runtime_preflight(smoke_output: Path) -> dict[str, object]:
    """Project claim runtime from a validated nonclaim smoke and apply the frozen 1.20 gate."""
    plan = read_json(smoke_output / "plan.json")
    state = _load_execution(smoke_output / "execution-state.json")
    validation = read_json(smoke_output / "validation.json")
    runtime = read_json(smoke_output / "validator-runtime.json")
    execution = cast(Mapping[str, Any], plan["execution"])
    if execution["claim"] is not False or execution["policy_mode"] != "learned-checkpoints":
        raise LeagueError("runtime preflight requires a learned-checkpoint nonclaim smoke")
    if state["completed"] is not True or validation.get("status") != "passed":
        raise LeagueError("runtime preflight requires a completed and validated smoke")
    scale = PAIRS_PER_FAMILY / int(execution["pair_count"])
    runner_seconds = float(state["active_seconds"])
    validator_seconds = float(runtime["elapsed_seconds"])
    projected = (runner_seconds + validator_seconds) * scale * PROJECTION_SAFETY_FACTOR / 60
    return sealed(
        PREFLIGHT_VERSION,
        smoke_plan_fingerprint=plan["plan_fingerprint"],
        code_fingerprint=cast(Mapping[str, object], plan["source"])["code_fingerprint"],
        registry_fingerprint=plan["registry_fingerprint"],
        smoke_pair_count=execution["pair_count"],
        linear_scale_to_claim=scale,
        runner_phase_seconds=state["phase_seconds"],
        runner_active_seconds=runner_seconds,
        validator_seconds=validator_seconds,
        validator_phase_seconds=runtime.get("phase_seconds"),
        safety_factor=PROJECTION_SAFETY_FACTOR,
        projected_claim_minutes=projected,
        limit_minutes=PROJECTION_LIMIT_MINUTES,
        claim_eligible=projected <= PROJECTION_LIMIT_MINUTES,
    )


def build_plan(
    config: LeagueRunConfig,
) -> tuple[dict[str, object], LeagueDesign, PolicyBundle, dict[str, Any]]:
    registry = load_registry(config.registry_path)
    root = repository_root() if config.inputs_root == Path(".") else config.inputs_root
    claim_requested = config.smoke_pairs is None
    design = (
        claim_design()
        if claim_requested
        else smoke_design(config.smoke_pairs or 1, root_seed=int(registry["root_seed"]))
    )
    source = source_identity()
    reasons: list[str] = []
    schedule_audit = verify_registry_schedule(registry)
    input_audit = (
        verify_league_inputs(registry, root, require_archives=claim_requested)
        if config.policy_mode == "learned-checkpoints"
        else sealed(
            "m7-independent-league-input-audit-v1", status="not-applicable-toy-smoke", failures=[]
        )
    )
    if claim_requested:
        if not is_claim_registry(registry):
            reasons.append("registry_is_not_committed_claim_registry")
        if config.policy_mode != "learned-checkpoints":
            reasons.append("claim_requires_learned_checkpoints")
        if schedule_audit["status"] != "passed":
            reasons.append("registry_schedule_reconstruction_failed")
        if input_audit["status"] != "passed":
            reasons.append("input_authentication_failed")
        if cast(Mapping[str, object], source)["tracked_tree_clean"] is not True:
            reasons.append("source_tree_not_clean")
        expected_output = (repository_root() / "runs" / CYCLE_ID).resolve()
        if config.output.resolve() != expected_output:
            reasons.append("claim_output_must_be_runs_cycle_directory")
        if tuple(path.resolve() for path in config.holdout_roots) != (
            (repository_root() / "runs").resolve(),
        ):
            reasons.append("claim_holdout_scope_must_be_runs")
        preflight, problem = validate_runtime_preflight(
            config.preflight_path,
            code=source["code_fingerprint"],
            registry_fingerprint=registry["artifact_fingerprint"],
        )
        if problem is not None:
            reasons.append(problem)
        if reasons:
            raise LeagueError("claim execution is not eligible: " + ", ".join(reasons))
    else:
        preflight = None
        if input_audit["status"] not in {"passed", "not-applicable-toy-smoke"}:
            raise LeagueError(
                "smoke input authentication failed: "
                + ", ".join(cast(list[str], input_audit["failures"]))
            )
    bundle = build_policy_bundle(registry, root, mode=config.policy_mode)
    plan_payload: dict[str, object] = {
        "cycle_id": CYCLE_ID,
        "registry_fingerprint": registry["artifact_fingerprint"],
        "registry_path": str(config.registry_path),
        "source": source,
        "schedule": schedule_data(design),
        "registry_schedule_audit": schedule_audit,
        "input_audit_fingerprint": input_audit["artifact_fingerprint"],
        "policies": {
            policy_id: {
                "config": dict(spec.config),
                "rng_identity": spec.rng_identity,
                "declared": next(
                    policy["config"] for policy in registry["policies"] if policy["id"] == policy_id
                ),
            }
            for policy_id, spec in bundle.specs.items()
        },
        "checkpoint_identities": {
            key: dict(value) for key, value in bundle.checkpoint_identities.items()
        },
        "execution": {
            "claim": claim_requested,
            "evidence_class": "claim-locked-final" if claim_requested else "nonclaim-smoke",
            "policy_mode": config.policy_mode,
            "pair_count": design.pair_count,
            "output": str(config.output.resolve()),
            "holdout_roots": [str(path) for path in config.holdout_roots],
            "claim_cutoff_seconds": CLAIM_CUTOFF_SECONDS,
            "hard_limit_seconds": HARD_LIMIT_SECONDS,
            "allowed_attempts": ALLOWED_ATTEMPTS,
        },
        "runtime_preflight_fingerprint": None
        if preflight is None
        else preflight["artifact_fingerprint"],
        "statistical_procedure": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "order_statistic_indices": [BOOTSTRAP_LOWER_INDEX, BOOTSTRAP_UPPER_INDEX],
            "matchup_domain": MATCHUP_BOOTSTRAP_DOMAIN,
            "anchor_domain": ANCHOR_BOOTSTRAP_DOMAIN,
            "m_family_domain": M_FAMILY_BOOTSTRAP_DOMAIN,
            "promotion_lift_threshold": PROMOTION_LIFT_THRESHOLD,
            "random_lower_threshold": RANDOM_LOWER_THRESHOLD,
            "seat_floor": SEAT_FLOOR,
        },
        "deployment_mutation": "never-performed-by-runner",
    }
    plan = {"version": PLAN_VERSION, **plan_payload, "plan_fingerprint": fingerprint(plan_payload)}
    return plan, design, bundle, {"registry": registry, "input_audit": input_audit}


def run_league(config: LeagueRunConfig) -> dict[str, object]:
    """Run the complete league (claim or nonclaim smoke) and emit the immutable decision."""
    output = config.output
    plan, design, bundle, context = build_plan(config)
    output.mkdir(parents=True, exist_ok=True)
    claim = cast(Mapping[str, object], plan["execution"])["claim"] is True
    cutoff = float(CLAIM_CUTOFF_SECONDS) if claim else None
    registry = cast(dict[str, Any], context["registry"])
    write_immutable(output / "plan.json", plan)
    write_immutable(output / "input-registry.json", registry)
    write_immutable(output / "input-audit.json", cast(Mapping[str, object], context["input_audit"]))
    write_immutable(output / "source.json", cast(Mapping[str, object], plan["source"]))
    write_immutable(output / "schedule.json", cast(Mapping[str, object], plan["schedule"]))
    state_path = begin_execution(output, plan)
    with timed_phase(state_path, "holdout", cutoff_seconds=cutoff):
        holdout_path = output / "holdout.json"
        if holdout_path.exists():
            holdout = read_json(holdout_path)
            if not seal_matches(holdout) or holdout.get("design") != design.to_data():
                raise LeagueError("retained holdout inventory is malformed")
        else:
            holdout = scan_setup_holdout(design, roots=config.holdout_roots, current_output=output)
            write_immutable(holdout_path, holdout)
        if holdout["status"] != "passed":
            raise LeagueError("setup holdout overlap blocks execution")
    cells: dict[str, dict[str, object]] = {}
    for cell in design.cells():
        with timed_phase(state_path, "cells", cutoff_seconds=cutoff):
            cells[cell.key] = run_league_cell(output, plan, cell, bundle.specs)
    with timed_phase(state_path, "prefix_audit", cutoff_seconds=cutoff):
        prefix = audit_aligned_prefixes(design, lambda cell: load_cell_records(output, cell))
        write_immutable(output / "prefix-audit.json", prefix)
    with timed_phase(state_path, "statistics", cutoff_seconds=cutoff):
        tactical = aggregate_tactical(cells)
        write_immutable(output / "tactical.json", tactical)
        statistics = league_statistics(
            design,
            {key: cast(Mapping[str, Any], artifact["summary"]) for key, artifact in cells.items()},
            {key: cast(Mapping[str, Any], artifact["report"]) for key, artifact in cells.items()},
            {
                key: cast(Mapping[str, Mapping[str, int]], artifact["tactical"])
                for key, artifact in cells.items()
            },
        )
        write_immutable(output / "statistics.json", statistics)
    state = _load_execution(state_path)
    integrity = {
        "registry_schedule": cast(Mapping[str, object], plan["registry_schedule_audit"])["status"]
        == "passed"
        or not claim,
        "inputs": cast(Mapping[str, object], context["input_audit"])["status"]
        in ({"passed"} if claim else {"passed", "not-applicable-toy-smoke"}),
        "holdout": holdout["status"] == "passed",
        "within_claim_cutoff": cutoff is None or float(state["active_seconds"]) < cutoff,
        "all_cells_reconstructed": len(cells) == len(design.cells()),
    }
    decision = promotion_decision(
        statistics=statistics, tactical=tactical, prefix=prefix, integrity=integrity
    )
    write_immutable(output / "promotion-decision.json", decision)
    result = sealed(
        RESULT_VERSION,
        plan_fingerprint=plan["plan_fingerprint"],
        evidence_class=cast(Mapping[str, object], plan["execution"])["evidence_class"],
        cells={
            key: {
                "artifact_fingerprint": artifact["artifact_fingerprint"],
                "corpus_fingerprint": artifact["corpus_fingerprint"],
            }
            for key, artifact in sorted(cells.items())
        },
        holdout_fingerprint=holdout["artifact_fingerprint"],
        prefix_fingerprint=prefix["artifact_fingerprint"],
        tactical_fingerprint=tactical["artifact_fingerprint"],
        statistics_fingerprint=statistics["artifact_fingerprint"],
        decision_fingerprint=decision["artifact_fingerprint"],
        decision=decision["decision"],
        independent_validation="required",
    )
    write_immutable(output / "result.json", result)
    state = _load_execution(state_path)
    write_immutable(
        output / "runtime.json",
        sealed(
            "m7-independent-league-runtime-v1",
            plan_fingerprint=plan["plan_fingerprint"],
            active_seconds=state["active_seconds"],
            phase_seconds=state["phase_seconds"],
            attempts=state["attempts"],
            games=sum(cell.games for cell in design.cells()),
        ),
    )
    write_immutable(output / "checksums.json", checksum_manifest(output))
    state["completed"] = True
    _store_execution(state_path, state)
    return result
