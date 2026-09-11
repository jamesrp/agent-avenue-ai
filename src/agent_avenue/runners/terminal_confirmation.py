"""Step-1 terminal-offense confirmation schedules, statistics, and replay audits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from agent_avenue.agents import (
    RNG_ALGORITHM,
    SEED_DERIVATION,
    DeterministicRandom,
    LearnedCandidateScores,
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
from agent_avenue.storage import GameRecord, game_record_fingerprint, load_corpus

from .arena import ArenaReport
from .public_win_oracle import independent_public_forced_win_oracle

CYCLE_ID: Final = "m7-terminal-offense-confirm-v1"
ROOT_SEED: Final = 2026091101
DEFAULT_PAIRS_PER_FAMILY: Final = 1_000
FAMILY_IDS: Final = ("seed-family-a", "seed-family-b")
CONTROL_ID: Final = "q0-terminal-safety-control-v1"
TREATMENT_ID: Final = "q0-terminal-offense-treatment-v1"
Q0_RNG_IDENTITY: Final = "q0-terminal-core-v1"
Q_IDS: Final = tuple(f"q{generation}-terminal-safety-v1" for generation in range(1, 5))
HISTORICAL_ID: Final = "historical-q0"
HEURISTIC_ID: Final = "greedy-public-v1"
RANDOM_ID: Final = "random"
ANCHOR_OPPONENTS: Final = (HISTORICAL_ID, HEURISTIC_ID, RANDOM_ID)
FIELD_OPPONENTS: Final = (*Q_IDS, *ANCHOR_OPPONENTS)
BOOTSTRAP_RESAMPLES: Final = 20_000
BOOTSTRAP_LOWER_INDEX: Final = 499
BOOTSTRAP_UPPER_INDEX: Final = 19_499
BOOTSTRAP_DOMAIN: Final = f"{CYCLE_ID}:stratified-common-block-bootstrap:v1"
PRACTICAL_LIFT_THRESHOLD: Final = 0.0025
CONFIRMATION_AUDIT_VERSION: Final = "terminal-offense-confirmation-replay-audit-v1"
DEFAULT_EXCLUDED_SETUP_ROOTS: Final = (
    Path("runs/m7-q0-strength-audit-v1"),
    Path("runs/terminal-safety-v1"),
    Path("runs/milestone6"),
    Path("runs/q0-corpus"),
    Path("runs/research-cycles/m7-heuristic-ranking-warmstart-v1"),
    Path("runs/research-cycles/m7-heuristic-ranking-warmstart-v1-repair1"),
)
CONFIRMATION_STATISTICS_VERSION: Final = "terminal-offense-confirmation-statistics-v1"

CandidateScorer = Callable[[PlayerObservation, tuple[Action, ...]], LearnedCandidateScores]


class TerminalConfirmationError(ValueError):
    """Raised when step-1 confirmation evidence violates its frozen structure."""


@dataclass(frozen=True, slots=True)
class ConfirmationCell:
    """One arena cell within one independently derived seed family."""

    family: str
    cell_id: str
    agent_a_id: str
    agent_b_id: str
    kind: str
    opponent_id: str | None

    @property
    def key(self) -> str:
        return f"{self.family}:{self.cell_id}"

    @property
    def run_id(self) -> str:
        return f"{CYCLE_ID}-{self.family}-{self.cell_id}"

    def to_data(self) -> dict[str, object]:
        return {
            "family": self.family,
            "cell_id": self.cell_id,
            "key": self.key,
            "run_id": self.run_id,
            "agent_a_id": self.agent_a_id,
            "agent_b_id": self.agent_b_id,
            "kind": self.kind,
            "opponent_id": self.opponent_id,
        }


def confirmation_cells() -> tuple[ConfirmationCell, ...]:
    """Return the frozen 15 cells per seed family in stable execution order."""
    cells: list[ConfirmationCell] = []
    for family in FAMILY_IDS:
        cells.append(
            ConfirmationCell(
                family,
                "direct-treatment-vs-control",
                TREATMENT_ID,
                CONTROL_ID,
                "direct",
                None,
            )
        )
        for opponent in FIELD_OPPONENTS:
            cells.append(
                ConfirmationCell(
                    family,
                    f"control-vs-{opponent}",
                    CONTROL_ID,
                    opponent,
                    "shared-opponent-control",
                    opponent,
                )
            )
            cells.append(
                ConfirmationCell(
                    family,
                    f"treatment-vs-{opponent}",
                    TREATMENT_ID,
                    opponent,
                    "shared-opponent-treatment",
                    opponent,
                )
            )
    return tuple(cells)


def family_seed_domain(family: str) -> str:
    if family not in FAMILY_IDS:
        raise TerminalConfirmationError(f"unknown seed family: {family}")
    return f"{CYCLE_ID}:seed-family:{family}"


def family_master_seed(family: str, *, root_seed: int = ROOT_SEED) -> int:
    """Derive one stable arena master seed for a declared seed family."""
    return derive_seed(root_seed, family_seed_domain(family)) & ((1 << 63) - 1)


def family_setup_seeds(
    family: str, pair_count: int, *, root_seed: int = ROOT_SEED
) -> tuple[int, ...]:
    """Recompute the common paired setup block used by every cell in a family."""
    if pair_count < 1:
        raise TerminalConfirmationError("pair count must be positive")
    master = family_master_seed(family, root_seed=root_seed)
    return tuple(
        derive_seed(master, f"arena:pair:{index}:setup") & ((1 << 64) - 1)
        for index in range(pair_count)
    )


def setup_seed_fingerprint(seeds: tuple[int, ...]) -> str:
    encoded = json.dumps(list(seeds), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_seed_families(pair_count: int, *, root_seed: int = ROOT_SEED) -> dict[str, object]:
    """Validate within-family uniqueness and between-family disjointness."""
    blocks = {
        family: family_setup_seeds(family, pair_count, root_seed=root_seed) for family in FAMILY_IDS
    }
    unique_within = all(len(set(seeds)) == len(seeds) for seeds in blocks.values())
    disjoint = set(blocks[FAMILY_IDS[0]]).isdisjoint(blocks[FAMILY_IDS[1]])
    return {
        "root_seed": root_seed,
        "family_seed_domains": {family: family_seed_domain(family) for family in FAMILY_IDS},
        "family_master_seeds": {
            family: family_master_seed(family, root_seed=root_seed) for family in FAMILY_IDS
        },
        "pair_count_per_family": pair_count,
        "setup_seed_fingerprints": {
            family: setup_seed_fingerprint(seeds) for family, seeds in blocks.items()
        },
        "unique_within_families": unique_within,
        "families_disjoint": disjoint,
        "status": "passed" if unique_within and disjoint else "failed",
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _setup_identity(game_config: Mapping[str, object], setup_seed: int) -> str:
    return hashlib.sha256(
        _canonical_json({"game_config": dict(game_config), "setup_seed": setup_seed})
    ).hexdigest()


def _identity_set_fingerprint(identities: Iterable[str]) -> str:
    return hashlib.sha256(_canonical_json(sorted(set(identities)))).hexdigest()


def scan_prior_setup_blocks(
    *,
    pair_count: int,
    excluded_roots: tuple[Path, ...] = DEFAULT_EXCLUDED_SETUP_ROOTS,
    current_output: Path | None = None,
    game_config: GameConfig | None = None,
) -> dict[str, object]:
    """Compare the confirmation setup block with completed corpora under excluded roots.

    Corpus loads deliberately skip code and replay execution because this is a fast holdout-identity
    scan; corpus/record fingerprints, JSON shape, ordering, counts, and current rules compatibility
    remain enforced by storage. A malformed completed corpus under a present root is fatal.
    """
    if pair_count < 1:
        raise TerminalConfirmationError("setup holdout scan requires a positive pair count")
    if not excluded_roots:
        raise TerminalConfirmationError("setup holdout scan requires excluded roots")
    actual_game_config = game_config or GameConfig()
    normalized_config = normalize_config(actual_game_config)
    current: dict[str, dict[str, object]] = {}
    for family in FAMILY_IDS:
        for pair_index, setup_seed in enumerate(family_setup_seeds(family, pair_count)):
            identity = _setup_identity(normalized_config, setup_seed)
            if identity in current:
                raise TerminalConfirmationError("confirmation setup families are not disjoint")
            current[identity] = {
                "family": family,
                "pair_index": pair_index,
                "setup_seed": setup_seed,
                "game_config": normalized_config,
            }

    output = current_output.resolve() if current_output is not None else None
    seen_manifests: set[Path] = set()
    prior_sources: dict[str, list[str]] = {}
    prior_material: dict[str, dict[str, object]] = {}
    root_rows: list[dict[str, object]] = []
    corpus_rows: list[dict[str, object]] = []
    scanned_record_count = 0
    for declared_root in excluded_roots:
        root = declared_root.resolve()
        if output is not None and root == output:
            root_rows.append(
                {
                    "declared_root": str(declared_root),
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
                    "declared_root": str(declared_root),
                    "resolved_root": str(root),
                    "status": "missing",
                    "corpus_count": 0,
                    "record_count": 0,
                    "ignored_manifest_count": 0,
                    "current_output_excluded": output is not None and output.is_relative_to(root),
                }
            )
            continue
        if not root.is_dir():
            raise TerminalConfirmationError(
                f"excluded setup root is present but is not a directory: {declared_root}"
            )
        corpus_count = 0
        record_count = 0
        ignored_count = 0
        for manifest_path in sorted(root.rglob("manifest.json")):
            resolved_manifest = manifest_path.resolve()
            if output is not None and resolved_manifest.is_relative_to(output):
                continue
            if resolved_manifest in seen_manifests:
                continue
            seen_manifests.add(resolved_manifest)
            records_path = manifest_path.parent / "games.jsonl.gz"
            try:
                manifest_data = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                if records_path.exists():
                    raise TerminalConfirmationError(
                        f"malformed completed corpus manifest: {manifest_path}"
                    ) from exc
                ignored_count += 1
                continue
            is_corpus = records_path.exists() or (
                isinstance(manifest_data, dict) and "records_file" in manifest_data
            )
            if not is_corpus:
                ignored_count += 1
                continue
            try:
                manifest, records = load_corpus(
                    manifest_path.parent,
                    verify_code=False,
                    verify_replays=False,
                )
            except (OSError, ValueError) as exc:
                raise TerminalConfirmationError(
                    f"unable to load completed excluded corpus: {manifest_path.parent}"
                ) from exc
            corpus_count += 1
            record_count += len(records)
            scanned_record_count += len(records)
            source = str(manifest_path.parent)
            corpus_rows.append(
                {
                    "path": source,
                    "corpus_fingerprint": manifest.corpus_fingerprint,
                    "record_count": len(records),
                }
            )
            for record in records:
                config_data = normalize_config(record.replay.config)
                identity = _setup_identity(config_data, record.replay.seed)
                prior_sources.setdefault(identity, []).append(source)
                prior_material.setdefault(
                    identity,
                    {
                        "setup_seed": record.replay.seed,
                        "game_config": config_data,
                    },
                )
        root_rows.append(
            {
                "declared_root": str(declared_root),
                "resolved_root": str(root),
                "status": "present",
                "corpus_count": corpus_count,
                "record_count": record_count,
                "ignored_manifest_count": ignored_count,
                "current_output_excluded": output is not None and output.is_relative_to(root),
            }
        )

    overlaps = sorted(set(current) & set(prior_sources))
    overlap_examples = [
        {
            "setup_identity": identity,
            "current": current[identity],
            "prior": prior_material[identity],
            "prior_corpora": sorted(set(prior_sources[identity])),
        }
        for identity in overlaps[:24]
    ]
    data: dict[str, object] = {
        "version": "terminal-offense-confirmation-setup-holdout-v1",
        "status": "passed" if not overlaps else "failed",
        "excluded_roots": root_rows,
        "current_output": None if current_output is None else str(current_output.resolve()),
        "current_setup_count": len(current),
        "current_setup_fingerprint": _identity_set_fingerprint(current),
        "prior_unique_setup_count": len(prior_sources),
        "prior_setup_fingerprint": _identity_set_fingerprint(prior_sources),
        "scanned_corpus_count": len(corpus_rows),
        "scanned_record_count": scanned_record_count,
        "scanned_corpora": sorted(corpus_rows, key=lambda row: str(row["path"])),
        "overlap_count": len(overlaps),
        "overlap_fingerprint": _identity_set_fingerprint(overlaps),
        "overlap_examples": overlap_examples,
        "artifact_fingerprint": "",
    }
    data["artifact_fingerprint"] = hashlib.sha256(
        _canonical_json(
            {key: value for key, value in data.items() if key != "artifact_fingerprint"}
        )
    ).hexdigest()
    return data


def validate_confirmation_corpora(
    corpus_directories: Mapping[str, Path],
    *,
    pair_count: int,
    rng_identities: Mapping[str, str],
    verify_replays: bool = True,
) -> dict[str, object]:
    """Recompute schedule, common-block, replay, and RNG-identity integrity from corpora."""
    expected_cells = confirmation_cells()
    if set(corpus_directories) != {cell.key for cell in expected_cells}:
        raise TerminalConfirmationError("integrity audit requires every confirmation corpus")
    required_agents = {
        agent_id for cell in expected_cells for agent_id in (cell.agent_a_id, cell.agent_b_id)
    }
    if set(rng_identities) != required_agents:
        raise TerminalConfirmationError("integrity audit RNG identities are incomplete")
    family_blocks: dict[str, tuple[int, ...]] = {}
    corpus_fingerprints: dict[str, str] = {}
    total_records = 0
    total_decisions = 0
    for cell in expected_cells:
        manifest, records = load_corpus(corpus_directories[cell.key], verify_replays=verify_replays)
        if len(records) != pair_count * 2:
            raise TerminalConfirmationError("confirmation corpus record count is incorrect")
        expected_setups = family_setup_seeds(cell.family, pair_count)
        actual_setups: list[int] = []
        master = family_master_seed(cell.family)
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
                raise TerminalConfirmationError("confirmation corpus schedule identity mismatch")
            expected_orders = (
                (cell.agent_a_id, cell.agent_b_id),
                (cell.agent_b_id, cell.agent_a_id),
            )
            for record, order in ((first, expected_orders[0]), (second, expected_orders[1])):
                if tuple(seat.agent_id for seat in record.seats) != order:
                    raise TerminalConfirmationError("confirmation corpus seat order mismatch")
                for seat in record.seats:
                    identity = rng_identities[seat.agent_id]
                    expected_seed = derive_seed(master, f"arena:pair:{pair_index}:agent:{identity}")
                    if (
                        seat.seed != expected_seed
                        or seat.rng_identity != identity
                        or seat.rng_domain != f"agent:{identity}"
                    ):
                        raise TerminalConfirmationError(
                            "confirmation corpus agent RNG identity mismatch"
                        )
            actual_setups.append(first.replay.seed)
        block = tuple(actual_setups)
        if cell.family in family_blocks and family_blocks[cell.family] != block:
            raise TerminalConfirmationError("cells do not share their family setup block")
        family_blocks[cell.family] = block
        corpus_fingerprints[cell.key] = manifest.corpus_fingerprint
        total_records += len(records)
        total_decisions += sum(record.decision_count for record in records)
    seed_validation = validate_seed_families(pair_count)
    passed = seed_validation["status"] == "passed" and set(family_blocks[FAMILY_IDS[0]]).isdisjoint(
        family_blocks[FAMILY_IDS[1]]
    )
    return {
        "version": "terminal-offense-confirmation-corpus-integrity-v1",
        "status": "passed" if passed else "failed",
        "cell_count": len(expected_cells),
        "record_count": total_records,
        "decision_count": total_decisions,
        "pair_count_per_family": pair_count,
        "family_setup_seed_fingerprints": {
            family: setup_seed_fingerprint(family_blocks[family]) for family in FAMILY_IDS
        },
        "family_blocks_disjoint": set(family_blocks[FAMILY_IDS[0]]).isdisjoint(
            family_blocks[FAMILY_IDS[1]]
        ),
        "rng_identities": dict(sorted(rng_identities.items())),
        "replay_verification": "full" if verify_replays else "preverified-upstream",
        "corpus_fingerprints": dict(sorted(corpus_fingerprints.items())),
    }


def _target_pair_wins(report: ArenaReport, target_id: str) -> tuple[int, ...]:
    if target_id not in {report.agent_a_id, report.agent_b_id}:
        raise TerminalConfirmationError("target policy is absent from arena report")
    values = tuple(row.agent_a_wins for row in report.paired_seed_outcomes)
    return values if report.agent_a_id == target_id else tuple(2 - value for value in values)


def _cell_by(family: str, *, kind: str, opponent: str | None = None) -> ConfirmationCell:
    matches = tuple(
        cell
        for cell in confirmation_cells()
        if cell.family == family and cell.kind == kind and cell.opponent_id == opponent
    )
    if len(matches) != 1:
        raise TerminalConfirmationError("confirmation cell lookup is not unique")
    return matches[0]


def _metric_blocks(
    reports: Mapping[str, ArenaReport],
) -> dict[str, dict[str, tuple[float, ...]]]:
    by_family: dict[str, dict[str, tuple[float, ...]]] = {}
    for family in FAMILY_IDS:
        metrics: dict[str, tuple[float, ...]] = {}
        opponent_differences: dict[str, tuple[float, ...]] = {}
        for opponent in FIELD_OPPONENTS:
            control = reports[
                _cell_by(family, kind="shared-opponent-control", opponent=opponent).key
            ]
            treatment = reports[
                _cell_by(family, kind="shared-opponent-treatment", opponent=opponent).key
            ]
            control_wins = _target_pair_wins(control, CONTROL_ID)
            treatment_wins = _target_pair_wins(treatment, TREATMENT_ID)
            if len(control_wins) != len(treatment_wins):
                raise TerminalConfirmationError("control/treatment block lengths differ")
            differences = tuple(
                (treatment_value - control_value) / 2
                for treatment_value, control_value in zip(treatment_wins, control_wins, strict=True)
            )
            opponent_differences[opponent] = differences
            metrics[f"opponent:{opponent}"] = differences
        block_count = len(next(iter(opponent_differences.values())))
        if any(len(values) != block_count for values in opponent_differences.values()):
            raise TerminalConfirmationError("opponent cells do not share one common block")
        metrics["anchor_macro"] = tuple(
            sum(opponent_differences[opponent][index] for opponent in ANCHOR_OPPONENTS)
            / len(ANCHOR_OPPONENTS)
            for index in range(block_count)
        )
        metrics["field_macro"] = tuple(
            sum(opponent_differences[opponent][index] for opponent in FIELD_OPPONENTS)
            / len(FIELD_OPPONENTS)
            for index in range(block_count)
        )
        direct = reports[_cell_by(family, kind="direct").key]
        metrics["direct_treatment_win_rate"] = tuple(
            wins / 2 for wins in _target_pair_wins(direct, TREATMENT_ID)
        )
        by_family[family] = metrics
    return by_family


def stratified_joint_bootstrap(
    metrics_by_family: Mapping[str, Mapping[str, tuple[float, ...]]],
    *,
    root_seed: int = ROOT_SEED,
) -> dict[str, object]:
    """Jointly resample common blocks within each fixed seed-family stratum."""
    if tuple(metrics_by_family) != FAMILY_IDS:
        raise TerminalConfirmationError("bootstrap requires both seed families in frozen order")
    metric_names = tuple(metrics_by_family[FAMILY_IDS[0]])
    if not metric_names or any(
        tuple(metrics_by_family[family]) != metric_names for family in FAMILY_IDS
    ):
        raise TerminalConfirmationError("bootstrap metric names are not aligned")
    rows_by_family: dict[str, tuple[tuple[float, ...], ...]] = {}
    total_blocks = 0
    for family in FAMILY_IDS:
        metrics = metrics_by_family[family]
        lengths = {len(metrics[name]) for name in metric_names}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise TerminalConfirmationError("bootstrap family metrics are not aligned")
        block_count = next(iter(lengths))
        rows_by_family[family] = tuple(
            tuple(metrics[name][index] for name in metric_names) for index in range(block_count)
        )
        total_blocks += block_count

    point_sums = [0.0] * len(metric_names)
    for rows in rows_by_family.values():
        for row in rows:
            for index, value in enumerate(row):
                point_sums[index] += value
    point_estimates = [value / total_blocks for value in point_sums]

    seed = derive_seed(root_seed, BOOTSTRAP_DOMAIN)
    rng = DeterministicRandom(seed, BOOTSTRAP_DOMAIN)
    samples: list[list[float]] = [[] for _ in metric_names]
    for _ in range(BOOTSTRAP_RESAMPLES):
        sums = [0.0] * len(metric_names)
        for family in FAMILY_IDS:
            rows = rows_by_family[family]
            for _ in range(len(rows)):
                sampled = rows[rng.randbelow(len(rows))]
                for metric_index, value in enumerate(sampled):
                    sums[metric_index] += value
        for metric_index, total in enumerate(sums):
            samples[metric_index].append(total / total_blocks)
    results: dict[str, object] = {}
    for index, name in enumerate(metric_names):
        ordered = sorted(samples[index])
        results[name] = {
            "point_estimate": point_estimates[index],
            "interval": [
                ordered[BOOTSTRAP_LOWER_INDEX],
                ordered[BOOTSTRAP_UPPER_INDEX],
            ],
        }
    return {
        "version": "stratified-joint-common-block-bootstrap-v1",
        "unit": "paired two-game setup block",
        "strata": list(FAMILY_IDS),
        "blocks_per_stratum": {family: len(rows_by_family[family]) for family in FAMILY_IDS},
        "resample_count": BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "order_statistic_indices": {
            "lower": BOOTSTRAP_LOWER_INDEX,
            "upper": BOOTSTRAP_UPPER_INDEX,
        },
        "bootstrap_seed": seed,
        "bootstrap_rng_domain": BOOTSTRAP_DOMAIN,
        "rng_algorithm": RNG_ALGORITHM,
        "seed_derivation": SEED_DERIVATION,
        "metrics": results,
    }


def confirmation_statistics(
    reports: Mapping[str, ArenaReport], *, root_seed: int = ROOT_SEED
) -> dict[str, object]:
    """Compute all frozen block-aware step-1 empirical statistics."""
    expected_keys = {cell.key for cell in confirmation_cells()}
    if set(reports) != expected_keys:
        raise TerminalConfirmationError("statistics require every frozen confirmation cell")
    bootstrap = stratified_joint_bootstrap(_metric_blocks(reports), root_seed=root_seed)
    raw_metrics = bootstrap["metrics"]
    if not isinstance(raw_metrics, dict):  # pragma: no cover - constructed above
        raise TypeError("bootstrap metrics must be an object")
    by_opponent = {opponent: raw_metrics[f"opponent:{opponent}"] for opponent in FIELD_OPPONENTS}
    anchor = raw_metrics["anchor_macro"]
    if not isinstance(anchor, dict):  # pragma: no cover - constructed above
        raise TypeError("anchor metric must be an object")
    interval = anchor["interval"]
    if not isinstance(interval, list):  # pragma: no cover - constructed above
        raise TypeError("anchor interval must be a list")
    practical = float(interval[0]) > PRACTICAL_LIFT_THRESHOLD
    return {
        "version": CONFIRMATION_STATISTICS_VERSION,
        "treatment_minus_control_by_opponent": by_opponent,
        "anchor_macro": anchor,
        "field_macro": raw_metrics["field_macro"],
        "direct_treatment_win_rate": raw_metrics["direct_treatment_win_rate"],
        "practical_lift_claim": {
            "criterion": "anchor-macro 95% lower bound > +0.25 percentage points",
            "threshold": PRACTICAL_LIFT_THRESHOLD,
            "passed": practical,
            "structural_adoption_blocking": False,
        },
        "bootstrap": {key: value for key, value in bootstrap.items() if key != "metrics"},
    }


def _empty_tactical_counts() -> Counter[str]:
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


def _empty_tie_counts() -> Counter[str]:
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


def _counts_data(template: Counter[str], counts: Mapping[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in template}


def _action_data(action: Action) -> dict[str, object]:
    if isinstance(action, PlayOfferAction):
        return {
            "type": "play_offer",
            "revision": action.revision,
            "actor": action.actor.value,
            "face_up": action.face_up.value,
            "face_down": action.face_down.value,
        }
    return {
        "type": "recruit",
        "revision": action.revision,
        "actor": action.actor.value,
        "slot": action.slot.value,
    }


def _resolved_chosen_state(
    state: GameState, action: Action, following_action: Action | None
) -> GameState:
    resolved = apply_action(state, action)
    if isinstance(action, PlayOfferAction):
        if not isinstance(following_action, RecruitAction):
            raise TerminalConfirmationError("play action is not followed by recruit resolution")
        resolved = apply_action(resolved, following_action)
    return resolved


def _learned_candidate_set(
    observation: PlayerObservation,
    *,
    treatment: bool,
) -> tuple[Action, ...]:
    actions = observation.legal_actions
    if treatment:
        offense = filter_immediate_win_actions(observation, actions)
        offense_observation = replace(observation, legal_actions=offense.allowed_actions)
        return filter_terminal_actions(offense_observation, offense.allowed_actions).allowed_actions
    return filter_terminal_actions(observation, actions).allowed_actions


def _update_nested(
    overall: Counter[str],
    phases: dict[str, Counter[str]],
    opponents: dict[str, Counter[str]],
    *,
    phase: str,
    opponent: str,
    values: Mapping[str, int],
    template: Callable[[], Counter[str]],
) -> None:
    rows = (
        overall,
        phases.setdefault(phase, template()),
        opponents.setdefault(opponent, template()),
    )
    for row in rows:
        row.update(values)


def _arm_audit_data(
    overall: Mapping[str, Counter[str]],
    phases: Mapping[str, Mapping[str, Counter[str]]],
    opponents: Mapping[str, Mapping[str, Counter[str]]],
    *,
    template: Callable[[], Counter[str]],
) -> dict[str, object]:
    empty = template()
    return {
        arm: {
            "counts": _counts_data(empty, overall[arm]),
            "by_phase": {
                phase: _counts_data(empty, counts)
                for phase, counts in sorted(phases.get(arm, {}).items())
            },
            "by_opponent": {
                opponent: _counts_data(empty, counts)
                for opponent, counts in sorted(opponents.get(arm, {}).items())
            },
        }
        for arm in ("control", "treatment")
    }


def _record_arm_index(record: GameRecord, arm_id: str) -> int:
    matches = tuple(index for index, seat in enumerate(record.seats) if seat.agent_id == arm_id)
    if len(matches) != 1:
        raise TerminalConfirmationError("matched record must contain its arm exactly once")
    return matches[0]


def _matched_records(
    records: tuple[GameRecord, ...], arm_id: str
) -> dict[tuple[str, int], GameRecord]:
    indexed: dict[tuple[str, int], GameRecord] = {}
    for record in records:
        if record.pair_id is None:
            raise TerminalConfirmationError("matched confirmation record is missing pair_id")
        arm_index = _record_arm_index(record, arm_id)
        key = (record.pair_id, arm_index)
        if key in indexed:
            raise TerminalConfirmationError("matched confirmation record key is duplicated")
        indexed[key] = record
    return indexed


def _prefix_comparison(control: GameRecord, treatment: GameRecord) -> dict[str, object]:
    control_index = _record_arm_index(control, CONTROL_ID)
    treatment_index = _record_arm_index(treatment, TREATMENT_ID)
    if control_index != treatment_index:
        raise TerminalConfirmationError("control and treatment arm seats are not aligned")
    opponent_index = 1 - control_index
    control_arm = control.seats[control_index]
    treatment_arm = treatment.seats[treatment_index]
    control_opponent = control.seats[opponent_index]
    treatment_opponent = treatment.seats[opponent_index]
    metadata_aligned = (
        control.replay.seed == treatment.replay.seed
        and control_arm.seed == treatment_arm.seed
        and control_arm.rng_identity == treatment_arm.rng_identity == Q0_RNG_IDENTITY
        and control_opponent.agent_id == treatment_opponent.agent_id
        and control_opponent.seed == treatment_opponent.seed
        and control_opponent.rng_domain == treatment_opponent.rng_domain
    )
    if not metadata_aligned:
        return {
            "metadata_aligned": False,
            "prefix_aligned": False,
            "guaranteed_win_endpoint_found": False,
            "endpoint_classification": None,
            "aligned_actions": 0,
            "failure_reason": "matched RNG/setup metadata differ",
            "failure_action_index": None,
            "control_converted_at_endpoint": False,
            "treatment_converted_at_endpoint": False,
        }

    control_state = new_game(control.replay.config, control.replay.seed)
    treatment_state = new_game(treatment.replay.config, treatment.replay.seed)
    control_actions = control.replay.actions
    treatment_actions = treatment.replay.actions
    index = 0
    while index < len(control_actions) and index < len(treatment_actions):
        if control_state != treatment_state:
            return {
                "metadata_aligned": True,
                "prefix_aligned": False,
                "guaranteed_win_endpoint_found": False,
                "endpoint_classification": None,
                "aligned_actions": index,
                "failure_reason": "authoritative states diverged before alignment endpoint",
                "failure_action_index": index,
                "control_converted_at_endpoint": False,
                "treatment_converted_at_endpoint": False,
            }
        actor = (
            control_state.active_player
            if control_state.phase is Phase.PLAY
            else control_state.active_player.other()
        )
        actor_agent = control.seats[player_index(actor)].agent_id
        control_action = control_actions[index]
        treatment_action = treatment_actions[index]
        if actor_agent == CONTROL_ID:
            observation = observe(control_state, actor)
            oracle = independent_public_forced_win_oracle(
                observation,
                observation.legal_actions,
                authoritative_state=control_state,
                exact_play_actions=(control_action,)
                if isinstance(control_action, PlayOfferAction)
                else (),
            )
            if oracle.forced_win_actions:
                control_converted = control_action in oracle.forced_win_actions
                treatment_converted = treatment_action in oracle.forced_win_actions
                classification = (
                    "both_converted"
                    if control_converted and treatment_converted
                    else "control_missed_treatment_converted"
                    if treatment_converted
                    else "treatment_failed_guaranteed_win"
                )
                return {
                    "metadata_aligned": True,
                    "prefix_aligned": True,
                    "guaranteed_win_endpoint_found": True,
                    "endpoint_classification": classification,
                    "aligned_actions": index,
                    "failure_reason": (
                        None
                        if treatment_converted
                        else "treatment did not choose a guaranteed action at alignment endpoint"
                    ),
                    "failure_action_index": None,
                    "endpoint_action_index": index,
                    "control_converted_at_endpoint": control_converted,
                    "treatment_converted_at_endpoint": treatment_converted,
                }
        if control_action != treatment_action:
            return {
                "metadata_aligned": True,
                "prefix_aligned": False,
                "guaranteed_win_endpoint_found": False,
                "endpoint_classification": None,
                "aligned_actions": index,
                "failure_reason": "actions diverged before guaranteed-win alignment endpoint",
                "failure_action_index": index,
                "control_converted_at_endpoint": False,
                "treatment_converted_at_endpoint": False,
            }
        control_state = apply_action(control_state, control_action)
        treatment_state = apply_action(treatment_state, treatment_action)
        index += 1
    complete = len(control_actions) == len(treatment_actions)
    return {
        "metadata_aligned": True,
        "prefix_aligned": complete,
        "guaranteed_win_endpoint_found": False,
        "endpoint_classification": None,
        "aligned_actions": index,
        "failure_reason": None if complete else "record lengths diverged before alignment endpoint",
        "failure_action_index": None if complete else index,
        "control_converted_at_endpoint": False,
        "treatment_converted_at_endpoint": False,
    }


def compare_matched_action_prefixes(
    control: GameRecord, treatment: GameRecord
) -> dict[str, object]:
    """Compare one same-setup/seat shared-opponent control/treatment game pair."""
    return _prefix_comparison(control, treatment)


def _prefix_row(
    family: str,
    opponent: str,
    control_records: tuple[GameRecord, ...],
    treatment_records: tuple[GameRecord, ...],
) -> tuple[Counter[str], list[dict[str, object]]]:
    controls = _matched_records(control_records, CONTROL_ID)
    treatments = _matched_records(treatment_records, TREATMENT_ID)
    if set(controls) != set(treatments):
        raise TerminalConfirmationError("matched prefix corpus keys differ")
    row = Counter(
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
    failures: list[dict[str, object]] = []
    for key in sorted(controls):
        comparison = _prefix_comparison(controls[key], treatments[key])
        aligned_actions = comparison["aligned_actions"]
        if type(aligned_actions) is not int:
            raise TerminalConfirmationError("prefix aligned-action count is malformed")
        endpoint_found = bool(comparison["guaranteed_win_endpoint_found"])
        metadata_aligned = bool(comparison["metadata_aligned"])
        prefix_aligned = bool(comparison["prefix_aligned"])
        treatment_converted = bool(comparison["treatment_converted_at_endpoint"])
        classification = comparison["endpoint_classification"]
        if classification is not None and not isinstance(classification, str):
            raise TerminalConfirmationError("prefix endpoint classification is malformed")
        values = {
            "matched_games": 1,
            "metadata_mismatches": int(not metadata_aligned),
            "pre_endpoint_prefix_mismatches": int(not prefix_aligned),
            "guaranteed_win_endpoints": int(endpoint_found),
            "control_missed_treatment_converted": int(
                classification == "control_missed_treatment_converted"
            ),
            "both_converted": int(classification == "both_converted"),
            "treatment_endpoint_failures": int(endpoint_found and not treatment_converted),
            "games_without_guaranteed_win_endpoint": int(not endpoint_found),
            "aligned_actions": aligned_actions,
        }
        row.update(values)
        if (
            not metadata_aligned
            or not prefix_aligned
            or (endpoint_found and not treatment_converted)
        ) and len(failures) < 24:
            failures.append(
                {
                    "family": family,
                    "opponent": opponent,
                    "pair_id": key[0],
                    "arm_seat_index": key[1],
                    "control_record_fingerprint": game_record_fingerprint(controls[key]),
                    "treatment_record_fingerprint": game_record_fingerprint(treatments[key]),
                    **comparison,
                }
            )
    return row, failures


def _prefix_data(
    rows: Mapping[str, Counter[str]], failures: list[dict[str, object]]
) -> dict[str, object]:
    overall: Counter[str] = Counter()
    for row in rows.values():
        overall.update(row)
    return {
        "definition": (
            "histories identical before the first control guaranteed-win opportunity; "
            "at that endpoint treatment must choose any guaranteed action"
        ),
        "counts": dict(overall),
        "by_family_opponent": {key: dict(value) for key, value in sorted(rows.items())},
        "failure_examples": failures[:24],
    }


def audit_matched_prefixes(
    corpus_directories: Mapping[str, Path], *, verify_replays: bool = True
) -> dict[str, object]:
    """Audit common-RNG control/treatment histories through the first missed control win."""
    rows: dict[str, Counter[str]] = {}
    failures: list[dict[str, object]] = []
    for family in FAMILY_IDS:
        for opponent in FIELD_OPPONENTS:
            control_cell = _cell_by(family, kind="shared-opponent-control", opponent=opponent)
            treatment_cell = _cell_by(family, kind="shared-opponent-treatment", opponent=opponent)
            _, control_records = load_corpus(
                corpus_directories[control_cell.key], verify_replays=verify_replays
            )
            _, treatment_records = load_corpus(
                corpus_directories[treatment_cell.key], verify_replays=verify_replays
            )
            row, row_failures = _prefix_row(family, opponent, control_records, treatment_records)
            rows[f"{family}:{opponent}"] = row
            failures.extend(row_failures[: max(0, 24 - len(failures))])
    return _prefix_data(rows, failures)


def audit_terminal_confirmation(
    corpus_directories: Mapping[str, Path],
    *,
    candidate_scorers: Mapping[str, CandidateScorer],
    source_label: str,
    verify_replays: bool = True,
) -> dict[str, object]:
    """Replay every cell and recompute tactical, oracle, safety, tie, and prefix evidence."""
    expected_keys = {cell.key for cell in confirmation_cells()}
    if set(corpus_directories) != expected_keys:
        raise TerminalConfirmationError("audit requires every frozen confirmation corpus")
    if set(candidate_scorers) != {CONTROL_ID, TREATMENT_ID}:
        raise TerminalConfirmationError("audit requires control and treatment candidate scorers")
    if not source_label:
        raise TerminalConfirmationError("audit requires a source label")

    classifier = Counter(
        {
            "decisions": 0,
            "agreement_decisions": 0,
            "disagreement_decisions": 0,
            "candidate_action_classifications": 0,
            "exact_play_transition_cross_checks": 0,
        }
    )
    tactical_overall = {
        "control": _empty_tactical_counts(),
        "treatment": _empty_tactical_counts(),
    }
    tactical_phases: dict[str, dict[str, Counter[str]]] = {
        "control": {},
        "treatment": {},
    }
    tactical_opponents: dict[str, dict[str, Counter[str]]] = {
        "control": {},
        "treatment": {},
    }
    tie_overall = {"control": _empty_tie_counts(), "treatment": _empty_tie_counts()}
    tie_phases: dict[str, dict[str, Counter[str]]] = {"control": {}, "treatment": {}}
    tie_opponents: dict[str, dict[str, Counter[str]]] = {"control": {}, "treatment": {}}
    disagreement_examples: list[dict[str, object]] = []
    invalid_tie_examples: list[dict[str, object]] = []
    prefix_rows: dict[str, Counter[str]] = {}
    prefix_failures: list[dict[str, object]] = []
    pending_control_records: dict[tuple[str, str], tuple[GameRecord, ...]] = {}
    record_count = 0
    decision_count = 0

    for cell in confirmation_cells():
        cell_key = cell.key
        _, records = load_corpus(corpus_directories[cell_key], verify_replays=verify_replays)
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
                oracle = independent_public_forced_win_oracle(
                    observation,
                    observation.legal_actions,
                    authoritative_state=state,
                    exact_play_actions=(action,) if isinstance(action, PlayOfferAction) else (),
                )
                production = filter_immediate_win_actions(observation, observation.legal_actions)
                agreed = oracle.forced_win_actions == production.forced_win_actions
                classifier.update(
                    {
                        "decisions": 1,
                        "agreement_decisions": int(agreed),
                        "disagreement_decisions": int(not agreed),
                        "candidate_action_classifications": len(observation.legal_actions),
                        "exact_play_transition_cross_checks": (
                            oracle.exact_transition_cross_checks
                        ),
                    }
                )
                if not agreed and len(disagreement_examples) < 24:
                    disagreement_examples.append(
                        {
                            "cell": cell_key,
                            "game_id": record.game_id,
                            "action_index": action_index,
                            "independent_forced_actions": [
                                _action_data(candidate) for candidate in oracle.forced_win_actions
                            ],
                            "production_forced_actions": [
                                _action_data(candidate)
                                for candidate in production.forced_win_actions
                            ],
                            "record_fingerprint": game_record_fingerprint(record),
                        }
                    )
                following = actions[action_index + 1] if action_index + 1 < len(actions) else None
                resolved = _resolved_chosen_state(state, action, following)
                chosen_immediate_win = (
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
                    opportunity = bool(oracle.forced_win_actions)
                    converted = action in oracle.forced_win_actions
                    tactical_values = {
                        "decisions": 1,
                        "guaranteed_win_opportunities": int(opportunity),
                        "guaranteed_win_conversions": int(opportunity and converted),
                        "guaranteed_win_misses": int(opportunity and not converted),
                        "false_guaranteed_wins": int(converted and not chosen_immediate_win),
                        "q0_avoidable_immediate_loss_violations": int(
                            action in safety.provable_loss_actions
                            and not safety.forced_loss_fallback
                        ),
                    }
                    _update_nested(
                        tactical_overall[arm],
                        tactical_phases[arm],
                        tactical_opponents[arm],
                        phase=state.phase.value,
                        opponent=opponent_id,
                        values=tactical_values,
                        template=_empty_tactical_counts,
                    )

                    candidates = _learned_candidate_set(observation, treatment=arm == "treatment")
                    scored_observation = replace(observation, legal_actions=candidates)
                    scores = candidate_scorers[agent_id](scored_observation, candidates)
                    expected_actions = tuple(sorted(candidates, key=semantic_action_key))
                    if scores.actions != expected_actions:
                        raise TerminalConfirmationError(
                            "candidate scorer did not expose semantic-order actions"
                        )
                    maxima = scores.maximum_actions
                    selected_in_maxima = action in maxima
                    forced_maxima = tuple(
                        candidate for candidate in maxima if candidate in oracle.forced_win_actions
                    )
                    nonforced_maxima = tuple(
                        candidate
                        for candidate in maxima
                        if candidate not in oracle.forced_win_actions
                    )
                    tied = scores.tie_count > 1
                    tie_values = {
                        "decisions": 1,
                        "exact_max_logit_ties": int(tied),
                        "exact_maximum_actions": scores.tie_count,
                        "ties_with_forced_maximum": int(bool(tied and forced_maxima)),
                        "ties_with_nonforced_maximum": int(bool(tied and nonforced_maxima)),
                        "ties_with_forced_and_nonforced_maxima": int(
                            bool(tied and forced_maxima and nonforced_maxima)
                        ),
                        "selected_in_exact_maxima": int(selected_in_maxima),
                        "selected_outside_exact_maxima": int(not selected_in_maxima),
                    }
                    _update_nested(
                        tie_overall[arm],
                        tie_phases[arm],
                        tie_opponents[arm],
                        phase=state.phase.value,
                        opponent=opponent_id,
                        values=tie_values,
                        template=_empty_tie_counts,
                    )
                    if not selected_in_maxima and len(invalid_tie_examples) < 24:
                        invalid_tie_examples.append(
                            {
                                "cell": cell_key,
                                "game_id": record.game_id,
                                "action_index": action_index,
                                "chosen_action": _action_data(action),
                                "maximum_actions": [
                                    _action_data(candidate) for candidate in maxima
                                ],
                                "maximum_indices": list(scores.maximum_indices),
                                "logits": list(scores.logits),
                                "record_fingerprint": game_record_fingerprint(record),
                            }
                        )
                decision_count += 1
                state = apply_action(state, action)
        if cell.kind == "shared-opponent-control":
            assert cell.opponent_id is not None
            pending_control_records[(cell.family, cell.opponent_id)] = records
        elif cell.kind == "shared-opponent-treatment":
            assert cell.opponent_id is not None
            key = (cell.family, cell.opponent_id)
            try:
                control_records = pending_control_records.pop(key)
            except KeyError as exc:
                raise TerminalConfirmationError(
                    "treatment prefix corpus appeared before its control corpus"
                ) from exc
            row, failures = _prefix_row(cell.family, cell.opponent_id, control_records, records)
            prefix_rows[f"{cell.family}:{cell.opponent_id}"] = row
            prefix_failures.extend(failures[: max(0, 24 - len(prefix_failures))])

    if pending_control_records:
        raise TerminalConfirmationError("unmatched control prefix corpora remain")
    prefix = _prefix_data(prefix_rows, prefix_failures)
    data: dict[str, object] = {
        "version": CONFIRMATION_AUDIT_VERSION,
        "source_label": source_label,
        "record_count": record_count,
        "decision_count": decision_count,
        "corpus_replay_verification": "full" if verify_replays else "preverified-upstream",
        "independent_production_oracle_agreement": {
            "counts": dict(classifier),
            "agreement_rate": (classifier["agreement_decisions"] / classifier["decisions"]),
            "disagreement_examples": disagreement_examples,
        },
        "guaranteed_win_and_safety": _arm_audit_data(
            tactical_overall,
            tactical_phases,
            tactical_opponents,
            template=_empty_tactical_counts,
        ),
        "exact_max_logit_ties": {
            "by_arm": _arm_audit_data(
                tie_overall,
                tie_phases,
                tie_opponents,
                template=_empty_tie_counts,
            ),
            "invalid_selection_examples": invalid_tie_examples,
        },
        "matched_action_prefixes": prefix,
        "artifact_fingerprint": "",
    }
    payload = {key: value for key, value in data.items() if key != "artifact_fingerprint"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    data["artifact_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return data
