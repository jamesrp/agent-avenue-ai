"""Held-out cross-policy record generation and chosen-action model evaluation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from agent_avenue.agents import (
    RNG_ALGORITHM,
    SEED_DERIVATION,
    DeterministicRandom,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    TerminalSafetyAgent,
    derive_seed,
)
from agent_avenue.engine import GameConfig
from agent_avenue.engine.setup import normalize_config
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    inspect_source_identity,
    load_corpus,
    repository_root,
    rules_fingerprint,
)

from .arena import ArenaConfig, run_resumable_arena, schedule_arena
from .game import AgentSpec
from .promotion import BOOTSTRAP_RESAMPLES, bootstrap_mean_interval
from .safety_audit import audit_terminal_safety

if TYPE_CHECKING:
    from agent_avenue.learning import LoadedCheckpoint

CROSSPLAY_SCHEMA_VERSION: Final = 1
CROSSPLAY_PLAN_VERSION: Final = "heldout-cross-policy-evaluation-v1"
CROSSPLAY_CORPUS_VERSION: Final = "heldout-cross-policy-records-v1"


class CrossplayError(ValueError):
    """Raised when a cross-policy diagnostic is malformed or non-reproducible."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: Mapping[str, object], *, exclude: tuple[str, ...] = ()) -> str:
    return hashlib.sha256(
        _canonical_json({key: item for key, item in value.items() if key not in exclude})
    ).hexdigest()


def _artifact_fingerprint(value: Mapping[str, object]) -> str:
    copied = json.loads(_canonical_json(value))
    if not isinstance(copied, dict):  # pragma: no cover - mapping guarantees this
        raise CrossplayError("crossplay report must be an object")
    copied.pop("artifact_fingerprint", None)
    copied.pop("runtime", None)
    return hashlib.sha256(_canonical_json(copied)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(_canonical_json(value) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossplayError(f"unable to read crossplay artifact {path}") from exc
    if not isinstance(value, dict):
        raise CrossplayError(f"crossplay artifact {path} must be an object")
    return value


def _portable_locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        return str(resolved)


def _policy_identity(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class CrossplayConfig:
    output: Path
    candidate_checkpoint: Path
    candidate_label: str
    generation: int
    prior_checkpoints: tuple[tuple[str, Path], ...]
    exclusion_corpora: tuple[Path, ...]
    root_seed: int
    pair_count: int = 200
    terminal_safety: bool = True
    game_config: GameConfig = field(default_factory=GameConfig)

    def __post_init__(self) -> None:
        if not 0 <= self.generation <= 4:
            raise CrossplayError("crossplay generation must be between zero and four")
        if self.candidate_label != f"q{self.generation}":
            raise CrossplayError("candidate label must match its q-generation")
        expected_labels = tuple(f"q{index}" for index in range(self.generation))
        if tuple(label for label, _ in self.prior_checkpoints) != expected_labels:
            raise CrossplayError("prior checkpoints must be ordered q0 through q(n-1)")
        if self.pair_count < 1:
            raise CrossplayError("crossplay pair count must be positive")
        if type(self.root_seed) is not int:
            raise CrossplayError("crossplay root seed must be an integer")


@dataclass(frozen=True, slots=True)
class CrossplayPolicy:
    label: str
    kind: str
    identity: str
    checkpoint_path: str | None
    checkpoint_fingerprint: str | None
    tensor_digest: str | None

    def to_data(self) -> dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "policy_identity": self.identity,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "tensor_digest": self.tensor_digest,
        }


@dataclass(frozen=True, slots=True)
class CrossplayPair:
    index: int
    left: CrossplayPolicy
    right: CrossplayPolicy
    master_seed: int

    @property
    def pair_id(self) -> str:
        return f"{self.index:02d}-{self.left.label}-vs-{self.right.label}"

    def to_data(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "left_policy_identity": self.left.identity,
            "right_policy_identity": self.right.identity,
            "master_seed": self.master_seed,
        }


@dataclass(frozen=True, slots=True)
class ExclusionCorpus:
    path: str
    corpus_fingerprint: str
    setup_seeds: tuple[int, ...]

    def to_data(self) -> dict[str, object]:
        return {
            "path": self.path,
            "corpus_fingerprint": self.corpus_fingerprint,
            "game_count": len(self.setup_seeds),
        }


@dataclass(frozen=True, slots=True)
class CrossplayPlan:
    config: CrossplayConfig
    candidate_fingerprint: str
    candidate_tensor_digest: str
    candidate_locator: str
    policies: tuple[CrossplayPolicy, ...]
    pairs: tuple[CrossplayPair, ...]
    exclusions: tuple[ExclusionCorpus, ...]
    source_identity: Mapping[str, object]
    fingerprint: str

    def to_data(self) -> dict[str, object]:
        return {
            "schema_version": CROSSPLAY_SCHEMA_VERSION,
            "version": CROSSPLAY_PLAN_VERSION,
            "plan_fingerprint": self.fingerprint,
            "diagnostic_only": True,
            "promotion_evidence": False,
            "candidate": {
                "label": self.config.candidate_label,
                "generation": self.config.generation,
                "checkpoint_path": self.candidate_locator,
                "checkpoint_fingerprint": self.candidate_fingerprint,
                "tensor_digest": self.candidate_tensor_digest,
            },
            "source": dict(self.source_identity),
            "root_seed": self.config.root_seed,
            "pair_count_per_matchup": self.config.pair_count,
            "matchup_count": len(self.pairs),
            "total_games": len(self.pairs) * self.config.pair_count * 2,
            "policy_shield": ("terminal-safety-v1" if self.config.terminal_safety else None),
            "policies": [policy.to_data() for policy in self.policies],
            "matchups": [pair.to_data() for pair in self.pairs],
            "excluded_training_corpora": [item.to_data() for item in self.exclusions],
            "metrics": {
                "target": "actor-relative terminal outcome for behavior-selected actions",
                "primary": "equal-matchup mean of paired-block equal-game log loss",
                "secondary": [
                    "paired-block Brier score",
                    "pooled and equal-game log loss",
                    "accuracy_at_0.5",
                    "calibration",
                    "phase metrics",
                ],
                "bootstrap": "20,000 deterministic paired-block resamples",
            },
            "game_config": normalize_config(self.config.game_config),
            "compatibility": {
                "rules_fingerprint": rules_fingerprint(),
                "code_fingerprint": code_fingerprint(),
            },
            "paths": {"records": "records", "report": "report.json"},
        }


def _learned_policy(
    label: str,
    path: Path,
    *,
    terminal_safety: bool,
) -> CrossplayPolicy:
    from agent_avenue.learning import inspect_checkpoint

    inspected = inspect_checkpoint(path)
    identity = _policy_identity(
        {
            "kind": "learned",
            "checkpoint_fingerprint": inspected.checkpoint_fingerprint,
            "tensor_digest": inspected.tensor_digest,
            "policy_shield": "terminal-safety-v1" if terminal_safety else None,
        }
    )
    return CrossplayPolicy(
        label,
        "learned",
        identity,
        _portable_locator(path),
        inspected.checkpoint_fingerprint,
        inspected.tensor_digest,
    )


def resolve_crossplay_plan(config: CrossplayConfig) -> CrossplayPlan:
    """Freeze candidate, prior policy pool, held-out schedules, and exclusions."""
    from agent_avenue.learning import inspect_checkpoint

    candidate = inspect_checkpoint(config.candidate_checkpoint)
    heuristic_config = GreedyHeuristicConfig().to_data()
    policies = [
        CrossplayPolicy(
            "heuristic",
            "heuristic",
            _policy_identity({"kind": "heuristic", "config": heuristic_config}),
            None,
            None,
            None,
        )
    ]
    policies.extend(
        _learned_policy(label, path, terminal_safety=config.terminal_safety)
        for label, path in config.prior_checkpoints
    )
    if len({policy.identity for policy in policies}) != len(policies):
        raise CrossplayError("crossplay policy identities must be unique")

    pair_values: list[CrossplayPair] = []
    index = 0
    for left_index, left in enumerate(policies):
        for right in policies[left_index:]:
            domain = (
                f"crossplay:{CROSSPLAY_PLAN_VERSION}:candidate:{candidate.checkpoint_fingerprint}:"
                f"left:{left.identity}:right:{right.identity}"
            )
            pair_values.append(
                CrossplayPair(
                    index,
                    left,
                    right,
                    derive_seed(config.root_seed, domain) & ((1 << 63) - 1),
                )
            )
            index += 1

    exclusions: list[ExclusionCorpus] = []
    excluded_setup: set[tuple[str, int]] = set()
    for path in config.exclusion_corpora:
        manifest, records = load_corpus(path)
        setup_seeds = tuple(record.replay.seed for record in records)
        for record in records:
            key = (
                _canonical_json(normalize_config(record.replay.config)).decode(),
                record.replay.seed,
            )
            excluded_setup.add(key)
        exclusions.append(
            ExclusionCorpus(_portable_locator(path), manifest.corpus_fingerprint, setup_seeds)
        )

    planned_setup: set[tuple[str, int]] = set()
    for pair in pair_values:
        placeholder_left = AgentSpec("left-placeholder", {}, GreedyHeuristicAgent)
        placeholder_right = AgentSpec("right-placeholder", {}, GreedyHeuristicAgent)
        arena = ArenaConfig(
            pair.pair_id,
            placeholder_left,
            placeholder_right,
            config.pair_count,
            pair.master_seed,
            config.game_config,
        )
        for spec in schedule_arena(arena):
            key = (_canonical_json(normalize_config(spec.config)).decode(), spec.setup_seed)
            if key in excluded_setup:
                raise CrossplayError("planned held-out setup overlaps an excluded training corpus")
            planned_setup.add(key)
    expected_unique = len(pair_values) * config.pair_count
    if len(planned_setup) != expected_unique:
        raise CrossplayError("crossplay matchup schedules contain duplicate setup blocks")

    source = inspect_source_identity()
    identity: dict[str, object] = {
        "version": CROSSPLAY_PLAN_VERSION,
        "candidate": {
            "label": config.candidate_label,
            "generation": config.generation,
            "checkpoint_fingerprint": candidate.checkpoint_fingerprint,
            "tensor_digest": candidate.tensor_digest,
        },
        "source": source.to_data(),
        "root_seed": config.root_seed,
        "pair_count": config.pair_count,
        "terminal_safety": config.terminal_safety,
        "policies": [policy.to_data() for policy in policies],
        "pairs": [pair.to_data() for pair in pair_values],
        "exclusions": [item.to_data() for item in exclusions],
        "game_config": normalize_config(config.game_config),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }
    return CrossplayPlan(
        config,
        candidate.checkpoint_fingerprint,
        candidate.tensor_digest,
        _portable_locator(config.candidate_checkpoint),
        tuple(policies),
        tuple(pair_values),
        tuple(exclusions),
        source.to_data(),
        _fingerprint(identity),
    )


def _agent_spec(
    policy: CrossplayPolicy,
    *,
    role: str,
    loaded: Mapping[str, LoadedCheckpoint],
    terminal_safety: bool,
) -> AgentSpec:
    agent_id = f"crossplay-{policy.label}-{role}"
    if policy.kind == "heuristic":
        config = GreedyHeuristicConfig()
        return AgentSpec(agent_id, config.to_data(), GreedyHeuristicAgent)
    from agent_avenue.agents.learned import LearnedValueAgent

    checkpoint = loaded.get(policy.label)
    if checkpoint is None:
        raise CrossplayError(f"missing loaded checkpoint for {policy.label}")
    base = LearnedValueAgent.from_checkpoint(checkpoint)
    if terminal_safety:
        agent = TerminalSafetyAgent(base)
        return AgentSpec(agent_id, agent.config_to_data(), lambda: agent)
    return AgentSpec(agent_id, base.config.to_data(), lambda: base)


def _paired_values(games: Sequence[object], attribute: str) -> tuple[float, ...]:
    grouped: dict[str, list[float]] = {}
    for game in games:
        pair_id = getattr(game, "pair_id", None)
        value = getattr(game, attribute, None)
        if not isinstance(pair_id, str) or not isinstance(value, float):
            raise CrossplayError("model game metrics are missing paired values")
        grouped.setdefault(pair_id, []).append(value)
    if any(len(values) != 2 for values in grouped.values()):
        raise CrossplayError("every held-out setup block must contain two seat-swapped games")
    return tuple(sum(grouped[pair_id]) / 2.0 for pair_id in sorted(grouped))


def _stratified_interval(
    groups: Sequence[Sequence[float]], *, master_seed: int, domain: str
) -> dict[str, object]:
    normalized = tuple(tuple(float(value) for value in group) for group in groups)
    if not normalized or any(not group for group in normalized):
        raise CrossplayError("stratified bootstrap requires non-empty matchup blocks")
    seed = derive_seed(master_seed, domain)
    rng = DeterministicRandom(seed, domain)
    means: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        matchup_means = []
        for group in normalized:
            matchup_means.append(
                sum(group[rng.randbelow(len(group))] for _ in range(len(group))) / len(group)
            )
        means.append(sum(matchup_means) / len(matchup_means))
    means.sort()
    return {
        "method": "deterministic-stratified-paired-block-percentile-bootstrap-v1",
        "statistic": "equal-matchup mean",
        "confidence_level": 0.95,
        "point_estimate": sum(sum(group) / len(group) for group in normalized) / len(normalized),
        "matchup_count": len(normalized),
        "blocks_per_matchup": [len(group) for group in normalized],
        "resample_count": BOOTSTRAP_RESAMPLES,
        "order_statistic_indices": {"lower": 499, "upper": 19_499},
        "interval": [means[499], means[19_499]],
        "bootstrap_seed": seed,
        "bootstrap_rng_domain": domain,
        "rng_algorithm": RNG_ALGORITHM,
        "seed_derivation": SEED_DERIVATION,
    }


def run_crossplay(plan: CrossplayPlan) -> Path:
    """Generate all prior-policy pair corpora and evaluate the frozen candidate on them."""
    from agent_avenue.learning import evaluate_records, load_checkpoint

    output = plan.config.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".crossplay.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise CrossplayError("crossplay evaluation is already running") from exc
    try:
        if inspect_source_identity().to_data() != dict(plan.source_identity):
            raise CrossplayError("current Git revision or uv.lock no longer matches the plan")
        plan_path = output / "plan.json"
        plan_data = plan.to_data()
        if plan_path.exists():
            if _read_json(plan_path) != plan_data:
                raise CrossplayError("existing crossplay plan does not match configuration")
        else:
            _atomic_json(plan_path, plan_data)

        candidate = load_checkpoint(plan.config.candidate_checkpoint)
        if (
            candidate.checkpoint_fingerprint != plan.candidate_fingerprint
            or candidate.manifest.get("tensor_digest") != plan.candidate_tensor_digest
        ):
            raise CrossplayError("candidate checkpoint no longer matches the plan")
        loaded: dict[str, LoadedCheckpoint] = {}
        by_label = dict(plan.config.prior_checkpoints)
        for policy in plan.policies:
            if policy.kind != "learned":
                continue
            checkpoint = load_checkpoint(by_label[policy.label])
            if (
                checkpoint.checkpoint_fingerprint != policy.checkpoint_fingerprint
                or checkpoint.manifest.get("tensor_digest") != policy.tensor_digest
            ):
                raise CrossplayError(f"prior checkpoint {policy.label} no longer matches the plan")
            loaded[policy.label] = checkpoint

        started = time.perf_counter()
        cell_reports: list[dict[str, object]] = []
        all_records: list[GameRecord] = []
        log_groups: list[tuple[float, ...]] = []
        brier_groups: list[tuple[float, ...]] = []
        records_root = output / "records"
        for pair in plan.pairs:
            agent_left = _agent_spec(
                pair.left,
                role="left",
                loaded=loaded,
                terminal_safety=plan.config.terminal_safety,
            )
            agent_right = _agent_spec(
                pair.right,
                role="right",
                loaded=loaded,
                terminal_safety=plan.config.terminal_safety,
            )
            arena = ArenaConfig(
                f"{plan.config.candidate_label}-{pair.pair_id}",
                agent_left,
                agent_right,
                plan.config.pair_count,
                pair.master_seed,
                plan.config.game_config,
            )
            records_path = records_root / pair.pair_id
            retained = run_resumable_arena(
                records_path,
                arena,
                generation=plan.config.generation,
                corpus_configuration={
                    "version": CROSSPLAY_CORPUS_VERSION,
                    "plan_fingerprint": plan.fingerprint,
                    "candidate_label": plan.config.candidate_label,
                    "left_policy_identity": pair.left.identity,
                    "right_policy_identity": pair.right.identity,
                    "diagnostic_only": True,
                },
            )
            manifest, records = load_corpus(records_path)
            if manifest.corpus_fingerprint != retained.records_manifest.corpus_fingerprint:
                raise CrossplayError("crossplay corpus changed between generation and loading")
            prediction = evaluate_records(candidate.model, records)
            log_values = _paired_values(prediction.games, "log_loss")
            brier_values = _paired_values(prediction.games, "brier_score")
            log_groups.append(log_values)
            brier_groups.append(brier_values)
            safety = audit_terminal_safety(
                records, source_corpus_fingerprint=manifest.corpus_fingerprint
            )
            all_records.extend(records)
            cell_reports.append(
                {
                    "pair_id": pair.pair_id,
                    "left": pair.left.label,
                    "right": pair.right.label,
                    "master_seed": pair.master_seed,
                    "records": {
                        "path": records_path.relative_to(output).as_posix(),
                        "corpus_fingerprint": manifest.corpus_fingerprint,
                        "declaration_fingerprint": manifest.declaration_fingerprint,
                        "game_count": manifest.record_count,
                        "decision_count": manifest.decision_count,
                    },
                    "model": prediction.to_data(),
                    "paired_block_log_loss": bootstrap_mean_interval(
                        log_values,
                        master_seed=pair.master_seed,
                        domain=f"crossplay:{plan.fingerprint}:{pair.pair_id}:log-loss:v1",
                    ).to_data(),
                    "paired_block_brier_score": bootstrap_mean_interval(
                        brier_values,
                        master_seed=pair.master_seed,
                        domain=f"crossplay:{plan.fingerprint}:{pair.pair_id}:brier:v1",
                    ).to_data(),
                    "safety_diagnostics": safety,
                }
            )

        overall = evaluate_records(candidate.model, all_records)
        report: dict[str, object] = {
            "schema_version": CROSSPLAY_SCHEMA_VERSION,
            "version": CROSSPLAY_PLAN_VERSION,
            "plan_fingerprint": plan.fingerprint,
            "diagnostic_only": True,
            "promotion_evidence": False,
            "candidate": {
                "label": plan.config.candidate_label,
                "generation": plan.config.generation,
                "checkpoint_fingerprint": plan.candidate_fingerprint,
                "tensor_digest": plan.candidate_tensor_digest,
            },
            "source": dict(plan.source_identity),
            "counts": {
                "matchups": len(plan.pairs),
                "paired_blocks_per_matchup": plan.config.pair_count,
                "games": len(all_records),
                "decisions": sum(record.decision_count for record in all_records),
            },
            "heldout_verification": {
                "excluded_training_corpora": [item.to_data() for item in plan.exclusions],
                "setup_overlap_count": 0,
                "all_records_replay_verified": True,
                "all_corpora_schedule_verified": True,
            },
            "overall_pooled": overall.to_data(),
            "macro_paired_block_log_loss": _stratified_interval(
                log_groups,
                master_seed=plan.config.root_seed,
                domain=f"crossplay:{plan.fingerprint}:macro-log-loss:v1",
            ),
            "macro_paired_block_brier_score": _stratified_interval(
                brier_groups,
                master_seed=plan.config.root_seed,
                domain=f"crossplay:{plan.fingerprint}:macro-brier:v1",
            ),
            "worst_matchup_log_loss": max(
                (
                    {
                        "pair_id": cell["pair_id"],
                        "point_estimate": cell["paired_block_log_loss"]["point_estimate"],  # type: ignore[index]
                    }
                    for cell in cell_reports
                ),
                key=lambda item: float(item["point_estimate"]),
            ),
            "matchups": cell_reports,
            "interpretation_limits": [
                (
                    "Only behavior-selected actions are labeled; this is not counterfactual "
                    "ranking data."
                ),
                "Targets include each generating pair's continuation policy.",
                (
                    "Prediction loss is a distribution-shift diagnostic, not playing-strength "
                    "evidence."
                ),
            ],
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "games_per_second": (
                    len(all_records) / (time.perf_counter() - started) if all_records else 0.0
                ),
            },
            "artifact_fingerprint": "",
        }
        report["artifact_fingerprint"] = _artifact_fingerprint(report)
        report_path = output / "report.json"
        if report_path.exists():
            existing = _read_json(report_path)
            if existing.get("artifact_fingerprint") != _artifact_fingerprint(existing):
                raise CrossplayError("existing crossplay report fingerprint mismatch")
            if _artifact_fingerprint(existing) != report["artifact_fingerprint"]:
                raise CrossplayError("existing crossplay report does not match regenerated data")
        else:
            _atomic_json(report_path, report)
        return report_path
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
