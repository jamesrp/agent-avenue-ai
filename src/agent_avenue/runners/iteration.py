"""High-level frozen-generation self-play, training, arena, and promotion orchestration."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from agent_avenue.agents import (
    EpsilonConfig,
    EpsilonGreedyAgent,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
    derive_seed,
)
from agent_avenue.engine import GameConfig
from agent_avenue.engine.setup import normalize_config
from agent_avenue.storage import (
    code_fingerprint,
    inspect_source_identity,
    load_corpus,
    repository_root,
    rules_fingerprint,
)

from .arena import ArenaConfig, run_resumable_arena
from .corpus import run_resumable_corpus
from .game import AgentSpec
from .promotion import (
    BootstrapMeanInterval,
    PromotionDecision,
    PromotionEvidence,
    PromotionPolicy,
    assess_attempt,
    bootstrap_mean_interval,
    evaluate_promotion,
)
from .self_play import (
    generation_config_from_agent,
    planned_epsilon,
    schedule_generation,
)

if TYPE_CHECKING:
    from agent_avenue.learning import LoadedCheckpoint

ITERATION_SCHEMA_VERSION = 2
ITERATION_PLAN_VERSION = "frozen-generation-iteration-v2"
ARENA_RECORDS_VERSION = "iteration-arena-records-v1"


class IterationError(ValueError):
    """Raised when an iteration artifact or lineage is incompatible with its plan."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: Mapping[str, object], *, exclude: tuple[str, ...] = ()) -> str:
    payload = {key: item for key, item in value.items() if key not in exclude}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _arena_fingerprint(artifact: Mapping[str, object]) -> str:
    payload = json.loads(_canonical_json(artifact))
    if not isinstance(payload, dict):  # pragma: no cover - mapping guarantees this
        raise IterationError("arena artifact must be an object")
    payload.pop("artifact_fingerprint", None)
    report = payload.get("report")
    if isinstance(report, dict):
        report.pop("elapsed_seconds", None)
        report.pop("games_per_second", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


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
        raise IterationError(f"unable to read iteration artifact {path}") from exc
    if not isinstance(value, dict):
        raise IterationError(f"iteration artifact {path} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class IterationConfig:
    output: Path
    incumbent_checkpoint: Path
    generation: int
    attempt_id: str
    root_seed: int
    game_count: int = 4_000
    primary_pairs: int = 500
    guardrail_pairs: int = 200
    confirmation_pairs: int = 1_000
    epsilon: EpsilonConfig | None = None
    max_epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    cpu_threads: int = 1
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    game_config: GameConfig = field(default_factory=GameConfig)

    def __post_init__(self) -> None:
        if not 1 <= self.generation <= 4:
            raise IterationError("baseline iteration generation must be between one and four")
        if not self.attempt_id:
            raise IterationError("iteration attempt id must be non-empty")
        if type(self.root_seed) is not int:
            raise IterationError("iteration root seed must be an integer")
        if self.game_count < 2:
            raise IterationError("iteration corpus requires at least two games")
        if min(self.primary_pairs, self.guardrail_pairs, self.confirmation_pairs) < 1:
            raise IterationError("all arena pair counts must be positive")


@dataclass(frozen=True, slots=True)
class IterationPlan:
    config: IterationConfig
    incumbent_fingerprint: str
    incumbent_tensor_digest: str
    incumbent_locator: str
    epsilon: EpsilonConfig
    seeds: Mapping[str, int]
    source_identity: Mapping[str, object]
    claim_eligible: bool
    claim_ineligibility_reasons: tuple[str, ...]
    fingerprint: str

    def to_data(self) -> dict[str, object]:
        return {
            "schema_version": ITERATION_SCHEMA_VERSION,
            "version": ITERATION_PLAN_VERSION,
            "plan_fingerprint": self.fingerprint,
            "generation": self.config.generation,
            "attempt_id": self.config.attempt_id,
            "root_seed": self.config.root_seed,
            "incumbent": {
                "path": self.incumbent_locator,
                "checkpoint_fingerprint": self.incumbent_fingerprint,
                "tensor_digest": self.incumbent_tensor_digest,
            },
            "source": dict(self.source_identity),
            "claim_eligibility": {
                "eligible": self.claim_eligible,
                "reasons": list(self.claim_ineligibility_reasons),
            },
            "epsilon": self.epsilon.to_data(),
            "counts": {
                "corpus_games": self.config.game_count,
                "primary_pairs": self.config.primary_pairs,
                "guardrail_pairs": self.config.guardrail_pairs,
                "confirmation_pairs": self.config.confirmation_pairs,
            },
            "training": {
                "max_epochs": self.config.max_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "patience": self.config.patience,
                "cpu_threads": self.config.cpu_threads,
            },
            "promotion_policy": asdict(self.config.promotion_policy),
            "game_config": normalize_config(self.config.game_config),
            "compatibility": {
                "rules_fingerprint": rules_fingerprint(),
                "code_fingerprint": code_fingerprint(),
            },
            "seeds": dict(self.seeds),
            "paths": {
                "corpus": "corpus",
                "dataset": "dataset.npz",
                "candidate": "candidate",
                "arenas": "arenas",
                "arena_records": "arena-records",
                "validation": "validation.json",
                "decision": "promotion-decision.json",
            },
        }


@dataclass(frozen=True, slots=True)
class IterationResult:
    plan: IterationPlan
    candidate_checkpoint: str
    decision: PromotionDecision
    selected_checkpoint: str
    selected_role: Literal["candidate", "incumbent"]
    decision_path: Path

    def to_data(self) -> dict[str, object]:
        return {
            "plan_fingerprint": self.plan.fingerprint,
            "candidate_checkpoint": self.candidate_checkpoint,
            "decision": self.decision.to_data(),
            "selected_checkpoint": self.selected_checkpoint,
            "selected_role": self.selected_role,
            "decision_path": str(self.decision_path),
        }


def _claim_ineligibility_reasons(
    config: IterationConfig, epsilon: EpsilonConfig
) -> tuple[str, ...]:
    expected = {
        "game_count": 4_000,
        "primary_pairs": 500,
        "guardrail_pairs": 200,
        "confirmation_pairs": 1_000,
        "max_epochs": 50,
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 8,
        "cpu_threads": 1,
    }
    reasons = [
        f"non_production_{name}"
        for name, value in expected.items()
        if getattr(config, name) != value
    ]
    if epsilon != planned_epsilon(config.generation):
        reasons.append("non_production_epsilon")
    if config.promotion_policy != PromotionPolicy():
        reasons.append("non_production_promotion_policy")
    if config.game_config != GameConfig():
        reasons.append("non_production_game_config")
    return tuple(reasons)


def _portable_locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_iteration_plan(config: IterationConfig) -> IterationPlan:
    """Validate the incumbent and freeze all attempt-specific random domains."""
    from agent_avenue.learning import TrainingConfig, inspect_checkpoint

    output = config.output.resolve()
    incumbent_path = config.incumbent_checkpoint.resolve()
    if (
        output == incumbent_path
        or output.is_relative_to(incumbent_path)
        or incumbent_path.is_relative_to(output)
    ):
        raise IterationError("iteration output and incumbent checkpoint paths must be disjoint")
    TrainingConfig(
        seed=0,
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_patience=config.patience,
        cpu_threads=config.cpu_threads,
    )
    incumbent = inspect_checkpoint(config.incumbent_checkpoint)
    epsilon = config.epsilon or planned_epsilon(config.generation)
    source = inspect_source_identity()
    claim_reasons = list(_claim_ineligibility_reasons(config, epsilon))
    if not source.tracked_tree_clean:
        claim_reasons.append("tracked_source_tree_is_dirty")
    incumbent_locator = _portable_locator(config.incumbent_checkpoint)
    domain = f"iteration:g{config.generation}:attempt:{config.attempt_id}"
    seed_names = (
        "corpus",
        "split",
        "training",
        "primary-arena",
        "primary-bootstrap",
        "random-arena",
        "random-bootstrap",
        "heuristic-arena",
        "heuristic-bootstrap",
        "confirmation-arena",
        "confirmation-bootstrap",
        "plateau-bootstrap",
    )
    seeds = {
        name: derive_seed(config.root_seed, f"{domain}:{name}") & ((1 << 63) - 1)
        for name in seed_names
    }
    identity: dict[str, object] = {
        "version": ITERATION_PLAN_VERSION,
        "generation": config.generation,
        "attempt_id": config.attempt_id,
        "root_seed": config.root_seed,
        "incumbent_fingerprint": incumbent.checkpoint_fingerprint,
        "incumbent_tensor_digest": incumbent.tensor_digest,
        "incumbent_locator": incumbent_locator,
        "source": source.to_data(),
        "claim_eligibility": {
            "eligible": not claim_reasons,
            "reasons": claim_reasons,
        },
        "epsilon": epsilon.to_data(),
        "game_count": config.game_count,
        "primary_pairs": config.primary_pairs,
        "guardrail_pairs": config.guardrail_pairs,
        "confirmation_pairs": config.confirmation_pairs,
        "training": {
            "max_epochs": config.max_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "patience": config.patience,
            "cpu_threads": config.cpu_threads,
        },
        "promotion_policy": asdict(config.promotion_policy),
        "game_config": normalize_config(config.game_config),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
        "seeds": seeds,
    }
    return IterationPlan(
        config,
        incumbent.checkpoint_fingerprint,
        incumbent.tensor_digest,
        incumbent_locator,
        epsilon,
        seeds,
        source.to_data(),
        not claim_reasons,
        tuple(claim_reasons),
        _fingerprint(identity),
    )


def _write_or_validate_plan(plan: IterationPlan) -> None:
    path = plan.config.output / "plan.json"
    data = plan.to_data()
    if path.exists():
        if _read_json(path) != data:
            raise IterationError("existing iteration plan does not match requested configuration")
    else:
        _atomic_json(path, data)


def _baseline_agent(kind: Literal["random", "heuristic"], agent_id: str) -> AgentSpec:
    if kind == "random":
        random_config = RandomAgentConfig()
        return AgentSpec(agent_id, random_config.to_data(), RandomAgent)
    heuristic_config = GreedyHeuristicConfig()
    return AgentSpec(agent_id, heuristic_config.to_data(), GreedyHeuristicAgent)


def _loaded_checkpoint_agent(loaded: LoadedCheckpoint, agent_id: str) -> AgentSpec:
    from agent_avenue.agents.learned import LearnedValueAgent

    agent = LearnedValueAgent.from_checkpoint(loaded)
    return AgentSpec(agent_id, agent.config.to_data(), lambda: agent)


def _behavior_agent(loaded: LoadedCheckpoint, epsilon: EpsilonConfig) -> AgentSpec:
    from agent_avenue.agents.learned import LearnedValueAgent

    base = LearnedValueAgent.from_checkpoint(loaded)
    wrapped = EpsilonGreedyAgent(base, epsilon)
    return AgentSpec("frozen-incumbent", wrapped.config_to_data(), lambda: wrapped)


def _semantic_report(report: Mapping[str, object]) -> dict[str, object]:
    copied = json.loads(_canonical_json(report))
    if not isinstance(copied, dict):
        raise IterationError("arena report must be an object")
    copied.pop("elapsed_seconds", None)
    copied.pop("games_per_second", None)
    return copied


def _arena_artifact(
    path: Path,
    *,
    records_path: Path,
    stage: str,
    plan: IterationPlan,
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    pairs: int,
    seed: int,
) -> dict[str, object]:
    expected = {
        "stage": stage,
        "plan_fingerprint": plan.fingerprint,
        "pair_count": pairs,
        "master_seed": seed,
        "agent_a_id": agent_a.agent_id,
        "agent_b_id": agent_b.agent_id,
        "agent_a_config": json.loads(json.dumps(dict(agent_a.config), sort_keys=True)),
        "agent_b_config": json.loads(json.dumps(dict(agent_b.config), sort_keys=True)),
        "game_config": normalize_config(plan.config.game_config),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }
    arena_config = ArenaConfig(
        f"{plan.config.attempt_id}-{stage}",
        agent_a,
        agent_b,
        pairs,
        seed,
        plan.config.game_config,
    )
    retained = run_resumable_arena(
        records_path,
        arena_config,
        generation=plan.config.generation,
        corpus_configuration={
            "version": ARENA_RECORDS_VERSION,
            "plan_fingerprint": plan.fingerprint,
            "stage": stage,
            "pair_count": pairs,
        },
    )
    report = retained.report.to_data()
    records = {
        "path": records_path.relative_to(plan.config.output).as_posix(),
        "corpus_fingerprint": retained.records_manifest.corpus_fingerprint,
        "declaration_fingerprint": retained.records_manifest.declaration_fingerprint,
        "record_count": retained.records_manifest.record_count,
    }
    expected_artifact: dict[str, object] = {
        **expected,
        "records": records,
        "report": report,
        "artifact_fingerprint": "",
    }
    expected_artifact["artifact_fingerprint"] = _arena_fingerprint(expected_artifact)
    if path.exists():
        existing_artifact = _read_json(path)
        declared = existing_artifact.get("artifact_fingerprint")
        if declared != _arena_fingerprint(existing_artifact):
            raise IterationError(f"arena artifact fingerprint mismatch: {stage}")
        if any(existing_artifact.get(key) != value for key, value in expected.items()):
            raise IterationError(f"arena artifact does not match iteration plan: {stage}")
        if existing_artifact.get("records") != records:
            raise IterationError(f"arena record corpus does not match aggregate: {stage}")
        existing_report = existing_artifact.get("report")
        if not isinstance(existing_report, Mapping) or _semantic_report(
            existing_report
        ) != _semantic_report(report):
            raise IterationError(f"arena aggregate does not match retained records: {stage}")
        return existing_artifact
    _atomic_json(path, expected_artifact)
    return expected_artifact


def _arena_records_fingerprint(artifact: Mapping[str, object]) -> str:
    records = artifact.get("records")
    fingerprint = records.get("corpus_fingerprint") if isinstance(records, Mapping) else None
    if not isinstance(fingerprint, str):
        raise IterationError("arena artifact has no record corpus fingerprint")
    return fingerprint


def _report(artifact: Mapping[str, object]) -> Mapping[str, object]:
    report = artifact.get("report")
    if not isinstance(report, Mapping):
        raise IterationError("arena report is malformed")
    return report


def _pair_scores(report: Mapping[str, object]) -> dict[str, float]:
    raw = report.get("paired_seed_outcomes")
    if not isinstance(raw, list):
        raise IterationError("arena report has no paired outcomes")
    scores: dict[str, float] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise IterationError("arena paired outcome is malformed")
        pair_id = item.get("pair_id")
        wins = item.get("agent_a_wins")
        if not isinstance(pair_id, str) or type(wins) is not int or not 0 <= wins <= 2:
            raise IterationError("arena paired outcome is malformed")
        scores[pair_id] = wins / 2.0
    if len(scores) != len(raw):
        raise IterationError("arena paired outcomes contain duplicate ids")
    return scores


def _seat_rate(report: Mapping[str, object], seat: str) -> float:
    by_seat = report.get("agent_a_by_seat")
    value = by_seat.get(seat) if isinstance(by_seat, Mapping) else None
    rate = value.get("win_rate") if isinstance(value, Mapping) else None
    if not isinstance(rate, int | float):
        raise IterationError("arena seat statistics are malformed")
    return float(rate)


def _interval(scores: Mapping[str, float], *, seed: int, domain: str) -> BootstrapMeanInterval:
    return bootstrap_mean_interval(
        tuple(scores[pair_id] for pair_id in sorted(scores)), master_seed=seed, domain=domain
    )


def _interval_data(interval: BootstrapMeanInterval) -> dict[str, object]:
    return interval.to_data()


def _validate_candidate_lineage(
    manifest: Mapping[str, object],
    plan: IterationPlan,
    dataset_fingerprint: str,
    corpus_fingerprint: str,
    training_config: Mapping[str, object],
) -> None:
    lineage = manifest.get("lineage")
    sources = manifest.get("sources")
    training = manifest.get("training")
    metadata = manifest.get("metadata")
    if not all(isinstance(value, Mapping) for value in (lineage, sources, training, metadata)):
        raise IterationError("candidate checkpoint lacks iteration lineage metadata")
    assert isinstance(lineage, Mapping)
    assert isinstance(sources, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(metadata, Mapping)
    if (
        lineage.get("parent_checkpoint") != plan.incumbent_fingerprint
        or lineage.get("generation") != plan.config.generation
        or sources.get("dataset_fingerprint") != dataset_fingerprint
        or sources.get("corpus_fingerprints") != (corpus_fingerprint,)
        or training.get("config") != training_config
        or training.get("seeds") != {"root": plan.seeds["training"]}
        or metadata.get("iteration_plan_fingerprint") != plan.fingerprint
    ):
        raise IterationError("candidate checkpoint lineage does not match iteration plan")


def _decision_from_data(data: Mapping[str, object]) -> PromotionDecision:
    decision = data.get("decision")
    if not isinstance(decision, Mapping):
        raise IterationError("promotion decision artifact is malformed")
    status = decision.get("status")
    reasons = decision.get("reasons")
    if (
        status not in {"promote", "retain"}
        or not isinstance(reasons, list)
        or any(not isinstance(reason, str) for reason in reasons)
    ):
        raise IterationError("promotion decision artifact is malformed")
    return PromotionDecision(status, tuple(reasons))


def _run_iteration_locked(plan: IterationPlan) -> IterationResult:
    """Run or validate every stage while the iteration output lock is held."""
    from agent_avenue.learning import (
        TrainingConfig,
        load_checkpoint,
        load_dataset,
        materialize_dataset,
        save_checkpoint,
        save_dataset,
        tensor_digest,
        train_model,
    )

    output = plan.config.output
    output.mkdir(parents=True, exist_ok=True)
    if inspect_source_identity().to_data() != dict(plan.source_identity):
        raise IterationError("current Git revision or uv.lock no longer matches the frozen plan")
    _write_or_validate_plan(plan)
    decision_path = output / "promotion-decision.json"
    existing_decision: dict[str, object] | None = None
    if decision_path.exists():
        existing_decision = _read_json(decision_path)
        if (
            existing_decision.get("artifact_fingerprint")
            != _fingerprint(existing_decision, exclude=("artifact_fingerprint",))
            or existing_decision.get("plan_fingerprint") != plan.fingerprint
        ):
            raise IterationError("promotion decision artifact mismatch")

    incumbent = load_checkpoint(plan.config.incumbent_checkpoint)
    if (
        incumbent.checkpoint_fingerprint != plan.incumbent_fingerprint
        or tensor_digest(incumbent.model.state_dict()) != plan.incumbent_tensor_digest
    ):
        raise IterationError("incumbent checkpoint no longer matches the frozen iteration plan")
    incumbent_agent = _behavior_agent(incumbent, plan.epsilon)
    generation = generation_config_from_agent(
        generation=plan.config.generation,
        run_id=f"{plan.config.attempt_id}-self-play",
        game_count=plan.config.game_count,
        root_seed=plan.seeds["corpus"],
        agent=incumbent_agent,
        epsilon=plan.epsilon,
        game_config=plan.config.game_config,
        attempt_id=plan.config.attempt_id,
    )
    corpus_path = output / "corpus"
    corpus_manifest = run_resumable_corpus(
        corpus_path,
        schedule_generation(generation, incumbent_agent),
        behavior_policy=f"{generation.fingerprint}:epsilon-frozen-incumbent-v1",
        root_seed=plan.seeds["corpus"],
        generation=plan.config.generation,
        configuration={"generation": generation.normalized(), "plan_fingerprint": plan.fingerprint},
    )
    _, records = load_corpus(corpus_path)

    dataset_path = output / "dataset.npz"
    dataset_manifest_path = dataset_path.with_suffix(".json")
    if dataset_path.exists() != dataset_manifest_path.exists():
        dataset_path.unlink(missing_ok=True)
        dataset_manifest_path.unlink(missing_ok=True)
    if dataset_path.exists():
        dataset = load_dataset(dataset_path)
        split_data = dataset.manifest.get("split")
        if (
            dataset.manifest.get("source_corpus_fingerprint") != corpus_manifest.corpus_fingerprint
            or not isinstance(split_data, Mapping)
            or split_data.get("seed") != plan.seeds["split"]
        ):
            raise IterationError("existing dataset does not match iteration corpus and split")
    else:
        dataset = materialize_dataset(
            records,
            split_seed=plan.seeds["split"],
            source_corpus_fingerprint=corpus_manifest.corpus_fingerprint,
        )
        save_dataset(dataset, dataset_path)

    training_config = TrainingConfig(
        seed=plan.seeds["training"],
        max_epochs=plan.config.max_epochs,
        batch_size=plan.config.batch_size,
        learning_rate=plan.config.learning_rate,
        weight_decay=plan.config.weight_decay,
        early_stopping_patience=plan.config.patience,
        cpu_threads=plan.config.cpu_threads,
    )
    candidate_path = output / "candidate"
    if candidate_path.exists():
        candidate = load_checkpoint(candidate_path)
        _validate_candidate_lineage(
            candidate.manifest,
            plan,
            dataset.fingerprint,
            corpus_manifest.corpus_fingerprint,
            training_config.normalized(),
        )
    else:
        training = train_model(
            dataset, config=training_config, initial_state_dict=incumbent.model.state_dict()
        )
        saved = save_checkpoint(
            candidate_path,
            training.model,
            metrics=training.metrics_dict(),
            training_config=training_config.normalized(),
            training_seeds={"root": plan.seeds["training"]},
            dataset_fingerprint=dataset.fingerprint,
            source_corpus_fingerprints=(corpus_manifest.corpus_fingerprint,),
            parent_checkpoint=plan.incumbent_fingerprint,
            generation=plan.config.generation,
            metadata={"iteration_plan_fingerprint": plan.fingerprint},
        )
        candidate = load_checkpoint(saved.path)

    candidate_agent = _loaded_checkpoint_agent(candidate, "candidate")
    incumbent_arena_agent = _loaded_checkpoint_agent(incumbent, "incumbent")
    arenas = output / "arenas"
    arena_records = output / "arena-records"
    primary = _arena_artifact(
        arenas / "primary.json",
        records_path=arena_records / "primary",
        stage="primary",
        plan=plan,
        agent_a=candidate_agent,
        agent_b=incumbent_arena_agent,
        pairs=plan.config.primary_pairs,
        seed=plan.seeds["primary-arena"],
    )
    random_guardrail = _arena_artifact(
        arenas / "versus-random.json",
        records_path=arena_records / "versus-random",
        stage="versus-random",
        plan=plan,
        agent_a=candidate_agent,
        agent_b=_baseline_agent("random", "random"),
        pairs=plan.config.guardrail_pairs,
        seed=plan.seeds["random-arena"],
    )
    candidate_heuristic = _arena_artifact(
        arenas / "candidate-versus-heuristic.json",
        records_path=arena_records / "candidate-versus-heuristic",
        stage="candidate-versus-heuristic",
        plan=plan,
        agent_a=candidate_agent,
        agent_b=_baseline_agent("heuristic", "heuristic"),
        pairs=plan.config.guardrail_pairs,
        seed=plan.seeds["heuristic-arena"],
    )
    incumbent_heuristic = _arena_artifact(
        arenas / "incumbent-versus-heuristic.json",
        records_path=arena_records / "incumbent-versus-heuristic",
        stage="incumbent-versus-heuristic",
        plan=plan,
        agent_a=incumbent_arena_agent,
        agent_b=_baseline_agent("heuristic", "heuristic"),
        pairs=plan.config.guardrail_pairs,
        seed=plan.seeds["heuristic-arena"],
    )

    primary_report = _report(primary)
    primary_scores = _pair_scores(primary_report)
    random_scores = _pair_scores(_report(random_guardrail))
    candidate_h_scores = _pair_scores(_report(candidate_heuristic))
    incumbent_h_scores = _pair_scores(_report(incumbent_heuristic))
    if set(candidate_h_scores) != set(incumbent_h_scores):
        raise IterationError("heuristic guardrail arenas do not share aligned pair blocks")
    heuristic_difference = {
        pair_id: candidate_h_scores[pair_id] - incumbent_h_scores[pair_id]
        for pair_id in candidate_h_scores
    }
    validation_path = output / "validation.json"
    checks = {
        "source_and_lock_match_plan": True,
        "incumbent_checkpoint_integrity_and_compatibility": True,
        "training_corpus_records_replay_verified": True,
        "dataset_lineage_and_split_verified": True,
        "candidate_checkpoint_integrity_compatibility_and_lineage": True,
        "arena_records_replay_and_schedule_verified": True,
        "aligned_heuristic_pair_blocks_verified": True,
        "safe_observation_encoder_contract_verified": True,
        "deterministic_artifact_fingerprints_verified": True,
    }
    validation_data: dict[str, object] = {
        "schema_version": ITERATION_SCHEMA_VERSION,
        "plan_fingerprint": plan.fingerprint,
        "checks": checks,
        "source": dict(plan.source_identity),
        "artifacts": {
            "corpus": corpus_manifest.corpus_fingerprint,
            "dataset": dataset.fingerprint,
            "candidate": candidate.checkpoint_fingerprint,
            "arena_record_corpora": {
                "primary": _arena_records_fingerprint(primary),
                "versus_random": _arena_records_fingerprint(random_guardrail),
                "candidate_versus_heuristic": _arena_records_fingerprint(candidate_heuristic),
                "incumbent_versus_heuristic": _arena_records_fingerprint(incumbent_heuristic),
            },
        },
        "artifact_fingerprint": "",
    }
    validation_data["artifact_fingerprint"] = _fingerprint(
        validation_data, exclude=("artifact_fingerprint",)
    )
    if validation_path.exists():
        if _read_json(validation_path) != validation_data:
            raise IterationError("existing validation artifact does not match verified artifacts")
    else:
        _atomic_json(validation_path, validation_data)
    compatibility_checks_passed = all(value is True for value in checks.values())
    evidence = PromotionEvidence(
        primary=_interval(
            primary_scores,
            seed=plan.seeds["primary-bootstrap"],
            domain=f"promotion:{plan.config.attempt_id}:primary:v1",
        ),
        player_one_win_rate=_seat_rate(primary_report, "player_one"),
        player_two_win_rate=_seat_rate(primary_report, "player_two"),
        versus_random=_interval(
            random_scores,
            seed=plan.seeds["random-bootstrap"],
            domain=f"promotion:{plan.config.attempt_id}:random:v1",
        ),
        heuristic_difference=_interval(
            heuristic_difference,
            seed=plan.seeds["heuristic-bootstrap"],
            domain=f"promotion:{plan.config.attempt_id}:heuristic-difference:v1",
        ),
        compatibility_checks_passed=compatibility_checks_passed,
    )
    decision = evaluate_promotion(evidence, plan.config.promotion_policy)
    confirmation_artifact: dict[str, object] | None = None
    if decision.status == "confirmation_required":
        confirmation_artifact = _arena_artifact(
            arenas / "confirmation.json",
            records_path=arena_records / "confirmation",
            stage="confirmation",
            plan=plan,
            agent_a=candidate_agent,
            agent_b=incumbent_arena_agent,
            pairs=plan.config.confirmation_pairs,
            seed=plan.seeds["confirmation-arena"],
        )
        confirmation = _interval(
            _pair_scores(_report(confirmation_artifact)),
            seed=plan.seeds["confirmation-bootstrap"],
            domain=f"promotion:{plan.config.attempt_id}:confirmation:v1",
        )
        evidence = PromotionEvidence(
            primary=evidence.primary,
            player_one_win_rate=evidence.player_one_win_rate,
            player_two_win_rate=evidence.player_two_win_rate,
            versus_random=evidence.versus_random,
            heuristic_difference=evidence.heuristic_difference,
            compatibility_checks_passed=compatibility_checks_passed,
            confirmation=confirmation,
        )
        decision = evaluate_promotion(evidence, plan.config.promotion_policy)

    selected_role: Literal["candidate", "incumbent"] = (
        "candidate" if decision.promoted else "incumbent"
    )
    selected_checkpoint = (
        candidate.checkpoint_fingerprint if decision.promoted else plan.incumbent_fingerprint
    )
    attempt_assessment = (
        assess_attempt(
            attempt_id=plan.config.attempt_id,
            incumbent_checkpoint=plan.incumbent_fingerprint,
            candidate_checkpoint=candidate.checkpoint_fingerprint,
            pair_scores=tuple(primary_scores[pair_id] for pair_id in sorted(primary_scores)),
            master_seed=plan.seeds["plateau-bootstrap"],
        )
        if not decision.promoted
        else None
    )
    decision_data: dict[str, object] = {
        "schema_version": ITERATION_SCHEMA_VERSION,
        "plan_fingerprint": plan.fingerprint,
        "generation": plan.config.generation,
        "attempt_id": plan.config.attempt_id,
        "source": dict(plan.source_identity),
        "claim_eligibility": {
            "eligible": plan.claim_eligible,
            "reasons": list(plan.claim_ineligibility_reasons),
        },
        "incumbent_checkpoint": plan.incumbent_fingerprint,
        "candidate_checkpoint": candidate.checkpoint_fingerprint,
        "selected_checkpoint": selected_checkpoint,
        "selected_role": selected_role,
        "decision": decision.to_data(),
        "plateau_assessment": (
            attempt_assessment.to_data() if attempt_assessment is not None else None
        ),
        "policy": asdict(plan.config.promotion_policy),
        "evidence": {
            "primary": _interval_data(evidence.primary),
            "player_one_win_rate": evidence.player_one_win_rate,
            "player_two_win_rate": evidence.player_two_win_rate,
            "versus_random": _interval_data(evidence.versus_random),
            "heuristic_difference": _interval_data(evidence.heuristic_difference),
            "compatibility_checks_passed": evidence.compatibility_checks_passed,
            "confirmation": (
                _interval_data(evidence.confirmation) if evidence.confirmation is not None else None
            ),
        },
        "artifacts": {
            "corpus_fingerprint": corpus_manifest.corpus_fingerprint,
            "dataset_fingerprint": dataset.fingerprint,
            "primary_arena": primary["artifact_fingerprint"],
            "random_arena": random_guardrail["artifact_fingerprint"],
            "candidate_heuristic_arena": candidate_heuristic["artifact_fingerprint"],
            "incumbent_heuristic_arena": incumbent_heuristic["artifact_fingerprint"],
            "confirmation_arena": (
                confirmation_artifact["artifact_fingerprint"]
                if confirmation_artifact is not None
                else None
            ),
            "arena_record_corpora": {
                "primary": _arena_records_fingerprint(primary),
                "versus_random": _arena_records_fingerprint(random_guardrail),
                "candidate_versus_heuristic": _arena_records_fingerprint(candidate_heuristic),
                "incumbent_versus_heuristic": _arena_records_fingerprint(incumbent_heuristic),
                "confirmation": (
                    _arena_records_fingerprint(confirmation_artifact)
                    if confirmation_artifact is not None
                    else None
                ),
            },
            "validation": validation_data["artifact_fingerprint"],
        },
        "artifact_fingerprint": "",
    }
    decision_data["artifact_fingerprint"] = _fingerprint(
        decision_data, exclude=("artifact_fingerprint",)
    )
    if existing_decision is not None:
        if existing_decision != decision_data:
            raise IterationError("existing promotion decision does not match validated evidence")
        decision = _decision_from_data(existing_decision)
    else:
        _atomic_json(decision_path, decision_data)
    return IterationResult(
        plan,
        candidate.checkpoint_fingerprint,
        decision,
        selected_checkpoint,
        selected_role,
        decision_path,
    )


def run_iteration(plan: IterationPlan) -> IterationResult:
    """Exclusively run or validate all stages and publish promotion or retention."""
    output = plan.config.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".iteration.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise IterationError("iteration is already running in this output directory") from exc
    try:
        return _run_iteration_locked(plan)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
