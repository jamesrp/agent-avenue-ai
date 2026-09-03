"""Resumable generation-zero bootstrap, training, and validation orchestration."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agent_avenue.agents import EpsilonConfig, derive_seed
from agent_avenue.engine import GameConfig
from agent_avenue.engine.setup import normalize_config
from agent_avenue.storage import (
    code_fingerprint,
    inspect_source_identity,
    load_corpus,
    repository_root,
    rules_fingerprint,
)

from .corpus import run_resumable_corpus
from .safety_audit import audit_terminal_safety
from .self_play import heuristic_bootstrap_agent, schedule_heuristic_bootstrap

BOOTSTRAP_SCHEMA_VERSION: Final = 1
BOOTSTRAP_PLAN_VERSION: Final = "generation-zero-bootstrap-v1"


class BootstrapError(ValueError):
    """Raised when a bootstrap artifact or frozen plan is incompatible."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: Mapping[str, object], *, exclude: tuple[str, ...] = ()) -> str:
    return hashlib.sha256(
        _canonical_json({key: item for key, item in value.items() if key not in exclude})
    ).hexdigest()


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
        raise BootstrapError(f"unable to read bootstrap artifact {path}") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"bootstrap artifact {path} must be an object")
    return value


def _portable_locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    output: Path
    attempt_id: str
    root_seed: int
    game_count: int = 4_000
    max_epochs: int = 50
    batch_size: int = 1_024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    cpu_threads: int = 1
    terminal_safety: bool = False
    experiment_id: str = "generation-zero-bootstrap"
    game_config: GameConfig = field(default_factory=GameConfig)

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.experiment_id:
            raise BootstrapError("bootstrap attempt and experiment ids must be non-empty")
        if type(self.root_seed) is not int:
            raise BootstrapError("bootstrap root seed must be an integer")
        if self.game_count < 2:
            raise BootstrapError("bootstrap requires at least two games")


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    config: BootstrapConfig
    seeds: Mapping[str, int]
    source_identity: Mapping[str, object]
    claim_eligible: bool
    claim_ineligibility_reasons: tuple[str, ...]
    fingerprint: str

    def to_data(self) -> dict[str, object]:
        return {
            "schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "version": BOOTSTRAP_PLAN_VERSION,
            "plan_fingerprint": self.fingerprint,
            "experiment_id": self.config.experiment_id,
            "attempt_id": self.config.attempt_id,
            "root_seed": self.config.root_seed,
            "source": dict(self.source_identity),
            "claim_eligibility": {
                "eligible": self.claim_eligible,
                "reasons": list(self.claim_ineligibility_reasons),
            },
            "behavior_policy": {
                "base": "greedy-public-v1",
                "epsilon": EpsilonConfig(1, 5).to_data(),
                "policy_shield": "terminal-safety-v1" if self.config.terminal_safety else None,
            },
            "counts": {"corpus_games": self.config.game_count},
            "training": {
                "max_epochs": self.config.max_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "patience": self.config.patience,
                "cpu_threads": self.config.cpu_threads,
            },
            "game_config": normalize_config(self.config.game_config),
            "compatibility": {
                "rules_fingerprint": rules_fingerprint(),
                "code_fingerprint": code_fingerprint(),
            },
            "seeds": dict(self.seeds),
            "paths": {
                "corpus": "corpus",
                "dataset": "dataset.npz",
                "checkpoint": "checkpoint",
                "validation": "validation.json",
                "result": "result.json",
            },
        }


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    plan: BootstrapPlan
    checkpoint_fingerprint: str
    checkpoint_path: Path
    result_path: Path

    def to_data(self) -> dict[str, object]:
        return {
            "plan_fingerprint": self.plan.fingerprint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "checkpoint_path": str(self.checkpoint_path),
            "result_path": str(self.result_path),
        }


def _claim_reasons(config: BootstrapConfig) -> tuple[str, ...]:
    expected = {
        "game_count": 4_000,
        "max_epochs": 50,
        "batch_size": 1_024,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 8,
        "cpu_threads": 1,
    }
    reasons = [
        f"non_production_{name}"
        for name, expected_value in expected.items()
        if getattr(config, name) != expected_value
    ]
    if config.game_config != GameConfig():
        reasons.append("non_production_game_config")
    return tuple(reasons)


def resolve_bootstrap_plan(config: BootstrapConfig) -> BootstrapPlan:
    """Freeze generation-zero source, configuration, and random domains."""
    from agent_avenue.learning import TrainingConfig

    TrainingConfig(
        seed=0,
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_patience=config.patience,
        cpu_threads=config.cpu_threads,
    )
    source = inspect_source_identity()
    reasons = list(_claim_reasons(config))
    if not source.tracked_tree_clean:
        reasons.append("tracked_source_tree_is_dirty")
    domain = f"bootstrap:experiment:{config.experiment_id}:attempt:{config.attempt_id}"
    seeds = {
        name: derive_seed(config.root_seed, f"{domain}:{name}") & ((1 << 63) - 1)
        for name in ("corpus", "split", "training")
    }
    identity: dict[str, object] = {
        "version": BOOTSTRAP_PLAN_VERSION,
        "experiment_id": config.experiment_id,
        "attempt_id": config.attempt_id,
        "root_seed": config.root_seed,
        "source": source.to_data(),
        "claim_eligibility": {"eligible": not reasons, "reasons": reasons},
        "terminal_safety": config.terminal_safety,
        "game_count": config.game_count,
        "training": {
            "max_epochs": config.max_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "patience": config.patience,
            "cpu_threads": config.cpu_threads,
        },
        "game_config": normalize_config(config.game_config),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
        "seeds": seeds,
    }
    return BootstrapPlan(
        config,
        seeds,
        source.to_data(),
        not reasons,
        tuple(reasons),
        _fingerprint(identity),
    )


def _validate_checkpoint(
    manifest: Mapping[str, object],
    *,
    plan: BootstrapPlan,
    dataset_fingerprint: str,
    corpus_fingerprint: str,
    training_config: Mapping[str, object],
) -> None:
    sources = manifest.get("sources")
    training = manifest.get("training")
    lineage = manifest.get("lineage")
    metadata = manifest.get("metadata")
    if not all(isinstance(value, Mapping) for value in (sources, training, lineage, metadata)):
        raise BootstrapError("bootstrap checkpoint lacks required lineage metadata")
    assert isinstance(sources, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(lineage, Mapping)
    assert isinstance(metadata, Mapping)
    if (
        sources.get("dataset_fingerprint") != dataset_fingerprint
        or sources.get("corpus_fingerprints") != (corpus_fingerprint,)
        or training.get("config") != training_config
        or training.get("seeds") != {"root": plan.seeds["training"]}
        or lineage.get("parent_checkpoint") is not None
        or lineage.get("generation") is not None
        or metadata.get("bootstrap_plan_fingerprint") != plan.fingerprint
        or metadata.get("experiment_id") != plan.config.experiment_id
    ):
        raise BootstrapError("bootstrap checkpoint lineage does not match its plan")


def run_bootstrap(plan: BootstrapPlan) -> BootstrapResult:
    """Run or validate generation-zero corpus, dataset, training, and diagnostics."""
    from agent_avenue.learning import (
        TrainingConfig,
        load_checkpoint,
        load_dataset,
        materialize_dataset,
        save_checkpoint,
        save_dataset,
        train_model,
    )

    output = plan.config.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".bootstrap.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise BootstrapError("bootstrap is already running in this output directory") from exc
    try:
        if inspect_source_identity().to_data() != dict(plan.source_identity):
            raise BootstrapError("current Git revision or uv.lock no longer matches the plan")
        plan_path = output / "plan.json"
        plan_data = plan.to_data()
        if plan_path.exists():
            if _read_json(plan_path) != plan_data:
                raise BootstrapError("existing bootstrap plan does not match configuration")
        else:
            _atomic_json(plan_path, plan_data)

        behavior = heuristic_bootstrap_agent(terminal_safety=plan.config.terminal_safety)
        run_id = f"{plan.config.attempt_id}-bootstrap"
        corpus_path = output / "corpus"
        manifest = run_resumable_corpus(
            corpus_path,
            schedule_heuristic_bootstrap(
                run_id=run_id,
                game_count=plan.config.game_count,
                root_seed=plan.seeds["corpus"],
                attempt_id=plan.config.attempt_id,
                agent=behavior,
                game_config=plan.config.game_config,
            ),
            behavior_policy=(
                "terminal-safety-v1:epsilon-1/5-greedy-public-v1"
                if plan.config.terminal_safety
                else "epsilon-1/5-greedy-public-v1"
            ),
            root_seed=plan.seeds["corpus"],
            generation=0,
            configuration={
                "version": BOOTSTRAP_PLAN_VERSION,
                "plan_fingerprint": plan.fingerprint,
                "experiment_id": plan.config.experiment_id,
                "attempt_id": plan.config.attempt_id,
                "epsilon": EpsilonConfig(1, 5).to_data(),
                "policy_shield": ("terminal-safety-v1" if plan.config.terminal_safety else None),
            },
        )
        _, records = load_corpus(corpus_path)
        safety = audit_terminal_safety(
            records, source_corpus_fingerprint=manifest.corpus_fingerprint
        )

        dataset_path = output / "dataset.npz"
        dataset_manifest_path = dataset_path.with_suffix(".json")
        if dataset_path.exists() != dataset_manifest_path.exists():
            dataset_path.unlink(missing_ok=True)
            dataset_manifest_path.unlink(missing_ok=True)
        expected_dataset = materialize_dataset(
            records,
            split_seed=plan.seeds["split"],
            source_corpus_fingerprint=manifest.corpus_fingerprint,
        )
        if dataset_path.exists():
            dataset = load_dataset(dataset_path)
            if dataset.fingerprint != expected_dataset.fingerprint:
                raise BootstrapError("existing dataset does not exactly match corpus and split")
        else:
            dataset = expected_dataset
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
        checkpoint_path = output / "checkpoint"
        if checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path)
            _validate_checkpoint(
                checkpoint.manifest,
                plan=plan,
                dataset_fingerprint=dataset.fingerprint,
                corpus_fingerprint=manifest.corpus_fingerprint,
                training_config=training_config.normalized(),
            )
        else:
            training = train_model(dataset, config=training_config)
            saved = save_checkpoint(
                checkpoint_path,
                training.model,
                metrics=training.metrics_dict(),
                training_config=training_config.normalized(),
                training_seeds={"root": plan.seeds["training"]},
                dataset_fingerprint=dataset.fingerprint,
                source_corpus_fingerprints=(manifest.corpus_fingerprint,),
                metadata={
                    "bootstrap_plan_fingerprint": plan.fingerprint,
                    "experiment_id": plan.config.experiment_id,
                    "policy_shield": (
                        "terminal-safety-v1" if plan.config.terminal_safety else None
                    ),
                },
            )
            checkpoint = load_checkpoint(saved.path)

        validation: dict[str, object] = {
            "schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "plan_fingerprint": plan.fingerprint,
            "checks": {
                "source_and_lock_match_plan": True,
                "corpus_records_replay_verified": True,
                "dataset_lineage_and_split_verified": True,
                "checkpoint_integrity_compatibility_and_lineage": True,
                "terminal_safety_audit_verified": True,
            },
            "artifacts": {
                "corpus": manifest.corpus_fingerprint,
                "dataset": dataset.fingerprint,
                "checkpoint": checkpoint.checkpoint_fingerprint,
                "safety_audit": safety["artifact_fingerprint"],
            },
            "safety_diagnostics": {"training_corpus": safety},
            "artifact_fingerprint": "",
        }
        validation["artifact_fingerprint"] = _fingerprint(
            validation, exclude=("artifact_fingerprint",)
        )
        validation_path = output / "validation.json"
        if validation_path.exists():
            if _read_json(validation_path) != validation:
                raise BootstrapError("existing bootstrap validation artifact mismatch")
        else:
            _atomic_json(validation_path, validation)

        result: dict[str, object] = {
            "schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "plan_fingerprint": plan.fingerprint,
            "experiment_id": plan.config.experiment_id,
            "attempt_id": plan.config.attempt_id,
            "source": dict(plan.source_identity),
            "claim_eligibility": {
                "eligible": plan.claim_eligible,
                "reasons": list(plan.claim_ineligibility_reasons),
            },
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "checkpoint_path": _portable_locator(checkpoint_path),
            "artifacts": {
                "corpus": manifest.corpus_fingerprint,
                "dataset": dataset.fingerprint,
                "safety_audit": safety["artifact_fingerprint"],
                "validation": validation["artifact_fingerprint"],
            },
            "artifact_fingerprint": "",
        }
        result["artifact_fingerprint"] = _fingerprint(result, exclude=("artifact_fingerprint",))
        result_path = output / "result.json"
        if result_path.exists():
            if _read_json(result_path) != result:
                raise BootstrapError("existing bootstrap result artifact mismatch")
        else:
            _atomic_json(result_path, result)
        return BootstrapResult(
            plan, checkpoint.checkpoint_fingerprint, checkpoint_path, result_path
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
