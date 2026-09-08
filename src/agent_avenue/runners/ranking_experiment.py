"""Bounded fixed-corpus heuristic-ranking warm-start experiment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

if TYPE_CHECKING:
    from torch import Tensor

    from agent_avenue.learning import LoadedCheckpoint, RankingDataset

from agent_avenue.agents import (
    DeterministicRandom,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
    TerminalSafetyAgent,
    derive_seed,
)
from agent_avenue.storage import (
    code_fingerprint,
    inspect_source_identity,
    load_corpus,
    repository_root,
    rules_fingerprint,
)

from .arena import ArenaConfig, arena_report_from_records, run_resumable_arena
from .game import AgentSpec
from .safety_audit import audit_terminal_safety

PLAN_VERSION: Final[str] = "m7-heuristic-ranking-warmstart-plan-v1"
RESULT_VERSION: Final[str] = "m7-heuristic-ranking-warmstart-result-v1"
NESTED_BOOTSTRAP_VERSION: Final[str] = "nested-training-paired-block-bootstrap-v1"
NESTED_BOOTSTRAP_RESAMPLES: Final[int] = 20_000
EXPECTED_CORPUS_FINGERPRINT: Final[str] = (
    "e995da2633a6027bba12eff88b233b7160f6e12a61ea3d75665122f7ea9ceb7f"
)
EXPECTED_CORPUS_RECORDS_SHA256: Final[str] = (
    "f810514e667be9737978e4a0f9c377e217c0f43a6da92a34c951c66c5359d9b3"
)
EXPECTED_MC_DATASET_FINGERPRINT: Final[str] = (
    "1ea1bc5c64efbdb8f08a0e4de72ec446dbb93d005deaf9b8099239cd0f09c052"
)
EXPECTED_MC_DATASET_ARRAY_SHA256: Final[str] = (
    "59239ad109a7b1e6267b0f4c8ccd0d611c7e0ad51241d2dd869787d56a259655"
)
EXPECTED_PARENT_CHECKPOINT: Final[str] = (
    "b511ae162b794da6450fd3151762d98b4e5c8475c1af23e5b9289d0140a0a06d"
)
EXPECTED_PARENT_TENSOR: Final[str] = (
    "b1562237220e3a59b1e514327f3a5acc3d60767676befc69970f88ef98253767"
)
EXPECTED_HISTORICAL_Q1_CHECKPOINT: Final[str] = (
    "77682d3c3a79730c8a3841c8dab7c2bef1ac41c7bcbdc889b2f049186980a0f0"
)
EXPECTED_HISTORICAL_Q1_TENSOR: Final[str] = (
    "14d6f9b2c3fbacf0100f497df48a87ec27594e6ff5c058549a34dafe72b07f48"
)
HISTORICAL_Q1_MC_SEED: Final[int] = 3768947516520230646
FIXED_CREATED_AT: Final[str] = "2026-09-08T00:00:00+00:00"


class RankingExperimentError(ValueError):
    """Raised when the frozen ranking experiment cannot be reproduced safely."""


@dataclass(frozen=True, slots=True)
class RankingExperimentConfig:
    output: Path
    corpus: Path
    mc_dataset: Path
    parent_checkpoint: Path
    historical_q1_checkpoint: Path
    root_seed: int = 2026090801
    replicates: int = 3
    direct_pairs: int = 500
    parent_pairs: int = 500
    heuristic_pairs: int = 300
    random_pairs: int = 200
    max_epochs: int = 50
    batch_size: int = 1024
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if type(self.root_seed) is not int or not 0 <= self.root_seed < 2**63:
            raise RankingExperimentError("root seed must be an integer in [0, 2**63)")
        for name in (
            "replicates",
            "direct_pairs",
            "parent_pairs",
            "heuristic_pairs",
            "random_pairs",
            "max_epochs",
            "batch_size",
            "patience",
            "cpu_threads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise RankingExperimentError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class RankingExperimentPlan:
    config: RankingExperimentConfig
    fingerprint: str
    source: dict[str, object]
    input_identities: dict[str, object]
    replicate_seeds: tuple[dict[str, int], ...]
    arena_seeds: dict[str, int]
    claim_eligible: bool
    claim_ineligibility_reasons: tuple[str, ...]

    def to_data(self) -> dict[str, object]:
        return {
            "version": PLAN_VERSION,
            "plan_fingerprint": self.fingerprint,
            "source": self.source,
            "inputs": self.input_identities,
            "root_seed": self.config.root_seed,
            "replicate_seeds": list(self.replicate_seeds),
            "arena_seeds": self.arena_seeds,
            "recipes": {
                "control": "parent -> selected-action-mc-v1",
                "treatment": "parent -> heuristic-pairwise-ranking-v1 -> selected-action-mc-v1",
            },
            "training": {
                "replicates": self.config.replicates,
                "max_epochs": self.config.max_epochs,
                "batch_size": self.config.batch_size,
                "patience": self.config.patience,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "cpu_threads": self.config.cpu_threads,
            },
            "arenas": {
                "direct_pairs": self.config.direct_pairs,
                "parent_pairs": self.config.parent_pairs,
                "heuristic_pairs": self.config.heuristic_pairs,
                "random_pairs": self.config.random_pairs,
                "terminal_safety": True,
            },
            "advancement_rule": {
                "direct_replicate_wins_required": 2,
                "direct_nested_lower_strictly_above": 0.5,
                "parent_nested_lower_strictly_above": 0.5,
                "heuristic_difference_lower_strictly_above": -0.05,
                "random_nested_lower_strictly_above": 0.5,
                "minimum_candidate_seat_point_any_arena": 0.45,
            },
            "claim_eligibility": {
                "eligible": self.claim_eligible,
                "reasons": list(self.claim_ineligibility_reasons),
            },
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
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
        raise RankingExperimentError(f"unable to read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RankingExperimentError(f"JSON artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_ranking_experiment_plan(config: RankingExperimentConfig) -> RankingExperimentPlan:
    """Freeze inputs, source, seeds, recipes, arenas, and claim eligibility."""
    from agent_avenue.learning import inspect_checkpoint, load_dataset

    source = inspect_source_identity()
    parent = inspect_checkpoint(config.parent_checkpoint)
    historical = inspect_checkpoint(config.historical_q1_checkpoint)
    mc_dataset = load_dataset(config.mc_dataset)
    corpus_manifest, _ = load_corpus(config.corpus, verify_code=False)
    inputs = {
        "corpus": {
            "locator": _portable(config.corpus),
            "fingerprint": corpus_manifest.corpus_fingerprint,
            "record_count": corpus_manifest.record_count,
            "records_sha256": _sha256(config.corpus / "games.jsonl.gz"),
            "manifest_sha256": _sha256(config.corpus / "manifest.json"),
        },
        "mc_dataset": {
            "locator": _portable(config.mc_dataset),
            "fingerprint": mc_dataset.fingerprint,
            "counts": mc_dataset.manifest.get("counts"),
            "arrays_sha256": _sha256(config.mc_dataset),
            "manifest_sha256": _sha256(config.mc_dataset.with_suffix(".json")),
        },
        "parent_checkpoint": {
            "locator": _portable(config.parent_checkpoint),
            "fingerprint": parent.checkpoint_fingerprint,
            "tensor_digest": parent.tensor_digest,
        },
        "historical_q1_checkpoint": {
            "locator": _portable(config.historical_q1_checkpoint),
            "fingerprint": historical.checkpoint_fingerprint,
            "tensor_digest": historical.tensor_digest,
        },
    }
    replicate_seeds: list[dict[str, int]] = []
    for index in range(config.replicates):
        replicate = index + 1
        mc_seed = (
            HISTORICAL_Q1_MC_SEED
            if replicate == 1
            else derive_seed(config.root_seed, f"ranking-cycle:replicate:{replicate}:mc")
            & ((1 << 63) - 1)
        )
        replicate_seeds.append(
            {
                "replicate": replicate,
                "mc": mc_seed,
                "ranking": derive_seed(
                    config.root_seed, f"ranking-cycle:replicate:{replicate}:ranking"
                )
                & ((1 << 63) - 1),
            }
        )
    arena_seeds = {
        "direct": derive_seed(config.root_seed, "ranking-cycle:arena:direct") & ((1 << 63) - 1),
        "parent": derive_seed(config.root_seed, "ranking-cycle:arena:parent") & ((1 << 63) - 1),
        "heuristic": derive_seed(config.root_seed, "ranking-cycle:arena:heuristic")
        & ((1 << 63) - 1),
        "random": derive_seed(config.root_seed, "ranking-cycle:arena:random") & ((1 << 63) - 1),
        "nested-bootstrap": derive_seed(config.root_seed, "ranking-cycle:nested-bootstrap")
        & ((1 << 63) - 1),
    }
    expected = {
        "root_seed": 2026090801,
        "replicates": 3,
        "direct_pairs": 500,
        "parent_pairs": 500,
        "heuristic_pairs": 300,
        "random_pairs": 200,
        "max_epochs": 50,
        "batch_size": 1024,
        "patience": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "cpu_threads": 1,
    }
    reasons = [
        f"non_production_{name}"
        for name, expected_value in expected.items()
        if getattr(config, name) != expected_value
    ]
    if not source.tracked_tree_clean:
        reasons.append("tracked_source_tree_is_dirty")
    if corpus_manifest.corpus_fingerprint != EXPECTED_CORPUS_FINGERPRINT:
        reasons.append("fixed_corpus_fingerprint_mismatch")
    corpus_input = cast(dict[str, object], inputs["corpus"])
    mc_dataset_input = cast(dict[str, object], inputs["mc_dataset"])
    if corpus_input["records_sha256"] != EXPECTED_CORPUS_RECORDS_SHA256:
        reasons.append("fixed_corpus_bytes_mismatch")
    if mc_dataset.fingerprint != EXPECTED_MC_DATASET_FINGERPRINT:
        reasons.append("fixed_mc_dataset_fingerprint_mismatch")
    if mc_dataset_input["arrays_sha256"] != EXPECTED_MC_DATASET_ARRAY_SHA256:
        reasons.append("fixed_mc_dataset_bytes_mismatch")
    if parent.checkpoint_fingerprint != EXPECTED_PARENT_CHECKPOINT:
        reasons.append("fixed_parent_checkpoint_mismatch")
    if parent.tensor_digest != EXPECTED_PARENT_TENSOR:
        reasons.append("fixed_parent_tensor_mismatch")
    if historical.checkpoint_fingerprint != EXPECTED_HISTORICAL_Q1_CHECKPOINT:
        reasons.append("historical_q1_checkpoint_mismatch")
    if historical.tensor_digest != EXPECTED_HISTORICAL_Q1_TENSOR:
        reasons.append("historical_q1_tensor_mismatch")
    config_identity = asdict(config)
    for path_field in (
        "output",
        "corpus",
        "mc_dataset",
        "parent_checkpoint",
        "historical_q1_checkpoint",
    ):
        config_identity[path_field] = _portable(getattr(config, path_field))
    identity = {
        "version": PLAN_VERSION,
        "source": source.to_data(),
        "inputs": inputs,
        "root_seed": config.root_seed,
        "replicate_seeds": replicate_seeds,
        "arena_seeds": arena_seeds,
        "configuration": config_identity,
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
    }
    return RankingExperimentPlan(
        config,
        _fingerprint(identity),
        source.to_data(),
        inputs,
        tuple(replicate_seeds),
        arena_seeds,
        not reasons,
        tuple(reasons),
    )


def _ranking_datasets_equal(left: RankingDataset, right: RankingDataset) -> bool:
    import numpy as np

    fields = (
        "candidate_features",
        "position_candidate_offsets",
        "preferred_candidate_index",
        "dispreferred_candidate_index",
        "position_pair_offsets",
        "game_index",
        "decision_index",
        "phase",
    )
    return left.fingerprint == right.fingerprint and all(
        np.array_equal(getattr(left_split, field), getattr(right_split, field))
        for left_split, right_split in (
            (left.train, right.train),
            (left.validation, right.validation),
        )
        for field in fields
    )


def _write_or_validate_json(path: Path, value: dict[str, object], *, label: str) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise RankingExperimentError(f"existing {label} differs")
    else:
        _atomic_json(path, value)


def _validate_ranking_dataset(
    plan: RankingExperimentPlan,
    dataset: RankingDataset,
    *,
    mc_manifest: dict[str, object],
    array_path: Path,
) -> None:
    manifest = getattr(dataset, "manifest", None)
    if not isinstance(manifest, dict):
        raise RankingExperimentError("ranking dataset has no mutable manifest")
    expected_teacher = GreedyHeuristicConfig().to_data()
    checks = {
        "source corpus": (
            manifest.get("source_corpus_fingerprint"),
            cast(dict[str, object], plan.input_identities["corpus"])["fingerprint"],
        ),
        "source MC dataset": (
            manifest.get("source_mc_dataset_fingerprint"),
            cast(dict[str, object], plan.input_identities["mc_dataset"])["fingerprint"],
        ),
        "MC split": (manifest.get("split"), mc_manifest.get("split")),
        "teacher": (manifest.get("teacher"), expected_teacher),
        "information boundary": (
            manifest.get("information_boundary"),
            "PlayerObservation + legal semantic Action only",
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise RankingExperimentError(f"ranking dataset {label} mismatch")
    audit_path = plan.config.output / "input-audit.json"
    if audit_path.exists():
        audit = _read_json(audit_path)
        declared = audit.get("artifact_fingerprint")
        payload = {key: value for key, value in audit.items() if key != "artifact_fingerprint"}
        if (
            declared != _fingerprint(payload)
            or audit.get("ranking_dataset_fingerprint") != dataset.fingerprint
            or audit.get("ranking_dataset_arrays_sha256") != _sha256(array_path)
        ):
            raise RankingExperimentError("ranking dataset changed after input audit")


def _validate_final_checkpoint(
    loaded: LoadedCheckpoint,
    *,
    plan: RankingExperimentPlan,
    recipe: str,
    replicate: int,
    seeds: dict[str, int],
    mc_dataset_fingerprint: str,
    corpus_fingerprint: str,
    ranking_dataset_fingerprint: str | None,
) -> None:
    from agent_avenue.learning import TrainingConfig

    manifest = _jsonable(loaded.manifest)
    if not isinstance(manifest, dict):
        raise RankingExperimentError("checkpoint manifest is malformed")
    lineage = manifest.get("lineage")
    sources = manifest.get("sources")
    training = manifest.get("training")
    metadata = manifest.get("metadata")
    if not all(isinstance(value, dict) for value in (lineage, sources, training, metadata)):
        raise RankingExperimentError("checkpoint lineage metadata is malformed")
    assert isinstance(lineage, dict)
    assert isinstance(sources, dict)
    assert isinstance(training, dict)
    assert isinstance(metadata, dict)
    expected_seeds = {"mc": seeds["mc"]}
    if recipe == "heuristic-ranking-warmstart":
        expected_seeds["ranking"] = seeds["ranking"]
    expected_config = TrainingConfig(
        seed=seeds["mc"],
        max_epochs=plan.config.max_epochs,
        batch_size=plan.config.batch_size,
        learning_rate=plan.config.learning_rate,
        weight_decay=plan.config.weight_decay,
        early_stopping_patience=plan.config.patience,
        cpu_threads=plan.config.cpu_threads,
    ).normalized()
    checks = {
        "parent": (lineage.get("parent_checkpoint"), EXPECTED_PARENT_CHECKPOINT),
        "generation": (lineage.get("generation"), 1),
        "dataset": (sources.get("dataset_fingerprint"), mc_dataset_fingerprint),
        "corpus": (sources.get("corpus_fingerprints"), [corpus_fingerprint]),
        "training config": (training.get("config"), expected_config),
        "training seeds": (training.get("seeds"), expected_seeds),
        "plan": (metadata.get("plan_fingerprint"), plan.fingerprint),
        "recipe": (metadata.get("recipe"), recipe),
        "replicate": (metadata.get("replicate"), replicate),
    }
    if recipe == "heuristic-ranking-warmstart":
        checks["ranking dataset"] = (
            metadata.get("ranking_dataset_fingerprint"),
            ranking_dataset_fingerprint,
        )
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise RankingExperimentError(f"checkpoint {label} mismatch")


def _write_or_validate_plan(plan: RankingExperimentPlan) -> None:
    path = plan.config.output / "plan.json"
    expected = plan.to_data()
    if path.exists():
        if _read_json(path) != expected:
            raise RankingExperimentError("existing ranking experiment plan differs")
    else:
        _atomic_json(path, expected)


def _save_ranking_initializer(
    path: Path,
    state: dict[str, Tensor],
    *,
    plan: RankingExperimentPlan,
    replicate: int,
    dataset_fingerprint: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    import torch

    from agent_avenue.learning import tensor_digest

    digest = tensor_digest(state)
    manifest = {
        "version": "ranking-training-initializer-v1",
        "semantics": "ordinal-heuristic-ranking-initializer-not-inference-value",
        "plan_fingerprint": plan.fingerprint,
        "replicate": replicate,
        "parent_checkpoint": plan.input_identities["parent_checkpoint"],
        "ranking_dataset_fingerprint": dataset_fingerprint,
        "tensor_digest": digest,
        "metrics": metrics,
    }
    if path.exists():
        existing = _read_json(path / "manifest.json")
        expected_without_file = {
            key: value for key, value in existing.items() if key != "weights_sha256"
        }
        if expected_without_file != manifest:
            raise RankingExperimentError("existing ranking initializer differs")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.tmp-", dir=path.parent))
    try:
        torch.save(state, temporary / "weights.pt")
        manifest["weights_sha256"] = hashlib.sha256(
            (temporary / "weights.pt").read_bytes()
        ).hexdigest()
        (temporary / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(temporary, path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _load_ranking_initializer(
    path: Path,
    *,
    plan: RankingExperimentPlan,
    replicate: int,
    dataset_fingerprint: str,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    import torch
    from torch import Tensor

    from agent_avenue.learning import tensor_digest
    from agent_avenue.learning.model import validate_state_dict

    manifest = _read_json(path / "manifest.json")
    expected_identity = {
        "version": "ranking-training-initializer-v1",
        "semantics": "ordinal-heuristic-ranking-initializer-not-inference-value",
        "plan_fingerprint": plan.fingerprint,
        "replicate": replicate,
        "parent_checkpoint": plan.input_identities["parent_checkpoint"],
        "ranking_dataset_fingerprint": dataset_fingerprint,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise RankingExperimentError(f"ranking initializer {key} mismatch")
    weights_path = path / "weights.pt"
    expected_file_digest = manifest.get("weights_sha256")
    if expected_file_digest != hashlib.sha256(weights_path.read_bytes()).hexdigest():
        raise RankingExperimentError("ranking initializer file digest mismatch")
    try:
        raw = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise RankingExperimentError("ranking initializer weights are malformed")
        validate_state_dict(raw)
        state = cast(dict[str, Tensor], raw)
    except Exception as exc:
        raise RankingExperimentError("unable to load ranking initializer") from exc
    if tensor_digest(state) != manifest.get("tensor_digest"):
        raise RankingExperimentError("ranking initializer tensor digest mismatch")
    return manifest, state


def _checkpoint_agent(path: Path, agent_id: str) -> AgentSpec:
    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import load_checkpoint

    loaded = load_checkpoint(path)
    learned = LearnedValueAgent.from_checkpoint(loaded)
    wrapped = TerminalSafetyAgent(learned)
    return AgentSpec(agent_id, wrapped.config_to_data(), lambda: wrapped)


def _baseline_agent(kind: Literal["heuristic", "random"], agent_id: str) -> AgentSpec:
    if kind == "heuristic":
        heuristic = GreedyHeuristicAgent(GreedyHeuristicConfig())
        wrapped = TerminalSafetyAgent(heuristic)
        return AgentSpec(agent_id, wrapped.config_to_data(), lambda: wrapped)
    random = RandomAgent(RandomAgentConfig())
    wrapped_random = TerminalSafetyAgent(random)
    return AgentSpec(agent_id, wrapped_random.config_to_data(), lambda: wrapped_random)


def _arena(
    plan: RankingExperimentPlan,
    *,
    stage: str,
    replicate: int | None,
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    pairs: int,
    seed: int,
) -> dict[str, object]:
    output = plan.config.output
    suffix = stage if replicate is None else f"r{replicate}-{stage}"
    artifact_path = output / "arenas" / f"{suffix}.json"
    records_path = output / "arena-records" / suffix
    config = ArenaConfig(f"ranking-{suffix}", agent_a, agent_b, pairs, seed)
    if artifact_path.exists():
        existing = _read_json(artifact_path)
        declared_fingerprint = existing.get("artifact_fingerprint")
        fingerprint_payload = {
            key: value for key, value in existing.items() if key != "artifact_fingerprint"
        }
        if (
            declared_fingerprint != _fingerprint(fingerprint_payload)
            or existing.get("plan_fingerprint") != plan.fingerprint
            or existing.get("stage") != stage
            or existing.get("replicate") != replicate
        ):
            raise RankingExperimentError(f"existing arena artifact differs: {suffix}")
        manifest, records = load_corpus(records_path)
        stored_records = existing.get("records")
        stored_report = existing.get("report")
        stored_safety = existing.get("safety_diagnostics")
        if not isinstance(stored_records, dict) or not isinstance(stored_report, dict):
            raise RankingExperimentError(f"existing arena artifact is malformed: {suffix}")
        if stored_records.get("corpus_fingerprint") != manifest.corpus_fingerprint:
            raise RankingExperimentError(f"arena record corpus changed: {suffix}")
        elapsed = stored_report.get("elapsed_seconds")
        if not isinstance(elapsed, int | float):
            raise RankingExperimentError(f"arena runtime is malformed: {suffix}")
        recomputed_report = arena_report_from_records(
            config, records, elapsed_seconds=float(elapsed)
        ).to_data()
        recomputed_safety = audit_terminal_safety(
            records, source_corpus_fingerprint=manifest.corpus_fingerprint
        )
        if recomputed_report != stored_report or recomputed_safety != stored_safety:
            raise RankingExperimentError(f"arena aggregate no longer matches records: {suffix}")
        return existing
    retained = run_resumable_arena(
        records_path,
        config,
        generation=1,
        corpus_configuration={
            "experiment": PLAN_VERSION,
            "plan_fingerprint": plan.fingerprint,
            "stage": stage,
            "replicate": replicate,
            "terminal_safety": True,
        },
    )
    _, records = load_corpus(records_path)
    safety = audit_terminal_safety(
        records, source_corpus_fingerprint=retained.records_manifest.corpus_fingerprint
    )
    artifact: dict[str, object] = {
        "version": "ranking-arena-artifact-v1",
        "plan_fingerprint": plan.fingerprint,
        "stage": stage,
        "replicate": replicate,
        "report": retained.report.to_data(),
        "records": {
            "locator": _portable(records_path),
            "corpus_fingerprint": retained.records_manifest.corpus_fingerprint,
        },
        "safety_diagnostics": safety,
    }
    artifact["artifact_fingerprint"] = _fingerprint(artifact)
    _atomic_json(artifact_path, artifact)
    return artifact


def _pair_scores(artifact: dict[str, object]) -> tuple[float, ...]:
    report = artifact.get("report")
    if not isinstance(report, dict):
        raise RankingExperimentError("arena report is malformed")
    outcomes = report.get("paired_seed_outcomes")
    if not isinstance(outcomes, list):
        raise RankingExperimentError("arena report lacks paired outcomes")
    return tuple(cast(int, cast(dict[str, object], item)["agent_a_wins"]) / 2 for item in outcomes)


def _agent_a_seat_rates(artifact: dict[str, object]) -> dict[str, float]:
    report = artifact.get("report")
    if not isinstance(report, dict):
        raise RankingExperimentError("arena report is malformed")
    raw = report.get("agent_a_by_seat")
    if not isinstance(raw, dict):
        raise RankingExperimentError("arena report lacks seat statistics")
    rates: dict[str, float] = {}
    for seat, item in raw.items():
        if not isinstance(seat, str) or not isinstance(item, dict):
            raise RankingExperimentError("arena seat statistics are malformed")
        value = item.get("win_rate")
        if not isinstance(value, int | float):
            raise RankingExperimentError("arena seat win rate is malformed")
        rates[seat] = float(value)
    return rates


def _nested_interval(
    replicate_blocks: tuple[tuple[float, ...], ...], *, seed: int, domain: str
) -> dict[str, object]:
    if not replicate_blocks or any(not blocks for blocks in replicate_blocks):
        raise RankingExperimentError("nested bootstrap requires non-empty replicate blocks")
    block_count = len(replicate_blocks[0])
    if any(len(blocks) != block_count for blocks in replicate_blocks):
        raise RankingExperimentError("nested bootstrap requires aligned block counts")
    replicate_count = len(replicate_blocks)
    point = sum(sum(blocks) / block_count for blocks in replicate_blocks) / replicate_count
    rng = DeterministicRandom(derive_seed(seed, domain), domain)
    values: list[float] = []
    for _ in range(NESTED_BOOTSTRAP_RESAMPLES):
        selected_replicates = [
            rng.randbelow(replicate_count) for _replicate in range(replicate_count)
        ]
        selected_blocks = [rng.randbelow(block_count) for _block in range(block_count)]
        replicate_means = [
            sum(replicate_blocks[index][block] for block in selected_blocks) / block_count
            for index in selected_replicates
        ]
        values.append(sum(replicate_means) / replicate_count)
    values.sort()
    return {
        "version": NESTED_BOOTSTRAP_VERSION,
        "unit": "training replicate and shared paired setup block",
        "replicate_count": replicate_count,
        "resamples": NESTED_BOOTSTRAP_RESAMPLES,
        "point_estimate": point,
        "interval": [values[499], values[19_499]],
        "seed": derive_seed(seed, domain),
        "domain": domain,
    }


def _nested_shared_reference_difference(
    treatment_blocks: tuple[tuple[float, ...], ...],
    reference_blocks: tuple[float, ...],
    *,
    seed: int,
    domain: str,
) -> dict[str, object]:
    if not treatment_blocks or not reference_blocks:
        raise RankingExperimentError("shared-reference bootstrap requires non-empty blocks")
    if any(len(blocks) != len(reference_blocks) for blocks in treatment_blocks):
        raise RankingExperimentError("shared-reference arena blocks are not aligned")
    differences = tuple(
        tuple(
            treatment - reference
            for treatment, reference in zip(blocks, reference_blocks, strict=True)
        )
        for blocks in treatment_blocks
    )
    result = _nested_interval(differences, seed=seed, domain=domain)
    result["shared_reference"] = True
    return result


def run_ranking_experiment(plan: RankingExperimentPlan) -> dict[str, object]:
    """Run or resume fixed input audit, paired training, arenas, and analysis."""
    from agent_avenue.learning import (
        RankingTrainingConfig,
        TrainingConfig,
        load_checkpoint,
        load_dataset,
        load_ranking_dataset,
        materialize_ranking_dataset,
        save_checkpoint,
        save_ranking_dataset,
        tensor_digest,
        train_model,
        train_ranking_model,
    )

    if not plan.claim_eligible:
        raise RankingExperimentError(
            f"plan is not claim eligible: {plan.claim_ineligibility_reasons}"
        )
    output = plan.config.output
    output.mkdir(parents=True, exist_ok=True)
    _write_or_validate_plan(plan)
    parent = load_checkpoint(plan.config.parent_checkpoint)
    mc_dataset = load_dataset(plan.config.mc_dataset)
    corpus_manifest, records = load_corpus(plan.config.corpus, verify_code=False)
    ranking_path = output / "ranking-dataset.npz"
    ranking_manifest_path = ranking_path.with_suffix(".json")
    audit_path = output / "input-audit.json"
    if audit_path.exists():
        ranking_dataset = load_ranking_dataset(ranking_path)
    else:
        expected_ranking_dataset = materialize_ranking_dataset(
            records,
            mc_manifest=mc_dataset.manifest,
            source_corpus_fingerprint=corpus_manifest.corpus_fingerprint,
            verify_code=False,
        )
        if ranking_path.exists() != ranking_manifest_path.exists():
            ranking_path.unlink(missing_ok=True)
            ranking_manifest_path.unlink(missing_ok=True)
        if ranking_path.exists():
            ranking_dataset = load_ranking_dataset(ranking_path)
            if not _ranking_datasets_equal(ranking_dataset, expected_ranking_dataset):
                raise RankingExperimentError(
                    "preexisting ranking dataset does not match deterministic extraction"
                )
        else:
            save_ranking_dataset(expected_ranking_dataset, ranking_path)
            ranking_dataset = load_ranking_dataset(ranking_path)
    _validate_ranking_dataset(
        plan,
        ranking_dataset,
        mc_manifest=mc_dataset.manifest,
        array_path=ranking_path,
    )
    input_audit = {
        "version": "ranking-fixed-input-audit-v1",
        "status": "completed",
        "plan_fingerprint": plan.fingerprint,
        "inputs": plan.input_identities,
        "ranking_dataset_fingerprint": ranking_dataset.fingerprint,
        "ranking_dataset_arrays_sha256": _sha256(ranking_path),
        "ranking_counts": ranking_dataset.manifest["counts"],
        "information_boundary": ranking_dataset.manifest["information_boundary"],
    }
    input_audit["artifact_fingerprint"] = _fingerprint(input_audit)
    _write_or_validate_json(output / "input-audit.json", input_audit, label="fixed-input audit")

    controls: list[Path] = []
    treatments: list[Path] = []
    training_summaries: list[dict[str, object]] = []
    for seeds in plan.replicate_seeds:
        replicate = seeds["replicate"]
        mc_config = TrainingConfig(
            seed=seeds["mc"],
            max_epochs=plan.config.max_epochs,
            batch_size=plan.config.batch_size,
            learning_rate=plan.config.learning_rate,
            weight_decay=plan.config.weight_decay,
            early_stopping_patience=plan.config.patience,
            cpu_threads=plan.config.cpu_threads,
        )
        replicate_dir = output / "training" / f"replicate-{replicate}"
        control_path = replicate_dir / "control"
        if not control_path.exists():
            control_training = train_model(
                mc_dataset,
                config=mc_config,
                initial_state_dict=parent.model.state_dict(),
            )
            save_checkpoint(
                control_path,
                control_training.model,
                metrics=control_training.metrics_dict(),
                training_config=mc_config.normalized(),
                training_seeds={"mc": seeds["mc"]},
                dataset_fingerprint=mc_dataset.fingerprint,
                source_corpus_fingerprints=(corpus_manifest.corpus_fingerprint,),
                parent_checkpoint=parent.checkpoint_fingerprint,
                generation=1,
                metadata={
                    "experiment": PLAN_VERSION,
                    "plan_fingerprint": plan.fingerprint,
                    "recipe": "mc-control",
                    "replicate": replicate,
                },
                created_at=FIXED_CREATED_AT,
            )
        control = load_checkpoint(control_path)
        _validate_final_checkpoint(
            control,
            plan=plan,
            recipe="mc-control",
            replicate=replicate,
            seeds=seeds,
            mc_dataset_fingerprint=mc_dataset.fingerprint,
            corpus_fingerprint=corpus_manifest.corpus_fingerprint,
            ranking_dataset_fingerprint=None,
        )
        if (
            replicate == 1
            and tensor_digest(control.model.state_dict()) != EXPECTED_HISTORICAL_Q1_TENSOR
        ):
            raise RankingExperimentError(
                "replicate-1 MC control failed exact q1 tensor reproduction"
            )

        initializer_path = replicate_dir / "ranking-initializer"
        treatment_path = replicate_dir / "treatment"
        if not treatment_path.exists():
            if initializer_path.exists():
                initializer, initializer_state = _load_ranking_initializer(
                    initializer_path,
                    plan=plan,
                    replicate=replicate,
                    dataset_fingerprint=ranking_dataset.fingerprint,
                )
            else:
                ranking_config = RankingTrainingConfig(
                    seed=seeds["ranking"],
                    learning_rate=plan.config.learning_rate,
                    weight_decay=plan.config.weight_decay,
                    batch_positions=plan.config.batch_size,
                    max_epochs=plan.config.max_epochs,
                    early_stopping_patience=plan.config.patience,
                    cpu_threads=plan.config.cpu_threads,
                )
                ranking_training = train_ranking_model(
                    ranking_dataset,
                    config=ranking_config,
                    initial_state_dict=parent.model.state_dict(),
                )
                initializer_state = {
                    name: value.detach().cpu().clone()
                    for name, value in ranking_training.model.state_dict().items()
                }
                initializer = _save_ranking_initializer(
                    initializer_path,
                    initializer_state,
                    plan=plan,
                    replicate=replicate,
                    dataset_fingerprint=ranking_dataset.fingerprint,
                    metrics=ranking_training.metrics_dict(),
                )
            treatment_training = train_model(
                mc_dataset,
                config=mc_config,
                initial_state_dict=initializer_state,
            )
            save_checkpoint(
                treatment_path,
                treatment_training.model,
                metrics=treatment_training.metrics_dict(),
                training_config=mc_config.normalized(),
                training_seeds={"mc": seeds["mc"], "ranking": seeds["ranking"]},
                dataset_fingerprint=mc_dataset.fingerprint,
                source_corpus_fingerprints=(corpus_manifest.corpus_fingerprint,),
                parent_checkpoint=parent.checkpoint_fingerprint,
                generation=1,
                metadata={
                    "experiment": PLAN_VERSION,
                    "plan_fingerprint": plan.fingerprint,
                    "recipe": "heuristic-ranking-warmstart",
                    "replicate": replicate,
                    "ranking_dataset_fingerprint": ranking_dataset.fingerprint,
                    "ranking_initializer_tensor_digest": initializer["tensor_digest"],
                    "ranking_loss": "pairwise-logistic-equal-position-v1",
                },
                created_at=FIXED_CREATED_AT,
            )
        treatment = load_checkpoint(treatment_path)
        _validate_final_checkpoint(
            treatment,
            plan=plan,
            recipe="heuristic-ranking-warmstart",
            replicate=replicate,
            seeds=seeds,
            mc_dataset_fingerprint=mc_dataset.fingerprint,
            corpus_fingerprint=corpus_manifest.corpus_fingerprint,
            ranking_dataset_fingerprint=ranking_dataset.fingerprint,
        )
        initializer, _ = _load_ranking_initializer(
            initializer_path,
            plan=plan,
            replicate=replicate,
            dataset_fingerprint=ranking_dataset.fingerprint,
        )
        treatment_manifest = _jsonable(treatment.manifest)
        assert isinstance(treatment_manifest, dict)
        treatment_metadata = treatment_manifest.get("metadata")
        if (
            not isinstance(treatment_metadata, dict)
            or treatment_metadata.get("ranking_initializer_tensor_digest")
            != initializer["tensor_digest"]
        ):
            raise RankingExperimentError("treatment ranking initializer lineage mismatch")
        controls.append(control_path)
        treatments.append(treatment_path)
        training_summaries.append(
            {
                "replicate": replicate,
                "seeds": seeds,
                "control": {
                    "checkpoint_fingerprint": control.checkpoint_fingerprint,
                    "tensor_digest": tensor_digest(control.model.state_dict()),
                    "metrics": _jsonable(control.metrics),
                },
                "treatment": {
                    "checkpoint_fingerprint": treatment.checkpoint_fingerprint,
                    "tensor_digest": tensor_digest(treatment.model.state_dict()),
                    "metrics": _jsonable(treatment.metrics),
                },
                "ranking_initializer": initializer,
            }
        )
        _atomic_json(replicate_dir / "summary.json", training_summaries[-1])
    first_control = cast(dict[str, object], training_summaries[0]["control"])
    _atomic_json(
        output / "training-summary.json",
        {
            "version": "ranking-training-summary-v1",
            "plan_fingerprint": plan.fingerprint,
            "replicates": training_summaries,
            "control_reproduction": {
                "expected_tensor_digest": EXPECTED_HISTORICAL_Q1_TENSOR,
                "actual_tensor_digest": first_control["tensor_digest"],
                "matched": True,
            },
        },
    )

    parent_agent = _checkpoint_agent(plan.config.parent_checkpoint, "parent")
    heuristic_agent = _baseline_agent("heuristic", "heuristic")
    random_agent = _baseline_agent("random", "random")
    parent_heuristic = _arena(
        plan,
        stage="parent-versus-heuristic",
        replicate=None,
        agent_a=parent_agent,
        agent_b=heuristic_agent,
        pairs=plan.config.heuristic_pairs,
        seed=plan.arena_seeds["heuristic"],
    )
    direct_blocks: list[tuple[float, ...]] = []
    parent_blocks: list[tuple[float, ...]] = []
    heuristic_blocks: list[tuple[float, ...]] = []
    random_blocks: list[tuple[float, ...]] = []
    replicate_results: list[dict[str, object]] = []
    reference_heuristic_blocks = _pair_scores(parent_heuristic)
    safety_passed = True
    minimum_seat = 1.0
    direct_positive = 0
    for replicate, (control_path, treatment_path) in enumerate(
        zip(controls, treatments, strict=True), start=1
    ):
        direct = _arena(
            plan,
            stage="treatment-versus-control",
            replicate=replicate,
            agent_a=_checkpoint_agent(treatment_path, "treatment"),
            agent_b=_checkpoint_agent(control_path, "control"),
            pairs=plan.config.direct_pairs,
            seed=plan.arena_seeds["direct"],
        )
        treatment_parent = _arena(
            plan,
            stage="treatment-versus-parent",
            replicate=replicate,
            agent_a=_checkpoint_agent(treatment_path, "candidate"),
            agent_b=parent_agent,
            pairs=plan.config.parent_pairs,
            seed=plan.arena_seeds["parent"],
        )
        control_parent = _arena(
            plan,
            stage="control-versus-parent",
            replicate=replicate,
            agent_a=_checkpoint_agent(control_path, "candidate"),
            agent_b=parent_agent,
            pairs=plan.config.parent_pairs,
            seed=plan.arena_seeds["parent"],
        )
        treatment_heuristic = _arena(
            plan,
            stage="treatment-versus-heuristic",
            replicate=replicate,
            agent_a=_checkpoint_agent(treatment_path, "candidate"),
            agent_b=heuristic_agent,
            pairs=plan.config.heuristic_pairs,
            seed=plan.arena_seeds["heuristic"],
        )
        control_heuristic = _arena(
            plan,
            stage="control-versus-heuristic",
            replicate=replicate,
            agent_a=_checkpoint_agent(control_path, "candidate"),
            agent_b=heuristic_agent,
            pairs=plan.config.heuristic_pairs,
            seed=plan.arena_seeds["heuristic"],
        )
        treatment_random = _arena(
            plan,
            stage="treatment-versus-random",
            replicate=replicate,
            agent_a=_checkpoint_agent(treatment_path, "candidate"),
            agent_b=random_agent,
            pairs=plan.config.random_pairs,
            seed=plan.arena_seeds["random"],
        )
        control_random = _arena(
            plan,
            stage="control-versus-random",
            replicate=replicate,
            agent_a=_checkpoint_agent(control_path, "candidate"),
            agent_b=random_agent,
            pairs=plan.config.random_pairs,
            seed=plan.arena_seeds["random"],
        )
        direct_score = _pair_scores(direct)
        treatment_parent_score = _pair_scores(treatment_parent)
        control_parent_score = _pair_scores(control_parent)
        treatment_heuristic_score = _pair_scores(treatment_heuristic)
        control_heuristic_score = _pair_scores(control_heuristic)
        treatment_random_score = _pair_scores(treatment_random)
        control_random_score = _pair_scores(control_random)
        direct_blocks.append(direct_score)
        parent_blocks.append(treatment_parent_score)
        heuristic_blocks.append(treatment_heuristic_score)
        random_blocks.append(treatment_random_score)
        if sum(direct_score) / len(direct_score) > 0.5:
            direct_positive += 1
        candidate_seat_rates: list[float] = []
        for candidate_artifact in (
            direct,
            treatment_parent,
            control_parent,
            treatment_heuristic,
            control_heuristic,
            treatment_random,
            control_random,
        ):
            candidate_seat_rates.extend(_agent_a_seat_rates(candidate_artifact).values())
        direct_seats = _agent_a_seat_rates(direct)
        candidate_seat_rates.extend(
            (
                1.0 - direct_seats["player_two"],
                1.0 - direct_seats["player_one"],
            )
        )
        replicate_minimum_seat = min(candidate_seat_rates)
        minimum_seat = min(minimum_seat, replicate_minimum_seat)
        for artifact in (
            direct,
            treatment_parent,
            control_parent,
            treatment_heuristic,
            control_heuristic,
            treatment_random,
            control_random,
        ):
            diagnostics = cast(dict[str, object], artifact["safety_diagnostics"])
            counts = cast(dict[str, int], diagnostics["counts"])
            safety_passed &= counts["executed_avoidable_provable_losses"] == 0
        replicate_results.append(
            {
                "replicate": replicate,
                "direct_treatment_win_rate": sum(direct_score) / len(direct_score),
                "treatment_parent_win_rate": sum(treatment_parent_score)
                / len(treatment_parent_score),
                "control_parent_win_rate": sum(control_parent_score) / len(control_parent_score),
                "treatment_heuristic_win_rate": sum(treatment_heuristic_score)
                / len(treatment_heuristic_score),
                "control_heuristic_win_rate": sum(control_heuristic_score)
                / len(control_heuristic_score),
                "treatment_random_win_rate": sum(treatment_random_score)
                / len(treatment_random_score),
                "control_random_win_rate": sum(control_random_score) / len(control_random_score),
                "minimum_candidate_seat_rate": replicate_minimum_seat,
            }
        )

    bootstrap_seed = plan.arena_seeds["nested-bootstrap"]
    direct_interval = _nested_interval(
        tuple(direct_blocks), seed=bootstrap_seed, domain="direct-treatment-versus-control"
    )
    parent_interval = _nested_interval(
        tuple(parent_blocks), seed=bootstrap_seed, domain="treatment-versus-parent"
    )
    heuristic_interval = _nested_shared_reference_difference(
        tuple(heuristic_blocks),
        reference_heuristic_blocks,
        seed=bootstrap_seed,
        domain="treatment-minus-parent-versus-heuristic",
    )
    random_interval = _nested_interval(
        tuple(random_blocks), seed=bootstrap_seed, domain="treatment-versus-random"
    )
    conditions = {
        "direct_two_of_three": direct_positive >= 2,
        "direct_nested_lower_above_half": cast(list[float], direct_interval["interval"])[0] > 0.5,
        "parent_nested_lower_above_half": cast(list[float], parent_interval["interval"])[0] > 0.5,
        "heuristic_non_regression": cast(list[float], heuristic_interval["interval"])[0] > -0.05,
        "random_nested_lower_above_half": cast(list[float], random_interval["interval"])[0] > 0.5,
        "minimum_seat_rate": minimum_seat >= 0.45,
        "terminal_safety": safety_passed,
        "control_reproduction": True,
    }
    decision = "advance" if all(conditions.values()) else "inconclusive"
    result = {
        "version": RESULT_VERSION,
        "status": "completed",
        "evidence_class": "fixed-corpus-development-recipe-comparison",
        "plan_fingerprint": plan.fingerprint,
        "replicates": replicate_results,
        "nested_intervals": {
            "direct_treatment_versus_control": direct_interval,
            "treatment_versus_parent": parent_interval,
            "heuristic_difference_versus_parent": heuristic_interval,
            "treatment_versus_random": random_interval,
        },
        "conditions": conditions,
        "decision": decision,
        "minimum_candidate_seat_rate": minimum_seat,
        "scientific_limit": (
            "Teacher preferences are heuristic imitation labels, not counterfactual outcomes; "
            "this development cycle does not promote a checkpoint or open locked-final seeds."
        ),
    }
    result["result_fingerprint"] = _fingerprint(result)
    _atomic_json(output / "result.json", result)
    return result
