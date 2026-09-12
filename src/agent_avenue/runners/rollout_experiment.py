"""Resumable frozen Step-4 counterfactual-rollout experiment runner.

This module owns the experiment orchestration only.  The safe panel, synthetic-world
transition, rollout targets, and auxiliary objective stay in their narrow rollout/learning
modules; the runner makes their frozen provenance and claim gates explicit.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, cast

from agent_avenue.agents import (
    Agent,
    GreedyHeuristicAgent,
    GreedyHeuristicConfig,
    RandomAgent,
    RandomAgentConfig,
    TerminalOffenseAgent,
    TerminalSafetyAgent,
    derive_seed,
)
from agent_avenue.engine.model import Action, PlayOfferAction, RecruitAction
from agent_avenue.engine.setup import normalize_config
from agent_avenue.learning import (
    LoadedCheckpoint,
    RolloutTrainingData,
    RolloutTrainingResult,
    RolloutTreatmentConfig,
    StructuredMaterializedDataset,
    StructuredTrainingConfig,
    StructuredTrainingData,
    inspect_structured_checkpoint,
    lambda_zero_trace_regression,
    load_checkpoint,
    load_structured_checkpoint,
    load_structured_dataset,
    save_structured_checkpoint,
    structured_tensor_digest,
    tensor_digest,
    train_rollout_treatment,
)
from agent_avenue.observation import observation_to_data
from agent_avenue.rollout import (
    PANEL_STRATA,
    PanelPosition,
    canonical_panel_order,
    extract_train_panel_candidates,
    generate_rollout_targets,
    sample_position_worlds,
    select_panel_result,
)
from agent_avenue.rollout.artifact import (
    assemble_rollout_target_shards,
    load_rollout_targets,
    save_rollout_targets,
)
from agent_avenue.rollout.latent import LatentRolloutState
from agent_avenue.rollout.targets import REQUIRED_POLICY_IDS, LeafScorer
from agent_avenue.runners.arena import (
    ArenaConfig,
    arena_report_from_records,
    run_resumable_arena,
    schedule_arena,
)
from agent_avenue.runners.game import AgentSpec
from agent_avenue.runners.population import population_setup_identity
from agent_avenue.runners.safety_audit import audit_terminal_safety
from agent_avenue.runners.strength_audit import audit_public_forced_wins
from agent_avenue.storage import (
    GameRecord,
    code_fingerprint,
    inspect_source_identity,
    load_corpus,
    repository_root,
    rules_fingerprint,
)

CYCLE_ID: Final = "m7-counterfactual-rollout-supervision-v1"
ROOT_SEED: Final = 2026091204
REPLICATES: Final = ("replicate-1", "replicate-2", "replicate-3")
PLAN_VERSION: Final = "m7-counterfactual-rollout-plan-v1"
NESTED_RESAMPLES: Final = 20_000
NESTED_LOWER: Final = 499
NESTED_UPPER: Final = 19_499
CLAIM_CUTOFF_SECONDS: Final = 7 * 60 * 60 + 45 * 60
HARD_BUDGET_SECONDS: Final = 8 * 60 * 60
EXECUTION_STATE_VERSION: Final = "m7-counterfactual-rollout-execution-v1"
FIXED_CREATED_AT: Final = "2026-09-12T00:00:00+00:00"
STEP3_CLAIM_SOURCE: Final = "a63f0819dd679308fe03ec23d619e609529b1166"


class RolloutExperimentError(ValueError):
    """Raised when frozen Step-4 evidence cannot be safely built or resumed."""


@dataclass(frozen=True, slots=True)
class Step4Input:
    replicate_id: str
    corpus: Mapping[str, object]
    dataset: Mapping[str, object]
    checkpoint: Mapping[str, object]

    @property
    def index(self) -> str:
        return self.replicate_id.removeprefix("replicate-")


@dataclass(frozen=True, slots=True)
class RolloutArenaCell:
    replicate_id: str
    key: str
    candidate: str | None
    opponent: str
    pair_count: int
    master_seed: int
    shared_group: str | None

    @property
    def run_id(self) -> str:
        return f"{CYCLE_ID}-{self.replicate_id}-{self.key}"

    def to_data(self) -> dict[str, object]:
        return {
            "replicate_id": self.replicate_id,
            "key": self.key,
            "run_id": self.run_id,
            "candidate": self.candidate,
            "opponent": self.opponent,
            "paired_blocks": self.pair_count,
            "games": self.pair_count * 2,
            "master_seed": self.master_seed,
            "shared_group": self.shared_group,
        }


@dataclass(frozen=True, slots=True)
class RolloutExperimentConfig:
    """Explicit Step-4 source/output paths and nonclaim smoke limits."""

    output: Path
    step3_root: Path
    input_manifest: Path = Path("research/cycles/m7-counterfactual-rollout-inputs.json")
    holdout_roots: tuple[Path, ...] = field(default_factory=lambda: (repository_root() / "runs",))
    claim: bool = False
    smoke_positions_per_stratum: int | None = None
    smoke_treatment_epochs: int | None = None
    smoke_arena_pairs: int | None = None

    def __post_init__(self) -> None:
        if not self.holdout_roots:
            raise RolloutExperimentError("at least one recursive holdout root is required")
        smoke_values = (
            self.smoke_positions_per_stratum,
            self.smoke_treatment_epochs,
            self.smoke_arena_pairs,
        )
        if self.claim and any(value is not None for value in smoke_values):
            raise RolloutExperimentError("claim configuration cannot use smoke reductions")
        if not self.claim and any(value is None for value in smoke_values):
            raise RolloutExperimentError("nonclaim execution requires all explicit smoke limits")
        if not self.claim and (
            self.smoke_positions_per_stratum != 1
            or self.smoke_treatment_epochs is None
            or self.smoke_treatment_epochs < 1
            or self.smoke_arena_pairs != 1
        ):
            raise RolloutExperimentError(
                "the only supported smoke is one position/stratum, positive treatment "
                "epochs, one arena pair"
            )

    @property
    def is_smoke(self) -> bool:
        return not self.claim

    @property
    def panel_quota(self) -> int:
        return 20 if self.claim else 1

    @property
    def treatment_epochs(self) -> int | None:
        return None if self.claim else self.smoke_treatment_epochs


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _read(path: Path) -> dict[str, object]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RolloutExperimentError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(result, dict):
        raise RolloutExperimentError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, object], result)


def _number(value: object, *, label: str) -> float:
    if not isinstance(value, int | float):
        raise RolloutExperimentError(f"{label} must be numeric")
    return float(value)


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise RolloutExperimentError(f"{label} must be an exact integer")
    return value


def _artifact(version: str, **values: object) -> dict[str, object]:
    result = {"version": version, **values}
    result["artifact_fingerprint"] = _fingerprint(result)
    return result


def _write_immutable(path: Path, value: Mapping[str, object], *, label: str) -> None:
    normalized = json.loads(_canonical(dict(value)))
    if not isinstance(normalized, dict):  # pragma: no cover - dict input is fixed
        raise RolloutExperimentError("immutable artifact normalization failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) != normalized:
            raise RolloutExperimentError(f"existing immutable {label} differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(_canonical(normalized) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, object]], *, label: str) -> None:
    """Write deterministic compressed immutable JSON lines (mtime zero)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical(dict(row)) + b"\n" for row in rows)
    if path.exists():
        with gzip.open(path, "rb") as source:
            if source.read() != payload:
                raise RolloutExperimentError(f"existing immutable {label} differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as destination:
                destination.write(payload)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _source_root(step3_root: Path) -> Path:
    root = step3_root.resolve()
    if root.name != "m7-structured-model-v2" or root.parent.name != "runs":
        raise RolloutExperimentError("step3_root must be retained runs/m7-structured-model-v2")
    return root.parent.parent


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root()).as_posix()
    except ValueError:
        return str(path.resolve())


def _manifest(config: RolloutExperimentConfig) -> tuple[dict[str, object], tuple[Step4Input, ...]]:
    data = _read(config.input_manifest)
    if data.get("artifact_fingerprint") != _fingerprint(
        {k: v for k, v in data.items() if k != "artifact_fingerprint"}
    ):
        raise RolloutExperimentError("committed Step-4 input manifest fingerprint mismatch")
    if (
        data.get("cycle_id") != CYCLE_ID
        or data.get("version") != "m7-counterfactual-rollout-inputs-v1"
    ):
        raise RolloutExperimentError("input manifest cycle/version mismatch")
    entries = data.get("replicates")
    if not isinstance(entries, list) or len(entries) != 3:
        raise RolloutExperimentError("input manifest must contain exactly three M-v2 replicates")
    parsed: list[Step4Input] = []
    for value in entries:
        if not isinstance(value, Mapping):
            raise RolloutExperimentError("replicate input is malformed")
        replicate = value.get("replicate_id")
        corpus = value.get("step2_mixed_corpus")
        dataset = value.get("step3_v2_dataset")
        checkpoint = value.get("step3_m_v2_checkpoint")
        if (
            replicate not in REPLICATES
            or not isinstance(corpus, Mapping)
            or not isinstance(dataset, Mapping)
            or not isinstance(checkpoint, Mapping)
        ):
            raise RolloutExperimentError("replicate source identity is malformed")
        parsed.append(Step4Input(cast(str, replicate), corpus, dataset, checkpoint))
    if {item.replicate_id for item in parsed} != set(REPLICATES):
        raise RolloutExperimentError("replicate identities are not unique/complete")
    return data, tuple(sorted(parsed, key=lambda item: item.replicate_id))


def _digest_field(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or len(result) != 64:
        raise RolloutExperimentError(f"frozen digest {key} is malformed")
    return result


def _verify_file(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise RolloutExperimentError(f"frozen {label} is missing or not a regular file: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RolloutExperimentError(f"frozen full-file digest mismatch for {label}")
    return actual


def _historical_compatibility() -> dict[str, object]:
    """Replay Step-3 historical evidence only under an explicit semantic allowlist."""
    try:
        changed = subprocess.run(
            ("git", "diff", "--name-only", f"{STEP3_CLAIM_SOURCE}..HEAD"),
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RolloutExperimentError("unable to inspect Step-3 historical compatibility") from exc
    protected = (
        "src/agent_avenue/engine/",
        "src/agent_avenue/observation/",
        "src/agent_avenue/encoding/candidate_structured_v2.py",
        "src/agent_avenue/learning/structured_",
        "src/agent_avenue/agents/structured.py",
        "src/agent_avenue/agents/learned.py",
        "src/agent_avenue/agents/terminal_",
        "src/agent_avenue/runners/arena.py",
        "src/agent_avenue/runners/game.py",
        "src/agent_avenue/runners/safety_audit.py",
        "src/agent_avenue/runners/strength_audit.py",
    )
    allowed_exact = {
        "src/agent_avenue/learning/__init__.py",
        "src/agent_avenue/runners/structured_experiment.py",
        "research/cycles/M7_STRUCTURED_MODEL_V2_REPAIR1.md",
        "docs/M7_STRUCTURED_MODEL_V2_RESULTS.md",
    }
    forbidden = [
        path
        for path in changed
        if path in protected or any(path.startswith(prefix) for prefix in protected)
    ]
    # The historical Step-3 repair changes only checksum scope and is explicitly reviewed.
    forbidden = [path for path in forbidden if path not in allowed_exact]
    if forbidden:
        raise RolloutExperimentError("Step-3 protected semantics changed: " + ", ".join(forbidden))
    return {
        "version": "step4-step3-historical-compatibility-v1",
        "step3_claim_source": STEP3_CLAIM_SOURCE,
        "current_revision": inspect_source_identity().git_revision,
        "rules_fingerprint": rules_fingerprint(),
        "protected_paths": list(protected),
        "explicit_repair_allowlist": sorted(allowed_exact),
        "changed_paths": changed,
        "protected_path_hits": forbidden,
        "passed": True,
    }


def _continuation_paths(root: Path, manifest: Mapping[str, object]) -> dict[str, Path]:
    values = manifest.get("continuation_policy_inputs")
    if not isinstance(values, Mapping):
        raise RolloutExperimentError("continuation policy identities are missing")
    paths: dict[str, Path] = {}
    for key in ("q0-parent", "q1", "q2", "q3", "q4", "historical-q0"):
        value = values.get(key)
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            raise RolloutExperimentError(f"continuation policy {key} is malformed")
        paths[key] = root / cast(str, value["path"])
    return paths


def verify_step3_inputs(
    config: RolloutExperimentConfig,
) -> tuple[dict[str, object], tuple[Step4Input, ...], dict[str, object]]:
    """Verify *every* frozen Step-3/continuation full-file and tensor identity first."""
    manifest, inputs = _manifest(config)
    root = _source_root(config.step3_root)
    rows: list[dict[str, object]] = []
    for item in inputs:
        corpus = root / cast(str, item.corpus["path"])
        dataset = root / cast(str, item.dataset["path"])
        checkpoint_path = root / cast(str, item.checkpoint["path"])
        files = {
            "corpus_manifest": (
                corpus / "manifest.json",
                _digest_field(item.corpus, "manifest_sha256"),
            ),
            "corpus_records": (
                corpus / "games.jsonl.gz",
                _digest_field(item.corpus, "records_sha256"),
            ),
            "dataset_npz": (dataset, _digest_field(item.dataset, "npz_sha256")),
            "dataset_manifest": (
                dataset.parent / "dataset.json",
                _digest_field(item.dataset, "manifest_sha256"),
            ),
            "checkpoint_weights": (
                checkpoint_path / "weights.pt",
                _digest_field(item.checkpoint, "weights_sha256"),
            ),
            "checkpoint_manifest": (
                checkpoint_path / "manifest.json",
                _digest_field(item.checkpoint, "manifest_sha256"),
            ),
            "checkpoint_metrics": (
                checkpoint_path / "metrics.json",
                _digest_field(item.checkpoint, "metrics_sha256"),
            ),
        }
        actual = {
            name: _verify_file(path, expected, label=f"{item.replicate_id}:{name}")
            for name, (path, expected) in files.items()
        }
        loaded = inspect_structured_checkpoint(checkpoint_path)
        if loaded.checkpoint_fingerprint != item.checkpoint.get(
            "fingerprint"
        ) or loaded.tensor_digest != item.checkpoint.get("tensor_digest"):
            raise RolloutExperimentError(
                f"retained M-v2 checkpoint identity differs: {item.replicate_id}"
            )
        dataset_manifest = _read(dataset.parent / "dataset.json")
        if dataset_manifest.get("dataset_fingerprint") != item.dataset.get("fingerprint"):
            raise RolloutExperimentError(
                f"retained M-v2 dataset identity differs: {item.replicate_id}"
            )
        rows.append(
            {
                "replicate_id": item.replicate_id,
                "files": {name: _relative(path) for name, (path, _) in files.items()},
                "digests": actual,
                "dataset_fingerprint": dataset_manifest.get("dataset_fingerprint"),
                "checkpoint_fingerprint": loaded.checkpoint_fingerprint,
                "tensor_digest": loaded.tensor_digest,
            }
        )
    step3_plan, step3_result = (
        _read(config.step3_root / "plan.json"),
        _read(config.step3_root / "result.json"),
    )
    step3_stats, step3_selection = (
        _read(config.step3_root / "statistics.json"),
        _read(config.step3_root / "selection.json"),
    )
    step3_validation = _read(config.step3_root / "validation.json")
    frozen = manifest.get("step3")
    if not isinstance(frozen, Mapping) or (
        step3_plan.get("plan_fingerprint") != frozen.get("plan_fingerprint")
        or step3_result.get("result_fingerprint") != frozen.get("result_fingerprint")
        or step3_stats.get("artifact_fingerprint") != frozen.get("statistics_fingerprint")
        or step3_selection.get("artifact_fingerprint") != frozen.get("selection_fingerprint")
        or step3_validation.get("artifact_fingerprint") != frozen.get("validation_fingerprint")
        or step3_selection.get("development_selected_step4_input_recipe") != "M"
    ):
        raise RolloutExperimentError(
            "Step-3 plan/result/statistics/selection/validation identity differs"
        )
    source = step3_plan.get("source")
    if not isinstance(source, Mapping) or source.get("git_revision") != frozen.get("claim_source"):
        raise RolloutExperimentError("Step-3 claim source differs")
    archive = root / "artifacts/archive/m7-structured-model-v2-2026-09-12.tar.gz"
    _verify_file(archive, _digest_field(frozen, "archive_sha256"), label="Step-3 archive")
    with tarfile.open(archive, "r:gz") as payload:
        member = payload.getmember("runs/m7-structured-model-v2/archive-manifest.json")
        stream = payload.extractfile(member)
        if stream is None:
            raise RolloutExperimentError("Step-3 archive manifest is missing")
        archived = json.loads(stream.read().decode())
    if not isinstance(archived, Mapping):
        raise RolloutExperimentError("Step-3 archived manifest is malformed")
    continuation_rows: dict[str, object] = {}
    for key, path in _continuation_paths(root, manifest).items():
        value = cast(
            Mapping[str, object],
            cast(Mapping[str, object], manifest["continuation_policy_inputs"])[key],
        )
        files = {
            "weights": (path / "weights.pt", _digest_field(value, "weights_sha256")),
            "manifest": (path / "manifest.json", _digest_field(value, "manifest_sha256")),
            "metrics": (path / "metrics.json", _digest_field(value, "metrics_sha256")),
        }
        actual = {
            name: _verify_file(file, expected, label=f"continuation:{key}:{name}")
            for name, (file, expected) in files.items()
        }
        checkpoint = load_checkpoint(path)
        actual_tensor = tensor_digest(checkpoint.model.state_dict())
        if checkpoint.checkpoint_fingerprint != value.get(
            "checkpoint_fingerprint"
        ) or actual_tensor != value.get("tensor_digest"):
            raise RolloutExperimentError(f"continuation checkpoint/tensor differs: {key}")
        continuation_rows[key] = {
            "path": _relative(path),
            "files": actual,
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "tensor_digest": actual_tensor,
        }
    return (
        manifest,
        inputs,
        _artifact(
            "m7-counterfactual-rollout-input-audit-v1",
            input_manifest_fingerprint=manifest["artifact_fingerprint"],
            step3_root=str(config.step3_root.resolve()),
            replicates=rows,
            step3={
                "plan": step3_plan["plan_fingerprint"],
                "result": step3_result["result_fingerprint"],
                "statistics": step3_stats["artifact_fingerprint"],
                "selection": step3_selection["artifact_fingerprint"],
                "validation": step3_validation["artifact_fingerprint"],
                "archive_manifest": archived.get("artifact_fingerprint"),
            },
            continuation=continuation_rows,
            historical_compatibility=_historical_compatibility(),
            status="passed",
        ),
    )


def _cells(config: RolloutExperimentConfig) -> tuple[RolloutArenaCell, ...]:
    if config.is_smoke:
        replicate = "replicate-1"
        key = "T1-vs-C1"
        return (
            RolloutArenaCell(
                replicate,
                key,
                "T",
                "C",
                1,
                derive_seed(ROOT_SEED, f"step4:arena:{key}:{replicate}") & ((1 << 63) - 1),
                "direct",
            ),
        )
    cells: list[RolloutArenaCell] = []
    for replicate in REPLICATES:
        index = replicate.removeprefix("replicate-")

        def add(
            key: str,
            candidate: str | None,
            opponent: str,
            pairs: int,
            shared: str | None,
            *,
            current_replicate: str = replicate,
        ) -> None:
            cells.append(
                RolloutArenaCell(
                    current_replicate,
                    key,
                    candidate,
                    opponent,
                    pairs,
                    derive_seed(ROOT_SEED, f"step4:arena:{key}:{current_replicate}")
                    & ((1 << 63) - 1),
                    shared,
                )
            )

        add(f"T{index}-vs-C{index}", "T", "C", 500, "direct")
        for candidate in ("T", "C"):
            add(f"{candidate}{index}-vs-q0-parent", candidate, "q0-parent", 500, "q0-parent")
            add(f"{candidate}{index}-vs-heuristic", candidate, "heuristic", 300, "heuristic")
        add(f"q0-parent-vs-heuristic-reference-{index}", None, "heuristic", 300, "heuristic")
        for candidate in ("T", "C"):
            add(f"{candidate}{index}-vs-random", candidate, "random", 200, "random")
            add(
                f"{candidate}{index}-vs-historical-q0",
                candidate,
                "historical-q0",
                200,
                "historical-q0",
            )
            for opponent in ("q1", "q2", "q3", "q4"):
                add(f"{candidate}{index}-vs-{opponent}", candidate, opponent, 100, opponent)
    return tuple(cells)


def _check_cardinality(cells: Sequence[RolloutArenaCell], *, claim: bool) -> None:
    games = sum(cell.pair_count * 2 for cell in cells)
    if claim and (len(cells) != 54 or games != 24_000):
        raise RolloutExperimentError("frozen Step-4 arena must contain 54 cells / 24,000 games")
    if not claim and (len(cells) != 1 or games != 2):
        raise RolloutExperimentError("smoke must contain one two-game arena pair")


def _holdout_scan(
    config: RolloutExperimentConfig, cells: Sequence[RolloutArenaCell]
) -> dict[str, object]:
    current: dict[str, dict[str, object]] = {}
    for cell in cells:
        config_cell = ArenaConfig(
            cell.run_id,
            AgentSpec("schedule-a", {"type": "schedule"}, RandomAgent),
            AgentSpec("schedule-b", {"type": "schedule"}, RandomAgent),
            cell.pair_count,
            cell.master_seed,
        )
        for spec in schedule_arena(config_cell):
            identity = population_setup_identity(normalize_config(spec.config), spec.setup_seed)
            previous = current.get(identity)
            current[identity] = {
                "replicate": cell.replicate_id,
                "cell": cell.key,
                "shared": cell.shared_group,
            }
            if previous is not None and previous != current[identity]:
                raise RolloutExperimentError("undeclared internal arena setup overlap")
    prior: dict[str, list[str]] = {}
    scanned: list[dict[str, object]] = []
    output = config.output.resolve()
    seen: set[Path] = set()
    for declared in config.holdout_roots:
        root = declared.resolve()
        if not root.exists():
            scanned.append({"root": str(root), "status": "missing"})
            continue
        for manifest in sorted(root.rglob("manifest.json")):
            if manifest.resolve() in seen or manifest.resolve().is_relative_to(output):
                continue
            seen.add(manifest.resolve())
            directory = manifest.parent
            if not (directory / "games.jsonl.gz").is_file():
                continue
            loaded, records = load_corpus(directory, verify_code=False, verify_replays=False)
            scanned.append(
                {
                    "path": str(directory),
                    "records": len(records),
                    "corpus": loaded.corpus_fingerprint,
                }
            )
            for record in records:
                identity = population_setup_identity(
                    normalize_config(record.replay.config), record.replay.seed
                )
                prior.setdefault(identity, []).append(str(directory))
    overlap = sorted(set(current).intersection(prior))
    return _artifact(
        "m7-counterfactual-rollout-holdout-v1",
        recursive=True,
        excludes_current_output=True,
        current_setup_count=len(current),
        current_setup_fingerprint=_fingerprint(sorted(current)),
        prior_setup_count=len(prior),
        prior_setup_fingerprint=_fingerprint(sorted(prior)),
        overlap_count=len(overlap),
        overlap_examples=[{"identity": value, "prior": prior[value]} for value in overlap[:24]],
        scanned=scanned,
        status="passed" if not overlap else "failed",
    )


def _panel_artifacts(
    output: Path, item: Step4Input, root: Path, *, quota: int
) -> tuple[tuple[PanelPosition, ...], dict[str, object]]:
    corpus_dir = root / cast(str, item.corpus["path"])
    dataset_path = root / cast(str, item.dataset["path"])
    dataset = load_structured_dataset(dataset_path)
    train = tuple(str(value) for value in dataset.train.record_fingerprint.tolist())
    manifest = dataset.manifest
    split = manifest.get("split")
    if not isinstance(split, Mapping) or set(train) != set(
        cast(list[str], split.get("train_game_fingerprints", []))
    ):
        raise RolloutExperimentError("Step-3 train dataset rows/split do not agree")
    loaded, records = load_corpus(corpus_dir, verify_code=False, verify_replays=True)
    if loaded.corpus_fingerprint != item.corpus.get("fingerprint"):
        raise RolloutExperimentError("Step-3 M corpus identity changed after input audit")
    candidates = extract_train_panel_candidates(
        records, replicate_id=item.replicate_id, train_record_fingerprints=train, verify_code=False
    )
    selection = select_panel_result(candidates, replicate_id=item.replicate_id)
    # Claim keeps exactly all 20 frozen selections.  Smoke is an explicitly nonclaim prefix of the
    # identical exact selection, so it cannot alter ranking/identity or hide a quota shortfall.
    full_positions = canonical_panel_order(selection.positions)
    positions = (
        tuple(
            position
            for stratum in PANEL_STRATA
            for position in full_positions
            if position.stratum == stratum
        )
        if quota == 20
        else tuple(
            next(position for position in full_positions if position.stratum == stratum)
            for stratum in PANEL_STRATA
        )
    )
    if len(positions) != len(PANEL_STRATA) * quota:
        raise RolloutExperimentError("panel quota cardinality differs")
    public_rows = (
        {
            "replicate_id": p.replicate_id,
            "stratum": p.stratum,
            "safe_identity": p.safe_identity,
            "selection_hash": p.selection_hash,
            "observation": observation_to_data(p.observation),
        }
        for p in positions
    )
    audit_rows = (
        {"safe_identity": p.safe_identity, "audit_record_fingerprint": p.audit_record_fingerprint}
        for p in positions
    )
    panel_dir = output / "panel" / item.replicate_id
    _write_gzip_jsonl(panel_dir / "positions.jsonl.gz", public_rows, label="public-safe panel")
    _write_gzip_jsonl(
        panel_dir / "audit-locators.jsonl.gz", audit_rows, label="panel audit locators"
    )
    artifact = _artifact(
        "m7-counterfactual-rollout-panel-v1",
        replicate_id=item.replicate_id,
        quota_per_stratum=quota,
        claim_quota_per_stratum=20,
        full_eligible_identity_counts=selection.eligible_identity_counts,
        positions=len(positions),
        panel_digest=_fingerprint(
            [
                {
                    "stratum": p.stratum,
                    "safe_identity": p.safe_identity,
                    "selection_hash": p.selection_hash,
                }
                for p in positions
            ]
        ),
        public_positions_sha256=_sha256(panel_dir / "positions.jsonl.gz"),
        audit_locators_sha256=_sha256(panel_dir / "audit-locators.jsonl.gz"),
        source_dataset_fingerprint=dataset.fingerprint,
        source_corpus_fingerprint=loaded.corpus_fingerprint,
        train_record_count=len(set(train)),
        validation_record_count=len(
            set(str(value) for value in dataset.validation.record_fingerprint.tolist())
        ),
        status="passed",
    )
    _write_immutable(panel_dir / "manifest.json", artifact, label="panel manifest")
    return positions, artifact


def _raw_v1_agent(path: Path) -> Any:
    from agent_avenue.agents.learned import LearnedValueAgent

    return LearnedValueAgent.from_checkpoint(load_checkpoint(path))


def _continuation_agents(paths: Mapping[str, Path]) -> tuple[dict[str, Agent], LeafScorer]:
    q0 = _raw_v1_agent(paths["q0-parent"])
    agents: dict[str, Agent] = {
        "q0": q0,
        "q1": _raw_v1_agent(paths["q1"]),
        "q2": _raw_v1_agent(paths["q2"]),
        "q3": _raw_v1_agent(paths["q3"]),
        "q4": _raw_v1_agent(paths["q4"]),
        "heuristic": GreedyHeuristicAgent(GreedyHeuristicConfig()),
        "random": RandomAgent(RandomAgentConfig()),
    }
    if set(agents) != REQUIRED_POLICY_IDS:
        raise RolloutExperimentError("continuation policy population differs")
    return agents, cast(LeafScorer, q0)


def _state_data(state: LatentRolloutState) -> dict[str, object]:
    """Quarantined hidden transcript representation; never written into targets/panel/training."""
    return {
        "deck": [card.value for card in state.deck],
        "hands": [[card.value for card in hand] for hand in state.hands],
        "recruited": [[card.value for card in cards] for cards in state.recruited],
        "scores": list(state.scores),
        "active_player": state.active_player.value,
        "turn": state.turn,
        "phase": state.phase.value,
        "revision": state.revision,
    }


def _action_data(action: Action) -> dict[str, object]:
    if isinstance(action, RecruitAction):
        return {
            "type": "recruit",
            "revision": action.revision,
            "actor": action.actor.value,
            "slot": action.slot.value,
        }
    if not isinstance(action, PlayOfferAction):
        raise RolloutExperimentError("unknown rollout action")
    return {
        "type": "play_offer",
        "revision": action.revision,
        "actor": action.actor.value,
        "face_up": action.face_up.value,
        "face_down": action.face_down.value,
    }


def _trusted_transcripts(
    positions: Sequence[PanelPosition], continuation: Mapping[str, Agent], leaf: LeafScorer
) -> tuple[dict[str, object], ...]:
    """Retain exactly one semantic-first root trajectory per stratum/world.

    The target rows intentionally retain only hidden digests.  This separate private artifact is
    an audit quarantine and is never loaded by the trainer, policy, UI, or validator safety path.
    """
    from agent_avenue.rollout.latent import rollout_transition_v1
    from agent_avenue.rollout.targets import _select_continuation_action, wrap_continuation_policy

    selected = {
        stratum: next(position for position in positions if position.stratum == stratum)
        for stratum in PANEL_STRATA
    }
    rows: list[dict[str, object]] = []
    for stratum, position in selected.items():
        candidate = position.observation.legal_actions[0]
        for world in sample_position_worlds(position):
            state = world.state
            steps = [
                {
                    "ordinal": 0,
                    "forced_root": True,
                    "action": _action_data(candidate),
                    "state_before": _state_data(state),
                }
            ]
            state = rollout_transition_v1(state, candidate)
            steps[-1]["state_after"] = _state_data(state)
            ordinal = 1
            while state.phase.value != "terminal" and ordinal < 9:
                actor = (
                    state.active_player
                    if state.phase.value == "play"
                    else state.active_player.other()
                )
                policy = wrap_continuation_policy(continuation[world.policies.policy_for(actor)])
                action, rng_seed = _select_continuation_action(
                    policy,
                    state,
                    root_seed=ROOT_SEED,
                    replicate_id=position.replicate_id,
                    safe_identity=position.safe_identity,
                    world_index=world.world_index,
                )
                row = {
                    "ordinal": ordinal,
                    "forced_root": False,
                    "rng_seed": rng_seed,
                    "action": _action_data(action),
                    "state_before": _state_data(state),
                }
                state = rollout_transition_v1(state, action)
                row["state_after"] = _state_data(state)
                steps.append(row)
                ordinal += 1
            rows.append(
                {
                    "replicate_id": position.replicate_id,
                    "stratum": stratum,
                    "safe_identity": position.safe_identity,
                    "candidate": _action_data(candidate),
                    "world_index": world.world_index,
                    "world_seed": world.seed,
                    "player_one_policy_id": world.policies.player_one_policy_id,
                    "player_two_policy_id": world.policies.player_two_policy_id,
                    "steps": steps,
                    "terminal": state.phase.value == "terminal",
                    "quarantine": "trusted-hidden-rollout-audit-only-v1",
                }
            )
    return tuple(rows)


def _rollout_artifacts(
    output: Path,
    item: Step4Input,
    positions: Sequence[PanelPosition],
    input_audit: Mapping[str, object],
    paths: Mapping[str, Path],
) -> tuple[RolloutTrainingData, dict[str, object]]:
    directory = output / "rollouts" / item.replicate_id
    target_path = directory / "targets.npz"
    continuation, leaf = _continuation_agents(paths)
    if not target_path.exists():
        # Fixed per-stratum shards are independently retained before the combined artifact.  A
        # resume reuses complete shard artifacts; the final combined NPZ is immutable.
        for stratum in PANEL_STRATA:
            shard_positions = tuple(
                position for position in positions if position.stratum == stratum
            )
            shard_path = directory / "shards" / stratum / "targets.npz"
            if not shard_path.exists():
                shard = generate_rollout_targets(
                    shard_positions,
                    continuation_policies=continuation,
                    leaf_scorer=leaf,
                    enforce_depth_leaf_gate=False,
                )
                save_rollout_targets(
                    shard,
                    shard_path,
                    input_identities={
                        "input_audit": input_audit["artifact_fingerprint"],
                        "replicate": item.replicate_id,
                        "shard": stratum,
                    },
                )
        shard_paths = tuple(
            directory / "shards" / stratum / "targets.npz" for stratum in PANEL_STRATA
        )
        assemble_rollout_target_shards(
            shard_paths,
            tuple(canonical_panel_order(positions)),
            target_path,
            input_identities={
                "input_audit": input_audit["artifact_fingerprint"],
                "replicate": item.replicate_id,
            },
        )
    loaded = load_rollout_targets(target_path)
    expected_positions = len(positions)
    counts = cast(Mapping[str, object], loaded.manifest["counts"])
    count_positions = counts.get("positions")
    if not isinstance(count_positions, int) or count_positions != expected_positions:
        raise RolloutExperimentError("resumed rollout target panel cardinality differs")
    samples = [json.loads(str(value)) for value in loaded.sample_json.tolist()]
    leaves = sum(1 for sample_list in samples for sample in sample_list if sample["depth_leaf"])
    total = sum(len(sample_list) for sample_list in samples)
    if total == 0 or leaves / total > 0.5:
        raise RolloutExperimentError("rollout depth-leaf fraction exceeds fixed 50% gate")
    transcripts = _trusted_transcripts(positions, continuation, leaf)
    expected_transcripts = len(PANEL_STRATA) * 10
    if len(transcripts) != expected_transcripts:
        raise RolloutExperimentError(
            "trusted transcript count differs from one-position/stratum/world contract"
        )
    _write_gzip_jsonl(
        directory / "trusted-world-transcripts.jsonl.gz",
        transcripts,
        label="trusted hidden audit transcripts",
    )
    artifact = _artifact(
        "m7-counterfactual-rollout-target-manifest-v1",
        replicate_id=item.replicate_id,
        target_artifact_fingerprint=loaded.fingerprint,
        target_npz_sha256=_sha256(target_path),
        target_schema_manifest_sha256=_sha256(target_path.with_suffix(".json")),
        rows=int(loaded.targets.size),
        positions=expected_positions,
        world_samples=int(loaded.targets.size) * 10,
        depth_leaf_fraction=leaves / total,
        terminal_fraction=1.0 - leaves / total,
        mechanical_cap_errors=0,
        transcript_count=len(transcripts),
        transcript_sha256=_sha256(directory / "trusted-world-transcripts.jsonl.gz"),
        rng_contract=loaded.manifest["rng_contract"],
        continuation_population=loaded.manifest["continuation_population"],
        status="passed",
    )
    _write_immutable(directory / "manifest.json", artifact, label="rollout manifest")
    return RolloutTrainingData(
        loaded.features, loaded.targets, loaded.position_index, expected_positions
    ), artifact


def _training_config(item: Step4Input) -> StructuredTrainingConfig:
    source = item.checkpoint.get("training_config")
    if not isinstance(source, Mapping):
        raise RolloutExperimentError("retained training configuration is malformed")
    return StructuredTrainingConfig(
        seed=cast(int, source["seed"]),
        projection_seed=cast(int, source["projection_seed"]),
        shuffle_seed=cast(int, source["shuffle_seed"]),
        batch_size=cast(int, source["batch_size"]),
        max_epochs=50,
        early_stopping_patience=8,
        learning_rate=float(cast(float, source["learning_rate"])),
        weight_decay=float(cast(float, source["weight_decay"])),
        cpu_threads=1,
        deterministic_algorithms=True,
    )


def _training_metrics(result: RolloutTrainingResult) -> dict[str, object]:
    return {
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "best_validation_loss": result.best_validation_loss,
        "train": result.train_metrics.to_dict(),
        "validation": result.validation_metrics.to_dict(),
        "history": [asdict(row) for row in result.history],
        "runtime": {
            "examples_per_second": result.examples_per_second,
            "wall_clock_seconds": result.wall_clock_seconds,
        },
        "trace": {
            "step_digests": list(result.trace.step_digests),
            "epoch_digests": list(result.trace.epoch_digests),
        },
    }


def _save_checkpoint(
    path: Path,
    result: RolloutTrainingResult,
    *,
    item: Step4Input,
    q0: LoadedCheckpoint,
    dataset: StructuredMaterializedDataset,
    plan: Mapping[str, object],
    arm: str,
    horizon: int,
) -> str:
    saved = save_structured_checkpoint(
        path,
        result.model,
        metrics=_training_metrics(result),
        q0_parent_checkpoint_fingerprint=q0.checkpoint_fingerprint,
        q0_parent_tensor_digest=structured_tensor_digest(q0.model.state_dict()),
        dataset_fingerprint=dataset.fingerprint,
        source_corpus_fingerprints=(cast(str, item.corpus["fingerprint"]),),
        split_identities=cast(Mapping[str, object], dataset.manifest["split"]),
        training_config=result.config.training.normalized(),
        training_seeds={
            "structured_init": result.config.training.effective_projection_seed,
            "shuffle": result.config.training.effective_shuffle_seed,
            "fresh_adamw": True,
        },
        input_manifest_fingerprint=cast(str, plan["input_manifest_fingerprint"]),
        source_rules_fingerprint=rules_fingerprint(),
        metadata={
            "plan_fingerprint": plan["plan_fingerprint"],
            "arm": arm,
            "replicate_id": item.replicate_id,
            "fixed_horizon": horizon,
            "initialization": "selected-q0+retained-step3-M-seeds+fresh-adamw",
        },
        created_at=FIXED_CREATED_AT,
    )
    return saved.checkpoint_fingerprint


def _train_replicate(
    output: Path,
    item: Step4Input,
    root: Path,
    rollout: RolloutTrainingData,
    plan: Mapping[str, object],
    *,
    treatment_epochs: int | None,
) -> dict[str, object]:
    dataset = load_structured_dataset(root / cast(str, item.dataset["path"]))
    train = StructuredTrainingData(
        dataset.train.features, dataset.train.targets, dataset.train.game_index, dataset.train.phase
    )
    validation = StructuredTrainingData(
        dataset.validation.features,
        dataset.validation.targets,
        dataset.validation.game_index,
        dataset.validation.phase,
    )
    q0 = load_checkpoint(cast(Path, plan["q0_path"]))
    horizon = cast(int, item.checkpoint["epochs_completed"])
    if horizon not in (26, 33, 31):
        raise RolloutExperimentError("retained M-v2 horizon is not one of frozen 26/33/31")
    training = _training_config(item)
    control_config = RolloutTreatmentConfig(training, fixed_epochs=horizon, lambda_roll=0.0)
    control = train_rollout_treatment(
        train,
        validation,
        rollout=rollout,
        q0_state_dict=q0.model.state_dict(),
        config=control_config,
    )
    control_digest = structured_tensor_digest(control.model.state_dict())
    expected = item.checkpoint.get("tensor_digest")
    if control_digest != expected:
        raise RolloutExperimentError(
            f"control tensor digest does not reproduce retained M-v2: {item.replicate_id}"
        )
    lambda_zero = lambda_zero_trace_regression(
        train,
        validation,
        rollout=rollout,
        q0_state_dict=q0.model.state_dict(),
        config=control_config,
    )
    if not lambda_zero.passed:
        raise RolloutExperimentError(
            "lambda_roll=0 does not reproduce every control optimizer state"
        )
    control_path = output / "checkpoints" / "control" / item.replicate_id
    if not control_path.exists():
        _save_checkpoint(
            control_path,
            control,
            item=item,
            q0=q0,
            dataset=dataset,
            plan=plan,
            arm="control",
            horizon=horizon,
        )
    treatment_horizon = horizon if treatment_epochs is None else min(horizon, treatment_epochs)
    treatment_config = RolloutTreatmentConfig(
        training, fixed_epochs=treatment_horizon, lambda_roll=0.20
    )
    treatment = train_rollout_treatment(
        train,
        validation,
        rollout=rollout,
        q0_state_dict=q0.model.state_dict(),
        config=treatment_config,
    )
    treatment_path = output / "checkpoints" / "treatment" / item.replicate_id
    if not treatment_path.exists():
        _save_checkpoint(
            treatment_path,
            treatment,
            item=item,
            q0=q0,
            dataset=dataset,
            plan=plan,
            arm="treatment",
            horizon=treatment_horizon,
        )
    reproduction = _artifact(
        "m7-counterfactual-rollout-control-reproduction-v1",
        replicate_id=item.replicate_id,
        retained_tensor_digest=expected,
        reproduced_tensor_digest=control_digest,
        byte_for_byte_tensor_match=True,
        retained_epochs=horizon,
        control_training=_training_metrics(control),
        lambda_zero={
            "passed": lambda_zero.passed,
            "control_step_digests": list(lambda_zero.control.step_digests),
            "treatment_step_digests": list(lambda_zero.treatment.step_digests),
            "control_epoch_digests": list(lambda_zero.control.epoch_digests),
            "treatment_epoch_digests": list(lambda_zero.treatment.epoch_digests),
        },
        status="passed",
    )
    _write_immutable(
        output / "control-reproduction" / item.replicate_id / "summary.json",
        reproduction,
        label="control reproduction",
    )
    control_summary = _artifact(
        "m7-counterfactual-rollout-training-v1",
        arm="control",
        replicate_id=item.replicate_id,
        checkpoint_fingerprint=inspect_structured_checkpoint(control_path).checkpoint_fingerprint,
        tensor_digest=control_digest,
        metrics=_training_metrics(control),
        fixed_horizon=horizon,
    )
    treatment_summary = _artifact(
        "m7-counterfactual-rollout-training-v1",
        arm="treatment",
        replicate_id=item.replicate_id,
        checkpoint_fingerprint=inspect_structured_checkpoint(treatment_path).checkpoint_fingerprint,
        tensor_digest=structured_tensor_digest(treatment.model.state_dict()),
        metrics=_training_metrics(treatment),
        fixed_horizon=treatment_horizon,
        declared_claim_horizon=horizon,
        lambda_roll=0.20,
    )
    _write_immutable(
        output / "training" / "control" / item.replicate_id / "history.json",
        control_summary,
        label="control training",
    )
    _write_immutable(
        output / "training" / "treatment" / item.replicate_id / "history.json",
        treatment_summary,
        label="treatment training",
    )
    return {
        "replicate_id": item.replicate_id,
        "control": control_summary,
        "treatment": treatment_summary,
        "reproduction": reproduction,
    }


def _agent_config(agent: object) -> dict[str, object]:
    source: Any = agent
    value = (
        source.config_to_data()
        if callable(getattr(source, "config_to_data", None))
        else source.config.to_data()
    )
    if not isinstance(value, dict):
        raise RolloutExperimentError("agent has no normalized configuration")
    return cast(dict[str, object], json.loads(_canonical(value)))


def _structured_spec(path: Path, agent_id: str, rng_identity: str) -> AgentSpec:
    from agent_avenue.agents.structured import StructuredValueAgent

    checkpoint = load_structured_checkpoint(path)

    def factory() -> Agent:
        return TerminalOffenseAgent(
            TerminalSafetyAgent(StructuredValueAgent.from_checkpoint(checkpoint))
        )

    return AgentSpec(agent_id, _agent_config(factory()), factory, rng_identity=rng_identity)


def _v1_spec(path: Path, agent_id: str, rng_identity: str, *, enveloped: bool) -> AgentSpec:
    from agent_avenue.agents.learned import LearnedValueAgent

    checkpoint = load_checkpoint(path)

    def factory() -> Agent:
        base: Agent = LearnedValueAgent.from_checkpoint(checkpoint)
        return TerminalOffenseAgent(TerminalSafetyAgent(base)) if enveloped else base

    return AgentSpec(agent_id, _agent_config(factory()), factory, rng_identity=rng_identity)


def _arena_agents(
    cell: RolloutArenaCell, output: Path, paths: Mapping[str, Path]
) -> tuple[AgentSpec, AgentSpec]:
    replicate = cell.replicate_id
    if cell.candidate is None:
        return _v1_spec(
            paths["q0-parent"],
            "q0-parent",
            f"step4:{replicate}:heuristic:candidate",
            enveloped=True,
        ), AgentSpec(
            "heuristic",
            _agent_config(GreedyHeuristicAgent(GreedyHeuristicConfig())),
            lambda: GreedyHeuristicAgent(GreedyHeuristicConfig()),
            rng_identity=f"step4:{replicate}:heuristic:opponent",
        )
    candidate_path = (
        output / "checkpoints" / ("treatment" if cell.candidate == "T" else "control") / replicate
    )
    left = _structured_spec(
        candidate_path,
        f"{cell.candidate}-{replicate}",
        f"step4:{replicate}:{cell.shared_group}:candidate",
    )
    if cell.opponent == "C":
        right = _structured_spec(
            output / "checkpoints" / "control" / replicate,
            f"C-{replicate}",
            f"step4:{replicate}:direct:control",
        )
    elif cell.opponent == "heuristic":
        right = AgentSpec(
            "heuristic",
            _agent_config(GreedyHeuristicAgent(GreedyHeuristicConfig())),
            lambda: GreedyHeuristicAgent(GreedyHeuristicConfig()),
            rng_identity=f"step4:{replicate}:heuristic:opponent",
        )
    elif cell.opponent == "random":
        right = AgentSpec(
            "random",
            _agent_config(RandomAgent(RandomAgentConfig())),
            lambda: RandomAgent(RandomAgentConfig()),
            rng_identity=f"step4:{replicate}:random:opponent",
        )
    else:
        right = _v1_spec(
            paths[cell.opponent],
            cell.opponent,
            f"step4:{replicate}:{cell.opponent}:opponent",
            enveloped=cell.opponent != "historical-q0",
        )
    return left, right


def _tactical(records: Sequence[GameRecord], corpus_fingerprint: str) -> dict[str, object]:
    safety = audit_terminal_safety(
        records, source_corpus_fingerprint=corpus_fingerprint, verify_code=False
    )
    offense = audit_public_forced_wins(
        records, source_label=corpus_fingerprint, verify_records=False
    )
    checked: dict[str, object] = {}
    for agent_id, value in cast(Mapping[str, object], safety["by_agent"]).items():
        if (
            not isinstance(value, Mapping)
            or not isinstance(value.get("config"), Mapping)
            or value["config"].get("type") != "terminal_offense"
        ):
            continue
        forced = cast(
            Mapping[str, object], cast(Mapping[str, object], offense["by_agent"])[agent_id]
        )
        losses = cast(Mapping[str, object], value["counts"])["executed_avoidable_provable_losses"]
        missed = cast(Mapping[str, object], forced["counts"])["missed_forced_wins"]
        checked[agent_id] = {
            "avoidable_losses": losses,
            "missed_guaranteed_wins": missed,
            "passed": losses == 0 and missed == 0,
        }
    return {
        "checked_agents": checked,
        "safety": safety,
        "offense": offense,
        "passed": bool(checked)
        and all(cast(Mapping[str, object], value)["passed"] is True for value in checked.values()),
    }


def _arena_artifact(
    output: Path, plan: Mapping[str, object], cell: RolloutArenaCell, paths: Mapping[str, Path]
) -> dict[str, object]:
    directory, artifact_path = (
        output / "arenas" / cell.key,
        output / "arenas" / cell.key / "report.json",
    )
    left, right = _arena_agents(cell, output, paths)
    arena = ArenaConfig(cell.run_id, left, right, cell.pair_count, cell.master_seed)
    if artifact_path.exists():
        artifact = _read(artifact_path)
        manifest, records = load_corpus(directory / "records", verify_code=False)
        report = artifact.get("report")
        if (
            not isinstance(report, Mapping)
            or arena_report_from_records(
                arena, records, elapsed_seconds=float(report["elapsed_seconds"])
            ).to_data()
            != report
        ):
            raise RolloutExperimentError(f"resumed arena report does not reconstruct: {cell.key}")
        if artifact.get("tactical") != _tactical(records, manifest.corpus_fingerprint):
            raise RolloutExperimentError(f"resumed arena tactical report differs: {cell.key}")
        return artifact
    retained = run_resumable_arena(
        directory / "records",
        arena,
        generation=4,
        corpus_configuration={
            "version": "m7-counterfactual-rollout-arena-v1",
            "plan_fingerprint": plan["plan_fingerprint"],
            "cell": cell.to_data(),
        },
    )
    manifest, records = load_corpus(directory / "records")
    artifact = _artifact(
        "m7-counterfactual-rollout-arena-v1",
        plan_fingerprint=plan["plan_fingerprint"],
        cell=cell.to_data(),
        records={
            "corpus_fingerprint": manifest.corpus_fingerprint,
            "locator": _relative(directory / "records"),
        },
        report=retained.report.to_data(),
        tactical=_tactical(records, manifest.corpus_fingerprint),
    )
    _write_immutable(artifact_path, artifact, label="arena report")
    return artifact


def _pair_scores(artifact: Mapping[str, object]) -> tuple[float, ...]:
    report = cast(Mapping[str, object], artifact["report"])
    outcomes = report.get("paired_seed_outcomes")
    if not isinstance(outcomes, list):
        raise RolloutExperimentError("arena lacks paired outcomes")
    return tuple(cast(int, cast(Mapping[str, object], row)["agent_a_wins"]) / 2 for row in outcomes)


def _nested(rows: tuple[tuple[float, ...], ...], domain: str) -> dict[str, object]:
    from agent_avenue.agents import DeterministicRandom

    if len(rows) != 3 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise RolloutExperimentError(
            "nested statistic needs three equally-sized nonempty replicate rows"
        )
    count, seed = len(rows[0]), derive_seed(ROOT_SEED, f"step4:nested-bootstrap:{domain}")
    rng = DeterministicRandom(seed, f"step4:nested-bootstrap:{domain}")
    values: list[float] = []
    for _ in range(NESTED_RESAMPLES):
        outer = tuple(rng.randbelow(3) for _ in range(3))
        values.append(
            sum(
                sum(rows[index][rng.randbelow(count)] for _ in range(count)) / count
                for index in outer
            )
            / 3
        )
    values.sort()
    point = sum(sum(row) / count for row in rows) / 3
    return {
        "version": "m7-counterfactual-rollout-nested-bootstrap-v1",
        "unit": "training-replicate-then-paired-block",
        "resamples": NESTED_RESAMPLES,
        "outer_draws_materialized_first": True,
        "order_statistic_indices": {"lower": NESTED_LOWER, "upper": NESTED_UPPER},
        "point_estimate": point,
        "interval": [values[NESTED_LOWER], values[NESTED_UPPER]],
        "seed": seed,
        "domain": domain,
        "blocks_per_replicate": count,
    }


def _aligned_difference(
    left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...], domain: str
) -> dict[str, object]:
    from agent_avenue.agents import DeterministicRandom

    if (
        len(left) != 3
        or len(right) != 3
        or not left[0]
        or any(len(a) != len(b) or len(a) != len(left[0]) for a, b in zip(left, right, strict=True))
    ):
        raise RolloutExperimentError("aligned heuristic contrast needs matched block rows")
    count, seed = len(left[0]), derive_seed(ROOT_SEED, f"step4:nested-bootstrap:{domain}")
    rng = DeterministicRandom(seed, f"step4:nested-bootstrap:{domain}")
    values: list[float] = []
    for _ in range(NESTED_RESAMPLES):
        outer = tuple(rng.randbelow(3) for _ in range(3))
        per_replicate = []
        for replicate in outer:
            indexes = tuple(rng.randbelow(count) for _ in range(count))
            per_replicate.append(
                sum(left[replicate][index] - right[replicate][index] for index in indexes) / count
            )
        values.append(sum(per_replicate) / 3)
    values.sort()
    point = sum((sum(a) / count - sum(b) / count) for a, b in zip(left, right, strict=True)) / 3
    return {
        "version": "m7-counterfactual-rollout-aligned-nested-bootstrap-v1",
        "unit": "training-replicate-then-identically-aligned-paired-block",
        "resamples": NESTED_RESAMPLES,
        "outer_draws_materialized_first": True,
        "same_inner_block_indexes_for_contrast": True,
        "order_statistic_indices": {"lower": NESTED_LOWER, "upper": NESTED_UPPER},
        "point_estimate": point,
        "interval": [values[NESTED_LOWER], values[NESTED_UPPER]],
        "seed": seed,
        "domain": domain,
        "blocks_per_replicate": count,
    }


def _statistics(
    cells: Sequence[RolloutArenaCell], arenas: Mapping[str, Mapping[str, object]], *, smoke: bool
) -> dict[str, object]:
    if smoke:
        direct = _pair_scores(arenas["T1-vs-C1"])
        return _artifact(
            "m7-counterfactual-rollout-statistics-v1",
            smoke_only=True,
            direct_treatment_vs_control={
                "point_estimate": sum(direct) / len(direct),
                "blocks": len(direct),
            },
            nested={},
            equal_macro_descriptive=None,
        )
    by_key = arenas
    rows: dict[str, list[tuple[float, ...]]] = {
        "direct": [],
        "parent": [],
        "random": [],
        "heur_t": [],
        "heur_c": [],
        "heur_q0": [],
    }
    macro_t: list[float] = []
    macro_c: list[float] = []
    seats: list[float] = []
    tactical_passed = True
    for replicate in REPLICATES:
        index = replicate.removeprefix("replicate-")
        direct = _pair_scores(by_key[f"T{index}-vs-C{index}"])
        parent = _pair_scores(by_key[f"T{index}-vs-q0-parent"])
        random = _pair_scores(by_key[f"T{index}-vs-random"])
        heur_t, heur_c, heur_q0 = (
            _pair_scores(by_key[f"{candidate}{index}-vs-heuristic"]) for candidate in ("T", "C")
        )
        heur_q0 = _pair_scores(by_key[f"q0-parent-vs-heuristic-reference-{index}"])
        rows["direct"].append(direct)
        rows["parent"].append(parent)
        rows["random"].append(random)
        rows["heur_t"].append(heur_t)
        rows["heur_c"].append(heur_c)
        rows["heur_q0"].append(heur_q0)
        opponents = ("q0-parent", "heuristic", "random", "historical-q0", "q1", "q2", "q3", "q4")
        macro_t.append(
            sum(
                cast(
                    float,
                    cast(Mapping[str, object], by_key[f"T{index}-vs-{opponent}"]["report"])[
                        "agent_a_win_rate"
                    ],
                )
                for opponent in opponents
            )
            / len(opponents)
        )
        macro_c.append(
            sum(
                cast(
                    float,
                    cast(Mapping[str, object], by_key[f"C{index}-vs-{opponent}"]["report"])[
                        "agent_a_win_rate"
                    ],
                )
                for opponent in opponents
            )
            / len(opponents)
        )
        for opponent in (f"C{index}", "q0-parent", "heuristic", "random"):
            key = f"T{index}-vs-{opponent}"
            report = cast(Mapping[str, object], by_key[key]["report"])
            seats.extend(
                cast(
                    float,
                    cast(Mapping[str, Mapping[str, object]], report["agent_a_by_seat"])[seat][
                        "win_rate"
                    ],
                )
                for seat in ("player_one", "player_two")
            )
        tactical_passed = tactical_passed and all(
            cast(Mapping[str, object], value["tactical"])["passed"] is True
            for key, value in by_key.items()
            if key.startswith(f"T{index}-")
        )
    nested = {
        "treatment_vs_control": _nested(tuple(rows["direct"]), "treatment-vs-control"),
        "treatment_vs_q0_parent": _nested(tuple(rows["parent"]), "treatment-vs-q0-parent"),
        "treatment_vs_random": _nested(tuple(rows["random"]), "treatment-vs-random"),
        "treatment_minus_q0_parent_heuristic": _aligned_difference(
            tuple(rows["heur_t"]), tuple(rows["heur_q0"]), "treatment-minus-q0-parent-heuristic"
        ),
        "control_minus_q0_parent_heuristic": _aligned_difference(
            tuple(rows["heur_c"]), tuple(rows["heur_q0"]), "control-minus-q0-parent-heuristic"
        ),
    }
    return _artifact(
        "m7-counterfactual-rollout-statistics-v1",
        smoke_only=False,
        bootstrap_contract={
            "root_seed": ROOT_SEED,
            "outer_draws_materialized_before_inner": True,
            "heuristic_contrasts_use_identical_aligned_block_indexes": True,
        },
        nested=nested,
        direct_replicate_points=[sum(row) / len(row) for row in rows["direct"]],
        candidate_treatment_seat_floor=min(seats),
        tactical_passed=tactical_passed,
        equal_macro_descriptive={
            "treatment": sum(macro_t) / 3,
            "control": sum(macro_c) / 3,
            "per_replicate_treatment": macro_t,
            "per_replicate_control": macro_c,
        },
    )


def _selection(
    statistics: Mapping[str, object], *, integrity: bool, smoke: bool
) -> dict[str, object]:
    if smoke:
        return _artifact(
            "m7-counterfactual-rollout-selection-v1",
            classification="nonclaim_smoke_no_disposition",
            step5_disposition="nonclaim; no advancement disposition",
            rollout_treatment_enters_step5=False,
            step3_M_descriptive_step5_entry=None,
            conditions={"nonclaim_smoke": True},
        )
    nested = cast(Mapping[str, Mapping[str, object]], statistics["nested"])
    conditions = {
        "treatment_vs_control_lower_strictly_above_50": cast(
            list[float], nested["treatment_vs_control"]["interval"]
        )[0]
        > 0.5,
        "two_of_three_direct_replicates_above_50": sum(
            value > 0.5 for value in cast(list[float], statistics["direct_replicate_points"])
        )
        >= 2,
        "treatment_vs_q0_parent_lower_above_50": cast(
            list[float], nested["treatment_vs_q0_parent"]["interval"]
        )[0]
        > 0.5,
        "treatment_vs_random_lower_above_50": cast(
            list[float], nested["treatment_vs_random"]["interval"]
        )[0]
        > 0.5,
        "heuristic_aligned_lower_strictly_above_minus_5pp": cast(
            list[float], nested["treatment_minus_q0_parent_heuristic"]["interval"]
        )[0]
        > -0.05,
        "candidate_treatment_seat_floor_at_least_45": cast(
            float, statistics["candidate_treatment_seat_floor"]
        )
        >= 0.45,
        "tactical_invariants": statistics.get("tactical_passed") is True,
        "source_split_panel_sampler_rollout_schedule_validator_integrity": integrity,
    }
    advances = all(conditions.values())
    direct_lower = cast(list[float], nested["treatment_vs_control"]["interval"])[0]
    classification = (
        "advances_step5_recipe_family"
        if advances
        else ("does_not_advance" if direct_lower <= 0.5 else "inconclusive_does_not_advance")
    )
    return _artifact(
        "m7-counterfactual-rollout-selection-v1",
        classification=classification,
        conditions=conditions,
        rollout_treatment_enters_step5=advances,
        step5_disposition=(
            "all three rollout treatments enter Step-5 as one family; "
            "controls remain paired references"
            if advances
            else (
                "rollout treatment does not advance; retained Step-3 M-v2 remains "
                "the descriptive Step-5 entry"
            )
        ),
        step5_entries=(
            ["M-v2-rollout-treatment-family:T1,T2,T3"]
            if advances
            else ["retained-Step3-M-v2-descriptive"]
        ),
        equal_macro_descriptive=statistics.get("equal_macro_descriptive"),
    )


def _checksums(output: Path) -> dict[str, object]:
    excluded = {
        "result.json",
        "execution-state.json",
        "checksums.json",
        "validation.json",
        "driver.stdout",
        "driver.stderr",
        "wrapper-exit.json",
        "validation.stdout",
        "validation.stderr",
    }
    entries: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in excluded or path.name == ".lock":
            continue
        entries[path.relative_to(output).as_posix()] = _sha256(path)
    return _artifact("m7-counterfactual-rollout-checksums-v1", files=entries)


def _write_mutable(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(dict(value)) + b"\n"
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


def _execution_fingerprint(value: Mapping[str, object]) -> str:
    return _fingerprint({key: item for key, item in value.items() if key != "artifact_fingerprint"})


def _load_execution(path: Path) -> dict[str, object]:
    value = _read(path)
    if value.get("version") != EXECUTION_STATE_VERSION or value.get(
        "artifact_fingerprint"
    ) != _execution_fingerprint(value):
        raise RolloutExperimentError("execution state is malformed")
    return value


def _begin_execution(output: Path, plan: Mapping[str, object]) -> Path:
    path = output / "execution-state.json"
    if not path.exists():
        state = {
            "version": EXECUTION_STATE_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "claim": cast(Mapping[str, object], plan["execution"])["claim"],
            "started_at_epoch_seconds": time.time(),
            "attempts": [{"attempt": 1, "started_at_epoch_seconds": time.time()}],
            "phase_timings_seconds": {},
            "completed": False,
        }
    else:
        state = _load_execution(path)
        if state.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise RolloutExperimentError("resume state belongs to a different plan")
        if state.get("completed") is True:
            return path
        attempts = cast(list[object], state["attempts"])
        if len(attempts) >= 2:
            raise RolloutExperimentError("only one Step-4 resume/repair attempt is authorized")
        attempts.append({"attempt": 2, "started_at_epoch_seconds": time.time()})
        state["attempts"] = attempts
    state["artifact_fingerprint"] = _execution_fingerprint(state)
    _write_mutable(path, state)
    return path


@contextmanager
def _timed(path: Path, stage: str, *, claim: bool) -> Iterator[None]:
    state = _load_execution(path)
    if (
        claim
        and time.time() - _number(state["started_at_epoch_seconds"], label="execution start")
        >= CLAIM_CUTOFF_SECONDS
    ):
        raise RolloutExperimentError("Step-4 claim cutoff reached before " + stage)
    started = time.perf_counter()
    try:
        yield
    finally:
        state = _load_execution(path)
        timings = dict(cast(Mapping[str, object], state["phase_timings_seconds"]))
        previous = _number(timings.get(stage, 0.0), label="phase timing")
        timings[stage] = previous + time.perf_counter() - started
        state["phase_timings_seconds"] = timings
        state["artifact_fingerprint"] = _execution_fingerprint(state)
        _write_mutable(path, state)
        if (
            claim
            and time.time() - _number(state["started_at_epoch_seconds"], label="execution start")
            >= CLAIM_CUTOFF_SECONDS
        ):
            raise RolloutExperimentError("Step-4 claim cutoff reached after " + stage)


def _source_identity() -> dict[str, object]:
    source = inspect_source_identity()
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"),
        cwd=repository_root(),
        check=True,
        capture_output=True,
    ).stdout
    return {
        **source.to_data(),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
        "runner_clean_check": {
            "tracked_and_nonignored_untracked_clean": not bool(status.strip()),
            "status_sha256": hashlib.sha256(status).hexdigest(),
        },
    }


def _exact_claim_config(config: RolloutExperimentConfig) -> bool:
    root = repository_root()
    return (
        config.claim
        and config.output.resolve() == (root / "runs" / CYCLE_ID).resolve()
        and config.step3_root.resolve() == (root / "runs" / "m7-structured-model-v2").resolve()
        and tuple(path.resolve() for path in config.holdout_roots) == ((root / "runs").resolve(),)
    )


def build_rollout_experiment_plan(config: RolloutExperimentConfig) -> dict[str, object]:
    manifest, inputs, input_audit = verify_step3_inputs(config)
    source, cells = _source_identity(), _cells(config)
    _check_cardinality(cells, claim=config.claim)
    reasons: list[str] = []
    if not config.claim:
        reasons.append("bounded_nonclaim_smoke")
    if config.claim and not _exact_claim_config(config):
        reasons.append("claim_paths_must_be_exact_frozen_defaults")
    if (
        source["tracked_tree_clean"] is not True
        or cast(Mapping[str, object], source["runner_clean_check"])[
            "tracked_and_nonignored_untracked_clean"
        ]
        is not True
    ):
        reasons.append("source_not_clean")
    plan: dict[str, object] = {
        "version": PLAN_VERSION,
        "cycle_id": CYCLE_ID,
        "root_seed": ROOT_SEED,
        "source": source,
        "input_manifest_fingerprint": manifest["artifact_fingerprint"],
        "frozen_input_audit": input_audit,
        "step3_root": str(config.step3_root.resolve()),
        "inputs": [
            {
                "replicate_id": item.replicate_id,
                "corpus": dict(item.corpus),
                "dataset": dict(item.dataset),
                "checkpoint": dict(item.checkpoint),
            }
            for item in inputs
        ],
        "q0_path": str(_continuation_paths(_source_root(config.step3_root), manifest)["q0-parent"]),
        "opponent_paths": {
            key: str(path)
            for key, path in _continuation_paths(_source_root(config.step3_root), manifest).items()
        },
        "rng_contract": {
            "version": "counterfactual-rng-contract-v1",
            "seed_derivation": "sha256-domain-v1",
            "rng_algorithm": "sha256-counter-rejection-v1",
            "expanded_deck": "EXPANDED_CANONICAL_DECK",
            "shuffle": "reverse-fisher-yates-v1",
        },
        "execution": {
            "claim": config.claim,
            "claim_eligible": not reasons,
            "ineligibility_reasons": reasons,
            "panel_quota_per_stratum": config.panel_quota,
            "treatment_epochs": config.treatment_epochs,
            "claim_cutoff_seconds": CLAIM_CUTOFF_SECONDS,
            "hard_budget_seconds": HARD_BUDGET_SECONDS,
        },
        "holdout_scope": {
            "roots": [str(path.resolve()) for path in config.holdout_roots],
            "recursive": True,
            "excludes_current_output": True,
        },
        "arenas": {
            "cells": [cell.to_data() for cell in cells],
            "total_physical_games": sum(cell.pair_count * 2 for cell in cells),
            "frozen_default_total_physical_games": 24_000,
        },
        "plan_fingerprint": "",
    }
    plan["plan_fingerprint"] = _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    )
    return plan


def validate_completed_rollout_result(
    output: Path, plan: Mapping[str, object], cells: Sequence[RolloutArenaCell]
) -> dict[str, object]:
    result = _read(output / "result.json")
    if (
        result.get("plan_fingerprint") != plan["plan_fingerprint"]
        or result.get("status") != "completed"
        or result.get("result_fingerprint")
        != _fingerprint(
            {key: value for key, value in result.items() if key != "result_fingerprint"}
        )
    ):
        raise RolloutExperimentError("existing Step-4 result is malformed")
    if set(cast(Mapping[str, object], result["arenas"])) != {cell.key for cell in cells}:
        raise RolloutExperimentError("existing Step-4 arena references are incomplete")
    checksums = _read(output / "checksums.json")
    if checksums != _checksums(output):
        raise RolloutExperimentError("completed Step-4 checksum manifest no longer matches")
    return result


def run_rollout_experiment(config: RolloutExperimentConfig) -> dict[str, object]:
    """Run/resume the complete frozen Step-4 pipeline or the explicit nonclaim smoke."""
    plan = build_rollout_experiment_plan(config)
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable(output / "plan.json", plan, label="plan")
    if config.claim and cast(Mapping[str, object], plan["execution"])["claim_eligible"] is not True:
        raise RolloutExperimentError("claim dispatch blocked by frozen source/path preflight")
    cells = _cells(config)
    if (output / "result.json").exists():
        return validate_completed_rollout_result(output, plan, cells)
    state_path = _begin_execution(output, plan)
    with _timed(state_path, "input-holdout", claim=config.claim):
        _manifest, inputs, input_audit = verify_step3_inputs(config)
        holdout = _holdout_scan(config, cells)
        _write_immutable(
            output / "inputs" / "step3-M-reference.json",
            _artifact(
                "m7-counterfactual-rollout-step3-reference-v1",
                plan_fingerprint=plan["plan_fingerprint"],
                frozen_inputs=input_audit,
                role_map={
                    item.replicate_id: {
                        "q0-parent": "leaf-and-external-reference",
                        "q1-q4": "continuation-and-external-reference",
                        "heuristic": "continuation-and-external-reference",
                        "random": "continuation-and-external-reference",
                        "historical-q0": "external-reference",
                        "retained-M-v2": item.checkpoint["tensor_digest"],
                        "reproduced-M-v2-matched-control": "checkpoints/control/"
                        + item.replicate_id,
                        "M-v2-rollout-treatment": "checkpoints/treatment/" + item.replicate_id,
                    }
                    for item in inputs
                },
                setup_holdout=holdout,
            ),
            label="Step-3 M role map",
        )
        _write_immutable(
            output / "source-identity.json",
            cast(Mapping[str, object], plan["source"]),
            label="source identity",
        )
    if holdout["status"] != "passed":
        raise RolloutExperimentError("fresh arena holdout overlaps prior repository runs")
    root, paths = (
        _source_root(config.step3_root),
        {
            key: Path(value)
            for key, value in cast(Mapping[str, str], plan["opponent_paths"]).items()
        },
    )
    panels: dict[str, tuple[PanelPosition, ...]] = {}
    panel_manifests: dict[str, Mapping[str, object]] = {}
    for item in inputs:
        with _timed(state_path, f"panel:{item.replicate_id}", claim=config.claim):
            panel, artifact = _panel_artifacts(output, item, root, quota=config.panel_quota)
        panels[item.replicate_id], panel_manifests[item.replicate_id] = panel, artifact
    rollouts: dict[str, RolloutTrainingData] = {}
    rollout_manifests: dict[str, Mapping[str, object]] = {}
    for item in inputs:
        with _timed(state_path, f"rollouts:{item.replicate_id}", claim=config.claim):
            data, artifact = _rollout_artifacts(
                output, item, panels[item.replicate_id], input_audit, paths
            )
        rollouts[item.replicate_id], rollout_manifests[item.replicate_id] = data, artifact
    if any(
        cast(float, value["depth_leaf_fraction"]) > 0.5 or value["mechanical_cap_errors"] != 0
        for value in rollout_manifests.values()
    ):
        raise RolloutExperimentError("rollout integrity eligibility gate failed")
    training_rows: list[dict[str, object]] = []
    for item in inputs:
        with _timed(state_path, f"training:{item.replicate_id}", claim=config.claim):
            training_rows.append(
                _train_replicate(
                    output,
                    item,
                    root,
                    rollouts[item.replicate_id],
                    plan,
                    treatment_epochs=config.treatment_epochs,
                )
            )
    training = _artifact(
        "m7-counterfactual-rollout-training-summary-v1",
        plan_fingerprint=plan["plan_fingerprint"],
        replicates=training_rows,
    )
    _write_immutable(output / "training" / "summary.json", training, label="training summary")
    arenas: dict[str, Mapping[str, object]] = {}
    for cell in cells:
        with _timed(state_path, f"arena:{cell.key}", claim=config.claim):
            arenas[cell.key] = _arena_artifact(output, plan, cell, paths)
    with _timed(state_path, "statistics", claim=config.claim):
        statistics = _statistics(cells, arenas, smoke=config.is_smoke)
        _write_immutable(output / "statistics.json", statistics, label="statistics")
        integrity = (
            all(
                cast(Mapping[str, object], row["reproduction"])["byte_for_byte_tensor_match"]
                is True
                for row in training_rows
            )
            and all(value["status"] == "passed" for value in panel_manifests.values())
            and all(value["status"] == "passed" for value in rollout_manifests.values())
        )
        smoke_arena = arenas["T1-vs-C1"]
        smoke_tactical = cast(Mapping[str, object], smoke_arena["tactical"])
        treatment_tactical_passed = (
            statistics.get("tactical_passed")
            if not config.is_smoke
            else smoke_tactical.get("passed")
        )
        safety = _artifact(
            "m7-counterfactual-rollout-safety-v1",
            plan_fingerprint=plan["plan_fingerprint"],
            treatment_tactical_passed=treatment_tactical_passed,
            source_split_panel_sampler_rollout_integrity=integrity,
        )
        _write_immutable(output / "safety-report.json", safety, label="safety report")
        selection = _selection(
            statistics,
            integrity=integrity and safety["treatment_tactical_passed"] is True,
            smoke=config.is_smoke,
        )
        _write_immutable(output / "selection.json", selection, label="selection")
    state = _load_execution(state_path)
    state["completed"] = True
    state["completed_at_epoch_seconds"] = time.time()
    state["artifact_fingerprint"] = _execution_fingerprint(state)
    _write_mutable(state_path, state)
    timing = cast(Mapping[str, object], state["phase_timings_seconds"])
    timing_values = {
        key: _number(value, label=f"phase timing {key}") for key, value in timing.items()
    }
    target_units = sum(
        _integer(item["rows"], label="rollout row count") * 10
        for item in rollout_manifests.values()
    )
    elapsed_target = sum(
        value for key, value in timing_values.items() if key.startswith("rollouts:")
    )
    units_per_second = target_units / elapsed_target if elapsed_target else float("inf")
    projection = _artifact(
        "m7-counterfactual-rollout-runtime-v1",
        mode="claim" if config.claim else "nonclaim-smoke",
        phase_timings_seconds=timing_values,
        rollout_work_units=target_units,
        rollout_units_per_second=units_per_second,
        required_minimum_units_per_second=67,
        leaf_fraction=max(
            cast(float, value["depth_leaf_fraction"]) for value in rollout_manifests.values()
        ),
        mechanical_cap_errors=0,
        projected_full_minutes=(
            sum(timing_values.values()) / 60
            if config.claim
            else (sum(timing_values.values()) * 20 / 60)
        ),
        claim_cutoff_minutes=465,
        claim_eligible_runtime=(
            config.claim
            or (
                units_per_second >= 67
                and max(
                    cast(float, value["depth_leaf_fraction"])
                    for value in rollout_manifests.values()
                )
                <= 0.5
            )
        ),
    )
    _write_immutable(output / "runtime-extrapolation.json", projection, label="runtime")
    checksums = _checksums(output)
    _write_immutable(output / "checksums.json", checksums, label="checksums")
    result = {
        "version": "m7-counterfactual-rollout-result-v1",
        "cycle_id": CYCLE_ID,
        "status": "completed",
        "plan_fingerprint": plan["plan_fingerprint"],
        "input_manifest_fingerprint": plan["input_manifest_fingerprint"],
        "panels": {key: value["artifact_fingerprint"] for key, value in panel_manifests.items()},
        "rollouts": {
            key: value["artifact_fingerprint"] for key, value in rollout_manifests.items()
        },
        "training_fingerprint": training["artifact_fingerprint"],
        "arenas": {key: value["artifact_fingerprint"] for key, value in arenas.items()},
        "statistics_fingerprint": statistics["artifact_fingerprint"],
        "safety_fingerprint": safety["artifact_fingerprint"],
        "selection_fingerprint": selection["artifact_fingerprint"],
        "runtime_fingerprint": projection["artifact_fingerprint"],
        "checksums_fingerprint": checksums["artifact_fingerprint"],
        "result_fingerprint": "",
    }
    result["result_fingerprint"] = _fingerprint(
        {key: value for key, value in result.items() if key != "result_fingerprint"}
    )
    _write_immutable(output / "result.json", result, label="result")
    return result


__all__ = [
    "CYCLE_ID",
    "RolloutArenaCell",
    "RolloutExperimentConfig",
    "RolloutExperimentError",
    "build_rollout_experiment_plan",
    "run_rollout_experiment",
    "validate_completed_rollout_result",
    "verify_step3_inputs",
]
