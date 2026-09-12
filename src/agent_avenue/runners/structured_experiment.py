"""Resumable runner for the frozen M7 structured-model v2 Step-3 experiment.

The runner deliberately treats the Step-2 archive as immutable external input.  It never regenerates
v1 controls or behaviour records: it verifies their committed full-file identities, replays them
under current rules through the narrow historical-compatibility path, and emits v2-only evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

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
from agent_avenue.encoding.candidate_structured_v2 import FEATURE_SCHEMA
from agent_avenue.engine.setup import normalize_config
from agent_avenue.learning.structured_model import create_structured_model
from agent_avenue.observation import observe
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
    game_record_fingerprint,
    inspect_source_identity,
    load_corpus,
    repository_root,
    rules_fingerprint,
)

CYCLE_ID: Final = "m7-structured-model-v2"
PLAN_VERSION: Final = "m7-structured-model-v2-plan-v1"
INPUT_AUDIT_VERSION: Final = "m7-structured-model-v2-input-audit-v1"
DATASET_AUDIT_VERSION: Final = "m7-structured-model-v2-dataset-audit-v1"
TRAINING_VERSION: Final = "m7-structured-model-v2-training-summary-v1"
ARENA_VERSION: Final = "m7-structured-model-v2-arena-v1"
STATISTICS_VERSION: Final = "m7-structured-model-v2-statistics-v1"
SAFETY_VERSION: Final = "m7-structured-model-v2-safety-v1"
SELECTION_VERSION: Final = "m7-structured-model-v2-selection-v1"
RESULT_VERSION: Final = "m7-structured-model-v2-result-v1"
ROOT_SEED: Final = 2026091203
REPLICATES: Final[tuple[str, str, str]] = ("replicate-1", "replicate-2", "replicate-3")
ARMS: Final[tuple[str, str]] = ("C", "M")
ARM_LABELS: Final[dict[str, str]] = {"C": "q0-only-control", "M": "mixed-population"}
MANIFEST_ARM: Final[dict[str, str]] = {"C": "control", "M": "treatment"}
NESTED_RESAMPLES: Final = 20_000
NESTED_LOWER: Final = 499
NESTED_UPPER: Final = 19_499
CLAIM_CUTOFF_SECONDS: Final = 7 * 60 * 60 + 45 * 60
HARD_BUDGET_SECONDS: Final = 8 * 60 * 60
ALLOWED_RESUME_RETRIES: Final = 1
EXECUTION_STATE_VERSION: Final = "m7-structured-model-v2-execution-state-v2"
RUNTIME_VERSION: Final = "m7-structured-model-v2-runtime-extrapolation-v1"
FIXED_CREATED_AT: Final = "2026-09-12T00:00:00+00:00"
STEP2_CLAIM_SOURCE: Final = "fde19b5d3c327e539c29913a973b5e4d76ffff5b"
_STEP2_PROTECTED_PREFIXES: Final[tuple[str, ...]] = (
    "src/agent_avenue/engine/",
    "src/agent_avenue/observation/",
    "src/agent_avenue/storage/game_record.py",
    "src/agent_avenue/storage/replay.py",
    "src/agent_avenue/runners/game.py",
    "src/agent_avenue/runners/arena.py",
    "src/agent_avenue/agents/terminal_offense.py",
    "src/agent_avenue/agents/terminal_safety.py",
    "src/agent_avenue/encoding/candidate_v1.py",
)
OPPONENTS: Final[tuple[str, ...]] = (
    "q0-parent",
    "heuristic",
    "random",
    "historical-q0",
    "q1",
    "q2",
    "q3",
    "q4",
)


class StructuredExperimentError(ValueError):
    """Raised when frozen Step-3 evidence cannot be safely constructed or resumed."""


@dataclass(frozen=True, slots=True)
class Step3Input:
    """One immutable Step-2 corpus/dataset/v1-checkpoint triple."""

    arm: str
    replicate_id: str
    corpus: Mapping[str, object]
    dataset: Mapping[str, object]
    checkpoint: Mapping[str, object]

    @property
    def key(self) -> str:
        return f"{self.arm}{self.replicate_id.removeprefix('replicate-')}"


@dataclass(frozen=True, slots=True)
class StructuredArenaCell:
    """One fresh paired arena; all logical lanes are explicit and stable."""

    replicate_id: str
    key: str
    candidate_arm: str | None
    opponent: str
    pair_count: int
    master_seed: int
    shared_group: str | None
    candidate_kind: str

    @property
    def run_id(self) -> str:
        return f"{CYCLE_ID}-{self.replicate_id}-{self.key}"

    def to_data(self) -> dict[str, object]:
        return {
            "replicate_id": self.replicate_id,
            "key": self.key,
            "run_id": self.run_id,
            "candidate_arm": self.candidate_arm,
            "candidate_kind": self.candidate_kind,
            "opponent": self.opponent,
            "paired_blocks": self.pair_count,
            "games": self.pair_count * 2,
            "master_seed": self.master_seed,
            "shared_group": self.shared_group,
        }


@dataclass(frozen=True, slots=True)
class StructuredExperimentConfig:
    """Explicit paths and bounded-smoke controls for the Step-3 runner.

    ``smoke_pairs`` is intentionally required for bounded evidence.  Claim defaults are available
    only by constructing this config without smoke values; the CLI does not dispatch that mode.
    """

    output: Path
    step2_root: Path
    input_manifest: Path = Path("research/cycles/m7-structured-model-v2-inputs.json")
    holdout_roots: tuple[Path, ...] = field(default_factory=lambda: (repository_root() / "runs",))
    q0_path: Path | None = None
    q_paths: Mapping[str, Path] | None = None
    smoke_pairs: int | None = None
    smoke_max_epochs: int | None = None

    def __post_init__(self) -> None:
        if not self.holdout_roots:
            raise StructuredExperimentError("at least one recursive holdout root is required")
        if self.smoke_pairs is not None and not 1 <= self.smoke_pairs <= 500:
            raise StructuredExperimentError("smoke_pairs must be in [1, 500]")
        if self.smoke_max_epochs is not None and self.smoke_max_epochs < 1:
            raise StructuredExperimentError("smoke_max_epochs must be positive")
        if (self.smoke_pairs is None) != (self.smoke_max_epochs is None):
            raise StructuredExperimentError("bounded smoke must set both pair and epoch limits")

    @property
    def claim_default(self) -> bool:
        return self.smoke_pairs is None

    @property
    def max_epochs(self) -> int:
        return 50 if self.smoke_max_epochs is None else self.smoke_max_epochs

    def pair_count(self, declared: int) -> int:
        return declared if self.smoke_pairs is None else min(declared, self.smoke_pairs)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


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
        raise StructuredExperimentError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise StructuredExperimentError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, object], value)


def _write_immutable(path: Path, value: Mapping[str, object], *, label: str) -> None:
    normalized = cast(dict[str, object], json.loads(_canonical(dict(value))))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path) != normalized:
            raise StructuredExperimentError(f"existing immutable {label} differs: {path}")
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


def _number(value: object, *, label: str) -> float:
    if not isinstance(value, int | float):
        raise StructuredExperimentError(f"{label} must be numeric")
    return float(value)


def _artifact(version: str, **values: object) -> dict[str, object]:
    result = {"version": version, **values}
    result["artifact_fingerprint"] = _fingerprint(result)
    return result


def _repository_for_step2(step2_root: Path) -> Path:
    root = step2_root.resolve()
    if root.name != "m7-population-replay-v1" or root.parent.name != "runs":
        raise StructuredExperimentError(
            "step2_root must be the retained runs/m7-population-replay-v1"
        )
    return root.parent.parent


def _full_source_identity() -> dict[str, object]:
    source = inspect_source_identity()
    root = repository_root()
    try:
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StructuredExperimentError("unable to inspect source cleanliness") from exc
    return {
        **source.to_data(),
        "rules_fingerprint": rules_fingerprint(),
        "code_fingerprint": code_fingerprint(),
        "runner_clean_check": {
            "version": "structured-model-v2-runner-clean-source-v1",
            "tracked_and_nonignored_untracked_clean": not bool(status.strip()),
            "status_sha256": hashlib.sha256(status).hexdigest(),
        },
    }


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root()).as_posix()
    except ValueError:
        return str(path.resolve())


def _manifest_inputs(
    config: StructuredExperimentConfig,
) -> tuple[dict[str, object], tuple[Step3Input, ...]]:
    data = _read(config.input_manifest)
    declared = data.get("artifact_fingerprint")
    if declared != _fingerprint(
        {key: value for key, value in data.items() if key != "artifact_fingerprint"}
    ):
        raise StructuredExperimentError("committed Step-3 input-manifest fingerprint mismatch")
    if data.get("cycle_id") != CYCLE_ID or data.get("rules_fingerprint") != rules_fingerprint():
        raise StructuredExperimentError("Step-3 input manifest cycle or rules identity differs")
    entries = data.get("inputs")
    if not isinstance(entries, list) or len(entries) != 6:
        raise StructuredExperimentError("input manifest must contain six retained Step-2 entries")
    inputs: list[Step3Input] = []
    seen: set[tuple[str, str]] = set()
    for value in entries:
        if not isinstance(value, Mapping):
            raise StructuredExperimentError("input manifest entry is malformed")
        source_arm = value.get("data_arm")
        replicate = value.get("replicate_id")
        arm = "C" if source_arm == "control" else "M" if source_arm == "treatment" else None
        corpus = value.get("corpus")
        dataset = value.get("v1_dataset")
        checkpoint = value.get("v1_checkpoint")
        if (
            arm is None
            or replicate not in REPLICATES
            or not isinstance(corpus, Mapping)
            or not isinstance(dataset, Mapping)
            or not isinstance(checkpoint, Mapping)
            or (arm, replicate) in seen
        ):
            raise StructuredExperimentError("input manifest entry identities are malformed")
        seen.add((arm, replicate))
        inputs.append(Step3Input(arm, replicate, corpus, dataset, checkpoint))
    if seen != {(arm, replicate) for replicate in REPLICATES for arm in ARMS}:
        raise StructuredExperimentError("input manifest lacks one of the six Step-2 controls")
    return data, tuple(sorted(inputs, key=lambda item: (item.replicate_id, item.arm)))


def _expected_digest(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or len(result) != 64:
        raise StructuredExperimentError(f"input manifest {key} is malformed")
    return result


def verify_step2_inputs(
    config: StructuredExperimentConfig,
) -> tuple[dict[str, object], tuple[Step3Input, ...], dict[str, object]]:
    """Verify every frozen full Step-2 input digest before re-encoding any historical record."""
    manifest, inputs = _manifest_inputs(config)
    root = _repository_for_step2(config.step2_root)
    rows: list[dict[str, object]] = []
    for item in inputs:
        corpus_path = root / cast(str, item.corpus["path"])
        dataset_path = root / cast(str, item.dataset["path"])
        checkpoint_path = root / cast(str, item.checkpoint["path"])
        files = {
            "corpus_manifest_sha256": corpus_path / "manifest.json",
            "corpus_records_sha256": corpus_path / "games.jsonl.gz",
            "dataset_npz_sha256": dataset_path,
            "dataset_manifest_sha256": dataset_path.with_suffix(".json"),
            "checkpoint_weights_sha256": checkpoint_path / "weights.pt",
            "checkpoint_manifest_sha256": checkpoint_path / "manifest.json",
            "checkpoint_metrics_sha256": checkpoint_path / "metrics.json",
        }
        expected = {
            "corpus_manifest_sha256": _expected_digest(item.corpus, "manifest_sha256"),
            "corpus_records_sha256": _expected_digest(item.corpus, "records_sha256"),
            "dataset_npz_sha256": _expected_digest(item.dataset, "npz_sha256"),
            "dataset_manifest_sha256": _expected_digest(item.dataset, "manifest_sha256"),
            "checkpoint_weights_sha256": _expected_digest(item.checkpoint, "weights_sha256"),
            "checkpoint_manifest_sha256": _expected_digest(item.checkpoint, "manifest_sha256"),
            "checkpoint_metrics_sha256": _expected_digest(item.checkpoint, "metrics_sha256"),
        }
        actual = {key: _sha256(path) for key, path in files.items()}
        if actual != expected:
            raise StructuredExperimentError(f"frozen full-file digest mismatch for {item.key}")
        corpus_json = _read(corpus_path / "manifest.json")
        dataset_json = _read(dataset_path.with_suffix(".json"))
        checkpoint_json = _read(checkpoint_path / "manifest.json")
        identities = {
            "corpus_fingerprint": corpus_json.get("corpus_fingerprint"),
            "dataset_fingerprint": dataset_json.get("dataset_fingerprint"),
            "checkpoint_fingerprint": checkpoint_json.get("checkpoint_fingerprint"),
            "tensor_digest": checkpoint_json.get("tensor_digest"),
        }
        if identities != {
            "corpus_fingerprint": item.corpus.get("fingerprint"),
            "dataset_fingerprint": item.dataset.get("fingerprint"),
            "checkpoint_fingerprint": item.checkpoint.get("fingerprint"),
            "tensor_digest": item.checkpoint.get("tensor_digest"),
        }:
            raise StructuredExperimentError(f"frozen object identity mismatch for {item.key}")
        rows.append(
            {
                "key": item.key,
                "paths": {key: str(path) for key, path in files.items()},
                "digests": actual,
                **identities,
            }
        )
    q0 = _q0_path(config)
    q0_json = _read(q0 / "manifest.json")
    q0_data = manifest.get("q0")
    if not isinstance(q0_data, Mapping) or (
        q0_json.get("checkpoint_fingerprint") != q0_data.get("checkpoint_fingerprint")
        or q0_json.get("tensor_digest") != q0_data.get("tensor_digest")
    ):
        raise StructuredExperimentError(
            "selected q0 checkpoint/tensor does not match frozen manifest"
        )
    global_audit = _step2_global_input_audit(config, manifest)
    audit = _artifact(
        INPUT_AUDIT_VERSION,
        input_manifest_fingerprint=manifest["artifact_fingerprint"],
        rules_fingerprint=rules_fingerprint(),
        step2_root=str(config.step2_root.resolve()),
        entries=rows,
        q0={
            "path": str(q0),
            "checkpoint_fingerprint": q0_json["checkpoint_fingerprint"],
            "tensor_digest": q0_json["tensor_digest"],
        },
        global_step2=global_audit,
        status="passed",
    )
    return manifest, inputs, audit


def _q0_path(config: StructuredExperimentConfig) -> Path:
    return (
        config.q0_path
        or (_repository_for_step2(config.step2_root) / "runs/terminal-safety-v1/q0-a1/checkpoint")
    ).resolve()


def _q_paths(config: StructuredExperimentConfig) -> dict[str, Path]:
    root = _repository_for_step2(config.step2_root)
    defaults = {
        "q0-parent": _q0_path(config),
        "q1": root / "runs/terminal-safety-v1/q1-a1/candidate",
        "q2": root / "runs/terminal-safety-v1/q2-a1/candidate",
        "q3": root / "runs/terminal-safety-v1/q3-a1/candidate",
        "q4": root / "runs/terminal-safety-v1/q4-a1/candidate",
        "historical-q0": root / "runs/terminal-safety-v1/inputs/historical-q0",
    }
    if config.q_paths is not None:
        defaults.update(config.q_paths)
    return {key: path.resolve() for key, path in defaults.items()}


def _historical_compatibility(source: Mapping[str, object]) -> dict[str, object]:
    """Allow historical corpus replay only when protected Step-2 semantics stayed unchanged."""
    revision = source.get("git_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise StructuredExperimentError("current claim source revision is malformed")
    try:
        changed = subprocess.run(
            ("git", "diff", "--name-only", f"{STEP2_CLAIM_SOURCE}..{revision}"),
            cwd=repository_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StructuredExperimentError("unable to inspect historical compatibility paths") from exc
    protected_hits = [
        path
        for path in changed
        if any(path.startswith(prefix) for prefix in _STEP2_PROTECTED_PREFIXES)
    ]
    if protected_hits:
        raise StructuredExperimentError(
            "Step-3 historical compatibility changed protected Step-2 semantics: "
            + ", ".join(protected_hits)
        )
    return {
        "version": "m7-structured-model-v2-historical-compatibility-v1",
        "step2_claim_source": STEP2_CLAIM_SOURCE,
        "current_claim_source": revision,
        "rules_fingerprint": rules_fingerprint(),
        "protected_prefixes": list(_STEP2_PROTECTED_PREFIXES),
        "changed_paths": changed,
        "protected_path_hits": protected_hits,
        "passed": True,
    }


def _step2_global_input_audit(
    config: StructuredExperimentConfig, manifest: Mapping[str, object]
) -> dict[str, object]:
    """Verify retained Step-2 plan/result/repair/archive and q-family policy identities."""
    root = _repository_for_step2(config.step2_root)
    step2 = manifest.get("step2")
    if not isinstance(step2, Mapping):
        raise StructuredExperimentError("frozen input manifest lacks Step-2 identities")
    plan = _read(config.step2_root / "plan.json")
    result = _read(config.step2_root / "result.json")
    repaired = _read(config.step2_root / "validation.json")
    split = _read(config.step2_root / "dataset-split-alignment.json")
    if (
        plan.get("plan_fingerprint") != step2.get("plan_fingerprint")
        or result.get("result_fingerprint") != step2.get("result_fingerprint")
        or repaired.get("artifact_fingerprint") != step2.get("validation_fingerprint")
        or split.get("artifact_fingerprint") != step2.get("split_alignment_fingerprint")
    ):
        raise StructuredExperimentError("retained Step-2 global artifact identities differ")
    plan_source = plan.get("source")
    if not isinstance(plan_source, Mapping) or plan_source.get("git_revision") != step2.get(
        "claim_source"
    ):
        raise StructuredExperimentError("retained Step-2 plan claim source differs")
    archive = root / "artifacts/archive/m7-population-replay-v1-2026-09-12.tar.gz"
    archive_sidecar = archive.with_suffix(".tar.gz.sha256")
    if (
        not archive.is_file()
        or not archive_sidecar.is_file()
        or _sha256(archive) != step2.get("archive_sha256")
    ):
        raise StructuredExperimentError("retained Step-2 archive SHA-256 differs")
    sidecar = archive_sidecar.read_text().split(maxsplit=1)
    if not sidecar or sidecar[0] != step2.get("archive_sha256"):
        raise StructuredExperimentError("retained Step-2 archive sidecar differs")
    try:
        with tarfile.open(archive, "r:gz") as payload:
            member = payload.getmember("runs/m7-population-replay-v1/archive-manifest.json")
            stream = payload.extractfile(member)
            if stream is None:
                raise StructuredExperimentError("archive manifest payload is missing")
            archive_manifest_value = json.loads(stream.read().decode("utf-8"))
    except (OSError, tarfile.TarError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if isinstance(exc, StructuredExperimentError):
            raise
        raise StructuredExperimentError(
            "unable to verify retained Step-2 archive manifest"
        ) from exc
    if not isinstance(archive_manifest_value, Mapping):
        raise StructuredExperimentError("retained Step-2 archive manifest is malformed")
    checkpoint_identities = plan.get("checkpoint_identities")
    if not isinstance(checkpoint_identities, Mapping):
        raise StructuredExperimentError("retained Step-2 checkpoint identities are malformed")
    from agent_avenue.learning import load_checkpoint, tensor_digest

    paths = _q_paths(config)
    policies: dict[str, object] = {}
    for policy in ("q0-parent", "q1", "q2", "q3", "q4", "historical-q0"):
        step2_name = "q0" if policy == "q0-parent" else policy
        expected = checkpoint_identities.get(step2_name)
        if not isinstance(expected, Mapping):
            raise StructuredExperimentError("retained Step-2 policy identity is missing")
        checkpoint = load_checkpoint(paths[policy])
        actual_tensor = tensor_digest(checkpoint.model.state_dict())
        if checkpoint.checkpoint_fingerprint != expected.get(
            "checkpoint_fingerprint"
        ) or actual_tensor != expected.get("tensor_digest"):
            raise StructuredExperimentError(f"retained {policy} checkpoint/tensor differs")
        policies[policy] = {
            "path": _relative_or_absolute(paths[policy]),
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "tensor_digest": actual_tensor,
        }
    return {
        "step2_global_artifacts": {
            "plan_fingerprint": plan["plan_fingerprint"],
            "result_fingerprint": result["result_fingerprint"],
            "repaired_validation_fingerprint": repaired["artifact_fingerprint"],
            "split_alignment_fingerprint": split["artifact_fingerprint"],
            "claim_source": plan_source["git_revision"],
        },
        "archive": {
            "path": _relative_or_absolute(archive),
            "sha256": _sha256(archive),
            "archive_manifest_fingerprint": archive_manifest_value.get("artifact_fingerprint"),
        },
        "policy_inputs": policies,
    }


def _cells(config: StructuredExperimentConfig) -> tuple[StructuredArenaCell, ...]:
    cells: list[StructuredArenaCell] = []
    for replicate in REPLICATES:
        index = replicate.removeprefix("replicate-")

        def add(
            key: str,
            arm: str | None,
            opponent: str,
            declared_pairs: int,
            shared: str | None,
            candidate_kind: str,
            current_replicate: str = replicate,
        ) -> None:
            seed_key = shared or key
            domain = f"step3:arena:{seed_key}:{current_replicate}"
            cells.append(
                StructuredArenaCell(
                    current_replicate,
                    key,
                    arm,
                    opponent,
                    config.pair_count(declared_pairs),
                    derive_seed(ROOT_SEED, domain) & ((1 << 63) - 1),
                    shared,
                    candidate_kind,
                )
            )

        for arm in ARMS:
            add(f"{arm}{index}-v2-vs-v1", arm, "matched-v1", 500, "architecture", "v2")
        add(f"M{index}-v2-vs-C{index}-v2", "M", f"C{index}-v2", 500, "v2-data", "v2")
        for arm in ARMS:
            add(f"{arm}{index}-v2-vs-q0-parent", arm, "q0-parent", 500, "q0-parent", "v2")
            add(f"{arm}{index}-v2-vs-heuristic", arm, "heuristic", 300, "heuristic", "v2")
        add(
            f"q0-parent-vs-heuristic-reference-{index}",
            None,
            "heuristic",
            300,
            "heuristic",
            "parent",
        )
        for arm in ARMS:
            add(f"{arm}{index}-v2-vs-random", arm, "random", 200, "random", "v2")
            add(
                f"{arm}{index}-v2-vs-historical-q0",
                arm,
                "historical-q0",
                200,
                "historical-q0",
                "v2",
            )
            for opponent in ("q1", "q2", "q3", "q4"):
                add(f"{arm}{index}-v2-vs-{opponent}", arm, opponent, 100, opponent, "v2")
    return tuple(cells)


def _count_cells(cells: Iterable[StructuredArenaCell]) -> int:
    return sum(cell.pair_count * 2 for cell in cells)


def _check_default_cardinality(cells: Sequence[StructuredArenaCell]) -> None:
    if len(cells) != 60 or _count_cells(cells) != 30_000:
        raise StructuredExperimentError(
            "frozen Step-3 schedule must contain 60 cells / 30,000 games"
        )


def _holdout_scan(
    config: StructuredExperimentConfig, cells: Sequence[StructuredArenaCell]
) -> dict[str, object]:
    current: dict[str, dict[str, object]] = {}
    for cell in cells:
        scheduling = ArenaConfig(
            cell.run_id,
            AgentSpec("schedule-a", {"type": "schedule"}, RandomAgent),
            AgentSpec("schedule-b", {"type": "schedule"}, RandomAgent),
            cell.pair_count,
            cell.master_seed,
        )
        for spec in schedule_arena(scheduling):
            identity = population_setup_identity(normalize_config(spec.config), spec.setup_seed)
            existing = current.get(identity)
            material: dict[str, object] = {
                "replicate_id": cell.replicate_id,
                "cell": cell.key,
                "shared_group": cell.shared_group,
                "pair_id": spec.pair_id,
            }
            if existing is not None and not (
                existing["replicate_id"] == cell.replicate_id
                and existing["shared_group"] is not None
                and existing["shared_group"] == cell.shared_group
            ):
                raise StructuredExperimentError(
                    "proposed Step-3 setup has undeclared internal overlap"
                )
            current.setdefault(identity, material)
    output = config.output.resolve()
    prior_setups: dict[str, list[str]] = {}
    scanned: list[dict[str, object]] = []
    seen: set[Path] = set()
    for declared_root in config.holdout_roots:
        root = declared_root.resolve()
        if not root.exists():
            scanned.append({"root": str(root), "status": "missing"})
            continue
        if not root.is_dir():
            raise StructuredExperimentError("holdout root is not a directory")
        for manifest in sorted(root.rglob("manifest.json")):
            if manifest.resolve().is_relative_to(output) or manifest.resolve() in seen:
                continue
            seen.add(manifest.resolve())
            directory = manifest.parent
            if not (directory / "games.jsonl.gz").is_file():
                continue
            loaded_manifest, records = load_corpus(
                directory, verify_code=False, verify_replays=False
            )
            scanned.append(
                {
                    "path": str(directory),
                    "records": len(records),
                    "corpus": loaded_manifest.corpus_fingerprint,
                }
            )
            for record in records:
                identity = population_setup_identity(
                    normalize_config(record.replay.config), record.replay.seed
                )
                prior_setups.setdefault(identity, []).append(str(directory))
    overlap = sorted(set(current).intersection(prior_setups))
    return _artifact(
        "m7-structured-model-v2-setup-holdout-v1",
        recursive=True,
        excludes_current_output=True,
        allowed_within_step3_shared_groups=sorted(
            {cell.shared_group for cell in cells if cell.shared_group}
        ),
        current_setup_count=len(current),
        current_setup_fingerprint=_fingerprint(sorted(current)),
        prior_setup_count=len(prior_setups),
        prior_setup_fingerprint=_fingerprint(sorted(prior_setups)),
        overlap_count=len(overlap),
        overlap_examples=[{"identity": key, "prior": prior_setups[key]} for key in overlap[:24]],
        scanned=scanned,
        status="passed" if not overlap else "failed",
    )


def _action_data(action: object) -> dict[str, object]:
    from agent_avenue.engine import PlayOfferAction, RecruitAction

    if isinstance(action, PlayOfferAction):
        return {
            "type": "play_offer",
            "face_up": action.face_up.value,
            "face_down": action.face_down.value,
        }
    if isinstance(action, RecruitAction):
        return {"type": "recruit", "slot": action.slot.value}
    raise StructuredExperimentError("fixture contains unsupported action")


def _feature_panel(
    records: Sequence[GameRecord],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Choose whole candidate groups that deterministically cover the frozen safety categories."""
    from agent_avenue.encoding.candidate_structured_v2 import (
        FEATURE_NAMES,
    )
    from agent_avenue.encoding.candidate_structured_v2 import (
        encode_candidate as encode_v2,
    )
    from agent_avenue.encoding.candidate_v1 import encode_candidate as encode_v1
    from agent_avenue.engine import Phase, PlayOfferAction, apply_action, new_game
    from agent_avenue.engine.cards import CardName

    required = {
        *(f"card:{card.value}" for card in CardName),
        "phase:play",
        "phase:recruit",
        "effect:score",
        "effect:win",
        "effect:lose",
        "terminal:win",
        "terminal:loss",
        "history:padding",
        "history:truncated",
    }
    covered: set[str] = set()
    groups: list[dict[str, object]] = []
    for record in records:
        state = new_game(record.replay.config, record.replay.seed)
        record_fingerprint = game_record_fingerprint(record)
        for decision_index, action in enumerate(record.replay.actions):
            actor = (
                state.active_player if state.phase is Phase.PLAY else state.active_player.other()
            )
            observation = observe(state, actor)
            tags = {
                f"phase:{state.phase.value}",
                "history:truncated"
                if len(observation.history) > 8
                else "history:padding"
                if len(observation.history) < 8
                else "history:exact-eight",
            }
            rows: list[dict[str, object]] = []
            for candidate in observation.legal_actions:
                v1 = encode_v1(observation, candidate).vector
                v2 = encode_v2(observation, candidate).vector
                rows.append({"action": _action_data(candidate), "v1": list(v1), "v2": list(v2)})
                if isinstance(candidate, PlayOfferAction):
                    tags.update(
                        (f"card:{candidate.face_up.value}", f"card:{candidate.face_down.value}")
                    )
                elif hasattr(observation.decision, "face_up"):
                    tags.add(f"card:{observation.decision.face_up.value}")
                for effect in ("score", "win", "lose"):
                    if any(
                        value == 1.0 and name.endswith(f"_effect_{effect}")
                        for name, value in zip(FEATURE_NAMES, v2, strict=True)
                    ):
                        tags.add(f"effect:{effect}")
                if any(
                    value == 1.0 and "terminal" in name and name.endswith("_win")
                    for name, value in zip(FEATURE_NAMES, v2, strict=True)
                ):
                    tags.add("terminal:win")
                if any(
                    value == 1.0 and "terminal" in name and name.endswith("_loss")
                    for name, value in zip(FEATURE_NAMES, v2, strict=True)
                ):
                    tags.add("terminal:loss")
            if tags.difference(covered):
                groups.append(
                    {
                        "group_id": f"{record_fingerprint}:{decision_index}",
                        "phase": state.phase.value,
                        "history_length": len(observation.history),
                        "coverage_tags": sorted(tags),
                        "rows": rows,
                    }
                )
                covered.update(tags)
            state = apply_action(state, action)
            if required <= covered:
                panel = {
                    "required_tags": sorted(required),
                    "covered_tags": sorted(covered),
                    "group_count": len(groups),
                    "candidate_count": sum(
                        len(cast(list[object], group["rows"])) for group in groups
                    ),
                }
                return groups, panel
    raise StructuredExperimentError(
        "coverage-complete q0 fixture panel missing: " + ", ".join(sorted(required - covered))
    )


def _initialization_audit(
    q0_path: Path, groups: Sequence[Mapping[str, object]], panel: Mapping[str, object], seed: int
) -> dict[str, object]:
    import torch

    from agent_avenue.learning import load_checkpoint, structured_tensor_digest, tensor_digest

    q0 = load_checkpoint(q0_path)
    model = create_structured_model(q0.model.state_dict(), projection_seed=seed)
    state = q0.model.state_dict()
    direct_mapping = {
        "base_hidden_weight": torch.equal(model.base_hidden.weight, state["hidden.weight"]),
        "base_hidden_bias": torch.equal(model.base_hidden.bias, state["hidden.bias"]),
        "base_value_weight": torch.equal(model.base_value.weight, state["output.weight"]),
        "base_value_bias": torch.equal(model.base_value.bias, state["output.bias"]),
        "play_residual_zero": bool(
            torch.count_nonzero(model.play_residual.weight) == 0
            and torch.count_nonzero(model.play_residual.bias) == 0
        ),
        "recruit_residual_zero": bool(
            torch.count_nonzero(model.recruit_residual.weight) == 0
            and torch.count_nonzero(model.recruit_residual.bias) == 0
        ),
    }
    if not all(direct_mapping.values()):
        raise StructuredExperimentError("q0 tensor mapping or zero residual initialization differs")
    maximum = 0.0
    summaries: list[dict[str, object]] = []
    for group in groups:
        rows = group.get("rows")
        if not isinstance(rows, list) or not rows:
            raise StructuredExperimentError("fixture group rows are malformed")
        v1 = torch.tensor(
            [cast(Mapping[str, object], row)["v1"] for row in rows], dtype=torch.float32
        )
        v2 = torch.tensor(
            [cast(Mapping[str, object], row)["v2"] for row in rows], dtype=torch.float32
        )
        with torch.inference_mode():
            q0_logits = q0.model(v1)
            v2_logits = model(v2)
        difference = float(torch.max(torch.abs(q0_logits - v2_logits)).item())
        maximum = max(maximum, difference)
        q0_values = q0_logits.tolist()
        v2_values = v2_logits.tolist()
        q0_maxima = tuple(index for index, value in enumerate(q0_values) if value == max(q0_values))
        v2_maxima = tuple(index for index, value in enumerate(v2_values) if value == max(v2_values))
        if difference > 1e-7 or q0_maxima != v2_maxima:
            raise StructuredExperimentError(
                "untrained v2 does not preserve q0 grouped greedy maxima"
            )
        summaries.append(
            {
                "group_id": group["group_id"],
                "phase": group["phase"],
                "history_length": group["history_length"],
                "actions": [cast(Mapping[str, object], row)["action"] for row in rows],
                "q0_maxima": list(q0_maxima),
                "v2_maxima": list(v2_maxima),
                "maximum_absolute_logit_difference": difference,
            }
        )
    return _artifact(
        "m7-structured-model-v2-q0-embedding-v2",
        q0_checkpoint_fingerprint=q0.checkpoint_fingerprint,
        q0_tensor_digest=tensor_digest(q0.model.state_dict()),
        structured_init_seed=seed,
        panel=panel,
        panel_digest=_fingerprint(groups),
        direct_tensor_mapping=direct_mapping,
        group_boundaries_and_maxima=summaries,
        maximum_absolute_logit_difference=maximum,
        greedy_choices_identical=True,
        structured_tensor_digest=structured_tensor_digest(model.state_dict()),
        passed=True,
    )


def _dataset_audit(
    output: Path,
    plan: Mapping[str, object],
    item: Step3Input,
    *,
    source_root: Path,
) -> tuple[Path, dict[str, object], tuple[GameRecord, ...]]:
    from agent_avenue.learning import (
        load_structured_dataset,
        load_verified_step2_corpus,
        materialize_structured_dataset,
        save_structured_dataset,
    )

    corpus_path = source_root / cast(str, item.corpus["path"])
    v1_path = source_root / cast(str, item.dataset["path"])
    manifest, records = load_verified_step2_corpus(
        corpus_path,
        expected_corpus_fingerprint=cast(str, item.corpus["fingerprint"]),
        caller_verified=True,
        verify_code=False,
    )
    corpus_fingerprint = getattr(manifest, "corpus_fingerprint", None)
    if not isinstance(corpus_fingerprint, str):
        raise StructuredExperimentError("historical corpus manifest lacks fingerprint")
    destination = output / "datasets" / item.key / "dataset.npz"
    if not destination.exists():
        dataset = materialize_structured_dataset(
            records,
            v1_dataset=v1_path,
            verify_code=False,
            source_corpus_fingerprint=corpus_fingerprint,
            source_corpus_manifest_fingerprint=cast(str, item.corpus["manifest_sha256"]),
        )
        save_structured_dataset(dataset, destination)
    dataset = load_structured_dataset(destination)
    source = __import__("agent_avenue.learning", fromlist=["load_dataset"]).load_dataset(v1_path)
    if (
        dataset.manifest.get("source_v1_dataset_fingerprint") != source.fingerprint
        or dataset.manifest.get("source_record_fingerprints")
        != source.manifest.get("source_record_fingerprints")
        or dataset.manifest.get("split")
        != {
            "source_algorithm": cast(Mapping[str, object], source.manifest["split"]).get(
                "algorithm"
            ),
            "train_game_fingerprints": cast(Mapping[str, object], source.manifest["split"])[
                "train_game_fingerprints"
            ],
            "validation_game_fingerprints": cast(Mapping[str, object], source.manifest["split"])[
                "validation_game_fingerprints"
            ],
        }
    ):
        raise StructuredExperimentError(
            f"v2 dataset did not retain full v1 row/split identity: {item.key}"
        )
    audit = _artifact(
        DATASET_AUDIT_VERSION,
        plan_fingerprint=plan["plan_fingerprint"],
        key=item.key,
        v1_dataset_fingerprint=source.fingerprint,
        v2_dataset_fingerprint=dataset.fingerprint,
        v1_npz_sha256=_sha256(v1_path),
        v2_npz_sha256=_sha256(destination),
        v2_manifest_sha256=_sha256(destination.with_suffix(".json")),
        row_identity={
            "train_rows": int(dataset.train.targets.size),
            "validation_rows": int(dataset.validation.targets.size),
            "labels_identical": bool(
                (dataset.train.targets == source.train.targets).all()
                and (dataset.validation.targets == source.validation.targets).all()
            ),
            "game_indexes_identical": bool(
                (dataset.train.game_index == source.train.game_index).all()
                and (dataset.validation.game_index == source.validation.game_index).all()
            ),
            "phases_identical": bool(
                (dataset.train.phase == source.train.phase).all()
                and (dataset.validation.phase == source.validation.phase).all()
            ),
        },
        split_groups=dataset.manifest["split"],
        source_corpus_fingerprint=corpus_fingerprint,
        passed=True,
    )
    _write_immutable(output / "datasets" / item.key / "audit.json", audit, label="dataset audit")
    return destination, audit, records


def _training_seed(replicate: str, arm: str, step2_root: Path) -> int:
    """Read the frozen Step-2 paired shuffle seed instead of generating one after results."""
    legacy = _read(step2_root / "training-summary.json")
    rows = legacy.get("replicates")
    if not isinstance(rows, list):
        raise StructuredExperimentError("Step-2 training summary is malformed")
    legacy_arm = MANIFEST_ARM[arm]
    for row in rows:
        if isinstance(row, Mapping) and row.get("replicate_id") == replicate:
            arms = row.get("arms")
            if isinstance(arms, Mapping) and isinstance(arms.get(legacy_arm), Mapping):
                config = cast(Mapping[str, object], arms[legacy_arm]).get("training_config")
                if isinstance(config, Mapping) and type(config.get("shuffle_seed")) is int:
                    return cast(int, config["shuffle_seed"])
    raise StructuredExperimentError("could not recover frozen Step-2 shuffle seed")


def _train_checkpoint(
    output: Path,
    plan: Mapping[str, object],
    item: Step3Input,
    dataset_path: Path,
    dataset_audit: Mapping[str, object],
    q0_path: Path,
    *,
    max_epochs: int,
) -> dict[str, object]:
    from agent_avenue.learning import (
        StructuredTrainingConfig,
        inspect_structured_checkpoint,
        load_checkpoint,
        load_structured_checkpoint,
        load_structured_dataset,
        save_structured_checkpoint,
        tensor_digest,
        train_structured_model,
    )

    init_seed = derive_seed(ROOT_SEED, f"step3:v2-structured-init:{item.replicate_id}") & (
        (1 << 63) - 1
    )
    shuffle_seed = _training_seed(item.replicate_id, item.arm, Path(cast(str, plan["step2_root"])))
    # The explicit plan path above is canonical.  It keeps the copied shuffle identity separate
    # from the new Step-3 root seed and is the same for C/M within a replicate.
    training = StructuredTrainingConfig(
        seed=init_seed,
        projection_seed=init_seed,
        shuffle_seed=shuffle_seed,
        max_epochs=max_epochs,
        early_stopping_patience=8,
        batch_size=1024,
        learning_rate=1e-3,
        weight_decay=1e-4,
        cpu_threads=1,
    )
    q0 = load_checkpoint(q0_path)
    destination = output / "checkpoints" / item.key
    if not destination.exists():
        dataset = load_structured_dataset(dataset_path)
        result = train_structured_model(
            dataset, q0_state_dict=q0.model.state_dict(), config=training
        )
        save_structured_checkpoint(
            destination,
            result.model,
            metrics=result.metrics_dict(),
            q0_parent_checkpoint_fingerprint=q0.checkpoint_fingerprint,
            q0_parent_tensor_digest=tensor_digest(q0.model.state_dict()),
            dataset_fingerprint=dataset.fingerprint,
            source_corpus_fingerprints=(cast(str, dataset_audit["source_corpus_fingerprint"]),),
            split_identities={
                "v1_train_pair_ids": item.dataset["train_pair_ids_fingerprint"],
                "v1_validation_pair_ids": item.dataset["validation_pair_ids_fingerprint"],
                "v1_split_groups": dataset.manifest["split"],
            },
            training_config=training.normalized(),
            training_seeds={
                "structured_init": init_seed,
                "shuffle": shuffle_seed,
                "fresh_adamw": True,
            },
            input_manifest_fingerprint=cast(str, plan["input_manifest_fingerprint"]),
            source_rules_fingerprint=rules_fingerprint(),
            metadata={
                "plan_fingerprint": plan["plan_fingerprint"],
                "arm": item.arm,
                "replicate_id": item.replicate_id,
                "initialization": "q0-base+arm-independent-xavier+zero-residual+fresh-adamw",
            },
            created_at=FIXED_CREATED_AT,
        )
    inspected = inspect_structured_checkpoint(destination)
    loaded = load_structured_checkpoint(destination)
    manifest = loaded.manifest
    sources = cast(Mapping[str, object], manifest["sources"])
    configured = cast(Mapping[str, object], manifest["training"])
    if (
        sources.get("dataset_fingerprint") != dataset_audit["v2_dataset_fingerprint"]
        or configured.get("config") != training.normalized()
        or cast(Mapping[str, object], manifest["metadata"]).get("plan_fingerprint")
        != plan["plan_fingerprint"]
    ):
        raise StructuredExperimentError(f"structured checkpoint lineage mismatch: {item.key}")
    training_artifact = _artifact(
        "m7-structured-model-v2-training-v1",
        key=item.key,
        checkpoint_fingerprint=inspected.checkpoint_fingerprint,
        tensor_digest=inspected.tensor_digest,
        training_config=training.normalized(),
        training_seeds={"structured_init": init_seed, "shuffle": shuffle_seed, "fresh_adamw": True},
        metrics=loaded.metrics,
    )
    _write_immutable(
        output / "training" / item.key / "summary.json", training_artifact, label="training summary"
    )
    return training_artifact


def _agent_config(agent: Agent) -> dict[str, object]:
    method = getattr(agent, "config_to_data", None)
    if callable(method):
        value = method()
    else:
        config = getattr(agent, "config", None)
        to_data = getattr(config, "to_data", None)
        if not callable(to_data):
            raise StructuredExperimentError("agent has no normalized configuration")
        value = to_data()
    if not isinstance(value, dict):
        raise StructuredExperimentError("agent configuration is malformed")
    return cast(dict[str, object], json.loads(_canonical(value)))


def _v1_agent(path: Path, agent_id: str, rng_identity: str) -> AgentSpec:
    from agent_avenue.agents.learned import LearnedValueAgent
    from agent_avenue.learning import load_checkpoint

    checkpoint = load_checkpoint(path)

    def factory() -> Agent:
        return TerminalOffenseAgent(
            TerminalSafetyAgent(LearnedValueAgent.from_checkpoint(checkpoint))
        )

    return AgentSpec(agent_id, _agent_config(factory()), factory, rng_identity=rng_identity)


def _v2_agent(path: Path, agent_id: str, rng_identity: str) -> AgentSpec:
    from agent_avenue.agents.structured import StructuredValueAgent
    from agent_avenue.learning import load_structured_checkpoint

    checkpoint = load_structured_checkpoint(path)

    def factory() -> Agent:
        return TerminalOffenseAgent(
            TerminalSafetyAgent(StructuredValueAgent.from_checkpoint(checkpoint))
        )

    return AgentSpec(agent_id, _agent_config(factory()), factory, rng_identity=rng_identity)


def _canonical_opponent(name: str, paths: Mapping[str, Path], replicate: str) -> AgentSpec:
    if name == "heuristic":
        heuristic = GreedyHeuristicAgent(GreedyHeuristicConfig())
        return AgentSpec(
            "heuristic",
            _agent_config(heuristic),
            lambda: GreedyHeuristicAgent(GreedyHeuristicConfig()),
            rng_identity=f"step3:{replicate}:heuristic",
        )
    if name == "random":
        random = RandomAgent(RandomAgentConfig())
        return AgentSpec(
            "random",
            _agent_config(random),
            lambda: RandomAgent(RandomAgentConfig()),
            rng_identity=f"step3:{replicate}:random",
        )
    if name == "historical-q0":
        from agent_avenue.agents.learned import LearnedValueAgent
        from agent_avenue.learning import load_checkpoint

        checkpoint = load_checkpoint(paths[name])

        def factory() -> Agent:
            return LearnedValueAgent.from_checkpoint(checkpoint)

        return AgentSpec(
            name, _agent_config(factory()), factory, rng_identity=f"step3:{replicate}:historical-q0"
        )
    return _v1_agent(paths[name], name, f"step3:{replicate}:{name}")


def _arena_agents(
    cell: StructuredArenaCell,
    output: Path,
    inputs: Mapping[str, Step3Input],
    paths: Mapping[str, Path],
) -> tuple[AgentSpec, AgentSpec]:
    replicate = cell.replicate_id
    index = replicate.removeprefix("replicate-")
    if cell.candidate_kind == "parent":
        return _canonical_opponent("q0-parent", paths, replicate), _canonical_opponent(
            "heuristic", paths, replicate
        )
    if cell.key.endswith("v2-vs-v1"):
        if cell.candidate_arm is None:
            raise StructuredExperimentError("architecture cell has no arm")
        item = inputs[f"{cell.candidate_arm}{index}"]
        source_root = paths["q0-parent"].parents[3]
        v1 = source_root / cast(str, item.checkpoint["path"])
        return _v2_agent(
            output / "checkpoints" / item.key,
            f"{item.key}-v2",
            f"step3:{replicate}:architecture-v2",
        ), _v1_agent(v1, f"{item.key}-v1", f"step3:{replicate}:architecture-v1")
    if "-v2-vs-C" in cell.key:
        return _v2_agent(
            output / "checkpoints" / f"M{index}", f"M{index}-v2", f"step3:{replicate}:M-candidate"
        ), _v2_agent(
            output / "checkpoints" / f"C{index}", f"C{index}-v2", f"step3:{replicate}:C-candidate"
        )
    if cell.candidate_arm is None:
        raise StructuredExperimentError("candidate cell has no arm")
    item = inputs[f"{cell.candidate_arm}{index}"]
    candidate_lane = (
        f"step3:{replicate}:{cell.shared_group}:candidate"
        if cell.shared_group is not None
        else f"step3:{replicate}:{item.key}:candidate"
    )
    return _v2_agent(
        output / "checkpoints" / item.key,
        f"{item.key}-v2",
        candidate_lane,
    ), _canonical_opponent(cell.opponent, paths, replicate)


def _tactical(records: Sequence[GameRecord], corpus_fingerprint: str) -> dict[str, object]:
    safety = audit_terminal_safety(
        records, source_corpus_fingerprint=corpus_fingerprint, verify_code=False
    )
    offense = audit_public_forced_wins(
        records, source_label=corpus_fingerprint, verify_records=False
    )
    checked: dict[str, object] = {}
    by_agent = cast(Mapping[str, object], safety["by_agent"])
    forced = cast(Mapping[str, object], offense["by_agent"])
    for agent_id, value in by_agent.items():
        if not isinstance(value, Mapping) or not isinstance(value.get("config"), Mapping):
            continue
        config = cast(Mapping[str, object], value["config"])
        if config.get("type") != "terminal_offense":
            continue
        counts = cast(Mapping[str, object], value["counts"])
        forced_counts = cast(
            Mapping[str, object], cast(Mapping[str, object], forced[agent_id])["counts"]
        )
        checked[agent_id] = {
            "avoidable_immediate_losses": counts["executed_avoidable_provable_losses"],
            "missed_guaranteed_wins": forced_counts["missed_forced_wins"],
            "passed": counts["executed_avoidable_provable_losses"] == 0
            and forced_counts["missed_forced_wins"] == 0,
        }
    return {
        "safety": safety,
        "offense": offense,
        "checked_agents": checked,
        "passed": bool(checked)
        and all(cast(Mapping[str, object], value)["passed"] is True for value in checked.values()),
    }


def _arena_artifact(
    output: Path,
    plan: Mapping[str, object],
    cell: StructuredArenaCell,
    inputs: Mapping[str, Step3Input],
    paths: Mapping[str, Path],
) -> dict[str, object]:
    directory = output / "arenas" / cell.key
    artifact_path = directory / "report.json"
    left, right = _arena_agents(cell, output, inputs, paths)
    arena = ArenaConfig(cell.run_id, left, right, cell.pair_count, cell.master_seed)
    records_directory = directory / "records"
    if artifact_path.exists():
        artifact = _read(artifact_path)
        loaded_manifest, records = load_corpus(records_directory, verify_code=False)
        report = artifact.get("report")
        elapsed = report.get("elapsed_seconds") if isinstance(report, Mapping) else None
        if (
            not isinstance(elapsed, int | float)
            or arena_report_from_records(arena, records, elapsed_seconds=float(elapsed)).to_data()
            != report
        ):
            raise StructuredExperimentError(
                f"resumed arena report does not reconstruct: {cell.key}"
            )
        if artifact.get("tactical") != _tactical(records, loaded_manifest.corpus_fingerprint):
            raise StructuredExperimentError(f"resumed arena tactical audit differs: {cell.key}")
        return artifact
    retained = run_resumable_arena(
        records_directory,
        arena,
        generation=3,
        corpus_configuration={
            "version": ARENA_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "cell": cell.to_data(),
        },
    )
    manifest, records = load_corpus(records_directory)
    artifact = _artifact(
        ARENA_VERSION,
        plan_fingerprint=plan["plan_fingerprint"],
        cell=cell.to_data(),
        records={
            "corpus_fingerprint": manifest.corpus_fingerprint,
            "locator": _relative_or_absolute(records_directory),
        },
        report=retained.report.to_data(),
        tactical=_tactical(records, manifest.corpus_fingerprint),
    )
    _write_immutable(artifact_path, artifact, label="arena")
    return artifact


def _pair_scores(artifact: Mapping[str, object]) -> tuple[float, ...]:
    report = artifact.get("report")
    outcomes = report.get("paired_seed_outcomes") if isinstance(report, Mapping) else None
    if not isinstance(outcomes, list):
        raise StructuredExperimentError("arena report lacks paired outcomes")
    output: list[float] = []
    for row in outcomes:
        wins = cast(Mapping[str, object], row).get("agent_a_wins")
        if type(wins) is not int:
            raise StructuredExperimentError("paired outcome is malformed")
        output.append(wins / 2)
    return tuple(output)


def _seed(domain: str) -> int:
    return derive_seed(ROOT_SEED, f"step3:nested-bootstrap:{domain}") & ((1 << 63) - 1)


def _interval(samples: list[float], point: float, domain: str, blocks: int) -> dict[str, object]:
    samples.sort()
    return {
        "version": "m7-structured-model-v2-nested-bootstrap-v1",
        "unit": "training-replicate-then-paired-block",
        "resamples": NESTED_RESAMPLES,
        "outer_draws_materialized_first": True,
        "order_statistic_indices": {"lower": NESTED_LOWER, "upper": NESTED_UPPER},
        "point_estimate": point,
        "interval": [samples[NESTED_LOWER], samples[NESTED_UPPER]],
        "seed": _seed(domain),
        "domain": domain,
        "blocks_per_replicate": blocks,
    }


def nested_bootstrap(rows: tuple[tuple[float, ...], ...], *, domain: str) -> dict[str, object]:
    """Bootstrap a regular estimand while materializing all outer draws before inner draws."""
    if len(rows) != 3 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise StructuredExperimentError("nested bootstrap needs three equally sized nonempty rows")
    from agent_avenue.agents import DeterministicRandom

    count = len(rows[0])
    rng = DeterministicRandom(_seed(domain), f"step3:nested-bootstrap:{domain}")
    values: list[float] = []
    for _ in range(NESTED_RESAMPLES):
        # The three outer replicate draws are intentionally all materialized before this
        # resample consumes any inner block randomness (the Step-2 validator repair lesson).
        selected = tuple(rng.randbelow(3) for _ in range(3))
        means = [
            sum(rows[index][rng.randbelow(count)] for _ in range(count)) / count
            for index in selected
        ]
        values.append(sum(means) / 3)
    return _interval(values, sum(sum(row) / count for row in rows) / 3, domain, count)


def _joint_architecture_bootstrap(
    control: tuple[tuple[float, ...], ...],
    mixed: tuple[tuple[float, ...], ...],
    *,
    interaction: bool,
) -> dict[str, object]:
    from agent_avenue.agents import DeterministicRandom

    count = len(control[0]) if control else 0
    if (
        len(control) != 3
        or len(mixed) != 3
        or count == 0
        or any(len(row) != count for row in (*control, *mixed))
    ):
        raise StructuredExperimentError("aligned architecture bootstrap requires matched blocks")
    domain = "interaction" if interaction else "pooled-architecture-main-effect"
    rng = DeterministicRandom(_seed(domain), f"step3:nested-bootstrap:{domain}")
    values: list[float] = []
    for _ in range(NESTED_RESAMPLES):
        selected = tuple(rng.randbelow(3) for _ in range(3))
        draws: list[float] = []
        for replicate in selected:
            indexes = tuple(rng.randbelow(count) for _ in range(count))
            # Exactly the same block indexes feed C and M for this replicate occurrence.
            c = sum(control[replicate][index] for index in indexes) / count
            m = sum(mixed[replicate][index] for index in indexes) / count
            draws.append((m - c) if interaction else (m + c) / 2)
        values.append(sum(draws) / 3)
    point_c = sum(sum(row) / count for row in control) / 3
    point_m = sum(sum(row) / count for row in mixed) / 3
    return _interval(
        values, point_m - point_c if interaction else (point_m + point_c) / 2, domain, count
    )


def _macro_bootstrap(
    rows: Mapping[str, tuple[tuple[float, ...], ...]], arm: str
) -> dict[str, object]:
    from agent_avenue.agents import DeterministicRandom

    if set(rows) != set(OPPONENTS):
        raise StructuredExperimentError("external macro must include exactly eight named opponents")
    domain = f"{arm}-equal-eight-opponent-macro"
    rng = DeterministicRandom(_seed(domain), f"step3:nested-bootstrap:{domain}")
    values: list[float] = []
    for resample_index in range(NESTED_RESAMPLES):
        selected = tuple(rng.randbelow(3) for _ in range(3))
        replicate_values: list[float] = []
        for occurrence_index, replicate in enumerate(selected):
            means: list[float] = []
            for opponent in OPPONENTS:
                row = rows[opponent][replicate]
                # Repeated outer replicate IDs remain independent occurrences.  The named seed
                # includes both resample and occurrence rather than just the replicate identifier.
                inner_domain = (
                    f"{domain}:opponent:{opponent}:resample:{resample_index}:"
                    f"occurrence:{occurrence_index}"
                )
                local = DeterministicRandom(
                    _seed(inner_domain), f"step3:nested-bootstrap:{inner_domain}"
                )
                means.append(
                    sum(row[local.randbelow(len(row))] for _ in range(len(row))) / len(row)
                )
            replicate_values.append(sum(means) / len(means))
        values.append(sum(replicate_values) / 3)
    point = sum(sum(sum(row) / len(row) for row in rows[name]) / 3 for name in OPPONENTS) / len(
        OPPONENTS
    )
    return _interval(values, point, domain, 0)


def _by_cell(artifacts: Iterable[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for artifact in artifacts:
        cell = artifact.get("cell")
        if not isinstance(cell, Mapping) or not isinstance(cell.get("key"), str):
            raise StructuredExperimentError("arena artifact has malformed cell")
        result[cast(str, cell["key"])] = artifact
    return result


def _same_block(left: Mapping[str, object], right: Mapping[str, object]) -> None:
    a = cast(Mapping[str, object], left["report"])["paired_seed_outcomes"]
    b = cast(Mapping[str, object], right["report"])["paired_seed_outcomes"]
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        raise StructuredExperimentError("aligned arena block malformed")
    for x, y in zip(a, b, strict=True):
        if cast(Mapping[str, object], x).get("pair_id") != cast(Mapping[str, object], y).get(
            "pair_id"
        ):
            raise StructuredExperimentError("aligned arena pair identifiers differ")


def structured_statistics(
    artifacts: Iterable[Mapping[str, object]], *, step2_root: Path
) -> dict[str, object]:
    """Compute all predeclared Step-3 estimands from paired blocks only."""
    by_key = _by_cell(artifacts)
    architecture: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    data_effect: list[tuple[float, ...]] = []
    parent: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    random_rows: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    heuristic_difference: dict[str, list[tuple[float, ...]]] = {arm: [] for arm in ARMS}
    external: dict[str, dict[str, list[tuple[float, ...]]]] = {
        arm: {opponent: [] for opponent in OPPONENTS} for arm in ARMS
    }
    per_replicate: list[dict[str, object]] = []
    minimum_seats: dict[str, float] = {arm: 1.0 for arm in ARMS}
    tactical = True
    for replicate in REPLICATES:
        index = replicate.removeprefix("replicate-")
        row: dict[str, object] = {"replicate_id": replicate, "arms": {}}
        parent_reference = by_key[f"q0-parent-vs-heuristic-reference-{index}"]
        for arm in ARMS:
            architecture_cell = by_key[f"{arm}{index}-v2-vs-v1"]
            v2_parent = by_key[f"{arm}{index}-v2-vs-q0-parent"]
            v2_random = by_key[f"{arm}{index}-v2-vs-random"]
            v2_heuristic = by_key[f"{arm}{index}-v2-vs-heuristic"]
            _same_block(v2_heuristic, parent_reference)
            architecture[arm].append(_pair_scores(architecture_cell))
            parent[arm].append(_pair_scores(v2_parent))
            random_rows[arm].append(_pair_scores(v2_random))
            heuristic_difference[arm].append(
                tuple(
                    a - b
                    for a, b in zip(
                        _pair_scores(v2_heuristic), _pair_scores(parent_reference), strict=True
                    )
                )
            )
            for opponent in OPPONENTS:
                key = f"{arm}{index}-v2-vs-{opponent}"
                external[arm][opponent].append(_pair_scores(by_key[key]))
            report = cast(Mapping[str, object], architecture_cell["report"])
            candidates = [architecture_cell, v2_parent, v2_random, v2_heuristic]
            seats: list[float] = []
            for candidate in candidates:
                report = cast(Mapping[str, object], candidate["report"])
                values = cast(Mapping[str, object], report["agent_a_by_seat"])
                for value in values.values():
                    rate = cast(Mapping[str, object], value).get("win_rate")
                    if not isinstance(rate, int | float):
                        raise StructuredExperimentError("arena seat rate is malformed")
                    seats.append(float(rate))
                tactical = (
                    tactical and cast(Mapping[str, object], candidate["tactical"])["passed"] is True
                )
            minimum_seats[arm] = min(minimum_seats[arm], min(seats))
            cast(dict[str, object], row["arms"])[arm] = {
                "architecture_v2_minus_v1": sum(architecture[arm][-1]) / len(architecture[arm][-1]),
                "v2_vs_q0_parent": sum(parent[arm][-1]) / len(parent[arm][-1]),
                "v2_vs_random": sum(random_rows[arm][-1]) / len(random_rows[arm][-1]),
                "v2_minus_q0_parent_heuristic": sum(heuristic_difference[arm][-1])
                / len(heuristic_difference[arm][-1]),
                "minimum_v2_candidate_seat": min(seats),
                "external_rates": {
                    opponent: sum(external[arm][opponent][-1]) / len(external[arm][opponent][-1])
                    for opponent in OPPONENTS
                },
            }
        data_cell = by_key[f"M{index}-v2-vs-C{index}-v2"]
        data_effect.append(_pair_scores(data_cell))
        tactical = tactical and cast(Mapping[str, object], data_cell["tactical"])["passed"] is True
        row["v2_data_effect_mixed_minus_control"] = sum(data_effect[-1]) / len(data_effect[-1])
        per_replicate.append(row)
    c_arch = tuple(architecture["C"])
    m_arch = tuple(architecture["M"])
    intervals: dict[str, object] = {
        "control_architecture_v2_minus_v1": nested_bootstrap(
            c_arch, domain="control-architecture-v2-minus-v1"
        ),
        "mixed_architecture_v2_minus_v1": nested_bootstrap(
            m_arch, domain="mixed-architecture-v2-minus-v1"
        ),
        "pooled_architecture_main_effect": _joint_architecture_bootstrap(
            c_arch, m_arch, interaction=False
        ),
        "v2_data_effect_mixed_minus_control": nested_bootstrap(
            tuple(data_effect), domain="v2-data-effect-mixed-minus-control"
        ),
        "data_x_architecture_interaction": _joint_architecture_bootstrap(
            c_arch, m_arch, interaction=True
        ),
    }
    for arm in ARMS:
        title = ARM_LABELS[arm]
        intervals[f"{arm}_v2_vs_q0_parent"] = nested_bootstrap(
            tuple(parent[arm]), domain=f"{arm}-v2-vs-q0-parent"
        )
        intervals[f"{arm}_v2_vs_random"] = nested_bootstrap(
            tuple(random_rows[arm]), domain=f"{arm}-v2-vs-random"
        )
        intervals[f"{arm}_v2_minus_q0_parent_heuristic"] = nested_bootstrap(
            tuple(heuristic_difference[arm]), domain=f"{arm}-v2-minus-parent-heuristic"
        )
        intervals[f"{arm}_equal_eight_opponent_macro"] = _macro_bootstrap(
            {name: tuple(rows) for name, rows in external[arm].items()}, arm
        )
        del title
    step2_statistics = _read(step2_root / "statistics.json")
    return {
        "version": STATISTICS_VERSION,
        "per_replicate": per_replicate,
        "nested": intervals,
        "minimum_v2_candidate_seat": minimum_seats,
        "tactical_invariants_passed": tactical,
        "step2_v1_data_effect_descriptive_only": cast(
            Mapping[str, object], step2_statistics.get("nested", {})
        ).get("treatment_vs_control"),
        "bootstrap_contract": {
            "root_seed": ROOT_SEED,
            "outer_draws_materialized_first": True,
            "shared_500_inner_indexes_for_architecture_and_interaction": True,
            "different_size_macro_uses_named_inner_streams": True,
        },
    }


def structured_selection(
    statistics: Mapping[str, object], *, integrity_passed: bool
) -> dict[str, object]:
    """Apply arm gates and the frozen robustness-floor Step-4 recipe choice."""
    nested = cast(Mapping[str, object], statistics["nested"])
    reps = cast(list[object], statistics["per_replicate"])
    seats = cast(Mapping[str, object], statistics["minimum_v2_candidate_seat"])
    tactical = statistics.get("tactical_invariants_passed") is True

    def point(name: str) -> float:
        row = nested.get(name)
        if not isinstance(row, Mapping):
            raise StructuredExperimentError("nested statistic is malformed")
        return _number(row.get("point_estimate"), label=f"{name}.point_estimate")

    def bounds(name: str) -> tuple[float, float]:
        row = nested.get(name)
        values = row.get("interval") if isinstance(row, Mapping) else None
        if not isinstance(values, list) or len(values) != 2:
            raise StructuredExperimentError("nested statistic interval is malformed")
        return (
            _number(values[0], label=f"{name}.interval[0]"),
            _number(values[1], label=f"{name}.interval[1]"),
        )

    def lower(name: str) -> float:
        return bounds(name)[0]

    def arm_row(replicate: object, arm: str) -> Mapping[str, object]:
        if not isinstance(replicate, Mapping):
            raise StructuredExperimentError("replicate statistic is malformed")
        values = replicate.get("arms")
        row = values.get(arm) if isinstance(values, Mapping) else None
        if not isinstance(row, Mapping):
            raise StructuredExperimentError("replicate arm statistic is malformed")
        return row

    arms: dict[str, object] = {}
    viable: list[tuple[float, str]] = []
    for arm in ARMS:
        arch = f"{arm.lower() if arm == 'M' else 'control'}_architecture_v2_minus_v1"
        if arm == "M":
            arch = "mixed_architecture_v2_minus_v1"
        conditions = {
            "architecture_lower_strictly_above_50": lower(arch) > 0.5,
            "two_of_three_architecture_replicates_above_50": sum(
                _number(
                    arm_row(row, arm).get("architecture_v2_minus_v1"),
                    label="replicate architecture effect",
                )
                > 0.5
                for row in reps
            )
            >= 2,
            "parent_lower_above_50": lower(f"{arm}_v2_vs_q0_parent") > 0.5,
            "random_lower_above_50": lower(f"{arm}_v2_vs_random") > 0.5,
            "parent_heuristic_difference_lower_above_minus_5pp": lower(
                f"{arm}_v2_minus_q0_parent_heuristic"
            )
            > -0.05,
            "v2_candidate_seat_floor": _number(seats.get(arm), label="minimum v2 seat") >= 0.45,
            "tactical_invariants": tactical,
            "integrity": integrity_passed,
        }
        floor = min(
            lower(arch) - 0.5,
            lower(f"{arm}_v2_vs_q0_parent") - 0.5,
            lower(f"{arm}_v2_vs_random") - 0.5,
            lower(f"{arm}_v2_minus_q0_parent_heuristic") + 0.05,
            _number(seats.get(arm), label="minimum v2 seat") - 0.45,
        )
        passed = all(conditions.values())
        if conditions["tactical_invariants"] and conditions["integrity"]:
            viable.append((floor, arm))
        arms[arm] = {
            "label": ARM_LABELS[arm],
            "conditions": conditions,
            "advancing_structured_recipe": passed,
            "robustness_floor": floor,
            "equal_opponent_macro": point(f"{arm}_equal_eight_opponent_macro"),
            "architecture_point": point(arch),
        }
    advancing = [
        arm
        for arm in ARMS
        if cast(Mapping[str, object], arms[arm])["advancing_structured_recipe"] is True
    ]
    pooled = lower("pooled_architecture_main_effect") > 0.5
    if len(advancing) == 2 and pooled:
        classification = "general_advancement"
    elif len(advancing) == 1:
        classification = "data_dependent_recipe_advance"
    else:
        primary = ("control_architecture_v2_minus_v1", "mixed_architecture_v2_minus_v1")
        crosses = any(lower(key) <= 0.5 <= bounds(key)[1] for key in primary)
        classification = "inconclusive_does_not_advance" if crosses else "does_not_advance"
    selected: str | None = None
    if viable:
        # Max floor, then macro, then direct architecture point, then C prior to M.
        selected = sorted(
            viable,
            key=lambda item: (
                item[0],
                _number(
                    cast(Mapping[str, object], arms[item[1]]).get("equal_opponent_macro"),
                    label="equal opponent macro",
                ),
                _number(
                    cast(Mapping[str, object], arms[item[1]]).get("architecture_point"),
                    label="architecture point",
                ),
                1 if item[1] == "C" else 0,
            ),
            reverse=True,
        )[0][1]
    return {
        "version": SELECTION_VERSION,
        "arm_gates": arms,
        "classification": classification,
        "pooled_architecture_lower_above_50": pooled,
        "development_selected_step4_input_recipe": selected,
        "selection_label": "development-selected input; not advance/promotion"
        if selected is not None
        else "blocked; retain q0 fallback",
        "q0_fallback": selected is None,
    }


def _safety_report(
    panel: Mapping[str, object],
    initialization: Mapping[str, object],
    artifacts: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    source = (
        repository_root() / "src/agent_avenue/encoding/candidate_structured_v2.py"
    ).read_text()
    forbidden = (
        "GameState",
        "apply_action",
        "current_decision",
        "legal_actions",
        "engine.transitions",
    )
    if any(token in source for token in forbidden):
        raise StructuredExperimentError("structured encoder source allowlist audit failed")
    tactical = all(
        cast(Mapping[str, object], artifact["tactical"])["passed"] is True for artifact in artifacts
    )
    return _artifact(
        SAFETY_VERSION,
        encoder_source_allowlist_passed=True,
        forbidden_encoder_tokens=list(forbidden),
        fixture_panel_digest=initialization["panel_digest"],
        q0_embedding=initialization,
        current_hidden_face_down_invariance=(
            "covered by retained hidden-equivalent fixture vectors/logits/actions and encoder tests"
        ),
        no_policy_belief_counterfactual_targets=True,
        all_learned_v1_v2_tactical_invariants_passed=tactical,
    )


def _checksums(output: Path) -> dict[str, object]:
    """Checksum immutable payloads, excluding mutable/cyclic result state."""
    entries: dict[str, str] = {}
    excluded = {"checksums.json", "execution-state.json", "result.json"}
    for path in sorted(output.rglob("*")):
        if (
            path.is_file()
            and path.name not in excluded
            and not path.name.startswith("validation")
            and path.name != ".lock"
        ):
            entries[path.relative_to(output).as_posix()] = _sha256(path)
    return _artifact("m7-structured-model-v2-checksums-v2", files=entries)


def _require_artifact(path: Path, *, label: str) -> dict[str, object]:
    value = _read(path)
    if value.get("artifact_fingerprint") != _fingerprint(
        {key: item for key, item in value.items() if key != "artifact_fingerprint"}
    ):
        raise StructuredExperimentError(f"{label} artifact fingerprint mismatch")
    return value


def validate_completed_structured_result(
    output: Path,
    plan: Mapping[str, object],
    cells: Sequence[StructuredArenaCell],
    *,
    require_execution_complete: bool = True,
) -> dict[str, object]:
    """Refuse reuse until every retained cardinality/reference/checksum boundary is complete."""
    state = _load_execution_state(output / "execution-state.json")
    if require_execution_complete and state.get("completed") is not True:
        raise StructuredExperimentError("existing result has no completed execution state")
    result = _read(output / "result.json")
    if (
        result.get("plan_fingerprint") != plan["plan_fingerprint"]
        or result.get("status") != "completed"
        or result.get("result_fingerprint")
        != _fingerprint(
            {key: value for key, value in result.items() if key != "result_fingerprint"}
        )
    ):
        raise StructuredExperimentError("existing Step-3 result identity is malformed")
    expected_keys = {f"{arm}{index}" for arm in ARMS for index in ("1", "2", "3")}
    datasets = result.get("datasets")
    checkpoints = result.get("checkpoints")
    arenas = result.get("arenas")
    if (
        not isinstance(datasets, Mapping)
        or set(datasets) != expected_keys
        or not isinstance(checkpoints, Mapping)
        or set(checkpoints) != expected_keys
        or not isinstance(arenas, Mapping)
        or set(arenas) != {cell.key for cell in cells}
    ):
        raise StructuredExperimentError("completed result artifact cardinalities are incomplete")
    for key in sorted(expected_keys):
        dataset = _require_artifact(output / "datasets" / key / "audit.json", label="dataset")
        if datasets.get(key) != dataset.get("artifact_fingerprint"):
            raise StructuredExperimentError("result dataset reference differs")
        from agent_avenue.learning import inspect_structured_checkpoint

        inspected = inspect_structured_checkpoint(output / "checkpoints" / key)
        if checkpoints.get(key) != inspected.checkpoint_fingerprint:
            raise StructuredExperimentError("result checkpoint reference differs")
    for cell in cells:
        arena = _require_artifact(output / "arenas" / cell.key / "report.json", label="arena")
        if arenas.get(cell.key) != arena.get("artifact_fingerprint"):
            raise StructuredExperimentError("result arena reference differs")
    statistics = _require_artifact(output / "statistics.json", label="statistics")
    safety = _require_artifact(output / "safety-report.json", label="safety")
    selection = _require_artifact(output / "selection.json", label="selection")
    runtime = _require_artifact(output / "runtime-extrapolation.json", label="runtime")
    checksums = _require_artifact(output / "checksums.json", label="checksums")
    if (
        result.get("statistics_fingerprint") != statistics.get("artifact_fingerprint")
        or result.get("safety_fingerprint") != safety.get("artifact_fingerprint")
        or result.get("selection_fingerprint") != selection.get("artifact_fingerprint")
        or result.get("runtime_fingerprint") != runtime.get("artifact_fingerprint")
        or result.get("checksums_fingerprint") != checksums.get("artifact_fingerprint")
    ):
        raise StructuredExperimentError("result derived-artifact references differ")
    expected_checksums = _checksums(output)
    if checksums != expected_checksums:
        raise StructuredExperimentError("checksum manifest no longer matches completed payload")
    return result


def _write_mutable(path: Path, value: Mapping[str, object]) -> None:
    """Atomically update resumable execution state; all scientific artifacts stay immutable."""
    normalized = cast(dict[str, object], json.loads(_canonical(dict(value))))
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _state_fingerprint(state: Mapping[str, object]) -> str:
    return _fingerprint(
        {key: value for key, value in state.items() if key != "artifact_fingerprint"}
    )


def _load_execution_state(path: Path) -> dict[str, object]:
    state = _read(path)
    if state.get("version") != EXECUTION_STATE_VERSION or state.get(
        "artifact_fingerprint"
    ) != _state_fingerprint(state):
        raise StructuredExperimentError("execution state fingerprint/version is malformed")
    return state


def _save_execution_state(path: Path, state: dict[str, object]) -> None:
    state["artifact_fingerprint"] = _state_fingerprint(state)
    _write_mutable(path, state)


def _begin_execution(output: Path, plan: Mapping[str, object], *, claim_mode: bool) -> Path:
    """Start one recorded execution attempt, permitting exactly one interrupted resume."""
    path = output / "execution-state.json"
    now = time.time()
    if not path.exists():
        state: dict[str, object] = {
            "version": EXECUTION_STATE_VERSION,
            "plan_fingerprint": plan["plan_fingerprint"],
            "claim_mode": claim_mode,
            "started_at_epoch_seconds": now,
            "attempts": [{"attempt": 1, "started_at_epoch_seconds": now}],
            "phase_timings_seconds": {},
            "completed": False,
            "complete_at_epoch_seconds": None,
            "artifact_fingerprint": "",
        }
        _save_execution_state(path, state)
        return path
    state = _load_execution_state(path)
    if (
        state.get("plan_fingerprint") != plan["plan_fingerprint"]
        or state.get("claim_mode") is not claim_mode
    ):
        raise StructuredExperimentError(
            "resume state belongs to a different plan or execution mode"
        )
    if state.get("completed") is True:
        return path
    attempts = state.get("attempts")
    if not isinstance(attempts, list) or len(attempts) > ALLOWED_RESUME_RETRIES + 1:
        raise StructuredExperimentError("execution state has malformed attempt accounting")
    if len(attempts) == ALLOWED_RESUME_RETRIES + 1:
        raise StructuredExperimentError("only one Step-3 resume is authorized")
    attempts.append({"attempt": len(attempts) + 1, "started_at_epoch_seconds": now})
    state["attempts"] = attempts
    _save_execution_state(path, state)
    return path


def _deadline_elapsed(state: Mapping[str, object]) -> float:
    started = state.get("started_at_epoch_seconds")
    if not isinstance(started, int | float):
        raise StructuredExperimentError("execution state start time is malformed")
    return time.time() - float(started)


def _enforce_deadline(path: Path, *, claim_mode: bool, stage: str) -> None:
    if not claim_mode:
        return
    state = _load_execution_state(path)
    elapsed = _deadline_elapsed(state)
    if elapsed >= CLAIM_CUTOFF_SECONDS:
        raise StructuredExperimentError(
            f"claim cutoff of {CLAIM_CUTOFF_SECONDS} seconds reached before {stage}"
        )


@contextmanager
def _timed_stage(path: Path, *, claim_mode: bool, stage: str) -> Iterator[None]:
    """Account every expensive stage before and after dispatch, including interrupted attempts."""
    _enforce_deadline(path, claim_mode=claim_mode, stage=f"{stage}:before")
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        state = _load_execution_state(path)
        timings = state.get("phase_timings_seconds")
        if not isinstance(timings, Mapping):
            raise StructuredExperimentError("execution state timing map is malformed")
        updated = {str(key): _number(value, label="phase timing") for key, value in timings.items()}
        updated[stage] = updated.get(stage, 0.0) + elapsed
        state["phase_timings_seconds"] = updated
        _save_execution_state(path, state)
        _enforce_deadline(path, claim_mode=claim_mode, stage=f"{stage}:after")


def _complete_execution(path: Path) -> dict[str, object]:
    state = _load_execution_state(path)
    state["completed"] = True
    state["complete_at_epoch_seconds"] = time.time()
    _save_execution_state(path, state)
    return state


def _runtime_extrapolation(
    execution: Mapping[str, object], *, smoke_pairs: int | None
) -> dict[str, object]:
    """Use observed phase timings and declared work ratios; never change frozen sample sizes."""
    timings = execution.get("phase_timings_seconds")
    if not isinstance(timings, Mapping):
        raise StructuredExperimentError("execution timing map is malformed")
    observed = {str(key): _number(value, label="phase timing") for key, value in timings.items()}
    if smoke_pairs is None:
        return _artifact(
            RUNTIME_VERSION,
            mode="claim-default",
            observed_phase_seconds=observed,
            projected_full_seconds=sum(observed.values()),
            claim_cutoff_seconds=CLAIM_CUTOFF_SECONDS,
            status="claim-run-observation",
        )
    arena_ratio = 30_000 / max(2 * 60 * smoke_pairs, 1)
    training_ratio = 50.0
    # Bootstrap work is proportional to named inner block draws.  The production suite has
    # regular, joint, and eight-opponent macro streams; smoke invokes those same stream families.
    full_bootstrap_blocks = 2 * 500 + 500 + 2 * 500 + 2 * 200 + 2 * 300 + 2 * 1600 + 2 * 500
    smoke_bootstrap_blocks = 2 + 1 + 2 + 2 + 2 + 16 + 2
    statistics_ratio = full_bootstrap_blocks / smoke_bootstrap_blocks
    projected: dict[str, float] = {}
    for phase, seconds in observed.items():
        if phase.startswith("training:"):
            projected[phase] = seconds * training_ratio
        elif phase.startswith("arena:"):
            projected[phase] = seconds * arena_ratio
        elif phase == "statistics":
            projected[phase] = seconds * statistics_ratio
        elif phase.startswith("validation:"):
            projected[phase] = seconds * arena_ratio
        else:
            projected[phase] = seconds
    total = sum(projected.values())
    return _artifact(
        RUNTIME_VERSION,
        mode="bounded-smoke-linear-extrapolation",
        smoke_pairs=smoke_pairs,
        observed_phase_seconds=observed,
        projected_phase_seconds=projected,
        projected_full_seconds=total,
        projected_full_minutes=total / 60,
        claim_cutoff_seconds=CLAIM_CUTOFF_SECONDS,
        status="fits_claim_cutoff"
        if total < CLAIM_CUTOFF_SECONDS
        else "blocked_exceeds_claim_cutoff",
        no_sample_reduction="all frozen claim quantities retained in projection",
    )


def _claim_config_is_exact(config: StructuredExperimentConfig) -> bool:
    root = repository_root()
    return (
        config.output.resolve() == (root / "runs" / CYCLE_ID).resolve()
        and config.step2_root.resolve() == (root / "runs" / "m7-population-replay-v1").resolve()
        and tuple(path.resolve() for path in config.holdout_roots) == ((root / "runs").resolve(),)
    )


def build_structured_experiment_plan(config: StructuredExperimentConfig) -> dict[str, object]:
    manifest, inputs, frozen_input_audit = verify_step2_inputs(config)
    source = _full_source_identity()
    compatibility = _historical_compatibility(source)
    cells = _cells(config)
    if config.claim_default:
        _check_default_cardinality(cells)
    reasons: list[str] = []
    if not config.claim_default:
        reasons.append("bounded_smoke")
    if (
        source["tracked_tree_clean"] is not True
        or cast(Mapping[str, object], source["runner_clean_check"])[
            "tracked_and_nonignored_untracked_clean"
        ]
        is not True
    ):
        reasons.append("source_not_tracked_clean_including_nonignored_untracked")
    if config.claim_default and not _claim_config_is_exact(config):
        reasons.append("claim_paths_must_be_exact_frozen_defaults")
    plan: dict[str, object] = {
        "version": PLAN_VERSION,
        "cycle_id": CYCLE_ID,
        "root_seed": ROOT_SEED,
        "source": source,
        "historical_compatibility": compatibility,
        "input_manifest_fingerprint": manifest["artifact_fingerprint"],
        "frozen_step2_input_audit": frozen_input_audit,
        "step2_root": str(config.step2_root.resolve()),
        "inputs": [
            {
                "key": item.key,
                "arm": item.arm,
                "replicate_id": item.replicate_id,
                "corpus": dict(item.corpus),
                "v1_dataset": dict(item.dataset),
                "v1_checkpoint": dict(item.checkpoint),
            }
            for item in inputs
        ],
        "q0_path": str(_q0_path(config)),
        "opponent_paths": {key: str(path) for key, path in _q_paths(config).items()},
        "encoder": {**FEATURE_SCHEMA.to_data(), "fingerprint": FEATURE_SCHEMA.fingerprint},
        "execution": {
            "claim_eligible": not reasons,
            "evidence_class": "claim-eligible-default" if not reasons else "smoke-only-nonclaim",
            "ineligibility_reasons": reasons,
            "max_epochs": config.max_epochs,
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
            "total_physical_games": _count_cells(cells),
            "frozen_default_total_physical_games": 30_000,
        },
        "plan_fingerprint": "",
    }
    plan["plan_fingerprint"] = _fingerprint(
        {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    )
    return plan


def run_structured_experiment(config: StructuredExperimentConfig) -> dict[str, object]:
    """Run/resume the complete non-claim or claim Step-3 evidence pipeline."""
    plan = build_structured_experiment_plan(config)
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable(output / "plan.json", plan, label="plan")
    claim_mode = config.claim_default
    execution_data = cast(Mapping[str, object], plan["execution"])
    if claim_mode and execution_data["claim_eligible"] is not True:
        raise StructuredExperimentError("claim dispatch blocked by frozen clean-source preflight")
    cells = _cells(config)
    if (output / "result.json").exists():
        validate_completed_structured_result(output, plan, cells, require_execution_complete=False)
        state_path = output / "execution-state.json"
        if _load_execution_state(state_path).get("completed") is not True:
            _complete_execution(state_path)
        return validate_completed_structured_result(output, plan, cells)
    state_path = _begin_execution(output, plan, claim_mode=claim_mode)
    with _timed_stage(state_path, claim_mode=claim_mode, stage="input-holdout"):
        input_manifest, inputs, audit = verify_step2_inputs(config)
        del input_manifest
        holdout = _holdout_scan(config, cells)
        input_artifact = _artifact(
            INPUT_AUDIT_VERSION,
            plan_fingerprint=plan["plan_fingerprint"],
            frozen_inputs=audit,
            setup_holdout=holdout,
        )
        _write_immutable(
            output / "inputs" / "step2-reference.json", input_artifact, label="Step-2 input audit"
        )
        _write_immutable(
            output / "source-identity.json",
            cast(Mapping[str, object], plan["source"]),
            label="source identity",
        )
    if holdout["status"] != "passed":
        raise StructuredExperimentError("Step-3 setup holdout overlaps prior/training evidence")
    root = _repository_for_step2(config.step2_root)
    inputs_by_key = {item.key: item for item in inputs}
    datasets: dict[str, Path] = {}
    dataset_audits: dict[str, dict[str, object]] = {}
    records_by_key: dict[str, tuple[GameRecord, ...]] = {}
    for item in inputs:
        with _timed_stage(state_path, claim_mode=claim_mode, stage=f"dataset:{item.key}"):
            path, dataset_audit, records = _dataset_audit(output, plan, item, source_root=root)
        datasets[item.key] = path
        dataset_audits[item.key] = dataset_audit
        records_by_key[item.key] = records
    fixture_groups, panel = _feature_panel(records_by_key["C1"])
    initialization = _initialization_audit(
        _q0_path(config),
        fixture_groups,
        panel,
        derive_seed(ROOT_SEED, "step3:v2-structured-init:replicate-1") & ((1 << 63) - 1),
    )
    _write_immutable(
        output / "q0-initialization.json", initialization, label="q0 initialization audit"
    )
    _write_immutable(
        output / "encoder-v2-schema.json",
        _artifact(
            "m7-structured-model-v2-encoder-schema-v1",
            schema=FEATURE_SCHEMA.to_data(),
            schema_fingerprint=FEATURE_SCHEMA.fingerprint,
            golden_fixture_panel_digest=initialization["panel_digest"],
            q0_embedding_fingerprint=initialization["artifact_fingerprint"],
        ),
        label="encoder schema",
    )
    training_rows: list[dict[str, object]] = []
    for replicate in REPLICATES:
        row: dict[str, object] = {"replicate_id": replicate, "arms": {}}
        for arm in ARMS:
            item = inputs_by_key[f"{arm}{replicate.removeprefix('replicate-')}"]
            with _timed_stage(state_path, claim_mode=claim_mode, stage=f"training:{item.key}"):
                report = _train_checkpoint(
                    output,
                    plan,
                    item,
                    datasets[item.key],
                    dataset_audits[item.key],
                    _q0_path(config),
                    max_epochs=config.max_epochs,
                )
            cast(dict[str, object], row["arms"])[arm] = report
        c = cast(Mapping[str, object], cast(Mapping[str, object], row["arms"])["C"])[
            "training_seeds"
        ]
        m = cast(Mapping[str, object], cast(Mapping[str, object], row["arms"])["M"])[
            "training_seeds"
        ]
        if c != m:
            raise StructuredExperimentError(
                "paired C/M structured initialization or shuffle seed differs"
            )
        training_rows.append(row)
    training = _artifact(
        TRAINING_VERSION, plan_fingerprint=plan["plan_fingerprint"], replicates=training_rows
    )
    _write_immutable(output / "training-summary.json", training, label="training summary")
    paths = _q_paths(config)
    artifacts: list[dict[str, object]] = []
    for cell in cells:
        with _timed_stage(state_path, claim_mode=claim_mode, stage=f"arena:{cell.key}"):
            artifacts.append(_arena_artifact(output, plan, cell, inputs_by_key, paths))
    with _timed_stage(state_path, claim_mode=claim_mode, stage="statistics"):
        statistics = structured_statistics(artifacts, step2_root=config.step2_root)
        stats_artifact = _artifact(
            STATISTICS_VERSION,
            plan_fingerprint=plan["plan_fingerprint"],
            **{key: value for key, value in statistics.items() if key != "version"},
        )
        _write_immutable(output / "statistics.json", stats_artifact, label="statistics")
    integrity = holdout["status"] == "passed" and all(
        cast(Mapping[str, object], artifact["tactical"])["passed"] is True for artifact in artifacts
    )
    with _timed_stage(state_path, claim_mode=claim_mode, stage="safety-selection"):
        selection = structured_selection(statistics, integrity_passed=integrity)
        selection_artifact = _artifact(
            SELECTION_VERSION,
            plan_fingerprint=plan["plan_fingerprint"],
            **{key: value for key, value in selection.items() if key != "version"},
        )
        _write_immutable(output / "selection.json", selection_artifact, label="selection")
        safety = _safety_report(panel, initialization, artifacts)
        _write_immutable(output / "safety-report.json", safety, label="safety report")
    runtime = _runtime_extrapolation(
        _load_execution_state(state_path), smoke_pairs=config.smoke_pairs
    )
    _write_immutable(output / "runtime-extrapolation.json", runtime, label="runtime extrapolation")
    with _timed_stage(state_path, claim_mode=claim_mode, stage="checksums"):
        checksums = _checksums(output)
        _write_immutable(output / "checksums.json", checksums, label="checksums")
    checkpoints: dict[str, object] = {}
    for row in training_rows:
        replicate_index = cast(str, row["replicate_id"]).removeprefix("replicate-")
        arms = cast(Mapping[str, object], row["arms"])
        for arm in ARMS:
            checkpoints[f"{arm}{replicate_index}"] = cast(Mapping[str, object], arms[arm])[
                "checkpoint_fingerprint"
            ]
    result: dict[str, object] = {
        "version": RESULT_VERSION,
        "cycle_id": CYCLE_ID,
        "status": "completed",
        "plan_fingerprint": plan["plan_fingerprint"],
        "input_manifest_fingerprint": plan["input_manifest_fingerprint"],
        "datasets": {
            key: dataset_audits[key]["artifact_fingerprint"] for key in sorted(dataset_audits)
        },
        "checkpoints": checkpoints,
        "arenas": {
            cast(str, cast(Mapping[str, object], artifact["cell"])["key"]): artifact[
                "artifact_fingerprint"
            ]
            for artifact in artifacts
        },
        "statistics_fingerprint": stats_artifact["artifact_fingerprint"],
        "safety_fingerprint": safety["artifact_fingerprint"],
        "selection_fingerprint": selection_artifact["artifact_fingerprint"],
        "runtime_fingerprint": runtime["artifact_fingerprint"],
        "checksums_fingerprint": checksums["artifact_fingerprint"],
        "selection": selection,
        "checksum_scope": (
            "all immutable payloads except result/execution-state/checksums/validation"
        ),
        "result_fingerprint": "",
    }
    result["result_fingerprint"] = _fingerprint(
        {key: value for key, value in result.items() if key != "result_fingerprint"}
    )
    with _timed_stage(state_path, claim_mode=claim_mode, stage="result"):
        _write_immutable(output / "result.json", result, label="result")
    _complete_execution(state_path)
    return validate_completed_structured_result(output, plan, cells)


__all__ = [
    "ARMS",
    "CYCLE_ID",
    "NESTED_RESAMPLES",
    "PLAN_VERSION",
    "REPLICATES",
    "ROOT_SEED",
    "StructuredArenaCell",
    "StructuredExperimentConfig",
    "StructuredExperimentError",
    "build_structured_experiment_plan",
    "nested_bootstrap",
    "run_structured_experiment",
    "structured_selection",
    "structured_statistics",
    "verify_step2_inputs",
]
