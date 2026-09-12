"""Immutable, self-verifying checkpoint bundles for structured model v2 only."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import torch
from torch import Tensor

from agent_avenue.encoding.candidate_structured_v2 import (
    ENCODER_FINGERPRINT,
    ENCODER_VERSION,
    FEATURE_NAMES,
    FEATURE_WIDTH,
)
from agent_avenue.storage.fingerprints import code_fingerprint, rules_fingerprint

from .model import CandidateMLP
from .structured_model import (
    MODEL_VERSION,
    StructuredModelError,
    StructuredResidualMLP,
    model_spec,
    validate_state_dict,
)

STRUCTURED_CHECKPOINT_SCHEMA_VERSION: Final[int] = 1
STRUCTURED_CHECKPOINT_ARTIFACT_KIND: Final[str] = "immutable-structured-v2-inference"
CHECKPOINT_FINGERPRINT_ALGORITHM: Final[str] = "sha256-canonical-json-v1"
MANIFEST_FILENAME: Final[str] = "manifest.json"
WEIGHTS_FILENAME: Final[str] = "weights.pt"
METRICS_FILENAME: Final[str] = "metrics.json"
_BUNDLE_FILES: Final[frozenset[str]] = frozenset(
    {MANIFEST_FILENAME, WEIGHTS_FILENAME, METRICS_FILENAME}
)


class StructuredCheckpointError(ValueError):
    """Raised for malformed or unsafe structured checkpoint artifacts."""


class StructuredCheckpointCompatibilityError(StructuredCheckpointError):
    """Raised when a valid bundle targets a different encoder/model/rules contract."""


class StructuredCheckpointIntegrityError(StructuredCheckpointError):
    """Raised when any manifest, file, or canonical tensor digest fails validation."""


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StructuredCheckpointError(
            "checkpoint metadata must be finite canonical JSON"
        ) from exc


def _json_copy(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    return json.loads(_canonical_json(value))


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(tensor: Tensor) -> bytes:
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise StructuredCheckpointError("canonical tensor digest requires dense CPU tensors")
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes(order="C")


def structured_tensor_digest(state_dict: Mapping[str, object]) -> str:
    """Hash state tensor names, dtypes, shapes, and bytes independent of torch serialization."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise StructuredCheckpointError("state dictionary must map string names to tensors")
        metadata = _canonical_json(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "layout": str(value.layout),
            }
        )
        raw = _tensor_bytes(value)
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


# Common term used in reports and tests.
tensor_digest = structured_tensor_digest


@dataclass(frozen=True, slots=True)
class StructuredCheckpointCompatibility:
    encoder_version: str = ENCODER_VERSION
    encoder_fingerprint: str = ENCODER_FINGERPRINT
    feature_names: tuple[str, ...] = FEATURE_NAMES
    input_width: int = FEATURE_WIDTH
    model_version: str = MODEL_VERSION
    rules_fingerprint: str | None = None
    code_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if (
            self.encoder_version != ENCODER_VERSION
            or self.encoder_fingerprint != ENCODER_FINGERPRINT
            or self.feature_names != FEATURE_NAMES
            or self.input_width != FEATURE_WIDTH
            or self.model_version != MODEL_VERSION
        ):
            raise StructuredCheckpointCompatibilityError(
                "compatibility is not candidate-public-structured-v2/model-v2"
            )
        for label, value in (("rules", self.rules_fingerprint), ("code", self.code_fingerprint)):
            if value is not None and not _valid_digest(value):
                raise StructuredCheckpointCompatibilityError(
                    f"{label} fingerprint must be a lowercase SHA-256 digest"
                )


@dataclass(frozen=True, slots=True)
class SavedStructuredCheckpoint:
    path: Path
    checkpoint_fingerprint: str
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LoadedStructuredCheckpoint:
    path: Path
    model: StructuredResidualMLP
    checkpoint_fingerprint: str
    manifest: Mapping[str, object]
    metrics: object


@dataclass(frozen=True, slots=True)
class StructuredCheckpointInspection:
    path: Path
    checkpoint_fingerprint: str
    manifest: Mapping[str, object]
    metrics: object
    tensor_digest: str


def current_structured_compatibility() -> StructuredCheckpointCompatibility:
    """Return the current v2 encoder/model/rules compatibility contract."""
    return StructuredCheckpointCompatibility(rules_fingerprint=rules_fingerprint())


def _checkpoint_identity(manifest: Mapping[str, object]) -> str:
    excluded = {"checkpoint_fingerprint", "created_at", "environment", "files"}
    payload = {key: value for key, value in manifest.items() if key not in excluded}
    metrics = payload.get("metrics_summary")
    if isinstance(metrics, Mapping):
        payload["metrics_summary"] = {
            key: value for key, value in metrics.items() if key != "runtime"
        }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _require_json_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StructuredCheckpointIntegrityError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _validate_sources(
    *,
    q0_parent_checkpoint_fingerprint: str,
    q0_parent_tensor_digest: str,
    dataset_fingerprint: str,
    source_corpus_fingerprints: Sequence[str],
    split_identities: Mapping[str, object],
    input_manifest_fingerprint: str | None,
) -> dict[str, object]:
    for label, value in (
        ("q0_parent_checkpoint_fingerprint", q0_parent_checkpoint_fingerprint),
        ("q0_parent_tensor_digest", q0_parent_tensor_digest),
        ("dataset_fingerprint", dataset_fingerprint),
    ):
        if not _valid_digest(value):
            raise StructuredCheckpointError(f"{label} must be a lowercase SHA-256 digest")
    if not source_corpus_fingerprints or any(
        not _valid_digest(value) for value in source_corpus_fingerprints
    ):
        raise StructuredCheckpointError(
            "source_corpus_fingerprints must contain one or more SHA-256 digests"
        )
    if not split_identities:
        raise StructuredCheckpointError("split_identities are required for a structured checkpoint")
    copied_split = _json_copy(dict(split_identities))
    if not isinstance(copied_split, dict):  # pragma: no cover - dict input guarantees this
        raise StructuredCheckpointError("split_identities must be canonical JSON")
    if input_manifest_fingerprint is not None and not _valid_digest(input_manifest_fingerprint):
        raise StructuredCheckpointError("input_manifest_fingerprint must be a SHA-256 digest")
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "corpus_fingerprints": list(source_corpus_fingerprints),
        "split_identities": copied_split,
        "input_manifest_fingerprint": input_manifest_fingerprint,
        "q0_parent": {
            "checkpoint_fingerprint": q0_parent_checkpoint_fingerprint,
            "tensor_digest": q0_parent_tensor_digest,
        },
    }


def save_structured_checkpoint(
    path: str | Path,
    model: StructuredResidualMLP,
    *,
    metrics: object,
    q0_parent_checkpoint_fingerprint: str,
    q0_parent_tensor_digest: str,
    dataset_fingerprint: str,
    source_corpus_fingerprints: Sequence[str] = (),
    split_identities: Mapping[str, object],
    training_config: object,
    training_seeds: Mapping[str, object],
    input_manifest_fingerprint: str | None = None,
    rules_compatibility_fingerprint: str | None = None,
    code_compatibility_fingerprint: str | None = None,
    source_rules_fingerprint: str | None = None,
    metadata: Mapping[str, object] | None = None,
    created_at: datetime | str | None = None,
) -> SavedStructuredCheckpoint:
    """Atomically save a v2-only checkpoint with immutable q0/data/split lineage."""
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise StructuredCheckpointError("structured checkpoint destination already exists")
    if next(model.parameters()).device.type != "cpu":
        raise StructuredCheckpointError("structured inference checkpoints must be CPU models")
    try:
        validate_state_dict(model.state_dict())
    except StructuredModelError as exc:
        raise StructuredCheckpointError("model is incompatible with structured v2") from exc
    rules_value = rules_compatibility_fingerprint or rules_fingerprint()
    code_value = code_compatibility_fingerprint or code_fingerprint()
    if not _valid_digest(rules_value) or not _valid_digest(code_value):
        raise StructuredCheckpointError(
            "rules/code compatibility fingerprints must be SHA-256 digests"
        )
    if source_rules_fingerprint is not None and not _valid_digest(source_rules_fingerprint):
        raise StructuredCheckpointError("source_rules_fingerprint must be a SHA-256 digest")
    sources = _validate_sources(
        q0_parent_checkpoint_fingerprint=q0_parent_checkpoint_fingerprint,
        q0_parent_tensor_digest=q0_parent_tensor_digest,
        dataset_fingerprint=dataset_fingerprint,
        source_corpus_fingerprints=source_corpus_fingerprints,
        split_identities=split_identities,
        input_manifest_fingerprint=input_manifest_fingerprint,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        state = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        }
        weights_path = temporary / WEIGHTS_FILENAME
        metrics_path = temporary / METRICS_FILENAME
        torch.save(state, weights_path)
        metrics_data = _json_copy(metrics)
        metrics_path.write_bytes(_canonical_json(metrics_data) + b"\n")
        if isinstance(created_at, datetime):
            created = created_at.astimezone(UTC).isoformat()
        elif isinstance(created_at, str):
            created = created_at
        elif created_at is None:
            created = datetime.now(UTC).isoformat()
        else:
            raise StructuredCheckpointError("created_at must be a datetime, string, or None")
        manifest: dict[str, object] = {
            "schema_version": STRUCTURED_CHECKPOINT_SCHEMA_VERSION,
            "artifact_kind": STRUCTURED_CHECKPOINT_ARTIFACT_KIND,
            "fingerprint_algorithm": CHECKPOINT_FINGERPRINT_ALGORITHM,
            "created_at": created,
            "model": model_spec(),
            "encoder": {
                "version": ENCODER_VERSION,
                "fingerprint": ENCODER_FINGERPRINT,
                "width": FEATURE_WIDTH,
                "feature_names": list(FEATURE_NAMES),
            },
            "compatibility": {
                "rules_fingerprint": rules_value,
                "code_fingerprint": code_value,
                "source_rules_fingerprint": source_rules_fingerprint,
            },
            "sources": sources,
            "training": {
                "config": _json_copy(training_config),
                "seeds": _json_copy(training_seeds),
                "deterministic_algorithms": True,
                "device": "cpu",
            },
            "metrics_summary": metrics_data,
            "metadata": _json_copy(metadata if metadata is not None else {}),
            "tensor_digest": structured_tensor_digest(state),
            "files": {
                WEIGHTS_FILENAME: {
                    "sha256": _sha256_file(weights_path),
                    "size": weights_path.stat().st_size,
                },
                METRICS_FILENAME: {
                    "sha256": _sha256_file(metrics_path),
                    "size": metrics_path.stat().st_size,
                },
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": "cpu",
                "cpu_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        }
        manifest["checkpoint_fingerprint"] = _checkpoint_identity(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(temporary, destination)
        return SavedStructuredCheckpoint(
            destination,
            str(manifest["checkpoint_fingerprint"]),
            cast(Mapping[str, object], _freeze(_json_copy(manifest))),
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _read_json(path: Path, label: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise StructuredCheckpointIntegrityError(f"{label} must be a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuredCheckpointIntegrityError(f"invalid {label}") from exc


def _manifest(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_dir():
        raise StructuredCheckpointIntegrityError(
            "structured checkpoint path must be a regular directory"
        )
    if {item.name for item in path.iterdir()} != _BUNDLE_FILES or any(
        item.is_symlink() for item in path.iterdir()
    ):
        raise StructuredCheckpointIntegrityError(
            "structured checkpoint bundle has unexpected files"
        )
    manifest = _require_json_mapping(_read_json(path / MANIFEST_FILENAME, "manifest"), "manifest")
    if manifest.get("schema_version") != STRUCTURED_CHECKPOINT_SCHEMA_VERSION:
        raise StructuredCheckpointCompatibilityError(
            "unsupported structured checkpoint schema version"
        )
    if manifest.get("artifact_kind") != STRUCTURED_CHECKPOINT_ARTIFACT_KIND:
        raise StructuredCheckpointCompatibilityError(
            "checkpoint is not a structured-v2 immutable artifact"
        )
    if manifest.get("fingerprint_algorithm") != CHECKPOINT_FINGERPRINT_ALGORITHM:
        raise StructuredCheckpointCompatibilityError(
            "unsupported structured checkpoint fingerprint algorithm"
        )
    if not _valid_digest(manifest.get("checkpoint_fingerprint")) or manifest.get(
        "checkpoint_fingerprint"
    ) != _checkpoint_identity(manifest):
        raise StructuredCheckpointIntegrityError("structured checkpoint fingerprint mismatch")
    return manifest


def _validate_manifest_contract(
    manifest: Mapping[str, object], compatibility: StructuredCheckpointCompatibility
) -> None:
    model = _require_json_mapping(manifest.get("model"), "model")
    if model != model_spec():
        raise StructuredCheckpointCompatibilityError("structured model specification mismatch")
    encoder = _require_json_mapping(manifest.get("encoder"), "encoder")
    expected_encoder = {
        "version": compatibility.encoder_version,
        "fingerprint": compatibility.encoder_fingerprint,
        "width": compatibility.input_width,
        "feature_names": list(compatibility.feature_names),
    }
    if dict(encoder) != expected_encoder:
        raise StructuredCheckpointCompatibilityError(
            "structured encoder version/fingerprint/names mismatch"
        )
    declared = _require_json_mapping(manifest.get("compatibility"), "compatibility")
    for field in ("rules_fingerprint", "code_fingerprint"):
        if not _valid_digest(declared.get(field)):
            raise StructuredCheckpointCompatibilityError(f"structured {field} is invalid")
    if (
        compatibility.rules_fingerprint is not None
        and declared.get("rules_fingerprint") != compatibility.rules_fingerprint
    ):
        raise StructuredCheckpointCompatibilityError("rules fingerprint mismatch")
    if (
        compatibility.code_fingerprint is not None
        and declared.get("code_fingerprint") != compatibility.code_fingerprint
    ):
        raise StructuredCheckpointCompatibilityError("code fingerprint mismatch")
    sources = _require_json_mapping(manifest.get("sources"), "sources")
    q0 = _require_json_mapping(sources.get("q0_parent"), "sources.q0_parent")
    if not (
        _valid_digest(q0.get("checkpoint_fingerprint"))
        and _valid_digest(q0.get("tensor_digest"))
        and _valid_digest(sources.get("dataset_fingerprint"))
        and isinstance(sources.get("corpus_fingerprints"), list)
        and all(
            _valid_digest(value) for value in cast(list[object], sources["corpus_fingerprints"])
        )
        and isinstance(sources.get("split_identities"), Mapping)
        and bool(sources["split_identities"])
    ):
        raise StructuredCheckpointCompatibilityError(
            "structured checkpoint q0/data/split lineage is invalid"
        )


def _validate_files(path: Path, manifest: Mapping[str, object]) -> object:
    files = _require_json_mapping(manifest.get("files"), "files")
    if set(files) != {WEIGHTS_FILENAME, METRICS_FILENAME}:
        raise StructuredCheckpointIntegrityError("structured checkpoint file list mismatch")
    for filename in (WEIGHTS_FILENAME, METRICS_FILENAME):
        file_data = _require_json_mapping(files.get(filename), f"files.{filename}")
        artifact = path / filename
        if (
            not _valid_digest(file_data.get("sha256"))
            or file_data.get("sha256") != _sha256_file(artifact)
            or type(file_data.get("size")) is not int
            or file_data.get("size") != artifact.stat().st_size
        ):
            raise StructuredCheckpointIntegrityError(f"{filename} digest or size mismatch")
    metrics = _read_json(path / METRICS_FILENAME, "metrics")
    if _json_copy(metrics) != manifest.get("metrics_summary"):
        raise StructuredCheckpointIntegrityError("structured metrics summary mismatch")
    return metrics


def _load_state(path: Path, manifest: Mapping[str, object]) -> Mapping[str, Tensor]:
    try:
        state = torch.load(path / WEIGHTS_FILENAME, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise StructuredCheckpointIntegrityError("safe structured weight loading failed") from exc
    if not isinstance(state, Mapping):
        raise StructuredCheckpointIntegrityError("structured weights must be a state dictionary")
    try:
        validate_state_dict(state)
        digest = structured_tensor_digest(state)
    except (StructuredModelError, StructuredCheckpointError) as exc:
        raise StructuredCheckpointIntegrityError("invalid structured model state") from exc
    if manifest.get("tensor_digest") != digest:
        raise StructuredCheckpointIntegrityError("structured canonical tensor digest mismatch")
    return cast(Mapping[str, Tensor], state)


def inspect_structured_checkpoint(
    path: str | Path,
    *,
    compatibility: StructuredCheckpointCompatibility | None = None,
) -> StructuredCheckpointInspection:
    """Fully verify a v2 bundle without exposing a mutable inference model."""
    compatibility = compatibility or current_structured_compatibility()
    checkpoint_path = Path(path)
    manifest = _manifest(checkpoint_path)
    _validate_manifest_contract(manifest, compatibility)
    metrics = _validate_files(checkpoint_path, manifest)
    state = _load_state(checkpoint_path, manifest)
    return StructuredCheckpointInspection(
        checkpoint_path,
        str(manifest["checkpoint_fingerprint"]),
        cast(Mapping[str, object], _freeze(_json_copy(manifest))),
        _freeze(_json_copy(metrics)),
        structured_tensor_digest(state),
    )


def load_structured_checkpoint(
    path: str | Path,
    *,
    compatibility: StructuredCheckpointCompatibility | None = None,
) -> LoadedStructuredCheckpoint:
    """Safely load a v2-only CPU inference model, loudly rejecting v1 bundles."""
    compatibility = compatibility or current_structured_compatibility()
    checkpoint_path = Path(path)
    manifest = _manifest(checkpoint_path)
    _validate_manifest_contract(manifest, compatibility)
    metrics = _validate_files(checkpoint_path, manifest)
    state = _load_state(checkpoint_path, manifest)
    model = StructuredResidualMLP(CandidateMLP(seed=0).state_dict(), projection_seed=0)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return LoadedStructuredCheckpoint(
        checkpoint_path,
        model,
        str(manifest["checkpoint_fingerprint"]),
        cast(Mapping[str, object], _freeze(_json_copy(manifest))),
        _freeze(_json_copy(metrics)),
    )


__all__ = [
    "CHECKPOINT_FINGERPRINT_ALGORITHM",
    "STRUCTURED_CHECKPOINT_ARTIFACT_KIND",
    "STRUCTURED_CHECKPOINT_SCHEMA_VERSION",
    "LoadedStructuredCheckpoint",
    "SavedStructuredCheckpoint",
    "StructuredCheckpointCompatibility",
    "StructuredCheckpointCompatibilityError",
    "StructuredCheckpointError",
    "StructuredCheckpointInspection",
    "StructuredCheckpointIntegrityError",
    "current_structured_compatibility",
    "inspect_structured_checkpoint",
    "load_structured_checkpoint",
    "save_structured_checkpoint",
    "structured_tensor_digest",
    "tensor_digest",
]
